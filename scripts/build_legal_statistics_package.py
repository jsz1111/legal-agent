"""Build a PostgreSQL-ready offline package from China Law Yearbook XLS files.

The source workbooks use legacy encrypted XLS containers that xlrd cannot open.
This builder reads them through a hidden, read-only Excel COM session and never
connects to PostgreSQL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pythoncom
import win32com.client


YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")
UNIT_RE = re.compile(r"单位\s*[:：]\s*([^）)\n]+)")
NOTE_RE = re.compile(r"^(?:注|备注|说明)\s*[:：]")
SEQUENCE_HEADERS = {"序号", "编号", "序列"}


DATASET_FIELDS = [
    "dataset_id", "yearbook_year", "statistical_year_start",
    "statistical_year_end", "title", "institution", "topic", "unit",
    "source_file", "source_sheet", "source_url", "source_sha256",
    "source_rows", "source_columns", "data_start_row", "year_quality",
    "extraction_status", "notes",
]
RECORD_FIELDS = [
    "record_id", "dataset_id", "source_row", "row_label", "row_data",
    "search_text",
]
FACT_FIELDS = [
    "fact_id", "dataset_id", "record_id", "yearbook_year",
    "statistical_year_start", "statistical_year_end", "institution", "topic",
    "dimension_label", "metric", "metric_path", "numeric_value", "text_value",
    "value_kind", "unit", "source_row", "source_column", "quality_flag",
]
CELL_FIELDS = [
    "dataset_id", "source_row", "source_column", "cell_text", "resolved_text",
    "cell_type", "is_merged", "merged_range",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_institution(title: str) -> str:
    rules = [
        ("检察", "人民检察院"),
        ("法院", "人民法院"),
        ("公安", "公安机关"),
        ("交通事故", "公安机关"),
        ("行政复议", "行政复议机关"),
        ("行政应诉", "行政机关"),
        ("民政", "民政部门"),
        ("条约", "外交与国际法"),
        ("司法解释", "司法机关"),
    ]
    return next((value for keyword, value in rules if keyword in title), "其他")


def classify_topic(title: str) -> str:
    rules = [
        ("婚姻家庭", "marriage_family"),
        ("民事", "civil_cases"),
        ("刑事", "criminal_cases"),
        ("行政复议", "administrative_reconsideration"),
        ("行政应诉", "administrative_litigation_response"),
        ("行政一审", "administrative_cases"),
        ("行政案件", "administrative_cases"),
        ("执行案件", "enforcement"),
        ("交通事故", "traffic_accidents"),
        ("检察", "procuratorate"),
        ("公安", "public_security"),
        ("民政", "civil_affairs"),
        ("社会救助", "social_assistance"),
        ("条约", "treaties"),
        ("废止", "abolished_documents"),
        ("法院", "court_statistics"),
    ]
    return next((value for keyword, value in rules if keyword in title), "other")


def extract_period(title: str, yearbook_year: int) -> tuple[int, int, str]:
    years = [int(value) for value in YEAR_RE.findall(title)]
    years = [value for value in years if 1980 <= value <= yearbook_year]
    if years:
        return min(years), max(years), "explicit_title"
    inferred = yearbook_year - 1
    return inferred, inferred, "inferred_yearbook_minus_one"


def extract_unit(rows: list[list[dict[str, Any]]]) -> str:
    for row in rows[:8]:
        text = " ".join(cell["resolved_text"] for cell in row if cell["resolved_text"])
        match = UNIT_RE.search(text)
        if match:
            return clean_text(match.group(1))
    return ""


def is_number_text(text: str) -> bool:
    compact = text.replace(" ", "")
    if compact.endswith("?"):
        compact = compact[:-1]
    return bool(NUMBER_RE.fullmatch(compact))


def parse_number(
    cell: dict[str, Any], metric: str, dimension_label: str
) -> tuple[str, str, bool] | None:
    text = clean_text(cell["cell_text"])
    compact = text.replace(" ", "")
    recovered_percent = False
    if compact.endswith("?") and any(
        keyword in f"{metric}/{dimension_label}"
        for keyword in ("百分比", "比重", "占比", "率")
    ):
        compact = compact[:-1] + "%"
        recovered_percent = True
    if not compact or not NUMBER_RE.fullmatch(compact):
        return None
    if metric in SEQUENCE_HEADERS or metric.endswith("/序号"):
        return None
    if compact.count(".") > 1:
        return None
    value_kind = "percentage" if compact.endswith("%") or any(
        keyword in metric for keyword in ("百分比", "比重", "占比", "率")
    ) else "number"
    numeric = compact.rstrip("%").replace(",", "")
    try:
        parsed = float(numeric)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer():
        numeric = str(int(parsed))
    else:
        numeric = format(parsed, ".15g")
    return numeric, value_kind, recovered_percent


def unique_join(values: list[str], separator: str = "/") -> str:
    result: list[str] = []
    for value in values:
        value = clean_text(value)
        if value and value not in result:
            result.append(value)
    return separator.join(result)


def find_title_row(rows: list[list[dict[str, Any]]]) -> int:
    for index, row in enumerate(rows):
        if any(cell["resolved_text"] for cell in row):
            return index
    return 0


def row_counts(row: list[dict[str, Any]]) -> tuple[int, int]:
    numeric = sum(is_number_text(cell["cell_text"]) for cell in row)
    textual = sum(
        bool(cell["cell_text"]) and not is_number_text(cell["cell_text"])
        for cell in row
    )
    return numeric, textual


def find_data_start(rows: list[list[dict[str, Any]]], title_row: int) -> int | None:
    for index in range(title_row + 1, len(rows)):
        raw_texts = [cell["cell_text"] for cell in rows[index]]
        joined = " ".join(value for value in raw_texts if value)
        if not joined or NOTE_RE.match(joined):
            continue
        numeric, textual = row_counts(rows[index])
        if numeric >= 1 and textual >= 1:
            return index
    return None


def build_headers(
    rows: list[list[dict[str, Any]]],
    title_row: int,
    data_start: int | None,
    title: str,
) -> list[str]:
    width = max((len(row) for row in rows), default=0)
    if data_start is None:
        header_rows = rows[title_row + 1 : title_row + 3]
    else:
        header_rows = rows[title_row + 1 : data_start]
    headers: list[str] = []
    for column in range(width):
        values = []
        for row in header_rows:
            if column >= len(row):
                continue
            value = row[column]["resolved_text"]
            if not value or UNIT_RE.search(value) or value.startswith("截至"):
                continue
            if re.fullmatch(r"表\s*\d+(?:-\d+)?", value):
                continue
            if len(value) >= 8 and value in title:
                continue
            values.append(value)
        headers.append(unique_join(values) or f"column_{column + 1}")
    return headers


def deduplicate_headers(headers: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    result: list[str] = []
    for header in headers:
        counts[header] += 1
        suffix = f"__{counts[header]}" if counts[header] > 1 else ""
        result.append(f"{header}{suffix}")
    return result


def closest_dimension(row: list[dict[str, Any]], column: int) -> str:
    for index in range(column - 1, -1, -1):
        text = row[index]["cell_text"]
        if text and not is_number_text(text) and text not in {"-", "—"}:
            return text
    return ""


def infer_fact_unit(metric: str, table_unit: str, value_kind: str) -> str:
    if value_kind == "percentage":
        return "%"
    parenthetical = re.findall(r"[（(]([^）)]+)[）)]", metric)
    if parenthetical:
        return clean_text(parenthetical[-1])
    return table_unit


def read_sheet(ws: Any) -> list[list[dict[str, Any]]]:
    used = ws.UsedRange
    start_row = int(used.Row)
    start_column = int(used.Column)
    row_count = int(used.Rows.Count)
    column_count = int(used.Columns.Count)
    rows: list[list[dict[str, Any]]] = []
    for row_offset in range(row_count):
        row: list[dict[str, Any]] = []
        for column_offset in range(column_count):
            cell = ws.Cells(start_row + row_offset, start_column + column_offset)
            raw_text = clean_text(cell.Text)
            is_merged = bool(cell.MergeCells)
            merged_range = ""
            resolved_text = raw_text
            if is_merged:
                merge_area = cell.MergeArea
                merged_range = clean_text(merge_area.Address)
                resolved_text = clean_text(merge_area.Cells(1, 1).Text)
            raw_value = cell.Value2
            if isinstance(raw_value, bool):
                cell_type = "boolean"
            elif isinstance(raw_value, (int, float)):
                cell_type = "number"
            elif raw_value is None:
                cell_type = "blank"
            else:
                cell_type = "text"
            row.append(
                {
                    "cell_text": raw_text,
                    "resolved_text": resolved_text,
                    "cell_type": cell_type,
                    "is_merged": is_merged,
                    "merged_range": merged_range,
                }
            )
        rows.append(row)
    return rows


def extract_dataset(
    path: Path,
    relative_path: str,
    yearbook_year: int,
    source_sha256: str,
    source_url: str,
    sheet: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_sheet(sheet)
    title_row = find_title_row(rows)
    worksheet_title = unique_join(
        [cell["resolved_text"] for cell in rows[title_row] if cell["resolved_text"]],
        " ",
    )
    title = clean_text(path.stem)
    start_year, end_year, year_quality = extract_period(title, yearbook_year)
    data_start = find_data_start(rows, title_row)
    headers = build_headers(rows, title_row, data_start, title)
    unique_headers = deduplicate_headers(headers)
    unit = extract_unit(rows)
    dataset_id = stable_id("dataset", relative_path, sheet.Name)
    institution = classify_institution(title)
    topic = classify_topic(title)

    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, 1):
        for column_index, cell in enumerate(row, 1):
            cells.append(
                {
                    "dataset_id": dataset_id,
                    "source_row": row_index,
                    "source_column": column_index,
                    **cell,
                }
            )

    records: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    if data_start is not None:
        for row_index in range(data_start, len(rows)):
            row = rows[row_index]
            texts = [cell["cell_text"] for cell in row]
            joined = " ".join(value for value in texts if value)
            if not joined or NOTE_RE.match(joined):
                continue

            row_data = {
                unique_headers[column]: texts[column]
                for column in range(min(len(unique_headers), len(texts)))
                if texts[column]
            }
            if not row_data:
                continue
            row_label = next(
                (
                    value for value in texts
                    if value and not is_number_text(value) and value not in {"-", "—"}
                ),
                "",
            )
            record_id = stable_id("record", dataset_id, row_index + 1)
            search_text = "；".join(
                [
                    title,
                    f"统计期间：{start_year}-{end_year}",
                    *[f"{key}：{value}" for key, value in row_data.items()],
                ]
            )
            records.append(
                {
                    "record_id": record_id,
                    "dataset_id": dataset_id,
                    "source_row": row_index + 1,
                    "row_label": row_label,
                    "row_data": json.dumps(row_data, ensure_ascii=False, separators=(",", ":")),
                    "search_text": search_text,
                }
            )

            for column, cell in enumerate(row):
                metric_path = headers[column] if column < len(headers) else f"column_{column + 1}"
                dimension_label = closest_dimension(row, column)
                parsed = parse_number(cell, metric_path, dimension_label)
                if parsed is None:
                    continue
                numeric_value, value_kind, recovered_percent = parsed
                quality_flag = (
                    "review_needed"
                    if metric_path.startswith("column_")
                    else "auto_extracted"
                )
                if year_quality != "explicit_title":
                    quality_flag += ";year_inferred"
                if recovered_percent:
                    quality_flag += ";percent_symbol_recovered"
                facts.append(
                    {
                        "fact_id": stable_id("fact", dataset_id, row_index + 1, column + 1),
                        "dataset_id": dataset_id,
                        "record_id": record_id,
                        "yearbook_year": yearbook_year,
                        "statistical_year_start": start_year,
                        "statistical_year_end": end_year,
                        "institution": institution,
                        "topic": topic,
                        "dimension_label": dimension_label,
                        "metric": metric_path.split("/")[-1],
                        "metric_path": metric_path,
                        "numeric_value": numeric_value,
                        "text_value": cell["cell_text"],
                        "value_kind": value_kind,
                        "unit": infer_fact_unit(metric_path, unit, value_kind),
                        "source_row": row_index + 1,
                        "source_column": column + 1,
                        "quality_flag": quality_flag,
                    }
                )

    dataset = {
        "dataset_id": dataset_id,
        "yearbook_year": yearbook_year,
        "statistical_year_start": start_year,
        "statistical_year_end": end_year,
        "title": title,
        "institution": institution,
        "topic": topic,
        "unit": unit,
        "source_file": relative_path,
        "source_sheet": clean_text(sheet.Name),
        "source_url": source_url,
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "source_columns": max((len(row) for row in rows), default=0),
        "data_start_row": (data_start + 1) if data_start is not None else "",
        "year_quality": year_quality,
        "extraction_status": "facts_extracted" if facts else "records_only",
        "notes": (
            f"worksheet_title={worksheet_title}"
            if worksheet_title and worksheet_title not in title
            else ""
        ),
    }
    return dataset, records, facts, cells


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            serializable = dict(row)
            if "row_data" in serializable and isinstance(serializable["row_data"], str):
                serializable["row_data"] = json.loads(serializable["row_data"])
            target.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def build_package(source_root: Path, output_dir: Path) -> dict[str, Any]:
    workbook_paths = sorted(source_root.rglob("*.xls"))
    if not workbook_paths:
        raise FileNotFoundError(f"No XLS workbooks found under {source_root}")

    datasets: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    pythoncom.CoInitialize()
    excel = win32com.client.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    excel.AskToUpdateLinks = False
    try:
        for index, path in enumerate(workbook_paths, 1):
            relative = path.relative_to(source_root).as_posix()
            year_match = YEAR_RE.search(path.parent.name)
            if not year_match:
                errors.append({"file": relative, "error": "yearbook year not found"})
                continue
            yearbook_year = int(year_match.group(1))
            source_sha256 = file_sha256(path)
            workbook = None
            try:
                workbook = excel.Workbooks.Open(
                    str(path),
                    UpdateLinks=0,
                    ReadOnly=True,
                    IgnoreReadOnlyRecommended=True,
                    AddToMru=False,
                )
                source_url = ""
                for worksheet in workbook.Worksheets:
                    if clean_text(worksheet.Name).upper() == "CNKI":
                        source_url = clean_text(worksheet.Cells(1, 1).Text)
                        break
                for worksheet in workbook.Worksheets:
                    if clean_text(worksheet.Name).upper() == "CNKI":
                        continue
                    dataset, sheet_records, sheet_facts, sheet_cells = extract_dataset(
                        path,
                        relative,
                        yearbook_year,
                        source_sha256,
                        source_url,
                        worksheet,
                    )
                    datasets.append(dataset)
                    records.extend(sheet_records)
                    facts.extend(sheet_facts)
                    cells.extend(sheet_cells)
            except Exception as exc:  # continue so the manifest records every failure
                errors.append({"file": relative, "error": clean_text(exc)})
            finally:
                if workbook is not None:
                    workbook.Close(False)
            if index % 10 == 0 or index == len(workbook_paths):
                print(f"[INFO] {index}/{len(workbook_paths)} workbooks processed")
    finally:
        excel.Quit()
        pythoncom.CoUninitialize()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    jsonl_dir = output_dir / "jsonl"
    csv_dir.mkdir(exist_ok=True)
    jsonl_dir.mkdir(exist_ok=True)

    write_csv(csv_dir / "datasets.csv", DATASET_FIELDS, datasets)
    write_csv(csv_dir / "records.csv", RECORD_FIELDS, records)
    write_csv(csv_dir / "facts.csv", FACT_FIELDS, facts)
    write_csv(csv_dir / "cells.csv", CELL_FIELDS, cells)
    write_jsonl(jsonl_dir / "datasets.jsonl", datasets)
    write_jsonl(jsonl_dir / "records.jsonl", records)
    write_jsonl(jsonl_dir / "facts.jsonl", facts)

    project_root = Path(__file__).resolve().parent.parent
    template_dir = project_root / "database" / "legal_statistics"
    shutil.copy2(template_dir / "create_database.sql", output_dir / "create_database.sql")
    shutil.copy2(template_dir / "schema.sql", output_dir / "schema.sql")
    shutil.copy2(template_dir / "README.md", output_dir / "README.md")
    shutil.copy2(
        project_root / "scripts" / "load_legal_statistics_postgres.py",
        output_dir / "load_postgres.py",
    )
    shutil.copy2(
        project_root / "scripts" / "validate_legal_statistics_package.py",
        output_dir / "validate_package.py",
    )

    manifest = {
        "package_name": "china_law_yearbook_statistics_2019_2021",
        "target_database": "legal_statistics_db",
        "target_schema": "legal_statistics",
        "source_root": str(source_root),
        "yearbook_years": sorted({row["yearbook_year"] for row in datasets}),
        "statistical_year_min": min(
            (row["statistical_year_start"] for row in datasets), default=None
        ),
        "statistical_year_max": max(
            (row["statistical_year_end"] for row in datasets), default=None
        ),
        "counts": {
            "workbooks": len(workbook_paths),
            "datasets": len(datasets),
            "records": len(records),
            "facts": len(facts),
            "cells": len(cells),
        },
        "extraction_status_counts": dict(
            sorted(Counter(row["extraction_status"] for row in datasets).items())
        ),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in datasets).items())),
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_package(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
