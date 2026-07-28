"""Validate a generated legal-statistics package without PostgreSQL."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def validate(package_dir: Path) -> dict:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    datasets = read_csv(package_dir / "csv" / "datasets.csv")
    records = read_csv(package_dir / "csv" / "records.csv")
    facts = read_csv(package_dir / "csv" / "facts.csv")
    cells = read_csv(package_dir / "csv" / "cells.csv")

    errors: list[str] = []
    warnings: list[str] = []
    tables = {
        "datasets": datasets,
        "records": records,
        "facts": facts,
        "cells": cells,
    }
    for table_name, rows in tables.items():
        expected = int(manifest["counts"][table_name])
        if len(rows) != expected:
            errors.append(f"{table_name}: expected {expected}, found {len(rows)}")

    dataset_ids = {row["dataset_id"] for row in datasets}
    record_ids = {row["record_id"] for row in records}
    if len(dataset_ids) != len(datasets):
        errors.append("datasets contains duplicate dataset_id values")
    if len(record_ids) != len(records):
        errors.append("records contains duplicate record_id values")

    dataset_by_id = {row["dataset_id"]: row for row in datasets}
    record_by_id = {row["record_id"]: row for row in records}
    for row in records:
        if row["dataset_id"] not in dataset_ids:
            errors.append(f"record {row['record_id']} references a missing dataset")
        try:
            json.loads(row["row_data"])
        except json.JSONDecodeError:
            errors.append(f"record {row['record_id']} has invalid row_data JSON")

    for row in facts:
        if row["dataset_id"] not in dataset_ids:
            errors.append(f"fact {row['fact_id']} references a missing dataset")
        if row["record_id"] not in record_ids:
            errors.append(f"fact {row['fact_id']} references a missing record")
        try:
            Decimal(row["numeric_value"])
        except InvalidOperation:
            errors.append(f"fact {row['fact_id']} has an invalid numeric value")
        dataset = dataset_by_id.get(row["dataset_id"])
        if dataset:
            if int(row["source_row"]) > int(dataset["source_rows"]):
                errors.append(f"fact {row['fact_id']} source_row is out of range")
            if int(row["source_column"]) > int(dataset["source_columns"]):
                errors.append(f"fact {row['fact_id']} source_column is out of range")

    for row in cells:
        dataset = dataset_by_id.get(row["dataset_id"])
        if dataset is None:
            errors.append("cell references a missing dataset")
            continue
        if int(row["source_row"]) > int(dataset["source_rows"]):
            errors.append("cell source_row is out of range")
        if int(row["source_column"]) > int(dataset["source_columns"]):
            errors.append("cell source_column is out of range")

    facts_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for fact in facts:
        facts_by_record[fact["record_id"]].append(fact)
    sample = []
    for record in records:
        dataset = dataset_by_id[record["dataset_id"]]
        if "民事一审" not in compact(dataset["title"]):
            continue
        if "劳动争议" not in compact(record["row_label"]):
            continue
        metrics = {
            compact(fact["metric"]): fact["numeric_value"]
            for fact in facts_by_record[record["record_id"]]
        }
        sample.append(
            {
                "statistical_year": int(dataset["statistical_year_end"]),
                "title": dataset["title"],
                "row_label": record["row_label"],
                "received": metrics.get("收案"),
                "closed": metrics.get("结案"),
                "source_file": dataset["source_file"],
                "source_row": int(record["source_row"]),
            }
        )
    sample.sort(key=lambda row: row["statistical_year"])
    if len(sample) < 3:
        warnings.append("cross-year civil first-instance labor-dispute sample is incomplete")

    report = {
        "status": "passed" if not errors else "failed",
        "counts": {name: len(rows) for name, rows in tables.items()},
        "yearbook_counts": dict(sorted(Counter(row["yearbook_year"] for row in datasets).items())),
        "year_quality_counts": dict(sorted(Counter(row["year_quality"] for row in datasets).items())),
        "quality_flag_counts": dict(sorted(Counter(row["quality_flag"] for row in facts).items())),
        "sample_query": sample,
        "warnings": warnings,
        "errors": errors[:100],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.package_dir.resolve())
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output, encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
