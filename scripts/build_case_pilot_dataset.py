"""Build a reviewable pilot dataset from the 2021-10 judgment CSV.

The source CSV is never modified. The processed dataset preserves every source
column except ``来源`` and adds a deterministic case ID, cleaned text, a short
retrieval text, and screening metadata. It does not call an LLM, PostgreSQL, or
Milvus.

Example:
    python scripts/build_case_pilot_dataset.py \
        --input "D:/BaiduNetdiskDownload/2021年10月裁判文书数据.csv" \
        --output-dir data/processed/legal_cases_2021_10_pilot \
        --per-cause 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "原始链接",
    "案号",
    "案件名称",
    "法院",
    "所属地区",
    "案件类型",
    "案件类型编码",
    "来源",
    "审理程序",
    "裁判日期",
    "公开日期",
    "当事人",
    "案由",
    "法律依据",
    "全文",
}

ALLOWED_PROCEDURES = {"民事一审", "民事二审", "民事审判监督"}
PROCEDURAL_OUTCOMES = (
    "准许撤回起诉",
    "按撤诉处理",
    "不予受理",
    "驳回起诉",
    "移送管辖",
    "管辖权异议",
    "终结本次执行程序",
    "终结执行",
    "中止诉讼",
)
FOOTER_MARKERS = (
    "审判长",
    "审判员",
    "人民陪审员",
    "书记员",
    "法官助理",
    "附相关法律条文",
    "附本案相关法律条文",
)
CONTENT_START_MARKERS = (
    "诉讼请求：",
    "诉讼请求:",
    "上诉请求：",
    "上诉请求:",
    "原告向本院提出诉讼请求",
    "上诉人上诉请求",
    "经审理查明",
    "本院查明",
    "经审理认定",
    "原审法院查明",
    "原审法院认定",
    "一审法院查明",
    "一审法院认定",
    "本院二审期间",
)
WATERMARK_PATTERNS = (
    re.compile(
        r"(?:\s*-\s*)?"
        r"(?:(?:更多数据[:：]?\s*(?:搜索)?\s*(?:来源)?|来自|来源)[:：]?\s*)?"
        r"(?:https?://)?(?:www\.)?macrodatas\.cn.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:百度搜索|关注公众号|微信公众号|来源[:：]?\s*(?:百度)?)"
        r"\s*[\"“']?\s*马克\s*数据网\s*[\"”']?"
    ),
    re.compile(r"[\"“']?\s*马克\s*数据网\s*[\"”']?"),
)

OUTPUT_COLUMNS = (
    "case_id",
    "原始链接",
    "案号",
    "案件名称",
    "法院",
    "所属地区",
    "案件类型",
    "案件类型编码",
    "审理程序",
    "裁判日期",
    "公开日期",
    "当事人",
    "案由",
    "法律依据",
    "全文",
    "全文字数",
    "retrieval_text",
    "筛选标签",
)


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    score: int


def clean_text(value: str) -> str:
    """Normalize common OCR/layout noise while preserving the legal text."""
    text = html.unescape(value or "")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    # Vendor footers may combine the brand name and domain. Two passes let one
    # pattern expose a suffix that another pattern can then remove.
    for _ in range(2):
        for pattern in WATERMARK_PATTERNS:
            text = pattern.sub("", text)
    text = re.sub(r"^\s*文书内容\s*", "", text)
    text = re.sub(r"[ \t\r\n]+", " ", text)
    text = re.sub(r"(?:来源[:：]?\s*百度|百度搜索|关注公众号)\s*[\"“”']*\s*$", "", text)
    text = re.sub(
        r"(?:\s*-\s*)?(?:更多数据[:：]?\s*(?:搜索)?|来自|来源)[:：]?\s*$",
        "",
        text,
    )
    return text.strip()


def stable_case_id(case_number: str, original_url: str, text: str) -> str:
    identity = f"{case_number.strip()}|{original_url.strip()}|{text[:500]}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _clip_head(text: str, limit: int) -> str:
    text = text.strip(" ，。；：")
    if len(text) <= limit:
        return text
    cut = max(text.rfind(mark, 0, limit + 1) for mark in "。；！？")
    if cut >= int(limit * 0.65):
        return text[: cut + 1]
    return text[:limit].rstrip(" ，；：") + "……"


def _clip_head_tail(text: str, limit: int) -> str:
    text = text.strip(" ，。；：")
    if len(text) <= limit:
        return text
    head_limit = int(limit * 0.58)
    tail_limit = limit - head_limit - 1
    head = _clip_head(text, head_limit).rstrip("……")
    tail = text[-tail_limit:].lstrip(" ，。；：")
    return f"{head}……{tail}"


def _find_footer(text: str, start: int) -> int:
    positions = [text.find(marker, start) for marker in FOOTER_MARKERS]
    positions = [pos for pos in positions if pos >= 0]
    return min(positions) if positions else len(text)


def extract_sections(text: str) -> tuple[str, str, str] | None:
    """Extract content, court reasoning, and judgment result without an LLM."""
    reasoning_start = text.find("本院认为")
    if reasoning_start < 0:
        return None
    result_start = text.find("判决如下", reasoning_start)
    if result_start < 0:
        return None

    prefix = text[:reasoning_start]
    starts = [prefix.find(marker) for marker in CONTENT_START_MARKERS]
    starts = [pos for pos in starts if pos >= 0]
    content_start = min(starts) if starts else max(0, reasoning_start - 900)

    content = text[content_start:reasoning_start].strip()
    reasoning = text[reasoning_start + len("本院认为") : result_start].strip()
    result_end = _find_footer(text, result_start + len("判决如下"))
    result = text[result_start + len("判决如下") : result_end].strip()
    if len(content) < 80 or len(reasoning) < 80 or len(result) < 20:
        return None
    return content, reasoning, result


def build_retrieval_text(row: dict[str, str], sections: tuple[str, str, str]) -> str:
    content, reasoning, result = sections
    parts = [
        f"案件名称：{_clip_head(row['案件名称'], 70)}",
        f"案由：{_clip_head(row['案由'], 40)}",
        f"案件内容：{_clip_head_tail(content, 260)}",
        f"法院认为：{_clip_head(reasoning, 260)}",
        f"裁判结果：{_clip_head(result, 150)}",
    ]
    text = "\n".join(parts)
    if len(text) > 800:
        text = text[:800].rstrip(" ，；：") + "……"
    return text


def _selection_score(row: dict[str, str], sections: tuple[str, str, str]) -> int:
    content, reasoning, result = sections
    length = len(row["全文"])
    score = 0
    if row["审理程序"] == "民事二审":
        score += 2
    if 2500 <= length <= 8000:
        score += 3
    elif 1800 <= length < 2500:
        score += 2
    if row["法律依据"].strip():
        score += 1
    if row["原始链接"].startswith("https://wenshu.court.gov.cn/"):
        score += 1
    if len(content) >= 400:
        score += 1
    if len(reasoning) >= 300:
        score += 2
    if len(result) >= 80:
        score += 1
    return score


def screen_row(
    raw: dict[str, str], min_length: int, max_length: int
) -> tuple[dict[str, str] | None, str]:
    """Return a cleaned candidate row or a stable rejection reason."""
    if raw.get("案件类型", "").strip() != "民事案件":
        return None, "案件类型"
    if raw.get("审理程序", "").strip() not in ALLOWED_PROCEDURES:
        return None, "审理程序"
    if "判决书" not in raw.get("案件名称", ""):
        return None, "非判决书"
    if not raw.get("案由", "").strip() or raw.get("案由", "").strip() in {"民事", "其他案由"}:
        return None, "案由无效"
    if not raw.get("原始链接", "").strip():
        return None, "原始链接缺失"

    row = {key: (value or "").strip() for key, value in raw.items() if key != "来源"}
    row["全文"] = clean_text(row.get("全文", ""))
    text_length = len(row["全文"])
    if text_length < min_length:
        return None, "全文过短"
    if text_length > max_length:
        return None, "全文过长"

    sections = extract_sections(row["全文"])
    if sections is None:
        return None, "缺少实体章节"
    result = sections[2]
    if any(marker in result for marker in PROCEDURAL_OUTCOMES):
        return None, "程序性裁判"

    row["case_id"] = stable_case_id(row["案号"], row["原始链接"], row["全文"])
    row["全文字数"] = str(text_length)
    row["retrieval_text"] = build_retrieval_text(row, sections)
    row["筛选标签"] = "民事实体判决;含法院说理;含裁判结果"
    row["_score"] = str(_selection_score(row, sections))
    return row, "合格"


def _dedupe_key(row: dict[str, str]) -> str:
    case_number = re.sub(r"\s+", "", row.get("案号", ""))
    if case_number:
        return f"case:{case_number}"
    url = row.get("原始链接", "").strip()
    if url:
        return f"url:{url}"
    return f"text:{hashlib.sha256(row.get('全文', '').encode('utf-8')).hexdigest()}"


def select_balanced(candidates: Iterable[Candidate], per_cause: int) -> list[dict[str, str]]:
    """Select high-quality rows while rotating across regions within each cause."""
    by_cause_region: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        cause = candidate.row["案由"]
        region = candidate.row["所属地区"] or "未知地区"
        by_cause_region[cause][region].append(candidate)

    selected: list[dict[str, str]] = []
    for cause in sorted(by_cause_region):
        queues: dict[str, deque[Candidate]] = {}
        for region, rows in by_cause_region[cause].items():
            rows.sort(key=lambda item: (-item.score, item.row["裁判日期"], item.row["案号"]))
            queues[region] = deque(rows)

        cause_rows: list[dict[str, str]] = []
        while queues and len(cause_rows) < per_cause:
            region_order = sorted(
                queues,
                key=lambda region: (-queues[region][0].score, region),
            )
            for region in region_order:
                if len(cause_rows) >= per_cause:
                    break
                queue = queues.get(region)
                if not queue:
                    continue
                cause_rows.append(queue.popleft().row)
                if not queue:
                    del queues[region]
        selected.extend(cause_rows)
    return selected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = {key: row.get(key, "") for key in OUTPUT_COLUMNS}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_sqlite(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a portable, indexed database without touching project services."""
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE legal_cases (
                case_id TEXT PRIMARY KEY,
                original_url TEXT NOT NULL,
                case_number TEXT NOT NULL UNIQUE,
                case_name TEXT NOT NULL,
                court TEXT NOT NULL,
                region TEXT NOT NULL,
                case_type TEXT NOT NULL,
                case_type_code TEXT NOT NULL,
                procedure TEXT NOT NULL,
                judgment_date TEXT NOT NULL,
                publication_date TEXT NOT NULL,
                parties TEXT NOT NULL,
                cause TEXT NOT NULL,
                legal_basis TEXT NOT NULL,
                full_text TEXT NOT NULL,
                full_text_length INTEGER NOT NULL,
                retrieval_text TEXT NOT NULL,
                selection_tags TEXT NOT NULL
            );
            CREATE INDEX ix_legal_cases_cause ON legal_cases(cause);
            CREATE INDEX ix_legal_cases_procedure ON legal_cases(procedure);
            CREATE INDEX ix_legal_cases_court ON legal_cases(court);
            CREATE INDEX ix_legal_cases_region ON legal_cases(region);
            CREATE INDEX ix_legal_cases_judgment_date ON legal_cases(judgment_date);
            """
        )
        connection.executemany(
            """
            INSERT INTO legal_cases (
                case_id, original_url, case_number, case_name, court, region,
                case_type, case_type_code, procedure, judgment_date,
                publication_date, parties, cause, legal_basis, full_text,
                full_text_length, retrieval_text, selection_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["case_id"],
                    row["原始链接"],
                    row["案号"],
                    row["案件名称"],
                    row["法院"],
                    row["所属地区"],
                    row["案件类型"],
                    row["案件类型编码"],
                    row["审理程序"],
                    row["裁判日期"],
                    row["公开日期"],
                    row["当事人"],
                    row["案由"],
                    row["法律依据"],
                    row["全文"],
                    int(row["全文字数"]),
                    row["retrieval_text"],
                    row["筛选标签"],
                )
                for row in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()


def write_report(
    path: Path,
    source: Path,
    rows_read: int,
    rejection_counts: Counter[str],
    eligible_counts: Counter[str],
    selected: list[dict[str, str]],
    causes: tuple[str, ...],
    min_length: int,
    max_length: int,
    per_cause: int,
) -> None:
    selected_counts = Counter(row["案由"] for row in selected)
    lines = [
        "# 裁判文书试验案例筛选报告",
        "",
        f"- 原始文件：`{source}`",
        f"- 原始行数：{rows_read}",
        f"- 入选总数：{len(selected)}",
        f"- 每个案由上限：{per_cause}",
        f"- 全文字数范围：{min_length}-{max_length}",
        "- 原始CSV未修改；处理结果删除了`来源`字段并保留`原始链接`。",
        "- 本批次未调用大模型、未生成向量、未导入PostgreSQL或Milvus。",
        "- 输出包含CSV、JSONL和带索引的独立SQLite数据库。",
        "",
        "## 选定案由",
        "",
        "| 案由 | 合格候选 | 最终入选 |",
        "|---|---:|---:|",
    ]
    for cause in causes:
        lines.append(f"| {cause} | {eligible_counts[cause]} | {selected_counts[cause]} |")

    lines.extend(
        [
            "",
            "## 筛选规则",
            "",
            "1. 案件类型为民事案件。",
            "2. 审理程序为民事一审、民事二审或民事审判监督。",
            "3. 案件名称包含“判决书”，案由和原始链接不为空。",
            f"4. 清洗后全文为{min_length}-{max_length}字。",
            "5. 全文包含“本院认为”和“判决如下”，且三个实体章节达到最低长度。",
            "6. 裁判结果不属于撤诉、不予受理、驳回起诉、移送管辖等程序性结果。",
            "7. 按案号去重，并在每个案由内跨地区均衡抽取。",
            "",
            "## 排除统计",
            "",
            "| 原因 | 数量 |",
            "|---|---:|",
        ]
    )
    for reason, count in rejection_counts.most_common():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_causes = tuple(dict.fromkeys(args.cause or ()))
    requested_set = set(requested_causes)

    rejection_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    candidates: list[Candidate] = []
    seen: set[str] = set()
    rows_read = 0

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV缺少字段: {', '.join(sorted(missing))}")

        for raw in reader:
            rows_read += 1
            row, reason = screen_row(raw, args.min_length, args.max_length)
            if row is None:
                rejection_counts[reason] += 1
                continue
            if requested_set and row["案由"] not in requested_set:
                rejection_counts["非目标案由"] += 1
                continue
            key = _dedupe_key(row)
            if key in seen:
                rejection_counts["重复案例"] += 1
                continue
            seen.add(key)
            eligible_counts[row["案由"]] += 1
            candidates.append(Candidate(row=row, score=int(row.pop("_score"))))

    selected = select_balanced(candidates, args.per_cause)
    selected_causes = tuple(sorted({row["案由"] for row in selected}))
    selected.sort(key=lambda row: (row["案由"], row["裁判日期"], row["案号"]))

    csv_path = output_dir / "cases_pilot.csv"
    jsonl_path = output_dir / "cases_pilot.jsonl"
    sqlite_path = output_dir / "cases_pilot.sqlite3"
    report_path = output_dir / "筛选报告.md"
    manifest_path = output_dir / "manifest.json"

    write_csv(csv_path, selected)
    write_jsonl(jsonl_path, selected)
    write_sqlite(sqlite_path, selected)
    write_report(
        report_path,
        source,
        rows_read,
        rejection_counts,
        eligible_counts,
        selected,
        selected_causes,
        args.min_length,
        args.max_length,
        args.per_cause,
    )

    manifest = {
        "source_file": str(source),
        "source_rows": rows_read,
        "selected_rows": len(selected),
        "per_cause_limit": args.per_cause,
        "cause_filter": list(requested_causes),
        "selected_causes": list(selected_causes),
        "selected_by_cause": dict(Counter(row["案由"] for row in selected)),
        "eligible_by_cause": {cause: eligible_counts[cause] for cause in selected_causes},
        "text_length": {"min": args.min_length, "max": args.max_length},
        "removed_source_column": True,
        "preserved_original_url": True,
        "llm_used": False,
        "vectorized": False,
        "database_imported": False,
        "files": {
            "csv": csv_path.name,
            "jsonl": jsonl_path.name,
            "sqlite": sqlite_path.name,
            "report": report_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选并清洗裁判文书试验案例")
    parser.add_argument("--input", type=Path, required=True, help="原始裁判文书CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--per-cause", type=int, default=50, help="每个案由最多保留数量")
    parser.add_argument("--min-length", type=int, default=1200, help="清洗后全文最小字数")
    parser.add_argument("--max-length", type=int, default=8000, help="清洗后全文最大字数")
    parser.add_argument(
        "--cause",
        action="append",
        help="可选案由过滤器，可重复指定；不指定时处理全部案由",
    )
    args = parser.parse_args()
    if args.per_cause <= 0:
        parser.error("--per-cause 必须大于0")
    if args.min_length <= 0 or args.max_length < args.min_length:
        parser.error("全文字数范围无效")
    return args


def main() -> None:
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
