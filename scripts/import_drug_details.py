"""
Batch import drug instruction files into PostgreSQL drug_details.

Supported inputs:
- txt / md: read directly
- pdf / doc / docx / ppt / pptx / html: parse via MinerU CLI

Typical usage:
    python scripts/import_drug_details.py --source-dir data/drug_instructions
    python scripts/import_drug_details.py --source-dir data/drug_instructions --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.infra.database import AsyncSessionLocal
from src.modules.medical.model import Drug, DrugDetail

TEXT_EXTS = {".txt", ".md"}
MINERU_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html"}
SUPPORTED_EXTS = TEXT_EXTS | MINERU_EXTS

HEADER_ALIASES = {
    "适应症": "indication",
    "功能主治": "indication",
    "用法用量": "usage_dosage",
    "不良反应": "adverse_reaction",
    "禁忌": "contraindication",
    "注意事项": "precaution",
    "药物相互作用": "interaction",
    "相互作用": "interaction",
    "贮藏": "storage",
    "储藏": "storage",
    "批准文号": "approval_number",
    "国药准字": "approval_number",
    "生产企业": "manufacturer",
    "生产厂家": "manufacturer",
    "上市许可持有人": "manufacturer",
    "药品名称": "drug_name",
    "通用名称": "drug_name",
    "商品名称": "alias",
}

FIELD_NAMES = {
    "indication",
    "usage_dosage",
    "adverse_reaction",
    "contraindication",
    "precaution",
    "interaction",
    "storage",
}

HEADER_PATTERN = re.compile(
    r"^(?:#+\s*)?(?:【|\[)?"
    r"(?P<label>适应症|功能主治|用法用量|不良反应|禁忌|注意事项|药物相互作用|相互作用|贮藏|储藏|批准文号|国药准字|生产企业|生产厂家|上市许可持有人|药品名称|通用名称|商品名称)"
    r"(?:】|\])?\s*[:：]?\s*(?P<rest>.*)$"
)

APPROVAL_PATTERN = re.compile(r"(国药准字[A-Z0-9一-龥\-]+)")
INLINE_FIELD_PATTERNS = {
    "manufacturer": re.compile(r"(?:生产企业|生产厂家|上市许可持有人)\s*[:：]\s*(.+)"),
    "drug_name": re.compile(r"(?:药品名称|通用名称)\s*[:：]\s*(.+)"),
    "alias": re.compile(r"(?:商品名称)\s*[:：]\s*(.+)"),
}

MINERU_SCRIPT = os.getenv(
    "MINERU_OPEN_API_SCRIPT",
    r"D:/develop/Miniconda/npm_global/node_modules/mineru-open-api/bin/mineru-open-api",
)

NAME_SUFFIXES = (
    "药品说明书",
    "说明书",
    "使用说明书",
    "药物说明书",
)


@dataclass
class DrugInfo:
    id: int
    name: str
    alias: str | None
    manufacturer: str | None
    approval_number: str | None


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def read_text_file(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    for encoding in encodings:
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def parse_with_mineru(path: Path) -> str:
    base_cmd = ["node", MINERU_SCRIPT]
    commands = [
        base_cmd + ["extract", str(path)],
        base_cmd + ["flash-extract", str(path)],
    ]
    last_error = None
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            text = normalize_text(result.stdout)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to parse with MinerU: {path}") from last_error


def load_instruction_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return read_text_file(path)
    if suffix in MINERU_EXTS:
        return parse_with_mineru(path)
    raise ValueError(f"unsupported file type: {path.suffix}")


def normalize_name(name: str) -> str:
    value = name.strip()
    for suffix in NAME_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    value = re.sub(r"[（）()\[\]【】\s_\-]+", "", value)
    return value.lower()


def collect_name_candidates(path: Path, parsed: dict[str, str | None]) -> list[str]:
    candidates = []
    stem = path.stem.strip()
    if stem:
        candidates.append(stem)
        for suffix in NAME_SUFFIXES:
            if stem.endswith(suffix):
                candidates.append(stem[: -len(suffix)].strip())
    for key in ("drug_name", "alias"):
        value = parsed.get(key)
        if value:
            candidates.append(value)
    deduped = []
    seen = set()
    for item in candidates:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def parse_instruction_fields(text: str) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {
        "drug_name": None,
        "alias": None,
        "approval_number": None,
        "manufacturer": None,
        "indication": None,
        "usage_dosage": None,
        "adverse_reaction": None,
        "contraindication": None,
        "precaution": None,
        "interaction": None,
        "storage": None,
        "full_instruction": text,
    }

    buffers = {name: [] for name in FIELD_NAMES}
    current_field: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header = HEADER_PATTERN.match(line)
        if header:
            label = header.group("label")
            rest = header.group("rest").strip()
            target = HEADER_ALIASES[label]
            current_field = target if target in FIELD_NAMES else None

            if target in FIELD_NAMES:
                if rest:
                    buffers[target].append(rest)
                continue

            if rest:
                if target == "approval_number":
                    approval = APPROVAL_PATTERN.search(rest)
                    parsed[target] = approval.group(1) if approval else rest
                else:
                    parsed[target] = rest
            continue

        if current_field:
            buffers[current_field].append(line)

    for field, values in buffers.items():
        if values:
            parsed[field] = "\n".join(values).strip()

    if not parsed["approval_number"]:
        approval = APPROVAL_PATTERN.search(text)
        if approval:
            parsed["approval_number"] = approval.group(1)

    for key, pattern in INLINE_FIELD_PATTERNS.items():
        if parsed.get(key):
            continue
        match = pattern.search(text)
        if match:
            parsed[key] = match.group(1).strip()

    return parsed


async def load_drug_map() -> tuple[dict[str, list[DrugInfo]], list[DrugInfo]]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                Drug.id,
                Drug.name,
                Drug.alias,
                Drug.manufacturer,
                Drug.approval_number,
            )
        )
        rows = [
            DrugInfo(
                id=row.id,
                name=row.name,
                alias=row.alias,
                manufacturer=row.manufacturer,
                approval_number=row.approval_number,
            )
            for row in result.all()
        ]

    drug_map: dict[str, list[DrugInfo]] = {}
    for drug in rows:
        drug_map.setdefault(normalize_name(drug.name), []).append(drug)
        if drug.alias:
            for alias in re.split(r"[、,，;/\s]+", drug.alias):
                alias = alias.strip()
                if alias:
                    drug_map.setdefault(normalize_name(alias), []).append(drug)
    return drug_map, rows


def resolve_drug(
    candidates: list[str],
    drug_map: dict[str, list[DrugInfo]],
    all_drugs: list[DrugInfo],
) -> tuple[DrugInfo | None, str | None]:
    for candidate in candidates:
        matched = drug_map.get(normalize_name(candidate), [])
        if len(matched) == 1:
            return matched[0], candidate

    fuzzy_matches: dict[int, DrugInfo] = {}
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if not normalized:
            continue
        for drug in all_drugs:
            drug_norm = normalize_name(drug.name)
            if normalized in drug_norm or drug_norm in normalized:
                fuzzy_matches[drug.id] = drug
    if len(fuzzy_matches) == 1:
        only = next(iter(fuzzy_matches.values()))
        return only, candidates[0] if candidates else None
    return None, None


def iter_source_files(source_dir: Path) -> list[Path]:
    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(files)


async def upsert_drug_detail(
    drug_id: int,
    parsed: dict[str, str | None],
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, bool]:
    async with AsyncSessionLocal() as db:
        drug = await db.get(Drug, drug_id)
        result = await db.execute(
            select(DrugDetail).where(DrugDetail.drug_id == drug_id)
        )
        detail = result.scalar_one_or_none()

        created = False
        if detail is None:
            detail = DrugDetail(drug_id=drug_id)
            db.add(detail)
            created = True

        for field in FIELD_NAMES | {"full_instruction"}:
            value = parsed.get(field)
            if not value:
                continue
            current = getattr(detail, field)
            if overwrite or not current:
                setattr(detail, field, value)

        for field in ("manufacturer", "approval_number"):
            value = parsed.get(field)
            if not value:
                continue
            current = getattr(drug, field)
            if overwrite or not current:
                setattr(drug, field, value)

        if dry_run:
            await db.rollback()
        else:
            await db.commit()

        return created, not created


async def main() -> None:
    parser = argparse.ArgumentParser(description="Import drug instruction files into drug_details")
    parser.add_argument(
        "--source-dir",
        default="data/drug_instructions",
        help="Directory containing instruction files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing detail fields when incoming values are present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and match files without writing to the database",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise SystemExit(f"source directory does not exist: {source_dir}")

    drug_map, all_drugs = await load_drug_map()
    files = iter_source_files(source_dir)
    if args.limit > 0:
        files = files[: args.limit]

    print(f"[INFO] source_dir={source_dir}")
    print(f"[INFO] files={len(files)}")

    stats = {
        "processed": 0,
        "created": 0,
        "updated": 0,
        "skipped_unmatched": 0,
        "failed_parse": 0,
    }

    for path in files:
        stats["processed"] += 1
        try:
            text = load_instruction_text(path)
        except Exception as exc:  # noqa: BLE001
            stats["failed_parse"] += 1
            print(f"[WARN] parse failed: {path.name} | {exc}")
            continue

        parsed = parse_instruction_fields(text)
        candidates = collect_name_candidates(path, parsed)
        drug, matched_by = resolve_drug(candidates, drug_map, all_drugs)
        if not drug:
            stats["skipped_unmatched"] += 1
            print(f"[WARN] unmatched: {path.name} | candidates={candidates}")
            continue

        created, updated = await upsert_drug_detail(
            drug_id=drug.id,
            parsed=parsed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if created:
            stats["created"] += 1
        if updated:
            stats["updated"] += 1

        print(
            f"[OK] {path.name} -> {drug.name}"
            f" | matched_by={matched_by}"
            f" | approval={parsed.get('approval_number') or '-'}"
            f" | manufacturer={parsed.get('manufacturer') or '-'}"
        )

    print("[DONE] import summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
