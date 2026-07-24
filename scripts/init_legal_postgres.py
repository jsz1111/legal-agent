"""初始化 legal_db 业务数据：laws / articles / channels / legal_cases"""
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Article, Channel, Law, LegalCase

# ── 数据路径 ──────────────────────────────────────────────────────────────────
BATCH04_JSON = ROOT / "data/sources/formal/2026-07-23/batch-04/download_report.json"
BATCH06_JSON = ROOT / "data/sources/formal/2026-07-23/batch-06-citizen-actions-all-regions/download_report.json"
CAIL_DIR     = ROOT / "data/sources/cail2019_scm"
CONCRETE_DIR = ROOT / "data/sources/formal/2026-07-23/concrete"

ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百千零\d]+)条[　\s]*(.*)", re.DOTALL)
BATCH_SIZE  = 200


async def import_laws(session) -> dict[str, int]:
    """batch-04/download_report.json → laws 表，返回 {doc_id: law.id}。"""
    with open(BATCH04_JSON, encoding="utf-8") as f:
        data = json.load(f)

    laws_map: dict[str, int] = {}
    for doc in data["documents"]:
        law = Law(
            title=doc["title"],
            category=doc.get("category", ""),
            authority=doc.get("authority"),
            domain=doc.get("domain", ""),
            effective_from=doc.get("effective_from"),
            file_path=doc.get("formats", {}).get("docx", {}).get("saved_path"),
        )
        session.add(law)
        await session.flush()
        laws_map[doc["id"]] = law.id

    await session.commit()
    print(f"  [laws] 导入 {len(laws_map)} 条")
    return laws_map


async def import_articles(session, laws_map: dict[str, int]) -> None:
    """batch-04/*.docx → articles 表。"""
    with open(BATCH04_JSON, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    buf: list[Article] = []

    for doc in data["documents"]:
        law_id = laws_map.get(doc["id"])
        if not law_id:
            continue
        docx_rel = doc.get("formats", {}).get("docx", {}).get("saved_path")
        if not docx_rel:
            continue
        docx_path = ROOT / docx_rel
        if not docx_path.exists():
            print(f"  [articles] 跳过（不存在）: {docx_path.name}")
            continue

        try:
            word_doc = Document(str(docx_path))
        except Exception as e:
            print(f"  [articles] 解析失败 {docx_path.name}: {e}")
            continue

        current_no: str | None = None
        current_lines: list[str] = []

        for para in word_doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            m = ARTICLE_RE.match(text)
            if m:
                if current_no and current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        buf.append(Article(law_id=law_id, article_no=current_no, content=content))
                        total += 1
                current_no = f"第{m.group(1)}条"
                current_lines = [text]
            elif current_no:
                current_lines.append(text)

        # flush last article of this doc
        if current_no and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                buf.append(Article(law_id=law_id, article_no=current_no, content=content))
                total += 1

        if len(buf) >= BATCH_SIZE:
            session.add_all(buf)
            await session.commit()
            buf.clear()

    if buf:
        session.add_all(buf)
        await session.commit()

    print(f"  [articles] 导入 {total} 条")


async def import_channels(session) -> None:
    """batch-06/download_report.json → channels 表。"""
    with open(BATCH06_JSON, encoding="utf-8") as f:
        data = json.load(f)

    channels: list[Channel] = []
    for doc in data["documents"]:
        contacts = doc.get("contacts", [])
        phone = next((c["value"] for c in contacts if c.get("kind") == "phone"), None)
        url   = next((c["value"] for c in contacts if c.get("kind") == "website"), None)
        ctypes = doc.get("channel_types", [])
        channels.append(Channel(
            name=doc["title"],
            domain=doc.get("domain", ""),
            channel_type=ctypes[0] if ctypes else "website",
            phone=phone,
            url=url,
            region_code=doc.get("region_code", "CN"),
        ))

    session.add_all(channels)
    await session.commit()
    print(f"  [channels] 导入 {len(channels)} 条")


async def import_cail_cases(session) -> None:
    """cail2019_scm/{train,valid,test}.json → legal_cases（刑事案例）。"""
    seen: set[str] = set()
    total = 0
    buf: list[LegalCase] = []

    for filename in ("train.json", "valid.json", "test.json"):
        path = CAIL_DIR / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                obj = json.loads(raw_line)
                for key in ("A", "B", "C"):
                    facts = obj.get(key, "").strip()
                    if not facts:
                        continue
                    h = hashlib.md5(facts.encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    buf.append(LegalCase(
                        facts=facts,
                        domain="criminal_public_security",
                        source="cail2019_scm",
                    ))
                    total += 1
                    if len(buf) >= BATCH_SIZE:
                        session.add_all(buf)
                        await session.commit()
                        buf.clear()

    if buf:
        session.add_all(buf)
        await session.commit()

    print(f"  [cail_cases] 导入 {total} 条（去重后）")


def _parse_html_cases(html_path: Path, domain: str) -> list[dict]:
    """解析 concrete HTML 案例文件，返回案例字典列表。"""
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    CASE_HEAD = re.compile(r"^案例[一二三四五六七八九十\d]+[：:]\s*(.+)")

    cases: list[dict] = []
    current: dict | None = None
    current_section: str | None = None

    for line in lines:
        m = CASE_HEAD.match(line)
        if m:
            if current and current["facts"]:
                cases.append(current)
            current = {"title": m.group(1).strip(), "facts": [], "gist": []}
            current_section = None
            continue

        if "【基本案情】" in line:
            current_section = "facts"
            after = line.split("【基本案情】", 1)[-1].strip()
            if after and current is not None:
                current["facts"].append(after)
            continue
        if "【裁判结果】" in line or "【法院认为】" in line:
            current_section = "result"
            continue
        if "【典型意义】" in line or "【裁判要旨】" in line:
            current_section = "gist"
            after = line.split("】", 1)[-1].strip()
            if after and current is not None:
                current["gist"].append(after)
            continue

        if current is None:
            continue
        if current_section == "facts":
            current["facts"].append(line)
        elif current_section == "gist":
            current["gist"].append(line)

    if current and current["facts"]:
        cases.append(current)

    return [
        {
            "title": c["title"] or None,
            "facts": "\n".join(c["facts"]).strip(),
            "gist":  "\n".join(c["gist"]).strip() or None,
            "domain": domain,
        }
        for c in cases
        if "\n".join(c["facts"]).strip()
    ]


async def import_concrete_cases(session) -> None:
    """concrete/ HTML 案例 → legal_cases（消费/劳动案例）。"""
    html_files = [
        (CONCRETE_DIR / "prepaid_consumption_cases_2025.html", "consumer_market"),
        (CONCRETE_DIR / "labor_cases_batch4.html", "labor_social_security"),
    ]

    total = 0
    for html_path, domain in html_files:
        if not html_path.exists():
            print(f"  [concrete] 跳过（不存在）: {html_path.name}")
            continue
        cases = _parse_html_cases(html_path, domain)
        for c in cases:
            session.add(LegalCase(
                facts=c["facts"],
                domain=c["domain"],
                source="concrete",
                title=c["title"],
                gist=c["gist"],
            ))
            total += 1
        await session.commit()
        print(f"  [concrete] {html_path.name}: {len(cases)} 条")

    print(f"  [concrete_cases] 合计 {total} 条")


async def main() -> None:
    print("=== init_legal_postgres 开始 ===\n")

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select, func
        row = await session.execute(select(func.count()).select_from(Law))
        if row.scalar_one() > 0:
            print("laws 表已有数据，跳过（如需重新导入请先清空表）")
            return

        print("→ 导入 laws ...")
        laws_map = await import_laws(session)

        print("→ 导入 articles ...")
        await import_articles(session, laws_map)

        print("→ 导入 channels ...")
        await import_channels(session)

        print("→ 导入 CAIL 刑事案例 ...")
        await import_cail_cases(session)

        print("→ 导入 concrete 案例 ...")
        await import_concrete_cases(session)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
