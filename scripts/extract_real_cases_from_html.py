"""
从现有 HTML 文件中提取官方发布的真实案例

数据源：
1. labor_cases_batch4.html - 人社部+最高法第四批劳动人事争议典型案例
2. prepaid_consumption_cases_2025.html - 预付消费典型案例
"""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import LegalCase


HTML_FILES = [
    {
        "file": ROOT / "data/sources/formal/2026-07-23/concrete/labor_cases_batch4.html",
        "domain": "labor_social_security",
        "source": "mohrss_court_batch4"
    },
    {
        "file": ROOT / "data/sources/formal/2026-07-23/concrete/prepaid_consumption_cases_2025.html",
        "domain": "consumer_market",
        "source": "prepaid_consumption_2025"
    },
]


def extract_cases_from_html(html_file: Path) -> list:
    """从 HTML 文件中提取案例"""
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # 查找主要内容区域
    content = soup.find('div', class_='detail_content')
    if not content:
        content = soup.find('div', class_='content')
    if not content:
        content = soup.find('article')
    if not content:
        # 尝试找包含最多文本的 div
        all_divs = soup.find_all('div')
        content = max(all_divs, key=lambda d: len(d.get_text())) if all_divs else None

    if not content:
        print(f"  [WARNING] Cannot find content in {html_file.name}")
        return []

    text = content.get_text()

    # 尝试提取案例（根据常见模式）
    cases = []

    # 模式1: "案例X：标题"
    pattern1 = re.compile(r'案例[一二三四五六七八九十\d]+[：:](.*?)(?=案例[一二三四五六七八九十\d]+|$)', re.DOTALL)
    matches = pattern1.findall(text)

    if matches:
        print(f"  Found {len(matches)} cases using pattern 1")
        for match in matches:
            cases.append(parse_case_text(match))
    else:
        # 模式2: 按段落分割，寻找案例特征
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        print(f"  Found {len(paragraphs)} paragraphs, analyzing...")

        current_case = None
        for para in paragraphs:
            # 查找案例标题特征
            if any(keyword in para for keyword in ['诉', '纠纷', '争议', '申请']):
                if current_case:
                    cases.append(current_case)
                current_case = {"raw_text": para}
            elif current_case:
                current_case["raw_text"] += "\n\n" + para

        if current_case:
            cases.append(current_case)

    return cases


def parse_case_text(text: str) -> dict:
    """解析案例文本，提取结构化信息"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    case_data = {
        "title": "",
        "facts": "",
        "gist": "",
        "court": "",
        "cause": ""
    }

    # 提取标题（通常是第一行或包含"诉"的行）
    for line in lines[:5]:
        if '诉' in line or '申请' in line:
            case_data["title"] = line
            break

    if not case_data["title"] and lines:
        case_data["title"] = lines[0][:100]  # 取前100字作为标题

    # 提取案由
    for line in lines:
        if '纠纷' in line or '争议' in line:
            case_data["cause"] = line.split('。')[0]
            break

    # 合并所有文本作为事实
    case_data["facts"] = '\n'.join(lines[:10])[:500]  # 前10行，最多500字

    # 查找裁判要旨
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in ['法院认为', '仲裁委认为', '裁决', '判决']):
            case_data["gist"] = '\n'.join(lines[i:min(i+5, len(lines))])[:300]
            break

    # 查找法院
    for line in lines:
        if '法院' in line or '仲裁委' in line:
            case_data["court"] = line.split('。')[0]
            break

    return case_data


async def import_cases_from_htmls():
    """从 HTML 文件导入真实案例"""
    total_imported = 0

    for config in HTML_FILES:
        html_file = config["file"]
        if not html_file.exists():
            print(f"\n[SKIP] File not found: {html_file}")
            continue

        print(f"\n[PROCESSING] {html_file.name}")
        print(f"  Domain: {config['domain']}")

        cases = extract_cases_from_html(html_file)
        print(f"  Extracted: {len(cases)} cases")

        if not cases:
            continue

        # 导入数据库
        async with AsyncSessionLocal() as session:
            added = 0
            for case_data in cases:
                if not case_data.get("title") or not case_data.get("facts"):
                    continue

                case = LegalCase(
                    title=case_data["title"][:300],
                    cause=case_data.get("cause", "")[:200] if case_data.get("cause") else None,
                    facts=case_data["facts"][:1000],
                    gist=case_data.get("gist", "")[:500] if case_data.get("gist") else None,
                    court=case_data.get("court", "")[:200] if case_data.get("court") else None,
                    domain=config["domain"],
                    source=config["source"]
                )
                session.add(case)
                added += 1

            await session.commit()
            print(f"  Imported: {added} cases to database")
            total_imported += added

    return total_imported


async def main():
    print("=" * 60)
    print("Extract Real Cases from HTML Files")
    print("=" * 60)

    total = await import_cases_from_htmls()

    print("\n" + "=" * 60)
    print(f"Total imported: {total} real cases")
    print("=" * 60)

    # 统计
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select, func
        stmt = select(
            LegalCase.domain,
            func.count(LegalCase.id).label('count')
        ).group_by(LegalCase.domain)
        result = await session.execute(stmt)

        print("\n[STATISTICS] Case distribution:")
        for row in result:
            print(f"  {row.domain}: {row.count}")


if __name__ == "__main__":
    asyncio.run(main())
