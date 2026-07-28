"""手动导入《中华人民共和国民法典》HTML"""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Article, Law


async def import_civil_code():
    html_path = ROOT / "data/sources/formal/2026-07-23/batch-02/civil_code_current.html"

    print(f"Reading {html_path}")
    with open(html_path, 'rb') as f:
        raw = f.read()

    soup = BeautifulSoup(raw.decode('utf-8'), 'html.parser')
    text = soup.get_text()

    # 提取所有条文
    article_pattern = re.compile(r'第([一二三四五六七八九十百千零\d]+)条\s+(.*?)(?=第[一二三四五六七八九十百千零\d]+条|$)', re.DOTALL)
    matches = article_pattern.findall(text)

    print(f"Found {len(matches)} articles")

    # 检查是否包含1218、1222条
    has_1218 = any('一千二百一十八' in m[0] for m in matches)
    has_1222 = any('一千二百二十二' in m[0] for m in matches)
    print(f"Contains article 1218: {has_1218}, 1222: {has_1222}")

    async with AsyncSessionLocal() as session:
        # 创建law记录
        law = Law(
            title="中华人民共和国民法典",
            category="法律",
            authority="全国人大",
            domain="",  # 民法典跨多个领域
            effective_from="2021-01-01",
            file_path=str(html_path.relative_to(ROOT)),
        )
        session.add(law)
        await session.flush()

        print(f"Created law: {law.title} (id={law.id})")

        # 导入条文（批量）
        articles = []
        for article_no_cn, content in matches:
            # 转换中文数字为阿拉伯数字（简化版）
            article_no = article_no_cn  # 保留中文
            content_clean = re.sub(r'\s+', ' ', content).strip()[:2000]  # 限制长度

            if content_clean:
                articles.append(Article(
                    law_id=law.id,
                    article_no=article_no,
                    content=content_clean,
                ))

        session.add_all(articles)
        await session.commit()

        print(f"Imported {len(articles)} articles")
        print(f"\nSample articles:")
        for a in articles[:3]:
            print(f"  {a.article_no}: {a.content[:80]}...")


if __name__ == "__main__":
    asyncio.run(import_civil_code())
