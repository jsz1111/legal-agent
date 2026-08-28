# -*- coding: utf-8 -*-
"""构建法条检索评测集：从已入库 articles 分层抽样，整理口语化问题，金标=条文本身。"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, r"D:\learn\legal-agent")

from sqlalchemy import text
from langchain_core.messages import SystemMessage

from src.agents.legal_guide.llm_runtime import build_chat_llm
from src.core.config import get_settings
from src.infra.database import engine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "statute_eval_dataset.json"
N_PER_DOMAIN = 2
DOMAINS = [
    "labor_social_security",
    "consumer_market",
    "contract_commercial",
    "real_estate_construction",
    "family_marriage",
]

settings = get_settings()
llm = build_chat_llm(temperature=0.3)

PROMPT = """你是法律咨询问题整理人员。请根据下面的法律条文，写出一条普通市民可能会问的口语化、场景化咨询问题。

要求：
- 不得包含法条原文、法律名称和条号；
- 不得直接暴露答案，要让问题像真实咨询；
- 只输出问题本身，不要任何解释或前后缀。

条文（节选）：
{content}
"""


async def sample_articles() -> list[dict]:
    rows = []
    async with engine.connect() as conn:
        for domain in DOMAINS:
            res = await conn.execute(
                text(
                    """
                    SELECT a.law_id, a.article_no, a.content, l.title
                    FROM articles a
                    JOIN laws l ON l.id = a.law_id
                    WHERE l.domain = :domain AND length(a.content) > 40
                    ORDER BY random() LIMIT :n
                    """
                ),
                {"domain": domain, "n": N_PER_DOMAIN},
            )
            for law_id, article_no, content, title in res.fetchall():
                rows.append(
                    {
                        "law_id": law_id,
                        "article_no": str(article_no),
                        "content": content,
                        "law_title": title,
                        "domain": domain,
                    }
                )
    await engine.dispose()
    return rows


async def make_question(content: str) -> str:
    msg = [SystemMessage(content=PROMPT.format(content=content[:300]))]
    resp = await llm.ainvoke(msg)
    return str(resp.content).strip()


async def main():
    articles = await sample_articles()
    dataset = []
    for item in articles:
        question = await make_question(item["content"])
        dataset.append(
            {
                "question": question,
                "law_title": item["law_title"],
                "article_no": item["article_no"],
                "expected_law_id": item["law_id"],
                "expected_article_no": item["article_no"],
                "domain": item["domain"],
                "reference": item["content"][:200],
            }
        )
        print(f"[{len(dataset)}/10] {item['domain']}: {question}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    asyncio.run(main())
