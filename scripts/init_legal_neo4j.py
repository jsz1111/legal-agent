"""
法律知识图谱初始化脚本。

节点：Law / Article / LegalCase / Channel / Domain
关系：HAS_ARTICLE / CITES / APPLIES_TO / HANDLES

用法：
    python scripts/init_legal_neo4j.py
"""
import asyncio
import logging
import sys
from pathlib import Path

from neo4j import GraphDatabase
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Article, Channel, Law, LawCase, LegalCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
BATCH = 500


async def fetch_pg_data():
    """从 PostgreSQL 读取全量数据。"""
    async with AsyncSessionLocal() as session:
        laws      = (await session.execute(select(Law))).scalars().all()
        articles  = (await session.execute(select(Article))).scalars().all()
        cases     = (await session.execute(select(LegalCase))).scalars().all()
        channels  = (await session.execute(select(Channel))).scalars().all()
        law_cases = (await session.execute(select(LawCase))).scalars().all()
    logger.info(
        f"PG读取完成 — laws={len(laws)}, articles={len(articles)}, "
        f"cases={len(cases)}, channels={len(channels)}, law_cases={len(law_cases)}"
    )
    return laws, articles, cases, channels, law_cases


def batch_run(session, cypher: str, params_list: list, batch_size: int = BATCH):
    """分批执行 UNWIND 写入，$params 作为参数名。"""
    for i in range(0, len(params_list), batch_size):
        session.run(cypher, params=params_list[i : i + batch_size])


def init_graph():
    laws, articles, cases, channels, law_cases = asyncio.run(fetch_pg_data())

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    with driver.session() as s:
        s.run("RETURN 1")
    logger.info("Neo4j 连接成功")

    # ── 约束 ──────────────────────────────────────────────────────────────────
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Law)       REQUIRE n.pg_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Article)   REQUIRE n.pg_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LegalCase) REQUIRE n.pg_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Channel)   REQUIRE n.pg_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Domain)    REQUIRE n.name  IS UNIQUE",
    ]
    with driver.session() as s:
        for c in constraints:
            s.run(c)
    logger.info("约束创建完成")

    # ── 节点写入 ──────────────────────────────────────────────────────────────
    domains = {l.domain for l in laws if l.domain} | {ch.domain for ch in channels if ch.domain}

    with driver.session() as s:
        # Domain
        s.run("UNWIND $names AS name MERGE (n:Domain {name: name})", names=list(domains))
        logger.info(f"  Domain  节点: {len(domains)}")

        # Law
        law_params = [
            {"pg_id": l.id, "title": l.title, "domain": l.domain or "",
             "category": l.category or "", "authority": l.authority or ""}
            for l in laws
        ]
        batch_run(s, """
            UNWIND $params AS p
            MERGE (n:Law {pg_id: p.pg_id})
            SET n.title=p.title, n.domain=p.domain,
                n.category=p.category, n.authority=p.authority
        """, law_params)
        logger.info(f"  Law     节点: {len(law_params)}")

        # Article（只存元信息，正文从PG按id回查）
        art_params = [
            {"pg_id": a.id, "law_pg_id": a.law_id, "article_no": a.article_no}
            for a in articles
        ]
        batch_run(s, """
            UNWIND $params AS p
            MERGE (n:Article {pg_id: p.pg_id})
            SET n.law_pg_id=p.law_pg_id, n.article_no=p.article_no
        """, art_params)
        logger.info(f"  Article 节点: {len(art_params)}")

        # LegalCase
        case_params = [
            {"pg_id": c.id, "domain": c.domain or "",
             "source": c.source or "", "title": c.title or ""}
            for c in cases
        ]
        batch_run(s, """
            UNWIND $params AS p
            MERGE (n:LegalCase {pg_id: p.pg_id})
            SET n.domain=p.domain, n.source=p.source, n.title=p.title
        """, case_params)
        logger.info(f"  LegalCase 节点: {len(case_params)}")

        # Channel
        ch_params = [
            {"pg_id": ch.id, "name": ch.name, "domain": ch.domain or "",
             "channel_type": ch.channel_type or "", "region_code": ch.region_code or ""}
            for ch in channels
        ]
        batch_run(s, """
            UNWIND $params AS p
            MERGE (n:Channel {pg_id: p.pg_id})
            SET n.name=p.name, n.domain=p.domain,
                n.channel_type=p.channel_type, n.region_code=p.region_code
        """, ch_params)
        logger.info(f"  Channel 节点: {len(ch_params)}")

    # ── 关系写入 ──────────────────────────────────────────────────────────────
    with driver.session() as s:
        # (Law)-[:HAS_ARTICLE]->(Article)
        ha = [{"law_id": a.law_id, "art_id": a.id} for a in articles]
        batch_run(s, """
            UNWIND $params AS p
            MATCH (l:Law     {pg_id: p.law_id})
            MATCH (a:Article {pg_id: p.art_id})
            MERGE (l)-[:HAS_ARTICLE]->(a)
        """, ha)
        logger.info(f"  HAS_ARTICLE: {len(ha)}")

        # (LegalCase)-[:CITES]->(Law)  从 law_cases 关联表
        cites = [{"case_id": lc.case_id, "law_id": lc.law_id} for lc in law_cases]
        if cites:
            batch_run(s, """
                UNWIND $params AS p
                MATCH (c:LegalCase {pg_id: p.case_id})
                MATCH (l:Law       {pg_id: p.law_id})
                MERGE (c)-[:CITES]->(l)
            """, cites)
        logger.info(f"  CITES: {len(cites)}")

        # (Law)-[:APPLIES_TO]->(Domain)
        at = [{"law_id": l.id, "domain": l.domain} for l in laws if l.domain]
        batch_run(s, """
            UNWIND $params AS p
            MATCH (l:Law    {pg_id: p.law_id})
            MATCH (d:Domain {name:  p.domain})
            MERGE (l)-[:APPLIES_TO]->(d)
        """, at)
        logger.info(f"  APPLIES_TO: {len(at)}")

        # (Channel)-[:HANDLES]->(Domain)
        h = [{"ch_id": ch.id, "domain": ch.domain} for ch in channels if ch.domain]
        batch_run(s, """
            UNWIND $params AS p
            MATCH (c:Channel {pg_id: p.ch_id})
            MATCH (d:Domain  {name:  p.domain})
            MERGE (c)-[:HANDLES]->(d)
        """, h)
        logger.info(f"  HANDLES: {len(h)}")

    driver.close()
    logger.info("法律知识图谱构建完成")
    logger.info("验证（Neo4j Browser: http://localhost:7475）:")
    logger.info("  MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
    logger.info("  MATCH (l:Law)-[:APPLIES_TO]->(d:Domain)<-[:HANDLES]-(c:Channel)")
    logger.info("  RETURN l.title, d.name, c.name LIMIT 10")


if __name__ == "__main__":
    init_graph()
