# -*- coding: utf-8 -*-
"""只读检查：Milvus / PG 数据量 + 项目代码可引用。不改任何数据。"""
import sys
sys.path.insert(0, r"D:\learn\legal-agent")
from pymilvus import MilvusClient
from sqlalchemy import text

from src.core.config import get_settings
from src.infra.database import engine

s = get_settings()
print("== Milvus ==")
c = MilvusClient(uri=f"http://{s.MILVUS_HOST}:{s.MILVUS_PORT}")
cols = c.list_collections()
print("collections:", cols)
if "statute_index" in cols:
    print("statute_index rows:", c.get_collection_stats("statute_index").get("row_count"))
if "case_index" in cols:
    print("case_index rows:", c.get_collection_stats("case_index").get("row_count"))

print("== PostgreSQL ==")
import asyncio
async def main():
    async with engine.connect() as conn:
        laws = (await conn.execute(text("SELECT count(*) FROM laws"))).scalar()
        arts = (await conn.execute(text("SELECT count(*) FROM articles"))).scalar()
        print("laws:", laws, "articles:", arts)
        rows = (await conn.execute(text("SELECT law_id, article_no, substr(content,1,40) FROM articles ORDER BY id LIMIT 3"))).fetchall() if arts else []
        # columns of articles?
        cols_rows = (await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='articles' ORDER BY ordinal_position"))).fetchall()
        print("articles columns:", [r[0] for r in cols_rows])
    await engine.dispose()
asyncio.run(main())
print("== import check ==")
from src.agents.legal_knowledge.statute_rag import search_statutes_raw
from src.infra.embedding import get_embedding_model
print("search_statutes_raw OK; embedding provider:", s.EMBEDDING_PROVIDER, s.EMBEDDING_MODEL)
