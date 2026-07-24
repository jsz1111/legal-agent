"""
知识文档索引初始化脚本。
从 PostgreSQL drug_details 表导入药品说明书到 Milvus knowledge_docs collection。

用法：
    python scripts/init_knowledge_docs.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.database import AsyncSessionLocal

settings = get_settings()


async def main():
    print("[INFO] 初始化知识文档索引...")

    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    async with AsyncSessionLocal() as db:
        from src.agents.knowledge.doc_ingestion import ingest_drug_instructions
        total = await ingest_drug_instructions(
            embedding_model=embedding_model,
            milvus_client=milvus_client,
            db_session=db,
        )

    print(f"[INFO] 初始化完成，共导入 {total} 个分块。")


if __name__ == "__main__":
    asyncio.run(main())