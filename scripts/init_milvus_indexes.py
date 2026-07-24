"""
Milvus 索引初始化：
  statute_index — 法条语义检索（Article.content → embedding）
  case_index    — 类案语义检索（LegalCase.facts[:512] → embedding）

用法：
    python scripts/init_milvus_indexes.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)
from sqlalchemy import select

from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Article, Law, LegalCase

settings = get_settings()
EMBEDDING_DIM = 1024
MILVUS_ALIAS  = "legal_init"
BATCH_EMBED   = 100


def ensure_collection(name: str, fields: list, alias: str) -> Collection:
    if utility.has_collection(name, using=alias):
        print(f"[INFO] {name} 已存在，删除重建")
        utility.drop_collection(name, using=alias)

    schema = CollectionSchema(fields, description=name)
    col = Collection(name, schema, using=alias)
    col.create_index("embedding", {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    })
    col.load()
    print(f"[INFO] {name} 创建成功")
    return col


def trunc_bytes(s: str, max_bytes: int = 3000) -> str:
    """按字节截断字符串，避免 Milvus VARCHAR 超长。"""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


async def fetch_articles() -> list[dict]:
    async with AsyncSessionLocal() as session:
        arts  = (await session.execute(select(Article))).scalars().all()
        laws  = {l.id: l.domain or "" for l in
                 (await session.execute(select(Law))).scalars().all()}
    return [
        {
            "id":         f"{a.law_id}_{a.article_no}",
            "domain":     laws.get(a.law_id, ""),
            "law_id":     str(a.law_id),
            "article_no": a.article_no,
            "text":       trunc_bytes(a.content or "", 2000),
            "_full_text": a.content or "",   # 仅用于embedding，不写入Milvus
        }
        for a in arts
    ]


async def fetch_cases(limit: int = 100) -> list[dict]:
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(LegalCase).limit(limit))).scalars().all()
    return [
        {
            "id":     str(c.id),
            "domain": c.domain or "",
            "source": c.source or "",
            "text":   (c.facts or "")[:512],
        }
        for c in rows
    ]


async def embed_and_upsert(col: Collection, rows: list[dict], text_key: str, embed_model) -> int:
    """向量化并分批 upsert，去重同批内重复主键。"""
    total = 0
    seen_ids: set[str] = set()
    for i in range(0, len(rows), BATCH_EMBED):
        batch = rows[i : i + BATCH_EMBED]
        texts = [r.get("_full_text", r[text_key]) for r in batch]
        embeddings = await embed_model.aembed_documents(texts)
        records = []
        for row, emb in zip(batch, embeddings):
            rid = row["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            record = {k: v for k, v in row.items() if not k.startswith("_")}
            record["embedding"] = emb
            records.append(record)
        if records:
            col.upsert(records)
            col.flush()
            total += len(records)
        print(f"  {min(i + BATCH_EMBED, len(rows))}/{len(rows)}")
    return total


async def build_indexes():
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    print(f"[INFO] Milvus 连接成功 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})")

    statute_col = ensure_collection("statute_index", [
        FieldSchema("id",         DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema("domain",     DataType.VARCHAR, max_length=100),
        FieldSchema("law_id",     DataType.VARCHAR, max_length=32),
        FieldSchema("article_no", DataType.VARCHAR, max_length=64),
        FieldSchema("text",       DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding",  DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ], MILVUS_ALIAS)

    case_col = ensure_collection("case_index", [
        FieldSchema("id",        DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("domain",    DataType.VARCHAR, max_length=100),
        FieldSchema("source",    DataType.VARCHAR, max_length=50),
        FieldSchema("text",      DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ], MILVUS_ALIAS)

    embed = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    print("[INFO] 向量化法条 → statute_index ...")
    articles = await fetch_articles()
    n = await embed_and_upsert(statute_col, articles, "text", embed)
    print(f"[INFO] statute_index 写入 {n} 条")

    print("[INFO] 向量化案例 → case_index ...")
    cases = await fetch_cases()
    n = await embed_and_upsert(case_col, cases, "text", embed)
    print(f"[INFO] case_index 写入 {n} 条")

    connections.disconnect(alias=MILVUS_ALIAS)
    print("[INFO] 完成")


if __name__ == "__main__":
    asyncio.run(build_indexes())
