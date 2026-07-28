"""同步追问依据到 PostgreSQL，并可选建立 Milvus 解释索引。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.agents.legal_guide.authority_registry import (
    PROJECT_ROOT,
    build_authority_index_rows,
    export_registry_payload,
    sync_authority_registry,
)
from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.infra.embedding import get_embedding_model


COLLECTION_NAME = "authority_basis_index"
EXPORT_PATH = PROJECT_ROOT / "data/legal_guide/authority_registry_export.json"


async def sync_postgres() -> dict[str, int]:
    async with AsyncSessionLocal() as session:
        return await sync_authority_registry(session)


async def sync_milvus() -> int:
    settings = get_settings()
    rows = build_authority_index_rows()
    embedding_model = get_embedding_model()
    embeddings = await embedding_model.aembed_documents([row["text"] for row in rows])
    if not embeddings:
        return 0
    dim = len(embeddings[0])
    alias = "authority_sync"
    connections.connect(alias=alias, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    try:
        if not utility.has_collection(COLLECTION_NAME, using=alias):
            schema = CollectionSchema([
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=40),
                FieldSchema("domain", DataType.VARCHAR, max_length=100),
                FieldSchema("rule_id", DataType.VARCHAR, max_length=120),
                FieldSchema("rule_type", DataType.VARCHAR, max_length=20),
                FieldSchema("source_key", DataType.VARCHAR, max_length=120),
                FieldSchema("title", DataType.VARCHAR, max_length=600),
                FieldSchema("source_url", DataType.VARCHAR, max_length=1500),
                FieldSchema("locator", DataType.VARCHAR, max_length=2000),
                FieldSchema("mapping_status", DataType.VARCHAR, max_length=40),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
            ], description="权威追问依据的派生解释索引；不参与确定性流程决策")
            collection = Collection(COLLECTION_NAME, schema=schema, using=alias)
            collection.create_index("embedding", {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 128},
            })
        else:
            collection = Collection(COLLECTION_NAME, using=alias)
            existing_dim = collection.schema.get_field("embedding").params.get("dim")
            if int(existing_dim) != dim:
                raise RuntimeError(f"{COLLECTION_NAME} 维度为 {existing_dim}，当前模型输出 {dim}")
        records = [{**row, "embedding": vector} for row, vector in zip(rows, embeddings)]
        collection.upsert(records)
        collection.flush()
        collection.load()
        return len(records)
    finally:
        connections.disconnect(alias=alias)


async def main(index_milvus: bool) -> None:
    payload = export_registry_payload()
    EXPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pg_counts = await sync_postgres()
    print(f"PostgreSQL: {pg_counts}")
    print(f"离线导出: {EXPORT_PATH}")
    if index_milvus:
        count = await sync_milvus()
        print(f"Milvus {COLLECTION_NAME}: {count} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-milvus", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.index_milvus))
