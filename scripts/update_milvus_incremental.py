"""
增量更新 Milvus 索引：只导入 PostgreSQL 中新增的数据

用法：
    python scripts/update_milvus_incremental.py --collection statute_index
    python scripts/update_milvus_incremental.py --collection case_index --samples 5
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import Collection, connections
from sqlalchemy import select

from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.infra.embedding import get_embedding_model
from src.modules.legal.model import Article, Law, LegalCase

settings = get_settings()
BATCH_EMBED = 20


async def get_existing_ids(collection_name: str) -> set[str]:
    """从 Milvus 获取已有的 ID 列表。"""
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    # 查询所有 ID（分批，避免内存溢出）
    existing_ids = set()
    offset = 0
    limit = 1000

    while True:
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id"],
            limit=limit,
            offset=offset,
        )
        if not results:
            break
        existing_ids.update(r["id"] for r in results)
        offset += limit
        if len(results) < limit:
            break

    return existing_ids


async def fetch_new_articles(existing_ids: set[str]) -> list[dict]:
    """获取 PostgreSQL 中新增的法条（不在 Milvus 的）。"""
    async with AsyncSessionLocal() as session:
        arts = (await session.execute(select(Article))).scalars().all()
        laws = {l.id: l.domain or "" for l in
                (await session.execute(select(Law))).scalars().all()}

    new_articles = []
    for a in arts:
        article_id = f"{a.law_id}_{a.article_no}"
        if article_id not in existing_ids:
            new_articles.append({
                "id": article_id,
                "domain": laws.get(a.law_id, ""),
                "law_id": str(a.law_id),
                "article_no": a.article_no,
                "text": (a.content or "")[:60000],
                "_full_text": a.content or "",
            })

    return new_articles


async def fetch_new_cases(existing_ids: set[str], samples_per_domain: int = 5) -> list[dict]:
    """获取 PostgreSQL 中新增的案例（每个领域取 N 条未索引的）。"""
    from sqlalchemy import func
    async with AsyncSessionLocal() as session:
        domains = (await session.execute(
            select(LegalCase.domain, func.count(LegalCase.id).label("cnt"))
            .group_by(LegalCase.domain)
        )).all()

        new_cases = []
        for domain, _ in domains:
            rows = (await session.execute(
                select(LegalCase)
                .where(LegalCase.domain == domain)
                .limit(samples_per_domain * 10)  # 多取一些，再过滤
            )).scalars().all()

            domain_new = [
                {
                    "id": str(c.id),
                    "domain": c.domain or "",
                    "source": c.source or "",
                    "text": (c.facts or "")[:300],
                    "_full_text": (c.facts or "") if not c.gist else f"{c.facts or ''}\n裁判要旨：{c.gist}",
                }
                for c in rows if str(c.id) not in existing_ids
            ]

            # 每个领域只取 samples_per_domain 条
            new_cases.extend(domain_new[:samples_per_domain])

        return new_cases


async def upsert_incremental(collection_name: str, rows: list[dict], embed_model) -> int:
    """增量插入（Dense + Sparse）。"""
    from pymilvus.model.sparse import BM25EmbeddingFunction

    if not rows:
        print("[INFO] 没有新数据需要插入")
        return 0

    # 连接
    connections.connect(
        alias="incremental",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    col = Collection(collection_name, using="incremental")

    # BM25 预训练
    all_texts = [r.get("_full_text", r["text"]) for r in rows]
    bm25_ef = BM25EmbeddingFunction()
    bm25_ef.fit(all_texts)

    total = 0
    for i in range(0, len(rows), BATCH_EMBED):
        batch = rows[i : i + BATCH_EMBED]
        texts = [r.get("_full_text", r["text"]) for r in batch]

        # Dense
        dense_embeddings = await embed_model.aembed_documents(texts)

        # Sparse
        sparse_embeddings = bm25_ef.encode_documents(texts)

        records = []
        for row, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings):
            record = {k: v for k, v in row.items() if not k.startswith("_")}
            record["embedding"] = dense_emb
            record["sparse_embedding"] = sparse_emb
            records.append(record)

        col.upsert(records)
        col.flush()
        total += len(records)
        print(f"  {min(i + BATCH_EMBED, len(rows))}/{len(rows)}")

    connections.disconnect(alias="incremental")
    return total


async def main(collection: str, samples: int = 5):
    print(f"=== 增量更新 {collection} ===\n")

    print("[1/4] 获取 Milvus 已有 ID...")
    existing_ids = await get_existing_ids(collection)
    print(f"  已有 {len(existing_ids)} 条")

    print("[2/4] 查询 PostgreSQL 新增数据...")
    if collection == "statute_index":
        new_rows = await fetch_new_articles(existing_ids)
    elif collection == "case_index":
        new_rows = await fetch_new_cases(existing_ids, samples_per_domain=samples)
    else:
        print(f"[ERROR] 不支持的 collection: {collection}")
        return

    print(f"  新增 {len(new_rows)} 条")

    if not new_rows:
        print("\n[完成] 无新增数据")
        return

    print("[3/4] 向量化并插入...")
    embed_model = get_embedding_model()
    n = await upsert_incremental(collection, new_rows, embed_model)

    print(f"[4/4] 完成，新增 {n} 条")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, choices=["statute_index", "case_index"])
    parser.add_argument("--samples", type=int, default=5, help="案例索引每个领域采样数量")
    args = parser.parse_args()

    asyncio.run(main(args.collection, args.samples))
