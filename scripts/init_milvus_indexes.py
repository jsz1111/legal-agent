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

from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)
from sqlalchemy import select

from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.infra.embedding import get_embedding_model
from src.modules.legal.model import Article, Law, LegalCase

settings = get_settings()
EMBEDDING_DIM = 1024
MILVUS_ALIAS  = "legal_init"
BATCH_EMBED   = 20


def ensure_collection(name: str, fields: list, alias: str, enable_sparse: bool = False, rebuild: bool = False) -> Collection:
    """
    确保 collection 存在。

    Args:
        rebuild: True=删除重建，False=保留已有数据（断点续传）
    """
    if utility.has_collection(name, using=alias):
        if rebuild:
            print(f"[INFO] {name} 已存在，删除重建")
            utility.drop_collection(name, using=alias)
        else:
            print(f"[INFO] {name} 已存在，断点续传模式（保留已有数据）")
            col = Collection(name, using=alias)
            col.load()
            return col

    schema = CollectionSchema(fields, description=name)
    col = Collection(name, schema, using=alias)

    # Dense 向量索引
    col.create_index("embedding", {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    })

    # Sparse 向量索引（使用 IP 距离，不用 BM25 metric）
    if enable_sparse:
        col.create_index("sparse_embedding", {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",  # BM25 向量用 IP 距离
        })

    col.load()
    print(f"[INFO] {name} 创建成功（sparse={enable_sparse}）")
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
            "text":       trunc_bytes(a.content or "", 60000),
            "_full_text": a.content or "",   # 仅用于embedding，不写入Milvus
        }
        for a in arts
    ]


async def fetch_cases(samples_per_domain: int = 2) -> list[dict]:
    """每个领域取样本案例（避免大量向量化消耗）。"""
    from sqlalchemy import func
    async with AsyncSessionLocal() as session:
        # 获取所有领域
        domains = (await session.execute(
            select(LegalCase.domain, func.count(LegalCase.id).label("cnt"))
            .group_by(LegalCase.domain)
        )).all()

        print(f"  [案例统计] 共 {len(domains)} 个领域")
        for domain, cnt in domains:
            print(f"    {domain}: {cnt} 条")

        # 每个领域取前 N 条
        all_cases = []
        for domain, _ in domains:
            rows = (await session.execute(
                select(LegalCase)
                .where(LegalCase.domain == domain)
                .limit(samples_per_domain)
            )).scalars().all()
            all_cases.extend(rows)

        print(f"  [采样结果] 共 {len(all_cases)} 条案例")

    return [
        {
            "id":     str(c.id),
            "domain": c.domain or "",
            "source": c.source or "",
            "text":   (c.facts or "")[:300],  # 展示摘要，回复用
            "_full_text": (c.facts or "") if not c.gist else f"{c.facts or ''}\n裁判要旨：{c.gist}",  # embedding 用完整 facts+gist
        }
        for c in all_cases
    ]


async def embed_and_upsert(col: Collection, rows: list[dict], text_key: str, embed_model) -> int:
    """向量化并分批 upsert，去重同批内重复主键。网络抖动自动重试3次。"""
    total = 0
    seen_ids: set[str] = set()
    for i in range(0, len(rows), BATCH_EMBED):
        batch = rows[i : i + BATCH_EMBED]
        texts = [r.get("_full_text", r[text_key]) for r in batch]
        for attempt in range(3):
            try:
                embeddings = await embed_model.aembed_documents(texts)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  [RETRY {attempt+1}] {type(e).__name__} → 等待{wait}s重试...")
                    await asyncio.sleep(wait)
                else:
                    raise
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


async def embed_and_upsert_hybrid(col: Collection, rows: list[dict], text_key: str, embed_model) -> int:
    """向量化并分批 upsert（Dense + Sparse BM25），去重同批内重复主键。网络抖动自动重试3次。

    支持断点续传：自动跳过已存在的记录。
    """
    from pymilvus import MilvusClient
    from milvus_model.sparse import BM25EmbeddingFunction
    import scipy.sparse as sp
    import jieba

    # 中文分词器（BM25 默认按空格分词，对中文无效）
    def chinese_tokenizer(text: str) -> list[str]:
        return jieba.lcut(text)

    # 获取已存在的 ID（断点续传）
    # TEMPORARILY DISABLED: 修复 BM25 需要全量重建 sparse_embedding
    milvus_client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    existing_ids = set()  # 强制为空，全量重建
    print(f"  [修复模式] BM25 全量重建，禁用断点续传")
    # try:
    #     offset = 0
    #     limit = 1000
    #     while True:
    #         results = milvus_client.query(
    #             collection_name=col.name,
    #             filter="",
    #             output_fields=["id"],
    #             limit=limit,
    #             offset=offset,
    #         )
    #         if not results:
    #             break
    #         existing_ids.update(r["id"] for r in results)
    #         offset += limit
    #         if len(results) < limit:
    #             break
    #     print(f"  [断点续传] 已索引 {len(existing_ids)} 条，跳过")
    # except Exception as e:
    #     print(f"  [断点续传] 查询失败（可能是空 collection）: {e}")

    # 过滤已存在的记录
    rows_to_process = [r for r in rows if r["id"] not in existing_ids]
    if not rows_to_process:
        print(f"  [完成] 所有记录已索引，无需处理")
        return len(existing_ids)

    print(f"  [待处理] {len(rows_to_process)} 条（总共 {len(rows)}）")

    # 初始化 BM25 模型（必须用全量数据训练，保证词表完整）
    all_texts_for_bm25 = [r.get("_full_text", r[text_key]) for r in rows]  # 用 rows 而非 rows_to_process
    bm25 = BM25EmbeddingFunction(analyzer=chinese_tokenizer)  # 使用中文分词器
    bm25.fit(all_texts_for_bm25)
    print(f"  [BM25] 训练语料: {len(all_texts_for_bm25)} 条（全量），使用 jieba 分词")

    # 保存模型供查询时加载
    from pathlib import Path
    _models_dir = Path(__file__).resolve().parent.parent / "models"
    _models_dir.mkdir(exist_ok=True)
    _model_name = "bm25_statute" if col.name == "statute_index" else "bm25_case"
    bm25.save(str(_models_dir / f"{_model_name}.json"))
    print(f"  [BM25] 模型已保存 → {_models_dir / (_model_name + '.json')}")

    # 预计算全量稀疏向量（必须用待处理的记录，不是全量 rows）
    all_texts_for_sparse = [r.get("_full_text", r[text_key]) for r in rows_to_process]
    all_sparse_coo = bm25.encode_documents(all_texts_for_sparse)
    all_sparse_dicts = [
        {int(idx): float(val) for idx, val in zip(sp.csr_array(sv).indices, sp.csr_array(sv).data)}
        for sv in all_sparse_coo
    ]

    total = len(existing_ids)  # 从已有数量开始计数
    seen_ids: set[str] = set()
    for i in range(0, len(rows_to_process), BATCH_EMBED):
        batch = rows_to_process[i : i + BATCH_EMBED]
        texts = [r.get("_full_text", r[text_key]) for r in batch]

        # Dense 向量
        for attempt in range(3):
            try:
                dense_embeddings = await embed_model.aembed_documents(texts)
                break
            except Exception as e:
                # 4xx（欠费/鉴权/参数错）重试不会好转，直接失败，除了 429 限流
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    print(f"  [FAIL] HTTP {status} 不可重试 → {str(e)[:200]}")
                    raise
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  [RETRY {attempt+1}] {type(e).__name__} → 等待{wait}s重试...")
                    await asyncio.sleep(wait)
                else:
                    raise

        # Sparse 向量（BM25）
        sparse_embeddings = all_sparse_dicts[i : i + BATCH_EMBED]

        records = []
        for row, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings):
            rid = row["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            record = {k: v for k, v in row.items() if not k.startswith("_")}
            record["embedding"] = dense_emb
            record["sparse_embedding"] = sparse_emb
            records.append(record)

        if records:
            col.upsert(records)
            col.flush()
            total += len(records)
        print(f"  {min(i + BATCH_EMBED, len(rows_to_process))}/{len(rows_to_process)}")

    return total


async def build_indexes():
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    print(f"[INFO] Milvus 连接成功 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})")

    # statute_index - 启用 RRF 混合检索（Dense + Sparse），断点续传模式
    statute_col = ensure_collection("statute_index", [
        FieldSchema("id",                DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema("domain",            DataType.VARCHAR, max_length=100),
        FieldSchema("law_id",            DataType.VARCHAR, max_length=32),
        FieldSchema("article_no",        DataType.VARCHAR, max_length=64),
        FieldSchema("text",              DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding",         DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("sparse_embedding",  DataType.SPARSE_FLOAT_VECTOR),  # BM25
    ], MILVUS_ALIAS, enable_sparse=True, rebuild=False)  # 断点续传

    # case_index - 启用 RRF 混合检索（每个领域取样本），断点续传模式
    case_col = ensure_collection("case_index", [
        FieldSchema("id",                DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("domain",            DataType.VARCHAR, max_length=100),
        FieldSchema("source",            DataType.VARCHAR, max_length=50),
        FieldSchema("text",              DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding",         DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("sparse_embedding",  DataType.SPARSE_FLOAT_VECTOR),  # BM25
    ], MILVUS_ALIAS, enable_sparse=True, rebuild=False)  # 断点续传

    embed = get_embedding_model()

    print("[INFO] 向量化法条 → statute_index (Dense + Sparse) ...")
    articles = await fetch_articles()
    n = await embed_and_upsert_hybrid(statute_col, articles, "text", embed)
    print(f"[INFO] statute_index 写入 {n} 条")

    print("[INFO] 向量化案例 → case_index (每个领域取样本 1-2 条) ...")
    try:
        cases = await fetch_cases(samples_per_domain=2)
        n = await embed_and_upsert_hybrid(case_col, cases, "text", embed)
        print(f"[INFO] case_index 写入 {n} 条")
    except Exception as e:
        print(f"[WARN] 案例向量化失败: {type(e).__name__}: {e}")
        print("[WARN] 法条已全部写入，案例部分跳过")

    connections.disconnect(alias=MILVUS_ALIAS)
    print("[INFO] 完成")


if __name__ == "__main__":
    asyncio.run(build_indexes())
