"""
修复 Milvus 中存储的 BM25 稀疏向量。

问题：init_milvus_indexes.py 原来用 BM25Okapi 的文档相似度分数作为稀疏向量，
      query 侧用 BM25EmbeddingFunction 输出 term-id 权重向量——两套空间完全不兼容。

修复：
  1. 从 Milvus 读取所有记录的 text + 已有 dense embedding（不重调 Volcengine API）
  2. 用 BM25EmbeddingFunction 对全语料 fit，建立统一 term 词表
  3. 重算所有记录的 sparse_embedding
  4. upsert 回 Milvus（dense embedding 原封不动复用）
  5. 把 fitted BM25 模型保存到 models/ 目录，供 statute_rag 查询时加载

用法：
    python scripts/repair_sparse_bm25.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient, connections, Collection, utility
from milvus_model.sparse import BM25EmbeddingFunction
from src.core.config import get_settings

settings = get_settings()
MILVUS_URI = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BATCH = 500


def query_all(client: MilvusClient, collection_name: str, output_fields: list) -> list[dict]:
    """分页读取 collection 全量数据。"""
    rows = []
    offset = 0
    limit = 500
    while True:
        batch = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=output_fields,
            limit=limit,
            offset=offset,
        )
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < limit:
            break
    return rows


def coo_to_dict(coo) -> dict:
    """coo_array → {int: float} Milvus 接受的稀疏向量格式。"""
    import scipy.sparse as sp
    csr = sp.csr_array(coo)
    return {int(idx): float(val) for idx, val in zip(csr.indices, csr.data)}


def repair_collection(client: MilvusClient, collection_name: str,
                      all_fields: list[str], model_path: Path) -> int:
    print(f"\n[{collection_name}] 读取记录...")
    rows = query_all(client, collection_name, all_fields)
    if not rows:
        print(f"  空集合，跳过")
        return 0
    print(f"  读取 {len(rows)} 条")

    texts = [r["text"] for r in rows]

    print(f"  拟合 BM25 语料（{len(texts)} 条）...")
    bm25 = BM25EmbeddingFunction()
    bm25.fit(texts)

    MODELS_DIR.mkdir(exist_ok=True)
    bm25.save(str(model_path))
    print(f"  模型已保存 → {model_path}")

    print(f"  生成稀疏向量...")
    sparse_vecs = bm25.encode_documents(texts)

    print(f"  回写 Milvus（{len(rows)} 条）...")
    updated = 0
    for i in range(0, len(rows), BATCH):
        batch_rows = rows[i : i + BATCH]
        batch_sparse = sparse_vecs[i : i + BATCH]
        records = []
        for row, sv in zip(batch_rows, batch_sparse):
            rec = {k: row[k] for k in all_fields if k != "sparse_embedding"}
            rec["sparse_embedding"] = coo_to_dict(sv)
            records.append(rec)
        client.upsert(collection_name=collection_name, data=records)
        updated += len(records)
        print(f"  {min(i + BATCH, len(rows))}/{len(rows)}")

    return updated


def main():
    client = MilvusClient(uri=MILVUS_URI)
    print(f"Milvus 连接成功：{MILVUS_URI}")

    # statute_index
    if "statute_index" in client.list_collections():
        n = repair_collection(
            client,
            collection_name="statute_index",
            all_fields=["id", "domain", "law_id", "article_no", "text", "embedding"],
            model_path=MODELS_DIR / "bm25_statute.json",
        )
        print(f"[statute_index] 完成，更新 {n} 条")
    else:
        print("[statute_index] 不存在，跳过")

    # case_index
    if "case_index" in client.list_collections():
        n = repair_collection(
            client,
            collection_name="case_index",
            all_fields=["id", "domain", "source", "text", "embedding"],
            model_path=MODELS_DIR / "bm25_case.json",
        )
        print(f"[case_index] 完成，更新 {n} 条")
    else:
        print("[case_index] 不存在，跳过")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
