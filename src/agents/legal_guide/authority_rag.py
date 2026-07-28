"""权威追问依据的可选语义检索，不参与工作流分支决策。"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from loguru import logger
from pymilvus import MilvusClient


COLLECTION_NAME = "authority_basis_index"


async def search_authority_basis_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    *,
    domain: str = "",
    top_k: int = 5,
) -> list[dict]:
    """检索派生依据说明，结果必须回链 PostgreSQL/官方来源后才能对外引用。"""
    query_vector = await embedding_model.aembed_query(question)
    filter_expr = f'domain == "{domain}"' if domain else None
    try:
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            anns_field="embedding",
            limit=top_k,
            output_fields=[
                "domain", "rule_id", "rule_type", "source_key", "title",
                "source_url", "locator", "mapping_status", "text",
            ],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            filter=filter_expr,
        )
    except Exception as exc:
        logger.warning("权威追问依据检索失败，降级为结构化来源摘要: {}", exc)
        return []
    if not results or not results[0]:
        return []
    return [
        {**hit["entity"], "score": float(hit.get("distance", 0.0))}
        for hit in results[0]
    ]


def format_authority_basis_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts = []
    for hit in hits:
        status_note = (
            "具体条款仍待人工精确标注"
            if hit.get("mapping_status") == "needs_pinpoint"
            else "来源定位已登记"
        )
        parts.append(
            f"[{hit.get('rule_id')}] {hit.get('title')}（{status_note}）\n"
            f"{hit.get('locator')}\n{hit.get('source_url')}"
        )
    return "\n\n".join(parts)
