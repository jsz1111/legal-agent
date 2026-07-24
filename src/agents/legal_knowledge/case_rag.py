"""类案语义检索：case_index → PG 补充完整案情 → 生成回答。"""
from __future__ import annotations

import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.agents.legal_knowledge.prompts import CASE_QA_PROMPT

COLLECTION_NAME = "case_index"
NO_DATA_DOMAINS = set()   # 无类案数据时动态添加，此处不预设


async def search_cases_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 5,
    domain: str = "",
) -> list[dict]:
    """类案向量检索，返回原始结果列表。"""
    query_vec = await embedding_model.aembed_query(question)

    filter_expr = f'domain == "{domain}"' if domain else None

    try:
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=["id", "domain", "source", "text"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            filter=filter_expr,
        )
    except Exception as e:
        logger.warning(f"类案检索失败: {e}")
        return []

    if not results or not results[0]:
        return []

    return [
        {
            "id": hit["entity"]["id"],
            "domain": hit["entity"]["domain"],
            "source": hit["entity"]["source"],
            "text": hit["entity"]["text"],
            "score": hit.get("distance", 0.0),
        }
        for hit in results[0]
    ]


async def _fetch_case_details(
    hits: list[dict], db_session: AsyncSession
) -> dict[int, dict]:
    """从 PG 批量取完整案例信息（title/gist），返回 {case_id: {...}}。"""
    from src.modules.legal.model import LegalCase
    case_ids = [int(h["id"]) for h in hits if h.get("id")]
    if not case_ids:
        return {}
    rows = (
        await db_session.execute(
            select(LegalCase.id, LegalCase.title, LegalCase.gist, LegalCase.source)
            .where(LegalCase.id.in_(case_ids))
        )
    ).all()
    return {
        row.id: {"title": row.title, "gist": row.gist, "source": row.source}
        for row in rows
    }


def format_case_context(hits: list[dict], details: dict[int, dict]) -> str:
    """格式化类案上下文字符串。"""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        detail = details.get(int(hit["id"]), {})
        title = detail.get("title") or f"案例{i}"
        gist = detail.get("gist") or ""
        source = hit.get("source", "")
        facts_snippet = hit["text"]

        lines = [f"案例{i}【{title}】（来源：{source}）"]
        lines.append(f"案情摘要：{facts_snippet}")
        if gist:
            lines.append(f"裁判要旨：{gist}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


async def search_cases(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    db_session: AsyncSession | None = None,
    domain: str = "",
) -> str:
    """类案 RAG 完整流程：检索 → 补充案情 → 生成回答。"""
    hits = await search_cases_raw(question, embedding_model, milvus_client, domain=domain)

    if not hits:
        return json.dumps(
            {"cases": [], "message": "暂无类案数据，建议通过法条检索了解相关法律规定"},
            ensure_ascii=False,
        )

    details: dict[int, dict] = {}
    if db_session is not None:
        details = await _fetch_case_details(hits, db_session)

    context = format_case_context(hits, details)
    prompt = CASE_QA_PROMPT.format(question=question, context=context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
