"""法条语义检索：statute_index → rerank → PG 补充 law title → 生成回答。"""
from __future__ import annotations

from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.agents.legal_knowledge.prompts import STATUTE_QA_PROMPT

COLLECTION_NAME = "statute_index"


async def search_statutes_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 20,
    rerank_top_k: int = 5,
    domain: str = "",
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
) -> list[dict]:
    """法条向量检索，返回原始结果列表。"""
    if use_hyde and llm is not None:
        from src.agents.knowledge.hyde import generate_hyde_embedding
        query_vec = await generate_hyde_embedding(question, llm, embedding_model)
    else:
        query_vec = await embedding_model.aembed_query(question)

    filter_expr = f'domain == "{domain}"' if domain else None

    try:
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=["law_id", "article_no", "domain", "text"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            filter=filter_expr,
        )
    except Exception as e:
        logger.warning(f"法条检索失败: {e}")
        return []

    if not results or not results[0]:
        return []

    hits = [
        {
            "law_id": hit["entity"]["law_id"],
            "article_no": hit["entity"]["article_no"],
            "domain": hit["entity"]["domain"],
            "text": hit["entity"]["text"],
            "score": hit.get("distance", 0.0),
        }
        for hit in results[0]
    ]

    from src.agents.knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    return reranked


async def _fetch_law_titles(
    hits: list[dict], db_session: AsyncSession
) -> dict[str, str]:
    """批量从 PG 取 law title，返回 {law_id_str: title}。"""
    from src.modules.legal.model import Law
    law_ids = list({int(h["law_id"]) for h in hits if h.get("law_id")})
    if not law_ids:
        return {}
    rows = (
        await db_session.execute(select(Law.id, Law.title).where(Law.id.in_(law_ids)))
    ).all()
    return {str(row.id): row.title for row in rows}


def format_statute_context(hits: list[dict], law_titles: dict[str, str]) -> str:
    """将法条检索结果格式化为 LLM 上下文字符串。"""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        title = law_titles.get(hit["law_id"], f"法律ID:{hit['law_id']}")
        parts.append(
            f"法条{i}【{title} {hit['article_no']}】\n{hit['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def search_statutes(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    db_session: AsyncSession | None = None,
    domain: str = "",
    use_hyde: bool = True,
    verify_grounding: bool = True,
) -> str:
    """法条 RAG 完整流程：检索 → 精排 → 补充标题 → 生成回答 → 自省校验。

    verify_grounding=True 时，生成回答后会用检索到的法条原文做幻觉校验，
    若回答含无法条支撑的陈述，追加免责提示并列出可疑内容——降低法条幻觉风险。
    """
    hits = await search_statutes_raw(
        question, embedding_model, milvus_client,
        domain=domain, llm=llm, use_hyde=use_hyde,
    )
    if not hits:
        return "当前法条库中未找到与您问题相关的法律条文。"

    law_titles: dict[str, str] = {}
    if db_session is not None:
        law_titles = await _fetch_law_titles(hits, db_session)

    context = format_statute_context(hits, law_titles)
    prompt = STATUTE_QA_PROMPT.format(question=question, context=context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    answer = response.content

    if verify_grounding:
        answer = await _apply_grounding_check(question, context, answer, llm)
    return answer


async def _apply_grounding_check(
    question: str, context: str, answer: str, llm: BaseChatModel,
) -> str:
    """对法条回答做幻觉校验，不可信时追加免责提示与可疑陈述清单。"""
    from src.agents.knowledge.hallucination_check import check_hallucination

    result = await check_hallucination(question, context, answer, llm)
    if result.get("is_grounded", True):
        return answer

    unsupported = result.get("unsupported_claims", [])
    conf = result.get("confidence", 0.0)
    logger.warning(
        "法条回答幻觉校验未通过 | confidence={} unsupported={}", conf, unsupported,
    )
    note_lines = [
        "\n\n---",
        "⚠️ **可信度提示**：以下内容未能在检索到的法条中找到直接依据，请谨慎参考，",
        "建议拨打 **12348** 法律援助热线向专业律师核实：",
    ]
    if unsupported:
        note_lines += [f"  - {c}" for c in unsupported[:5]]
    return answer + "\n".join(note_lines)
