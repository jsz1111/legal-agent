# src/agents/legal_knowledge/doc_rag.py（从 agents/knowledge/doc_rag.py 迁移）

from __future__ import annotations
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

from src.agents.legal_knowledge.knowledge_prompts import DOC_QA_PROMPT

COLLECTION_NAME = "knowledge_docs"


async def search_docs_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
) -> list[dict]:
    """
    文档向量检索，返回原始检索结果列表。
    支持 HyDE 增强检索和 Reranker 精排。
    """
    if use_hyde and llm is not None:
        from src.agents.legal_knowledge.hyde import generate_hyde_embedding
        query_vec = await generate_hyde_embedding(question, llm, embedding_model)
    else:
        query_vec = await embedding_model.aembed_query(question)

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    filter_expr = f'doc_type == "{doc_type}"' if doc_type else None

    try:
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=["doc_name", "doc_type", "page_number", "chunk_index", "text"],
            search_params=search_params,
            filter=filter_expr,
        )
    except Exception as e:
        logger.warning(f"文档检索失败: {e}")
        return []

    if not results or not results[0]:
        return []

    hits = [
        {**hit["entity"], "score": hit.get("distance", 0.0)}
        for hit in results[0]
    ]

    from src.agents.legal_knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    return reranked


def format_doc_context(hits: list[dict]) -> str:
    """将检索结果格式化为 LLM 可读的上下文字符串。"""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        source = f"[{hit['doc_name']}, 第{hit.get('page_number', '?')}页]"
        parts.append(f"片段{i} {source}:\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


async def search_docs(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    role: str = "user",
    use_hyde: bool = True,
) -> str:
    """文档 RAG 检索 + Reranker 精排 + HyDE 增强 + 生成回答。"""
    hits = await search_docs_raw(
        question, embedding_model, milvus_client,
        top_k=top_k, rerank_top_k=rerank_top_k, doc_type=doc_type,
        llm=llm, use_hyde=use_hyde,
    )
    if not hits:
        return "当前知识库中未找到与您问题相关的文档内容。"

    context = format_doc_context(hits)
    prompt = DOC_QA_PROMPT.format(question=question, context=context, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
