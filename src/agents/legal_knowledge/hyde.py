# src/agents/legal_knowledge/hyde.py（从 agents/knowledge/hyde.py 迁移）

from __future__ import annotations
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.agents.legal_knowledge.knowledge_prompts import (
    HYDE_PROMPT,
    LEGAL_STATUTE_HYDE_PROMPT,
    LEGAL_CASE_HYDE_PROMPT,
)

_PROMPT_MAP = {
    "statute": LEGAL_STATUTE_HYDE_PROMPT,
    "case":    LEGAL_CASE_HYDE_PROMPT,
    "general": HYDE_PROMPT,
}


async def generate_hyde_embedding(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    mode: str = "general",  # "statute" | "case" | "general"
) -> list[float]:
    """
    HyDE（Hypothetical Document Embeddings）：
    1. LLM 按模式生成对应格式的假设文档
       - statute：假设法条原文（《XXX》第X条：...）
       - case：假设案件事实陈述
       - general：通用假设回答
    2. 对假设文档做向量化
    3. 用假设文档向量去检索，召回率高于原始问句向量
    """
    prompt_tpl = _PROMPT_MAP.get(mode, HYDE_PROMPT)
    prompt = prompt_tpl.format(question=question)
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        hypothetical_doc = response.content.strip()
        logger.debug(f"HyDE 假设文档: {hypothetical_doc[:100]}...")
        return await embedding_model.aembed_query(hypothetical_doc)
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始查询向量: {e}")
        return await embedding_model.aembed_query(question)
