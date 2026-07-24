"""法律知识图谱 NL2Cypher 检索：实体提取 → Cypher 生成（带重试）→ 生成回答。"""
from __future__ import annotations

import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from neo4j import AsyncDriver

from src.agents.legal_knowledge.prompts import (
    LEGAL_ENTITY_EXTRACT_PROMPT,
    LEGAL_NL2CYPHER_PROMPT,
    LEGAL_GRAPH_QA_PROMPT,
)

MAX_CYPHER_RETRIES = 2


async def _extract_entities(question: str, llm: BaseChatModel) -> dict:
    prompt = LEGAL_ENTITY_EXTRACT_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"法律实体提取失败: {e}")
        return {"domain": [], "law": [], "legal_term": [], "channel": []}


async def _generate_cypher(
    question: str,
    entities: dict,
    llm: BaseChatModel,
    error_hint: str = "",
) -> str:
    extra = (
        f"\n\n上一次生成的 Cypher 执行报错：{error_hint}\n请修正后重新生成。"
        if error_hint
        else ""
    )
    prompt = (
        LEGAL_NL2CYPHER_PROMPT.format(
            question=question,
            entities=json.dumps(entities, ensure_ascii=False),
        )
        + extra
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    cypher = response.content.strip()
    if "```" in cypher:
        cypher = cypher.split("```")[1].lstrip("cypher").strip()
    return cypher


async def _execute_cypher(cypher: str, neo4j_driver: AsyncDriver) -> list[dict]:
    if not cypher:
        return []
    async with neo4j_driver.session() as session:
        result = await session.run(cypher)
        return await result.data()


async def search_graph_raw(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> list[dict]:
    """图谱检索，返回原始 Neo4j 查询结果（不经 LLM 生成）。"""
    entities = await _extract_entities(question, llm)
    logger.info(f"法律图谱实体提取: {entities}")

    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)
        logger.info(f"法律图谱 Cypher (attempt {attempt + 1}): {cypher}")
        try:
            records = await _execute_cypher(cypher, neo4j_driver)
            return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                return []
    return []


async def search_graph(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> str:
    """图谱 RAG 完整流程：实体提取 → NL2Cypher（带重试）→ LLM 整合。"""
    records = await search_graph_raw(question, neo4j_driver, llm)

    if not records:
        return "知识图谱中未找到与您问题相关的法律关系信息。"

    graph_result = json.dumps(records, ensure_ascii=False, indent=2)
    prompt = LEGAL_GRAPH_QA_PROMPT.format(
        question=question, graph_result=graph_result
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
