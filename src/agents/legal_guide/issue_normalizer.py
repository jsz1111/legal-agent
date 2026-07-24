"""三层法律问题标准化：LLM提取 → Neo4j领域确认 → Milvus语义兜底。"""
from __future__ import annotations

import json
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.agents.legal_guide.prompts import ISSUE_EXTRACT_PROMPT

# with_structured_output 失败时的降级提示词（要求输出 JSON）
_ISSUE_EXTRACT_FALLBACK_PROMPT = ISSUE_EXTRACT_PROMPT + """

请严格输出 JSON，格式如下（只输出 JSON，不要有其他文字）：
{{"issues": ["法律问题1", "法律问题2"], "domain": "领域代码"}}"""


class IssuesOutput(BaseModel):
    """LLM 结构化输出 Schema。"""
    issues: list[str] = Field(description="标准化的法律问题列表，无则空列表")
    domain: str = Field(description="推断的法律领域代码，如 labor_social_security")


async def extract_legal_issues(user_input: str, llm: BaseChatModel) -> IssuesOutput:
    """第一层：LLM with_structured_output 提取+标准化；失败时降级为手动 JSON 解析。"""
    structured_llm = llm.with_structured_output(IssuesOutput)
    prompt = ISSUE_EXTRACT_PROMPT.format(user_input=user_input)
    try:
        result: IssuesOutput = await structured_llm.ainvoke([SystemMessage(content=prompt)])
        logger.debug(f"LLM提取法律问题: {result.issues}, domain: {result.domain}")
        if result.issues:
            return result
        # issues 为空时也尝试降级，防止 with_structured_output 静默返回空
    except Exception as e:
        logger.warning(f"LLM结构化输出失败，尝试手动解析: {e}")

    # 降级：用明确要求 JSON 的 prompt 重试
    try:
        fallback_prompt = _ISSUE_EXTRACT_FALLBACK_PROMPT.format(user_input=user_input)
        response = await llm.ainvoke([SystemMessage(content=fallback_prompt)])
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        data = json.loads(content)
        result = IssuesOutput(
            issues=[i for i in data.get("issues", []) if i],
            domain=data.get("domain", "other") or "other",
        )
        logger.debug(f"手动解析法律问题: {result.issues}, domain: {result.domain}")
        return result
    except Exception as e2:
        logger.warning(f"手动解析也失败: {e2}")
        return IssuesOutput(issues=[], domain="other")


async def confirm_domain_in_neo4j(domain: str, neo4j_driver: AsyncDriver) -> str:
    """第二层：Neo4j 确认领域节点存在。节点不存在时仍保留LLM推断值。"""
    if not domain or domain == "other":
        return domain
    try:
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (d:Domain {name: $name}) RETURN d.name AS name LIMIT 1",
                name=domain,
            )
            record = await result.single()
            confirmed = record is not None
            logger.debug(f"Neo4j领域确认 {domain}: {'命中' if confirmed else '未找到节点，保留推断值'}")
            return domain  # 无论是否命中，保留LLM推断值（图谱可能未建全节点）
    except Exception as e:
        logger.warning(f"Neo4j领域确认失败: {e}")
        return domain


SEMANTIC_THRESHOLD = 0.70


async def semantic_fallback(
    issues: list[str],
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> tuple[list[str], list[str]]:
    """第三层：statute_index 语义兜底。返回 (可分类, 真正无法分类)。"""
    if not issues:
        return [], []
    query_embeddings = await embedding_model.aembed_documents(issues)
    classifiable, unmatched = [], []
    for issue, vec in zip(issues, query_embeddings):
        try:
            results = milvus_client.search(
                collection_name="statute_index",
                data=[vec],
                limit=1,
                output_fields=["domain"],
            )
            if results and results[0]:
                score = results[0][0]["distance"]
                if score >= SEMANTIC_THRESHOLD:
                    classifiable.append(issue)
                    logger.debug(f"语义兜底命中: '{issue}' score={score:.3f}")
                else:
                    unmatched.append(issue)
            else:
                unmatched.append(issue)
        except Exception as e:
            logger.warning(f"Milvus语义兜底失败 '{issue}': {e}")
            unmatched.append(issue)
    return classifiable, unmatched


async def normalize_legal_issues(
    user_input: str,
    llm: BaseChatModel,
    neo4j_driver: AsyncDriver,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> dict:
    """
    完整三层法律问题标准化入口。

    Returns:
        confirmed : 标准化后可处理的法律问题列表
        unmatched : 真正无法分类的描述（语义距离过低）
        domain    : 确认后的法律领域代码
    """
    extracted = await extract_legal_issues(user_input, llm)
    issues = [i.strip() for i in extracted.issues if i.strip()]
    domain = await confirm_domain_in_neo4j(extracted.domain, neo4j_driver)

    if not issues:
        return {"confirmed": [], "unmatched": [], "domain": domain}

    classifiable, unmatched = await semantic_fallback(issues, embedding_model, milvus_client)
    # 语义兜底失败时仍保留LLM提取结果，不丢弃
    return {
        "confirmed": classifiable if classifiable else issues,
        "unmatched": unmatched if classifiable else [],
        "domain": domain,
    }
