"""法律 NL2SQL：渠道/法律结构化查询，带安全校验和重试。"""
from __future__ import annotations

import asyncio
import json
import re
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.agents.legal_knowledge.prompts import LEGAL_NL2SQL_PROMPT, LEGAL_SQL_QA_PROMPT

MAX_SQL_RETRIES = 2
SQL_TIMEOUT_SECONDS = 10

_FORBIDDEN = [
    re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    ),
]


def _validate_sql(sql: str) -> tuple[bool, str]:
    stripped = sql.strip().rstrip(";")
    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"
    for pattern in _FORBIDDEN:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作"
    if "LIMIT" not in stripped.upper():
        stripped += " LIMIT 20"
    return True, stripped


async def _generate_sql(
    question: str, llm: BaseChatModel, error_hint: str = ""
) -> str:
    extra = (
        f"\n\n上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。"
        if error_hint
        else ""
    )
    prompt = LEGAL_NL2SQL_PROMPT.format(question=question) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql


async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> tuple[list[dict], str]:
    """生成 SQL → 校验 → 执行（带重试），返回 (rows, validated_sql)。"""
    error_hint = ""
    validated_sql = ""
    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(question, llm, error_hint)
        logger.info(f"法律 NL2SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated_sql = _validate_sql(raw_sql)
        if not valid:
            logger.warning(f"SQL 安全校验失败: {validated_sql}")
            return [], ""

        try:
            result = await asyncio.wait_for(
                db.execute(text(validated_sql)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
            rows = [dict(row) for row in result.mappings().all()]
            return rows, validated_sql
        except asyncio.TimeoutError:
            logger.warning(f"SQL 执行超时: {validated_sql}")
            return [], validated_sql
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_SQL_RETRIES:
                return [], validated_sql
    return [], validated_sql


async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> str:
    """NL2SQL 完整流程：生成 → 执行 → LLM 整合回答。"""
    rows, sql = await search_sql_raw(question, llm, db)

    if not rows:
        return "未查询到相关数据，建议换一种描述方式。"

    result_str = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
    prompt = LEGAL_SQL_QA_PROMPT.format(question=question, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
