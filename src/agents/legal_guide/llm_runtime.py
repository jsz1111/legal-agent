"""Bounded LLM invocation helpers for the legal-guide request path."""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from src.core.config import get_settings


def build_chat_llm(*, temperature: float = 0.3, model: str | None = None) -> Any:
    """数据驱动构造聊天模型：DashScope OpenAI 兼容模式。

    供应商/模型切换只需改 .env 的 CHAT_MODEL / BASE_URL_CHAT / DASHSCOPE_API_KEY。
    """
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=temperature,
    )


def llm_for_stage(
    llm: Any,
    *,
    max_tokens: int,
) -> Any:
    """Configure short structured stages without changing test doubles.

    Only a token cap is bound here.  Vendor switching is handled by
    build_chat_llm; no vendor-specific reasoning parameters (e.g. DeepSeek's
    reasoning_effort) are bound, since DashScope's OpenAI-compatible mode does
    not recognize them.
    """

    if isinstance(llm, BaseChatModel):
        return llm.bind(max_tokens=max(int(max_tokens), 1))
    return llm


async def ainvoke_bounded(
    llm: Any,
    messages: list[Any],
    *,
    timeout: float | None,
    stage: str,
) -> Any:
    """Invoke an LLM within a stage-owned latency budget.

    Callers remain responsible for a domain-safe fallback.  Keeping the helper
    small makes timeout behavior explicit at each trust boundary instead of
    silently retrying and multiplying end-to-end latency.
    """

    if timeout is None or float(timeout) <= 0:
        return await llm.ainvoke(messages)
    try:
        return await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=float(timeout),
        )
    except asyncio.TimeoutError:
        logger.warning("LLM阶段超时 | stage={} timeout={}s", stage, timeout)
        raise
