"""Bounded LLM invocation helpers for the legal-guide request path."""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from loguru import logger


def llm_for_stage(
    llm: Any,
    *,
    max_tokens: int,
    reasoning_effort: str = "low",
) -> Any:
    """Configure short structured stages without changing test doubles.

    The production model otherwise spends most of a small stage's latency and
    token budget on hidden reasoning, which can truncate the JSON body.
    """

    if isinstance(llm, BaseChatModel):
        return llm.bind(
            max_tokens=max(int(max_tokens), 1),
            reasoning_effort=reasoning_effort,
        )
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
