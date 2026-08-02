"""Public, auditable progress events for the legal-guide workflow.

These events describe application stages and data sources.  They deliberately
do not expose model chain-of-thought, hidden prompts, or private reasoning.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator


ProgressEmitter = Callable[[dict], None]
_progress_emitter: ContextVar[ProgressEmitter | None] = ContextVar(
    "legal_guide_progress_emitter",
    default=None,
)


@contextmanager
def guide_progress_scope(emitter: ProgressEmitter) -> Iterator[None]:
    """Route progress emitted in the current async task tree to ``emitter``."""

    token = _progress_emitter.set(emitter)
    try:
        yield
    finally:
        _progress_emitter.reset(token)


def emit_guide_progress(
    stage: str,
    label: str,
    detail: str = "",
    *,
    status: str = "active",
) -> None:
    """Emit a safe workflow milestone when a streaming caller is listening."""

    emitter = _progress_emitter.get()
    if emitter is None:
        return
    emitter({
        "type": "progress",
        "stage": str(stage or "processing"),
        "label": str(label or "正在处理"),
        "detail": str(detail or ""),
        "status": status if status in {"active", "completed"} else "active",
    })
