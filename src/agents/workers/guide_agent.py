"""公民法律指引 Worker — 调用 legal_guide/graph.py 状态机。"""
import json

from src.agents.legal_guide.graph import run_guide, build_guide_deps
from src.agents.legal_guide.state import GuidePhase
from src.core.config import get_settings
from src.infra.redis_cache import get_checkpointer_redis
from src.infra.database import AsyncSessionLocal

_DEBUG_TTL = 120  # 调试信息保留 2 分钟，供路由层读取后展示
settings = get_settings()


def _save_debug_key(user_id: str, session_id: str) -> str:
    return f"guide_last_debug:{user_id}:{session_id}"


def _save_reply_key(user_id: str, session_id: str) -> str:
    return f"guide_last_reply:{user_id}:{session_id}"


async def call_guide_agent_impl(
    message: str,
    user_id: str,
    session_id: str,
    long_term_memories: list[str] | None = None,
) -> str:
    """
    执行公民法律指引首轮对话，保存状态并设置活跃标记。
    供 worker_tools.call_guide_agent 直接调用。
    """
    thread_id = f"{user_id}:{session_id}"

    async with AsyncSessionLocal() as db_session:
        deps = build_guide_deps(db_session=db_session)
        reply, new_state = await run_guide(
            user_message=message,
            thread_id=thread_id,
            deps=deps,
            existing_state=None,
            user_id=user_id,
            long_term_memories=long_term_memories or [],
        )

    redis = get_checkpointer_redis()

    # 保存调试信息 + guide_agent原始回复（供路由层透传，短TTL）
    try:
        debug_data = {
            "domain":           new_state.legal_domain or "",
            "confidence_tier":  new_state.confidence_tier or "",
            "statute_hits":     new_state.law_context_str or "",
            "case_hits":        new_state.case_context_str or "",
            "graph_laws":       new_state.candidate_laws or [],
            "graph_channels":   new_state.relevant_channels or [],
            "fallback_guide":   new_state.fallback_guide,
        }
        await redis.set(
            _save_debug_key(user_id, session_id),
            json.dumps(debug_data, ensure_ascii=False),
            ex=_DEBUG_TTL,
        )
        # 原始回复存 Redis，让 chat.py 直接取用，绕过 Supervisor 重写
        await redis.set(
            _save_reply_key(user_id, session_id),
            reply,
            ex=_DEBUG_TTL,
        )
    except Exception:
        pass

    state_key = f"guide_state:{user_id}:{session_id}"
    active_key = f"guide_active:{user_id}:{session_id}"
    ttl = settings.GUIDE_SESSION_TTL

    # 已识别具体法律问题的终态仍需保留，下一轮可生成或重生成参考文书。
    if new_state.phase == GuidePhase.END:
        if new_state.confirmed_issues:
            await redis.set(state_key, new_state.model_dump_json(), ex=ttl)
            await redis.set(active_key, "1", ex=ttl)
        return reply

    # 指引未结束：保存状态，设置活跃标记，等待后续轮次
    await redis.set(state_key, new_state.model_dump_json(), ex=ttl)
    await redis.set(active_key, "1", ex=ttl)

    return reply
