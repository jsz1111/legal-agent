"""公民法律指引 Worker — 对标 inquiry_agent.py，调用 legal_guide/graph.py 状态机。"""
from src.agents.legal_guide.graph import run_guide, build_guide_deps
from src.agents.legal_guide.state import GuidePhase
from src.infra.redis_cache import get_checkpointer_redis


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
    deps = build_guide_deps()
    thread_id = f"{user_id}:{session_id}"

    reply, new_state = await run_guide(
        user_message=message,
        thread_id=thread_id,
        deps=deps,
        existing_state=None,
        user_id=user_id,
        long_term_memories=long_term_memories or [],
    )

    # 首轮即收敛（CRITICAL 紧急情形）：不设活跃标记
    if new_state.phase == GuidePhase.END:
        return reply

    # 指引未结束：保存状态，设置活跃标记，等待后续轮次
    redis = get_checkpointer_redis()
    state_key = f"guide_state:{user_id}:{session_id}"
    await redis.set(state_key, new_state.model_dump_json(), ex=3600)

    active_key = f"guide_active:{user_id}:{session_id}"
    await redis.set(active_key, "1", ex=3600)

    return reply
