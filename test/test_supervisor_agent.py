"""监督 Agent 的短期与长期法律记忆集成测试。"""

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor_agent import _exact_name_recall, chat_endpoint
from src.infra.milvus_store import get_milvus_store
from src.infra.redis_cache import get_checkpointer_redis


def _test_identity(prefix: str) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:10]
    return f"{prefix}_{suffix}", uuid.uuid4().hex[:8]


async def _cleanup_test_state(user_id: str) -> None:
    redis = get_checkpointer_redis()
    keys = [key async for key in redis.scan_iter(match=f"*{user_id}*")]
    if keys:
        await redis.delete(*keys)

    store = get_milvus_store()
    namespace = ("users", user_id, "memories")
    items = await store.asearch(namespace, query=None, limit=100)
    for item in items:
        await store.aput(namespace, item.key, None)


async def test_agent_memory():
    """同一会话的第二轮应记得用户刚提供的称呼。"""
    user_id, session_id = _test_identity("test_short_memory")
    try:
        await chat_endpoint(user_id, session_id, "你好，我叫雷丰阳")
        reply = await chat_endpoint(user_id, session_id, "我刚才说我叫什么？")

        assert "雷丰阳" in reply
    finally:
        await _cleanup_test_state(user_id)


def test_exact_name_recall_preserves_full_name_from_same_session():
    messages = [
        HumanMessage(content="你好，我叫雷丰阳"),
        AIMessage(content="你好。"),
        HumanMessage(content="我刚才说我叫什么？"),
        AIMessage(content="雷先生。"),
    ]

    assert _exact_name_recall("我刚才说我叫什么？", messages) == "你刚才说你叫雷丰阳。"


async def test_agent_store():
    """法律案情摘要应能跨会话检索，且测试数据在结束后清理。"""
    user_id, first_session = _test_identity("test_legal_memory")
    second_session = uuid.uuid4().hex[:8]
    try:
        await chat_endpoint(
            user_id,
            first_session,
            "请记住：我在上海有劳动争议，老板已经拖欠我三个月工资。",
        )
        reply = await chat_endpoint(
            user_id,
            second_session,
            "我之前说的劳动争议是什么情况？",
        )

        assert "拖欠" in reply and ("三个月" in reply or "3个月" in reply)
    finally:
        await _cleanup_test_state(user_id)
