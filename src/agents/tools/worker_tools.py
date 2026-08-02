# src/agents/tools/worker_tools.py

import json
from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.workers.guide_agent import call_guide_agent_impl
from src.agents.workers.legal_qa_agent import create_legal_qa_agent
from src.agents.workers.professional_agent import handle_professional
from src.agents.workers.exam_agent import handle_exam
from src.agents.workers.operation_agent import get_operation_agent
from src.infra.redis_cache import get_checkpointer_redis
from src.core.config import get_settings

settings = get_settings()


_INLINE_MEMORY_MARKER = "[长期记忆]"


def _separate_inline_memory_context(message: str) -> tuple[str, list[str]]:
    """Separate Supervisor metadata from the user's original utterance.

    Older Supervisor prompts could append a retrieved memory to ``message``.  The
    guide should receive that memory as context, not treat it as something the
    user stated again in the current session.
    """
    raw_message = str(message or "")
    if _INLINE_MEMORY_MARKER not in raw_message:
        return raw_message, []
    user_message, _, memory_text = raw_message.partition(_INLINE_MEMORY_MARKER)
    memory_text = memory_text.strip()
    return user_message.rstrip(), [memory_text] if memory_text else []


@dataclass
class UserContext:
    user_id: str
    session_id: str


async def _search_relevant_memories(message: str, runtime: ToolRuntime[UserContext]) -> list[str]:
    """在进入维权图前确定性检索长期记忆，不依赖 Supervisor 自主决定是否调用工具。"""
    store = getattr(runtime, "store", None)
    context = getattr(runtime, "context", None)
    user_id = getattr(context, "user_id", "")
    if not store or not user_id:
        return []
    try:
        results = await store.asearch(
            ("users", str(user_id), "memories"),
            query=message,
            limit=5,
        )
    except Exception:
        return []

    memories: list[str] = []
    for item in results:
        value = getattr(item, "value", {}) or {}
        content = value.get("content") or value.get("text")
        if content and str(content).strip():
            memories.append(str(content).strip())
    return memories


@tool
async def call_guide_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    启动公民法律指引流程。
    适用场景：用户描述具体法律纠纷或事件（拖欠工资、消费维权、合同违约、家庭纠纷等），
    需要法律依据、证据清单、维权路径和可操作步骤时。
    后续多轮追问由系统自动路由，无需再次调用此工具。

    Args:
        message: 用户描述的具体纠纷情况或维权诉求
    """
    session_id = runtime.context.session_id
    user_id = runtime.context.user_id
    user_message, inline_memories = _separate_inline_memory_context(message)
    print(f"[TOOL] call_guide_agent: session={session_id}, msg={user_message[:50]}")
    searched_memories = await _search_relevant_memories(user_message, runtime)
    long_term_memories = list(dict.fromkeys(inline_memories + searched_memories))

    return await call_guide_agent_impl(
        message=user_message,
        user_id=user_id,
        session_id=session_id,
        long_term_memories=long_term_memories,
    )


def _decode_redis_json(raw) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _extract_statistics_artifact(messages: list) -> dict | None:
    """从法律问答 Agent 的工具轨迹提取 ChatBI 产物。"""
    for item in reversed(messages):
        if getattr(item, "name", None) != "search_legal_statistics":
            continue
        content = getattr(item, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        statistics = payload.get("statistics") if isinstance(payload, dict) else None
        if isinstance(statistics, dict):
            return statistics
    return None


async def _persist_legal_qa_turn(
    redis,
    *,
    history_key: str,
    reply_key: str,
    statistics_key: str,
    statistics_context_key: str,
    history: list[dict],
    message: str,
    reply: str,
    artifact: dict | None,
) -> None:
    if artifact:
        serialized = json.dumps(artifact, ensure_ascii=False)
        await redis.set(statistics_key, serialized, ex=settings.REDIS_SESSION_TTL)
        await redis.set(
            statistics_context_key,
            serialized,
            ex=settings.REDIS_SESSION_TTL,
        )
    await redis.set(reply_key, reply, ex=settings.REDIS_SESSION_TTL)
    history.extend(
        [
            {"role": "user", "content": message[:4000]},
            {"role": "assistant", "content": reply[:8000]},
        ]
    )
    await redis.set(
        history_key,
        json.dumps(history[-6:], ensure_ascii=False),
        ex=settings.REDIS_SESSION_TTL,
    )


async def call_legal_qa_agent_impl(
    message: str,
    *,
    user_id: str,
    session_id: str,
) -> str:
    """Run one isolated legal-Q&A turn outside the Supervisor when mode is locked."""
    redis = get_checkpointer_redis()
    original_message_key = f"current_user_message:{user_id}:{session_id}"
    raw_original_message = await redis.get(original_message_key)
    if raw_original_message:
        if isinstance(raw_original_message, bytes):
            raw_original_message = raw_original_message.decode("utf-8")
        message = str(raw_original_message)
        await redis.delete(original_message_key)
    history_key = f"legal_qa_history:{user_id}:{session_id}"
    reply_key = f"legal_qa_last_reply:{user_id}:{session_id}"
    statistics_key = f"legal_statistics_last:{user_id}:{session_id}"
    statistics_context_key = f"legal_statistics_context:{user_id}:{session_id}"

    await redis.delete(statistics_key, reply_key)
    history = _decode_redis_json(await redis.get(history_key))[-6:]
    raw_context = await redis.get(statistics_context_key)
    if isinstance(raw_context, bytes):
        raw_context = raw_context.decode("utf-8")
    try:
        statistics_context = json.loads(raw_context) if raw_context else {}
    except json.JSONDecodeError:
        statistics_context = {}
    previous_sql = str(statistics_context.get("sql") or "")
    messages = history + [{"role": "user", "content": message}]

    from src.agents.legal_knowledge.legal_statistics_chatbi import (
        is_statistics_followup,
    )

    if is_statistics_followup(message, previous_sql):
        from src.agents.legal_knowledge.legal_statistics_chatbi import (
            run_legal_statistics_chatbi,
        )
        from src.agents.legal_knowledge.runtime import get_shared_legal_runtime

        llm = get_shared_legal_runtime()[0]
        chatbi_result = await run_legal_statistics_chatbi(
            message,
            llm,
            previous_sql=previous_sql,
        )
        artifact = chatbi_result.model_dump(mode="json")
        reply = chatbi_result.answer
        await _persist_legal_qa_turn(
            redis,
            history_key=history_key,
            reply_key=reply_key,
            statistics_key=statistics_key,
            statistics_context_key=statistics_context_key,
            history=history,
            message=message,
            reply=reply,
            artifact=artifact,
        )
        return reply

    agent = create_legal_qa_agent(
        user_id=user_id,
        statistics_previous_sql=previous_sql,
    )
    result = await agent.ainvoke(
        {"messages": messages}
    )
    reply = str(result["messages"][-1].content)

    artifact = _extract_statistics_artifact(result.get("messages", []))
    if artifact:
        # 统计回答已经在受约束的 ChatBI 回答阶段生成。外层 Agent 的二次改写
        # 可能引入无数据支撑的原因分析，因此这里强制透传原始 answer。
        reply = str(artifact.get("answer") or reply)
    await _persist_legal_qa_turn(
        redis,
        history_key=history_key,
        reply_key=reply_key,
        statistics_key=statistics_key,
        statistics_context_key=statistics_context_key,
        history=history,
        message=message,
        reply=reply,
        artifact=artifact,
    )
    return reply


@tool
async def call_legal_qa_agent(
    message: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    调用法律知识问答Agent，回答法律知识类问题。
    适用场景：询问法律概念、法条含义、制度性知识、维权流程等通用法律知识时。
    示例："劳动仲裁的流程是什么"、"什么是诉讼时效"、"合同解除需要哪些条件"

    Args:
        message: 用户的法律知识问题
    """
    return await call_legal_qa_agent_impl(
        message,
        user_id=runtime.context.user_id,
        session_id=runtime.context.session_id,
    )


@tool
async def call_professional_agent(message: str) -> str:
    """
    调用专业法律助手，面向法律从业者提供裁决预测、案件分析、文书摘要等服务。
    适用场景：律师或法务人员需要专业法律分析时。

    Args:
        message: 法律从业者的专业分析需求
    """
    return await handle_professional(message)


@tool
async def call_exam_agent(message: str) -> str:
    """
    调用法考助手，提供法考真题练习和知识点讲解。
    适用场景：法学学生或备考人员需要法考题目解析时。

    Args:
        message: 法考相关问题或练习需求
    """
    return await handle_exam(message)


@tool
async def call_operation_agent(message: str) -> str:
    """
    调用运营数据Agent，查询平台运营统计数据（仅限内部运营人员）。
    适用场景：运营人员查询用户量、咨询量、领域分布等运营数据时。

    Args:
        message: 运营人员的数据查询需求（自然语言）
    """
    agent = get_operation_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_guide_agent,
    call_legal_qa_agent,
    call_professional_agent,
    call_exam_agent,
    call_operation_agent,
]
