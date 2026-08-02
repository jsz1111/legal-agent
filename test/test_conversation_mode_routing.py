import asyncio
import json

import pytest

from src.api.routers import chat as chat_router


class _RedisStub:
    def __init__(self, values=None):
        self.values = dict(values or {})

    async def exists(self, key):
        return int(key in self.values)

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **_kwargs):
        self.values[key] = value
        return True

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
        return len(keys)


def _payloads(response):
    async def collect():
        result = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            result.append(json.loads(chunk.removeprefix("data: ").strip()))
        return result

    return collect()


@pytest.mark.asyncio
async def test_explicit_qa_mode_bypasses_supervisor(monkeypatch):
    redis = _RedisStub()
    calls = []

    async def run_qa(message, **kwargs):
        calls.append((message, kwargs["user_id"], kwargs["session_id"]))
        return "独立法律问答", None

    async def fail_supervisor():
        raise AssertionError("locked Q&A must not enter Supervisor")

    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(chat_router, "_run_legal_qa_turn", run_qa)
    monkeypatch.setattr(chat_router, "get_supervisor_agent", fail_supervisor)

    response = await chat_router.chat_stream(
        chat_router.ChatRequest(
            user_id="mode-user",
            session_id="qa-session",
            message="诉讼时效是什么？",
            mode="qa",
        ),
        db=None,
    )
    payloads = await _payloads(response)

    assert calls == [("诉讼时效是什么？", "mode-user", "qa-session")]
    assert [item["content"] for item in payloads if item["type"] == "token"] == [
        "独立法律问答"
    ]
    assert payloads[-1]["resolved_mode"] == "qa"
    assert payloads[-1]["mode_locked"] is True
    assert redis.values["conversation_mode:mode-user:qa-session"] == "qa"


@pytest.mark.asyncio
async def test_explicit_case_mode_creates_isolated_case(monkeypatch):
    redis = _RedisStub()
    calls = []

    async def run_case(message, thread_id, _redis, _db, **_structured_action):
        calls.append((message, thread_id))
        return "案件首轮回复", chat_router.DebugInfo(case_id="case-1"), None

    async def fail_supervisor():
        raise AssertionError("locked case must not enter Supervisor")

    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(chat_router, "_run_guide_turn", run_case)
    monkeypatch.setattr(chat_router, "get_supervisor_agent", fail_supervisor)

    response = await chat_router.chat_stream(
        chat_router.ChatRequest(
            user_id="mode-user",
            session_id="case-session",
            message="公司拖欠我三个月工资",
            mode="case",
        ),
        db=None,
    )
    payloads = await _payloads(response)

    assert calls == [("公司拖欠我三个月工资", "mode-user:case-session")]
    assert payloads[-1]["resolved_mode"] == "case"
    assert payloads[-1]["mode_locked"] is True
    assert payloads[-1]["debug"]["case_id"] == "case-1"
    assert redis.values["conversation_mode:mode-user:case-session"] == "case"


@pytest.mark.asyncio
async def test_server_saved_mode_cannot_be_overridden_by_later_request():
    redis = _RedisStub({"conversation_mode:u:s": "case"})
    request = chat_router.ChatRequest(
        user_id="u",
        session_id="s",
        message="只问一个问题",
        mode="qa",
    )

    resolved = await chat_router._resolve_conversation_mode(
        redis,
        request,
        active_key="guide_active:u:s",
        state_key="guide_state:u:s",
    )

    assert resolved == "case"


@pytest.mark.asyncio
async def test_different_sessions_can_run_legal_qa_concurrently(monkeypatch):
    redis = _RedisStub()
    started: list[str] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def run_qa(message, **kwargs):
        started.append(kwargs["session_id"])
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return f"{kwargs['session_id']}:{message}", None

    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(chat_router, "_run_legal_qa_turn", run_qa)

    first = await chat_router.chat_stream(
        chat_router.ChatRequest(
            user_id="parallel-user",
            session_id="qa-one",
            message="问题一",
            mode="qa",
        ),
        db=None,
    )
    second = await chat_router.chat_stream(
        chat_router.ChatRequest(
            user_id="parallel-user",
            session_id="qa-two",
            message="问题二",
            mode="qa",
        ),
        db=None,
    )

    first_task = asyncio.create_task(_payloads(first))
    second_task = asyncio.create_task(_payloads(second))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert set(started) == {"qa-one", "qa-two"}
    release.set()
    first_payloads, second_payloads = await asyncio.gather(first_task, second_task)

    assert [item["content"] for item in first_payloads if item["type"] == "token"] == [
        "qa-one:问题一"
    ]
    assert [item["content"] for item in second_payloads if item["type"] == "token"] == [
        "qa-two:问题二"
    ]
