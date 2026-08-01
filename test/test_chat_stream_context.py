import json

import pytest

from src.api.routers import chat as chat_router


class _RedisStub:
    def __init__(self):
        self.values = {
            "guide_last_reply:web-user:case-session": "正式业务回复",
            "guide_last_debug:web-user:case-session": json.dumps(
                {"domain": "consumer_market"},
                ensure_ascii=False,
            ),
        }

    async def exists(self, _key):
        return 0

    async def set(self, *_args, **_kwargs):
        return True

    async def delete(self, *_keys):
        for key in _keys:
            self.values.pop(key, None)
        return 1

    async def get(self, key):
        return self.values.get(key)


class _AgentStub:
    def __init__(self):
        self.context = None

    async def ainvoke(self, _input, **kwargs):
        self.context = kwargs.get("context")
        return {
            "messages": [
                _Message("已记住：内部记忆结果"),
                _Message('{"urgency":"NORMAL"}'),
                _Message("Supervisor 改写回复"),
            ]
        }


class _Message:
    def __init__(self, content):
        self.content = content


@pytest.mark.asyncio
async def test_chat_stream_passes_user_context_to_supervisor(monkeypatch):
    agent = _AgentStub()
    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: _RedisStub())
    monkeypatch.setattr(chat_router, "get_supervisor_agent", lambda: _return(agent))
    monkeypatch.setattr(chat_router, "_pop_statistics_artifact", _no_statistics)

    response = await chat_router.chat_stream(
        chat_router.ChatRequest(
            user_id="web-user",
            session_id="case-session",
            message="test question",
        ),
        db=None,
    )

    chunks = [chunk async for chunk in response.body_iterator]
    payloads = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        payloads.append(json.loads(chunk.removeprefix("data: ").strip()))

    assert agent.context.user_id == "web-user"
    assert agent.context.session_id == "case-session"
    assert [item["content"] for item in payloads if item["type"] == "token"] == [
        "正式业务回复"
    ]
    assert payloads[-1]["type"] == "done"
    assert payloads[-1]["debug"]["domain"] == "consumer_market"


async def _return(value):
    return value


async def _no_statistics(*_args, **_kwargs):
    return None
