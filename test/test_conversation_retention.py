import json

import pytest
from fastapi import HTTPException

from src.agents import supervisor_agent
from src.api.routers import chat as chat_router
from src.infra import milvus_store
from src.infra.redis_cache import set_with_optional_ttl


class _RedisStub:
    def __init__(self):
        self.set_calls = []
        self.deleted = []
        self.values = {
            b"legal_document_meta:doc123": json.dumps(
                {
                    "user_id": "web-user",
                    "session_id": "web-user:case-session",
                }
            ).encode(),
        }

    async def set(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    async def scan_iter(self, match, count=100):
        del count
        if match == "guide_case_archive:web-user:case-session:*":
            yield b"guide_case_archive:web-user:case-session:old-case"
        if match == "legal_document_meta:*":
            yield b"legal_document_meta:doc123"


class _MemoryStoreStub:
    def __init__(self):
        self.deleted = []

    async def adelete(self, namespace, key):
        self.deleted.append((namespace, key))


@pytest.mark.asyncio
async def test_optional_ttl_zero_persists_until_manual_delete():
    redis = _RedisStub()

    await set_with_optional_ttl(redis, "guide_state:user:session", "{}", 0)
    await set_with_optional_ttl(redis, "guide_active:user:session", "1", 3600)

    assert redis.set_calls[0][2] == {}
    assert redis.set_calls[1][2] == {"ex": 3600}


@pytest.mark.asyncio
async def test_delete_conversation_clears_all_owned_state(monkeypatch):
    redis = _RedisStub()
    memory_store = _MemoryStoreStub()
    deleted_threads = []

    async def delete_thread(thread_id):
        deleted_threads.append(thread_id)

    monkeypatch.setattr(chat_router, "get_checkpointer_redis", lambda: redis)
    monkeypatch.setattr(supervisor_agent, "delete_supervisor_thread", delete_thread)
    monkeypatch.setattr(milvus_store, "get_milvus_store", lambda: memory_store)

    result = await chat_router.delete_conversation(
        "case-session",
        chat_router.DeleteConversationRequest(user_id="web-user"),
    )

    assert result == {
        "deleted": True,
        "session_id": "case-session",
        "warnings": [],
    }
    deleted = {
        key.decode() if isinstance(key, bytes) else key
        for key in redis.deleted
    }
    assert "guide_state:web-user:case-session" in deleted
    assert "guide_active:web-user:case-session" in deleted
    assert "guide_case_archive:web-user:case-session:old-case" in deleted
    assert "legal_statistics_context:web-user:case-session" in deleted
    assert "legal_document_meta:doc123" in deleted
    assert "legal_document_file:doc123" in deleted
    assert deleted_threads == ["web-user:case-session"]
    assert memory_store.deleted == [
        (("users", "web-user", "memories"), "guide_web-user_case-session")
    ]


@pytest.mark.asyncio
async def test_delete_conversation_rejects_unsafe_identifiers():
    with pytest.raises(HTTPException) as exc_info:
        await chat_router.delete_conversation(
            "case*",
            chat_router.DeleteConversationRequest(user_id="web-user"),
        )

    assert exc_info.value.status_code == 400
