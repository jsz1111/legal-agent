from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.agents.legal_guide.state import GuidePhase, GuideState
from src.api.routers import chat as chat_router


class _FakeRedis:
    def __init__(self, values: dict[str, object] | None = None):
        self.values = dict(values or {})

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value, **_kwargs):
        self.values[key] = value
        return True

    async def exists(self, *keys: str):
        return any(key in self.values for key in keys)

    async def delete(self, *keys: str):
        for key in keys:
            self.values.pop(key, None)


def test_document_request_bypasses_guide_graph_and_does_not_become_case_fact(
    monkeypatch,
):
    thread_id = "user:session"
    state_key = f"guide_state:{thread_id}"
    state = GuideState(
        session_id=thread_id,
        phase=GuidePhase.DETAIL_GATHER,
        legal_domain="consumer_market",
        confirmed_issues=["网络购物未发货纠纷"],
        draftable_facts=["2026年7月18日支付800元后卖家未发货"],
        collected_facts=["2026年7月18日支付800元后卖家未发货"],
        messages=[HumanMessage(content="我付款800元后卖家没有发货")],
    )
    redis = _FakeRedis({state_key: state.model_dump_json()})
    captured = {}

    def fake_export_plan_word(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            text="消费者投诉方案",
            docx_bytes=b"PK-test",
            filename="维权行动方案_法护通.docx",
            doc_type="维权行动方案（Word 版）",
            official_template=None,
            related_official_template=None,
            missing_fields=[],
        )

    async def fail_run_guide(**_kwargs):
        raise AssertionError("document control intent must not enter GuideGraph")

    monkeypatch.setattr(
        chat_router,
        "build_guide_deps",
        lambda db_session=None: SimpleNamespace(llm=object()),
    )
    monkeypatch.setattr(chat_router, "run_guide", fail_run_guide)
    monkeypatch.setattr(
        "src.agents.legal_guide.doc_generator.export_plan_word",
        fake_export_plan_word,
    )

    reply, _debug, document = asyncio.run(
        chat_router._run_guide_turn(
            "生成文书",
            thread_id,
            redis,
            None,
        )
    )

    saved = GuideState.model_validate_json(redis.values[state_key])
    assert reply == "消费者投诉方案"
    assert document and document["doc_type"] == "维权行动方案（Word 版）"
    assert captured["legal_domain"] == state.legal_domain
    assert captured["confirmed_issues"] == state.confirmed_issues
    assert captured["collected_facts"] == state.draftable_facts
    assert saved.phase == GuidePhase.DETAIL_GATHER
    assert saved.doc_draft == "消费者投诉方案"
    assert all(
        "生成文书" not in str(message.content)
        for message in saved.messages
    )
