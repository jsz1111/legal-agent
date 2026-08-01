"""Contracts for the update_facts dynamic fact-blackboard node."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.graph import GuideDeps, node_update_facts
from src.agents.legal_guide.state import GuideState


def _deps(payload: dict) -> MagicMock:
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return deps


def _payload(*updates: dict, issues: list[str] | None = None) -> dict:
    return {
        "issues": issues or ["网络购物交易纠纷"],
        "domain": "consumer_market",
        "facts": [item["statement"] for item in updates],
        "case_updates": list(updates),
        "evidence_details": [],
        "region": "",
        "time_info": "",
    }


def _state(text: str, **updates) -> GuideState:
    values = {
        "case_id": "case-facts",
        "round": 1,
        "event_sequence": 1,
        "current_message_id": "message-1",
        "current_message_text": text,
        "fact_payload": {"text": text, "source_message_id": "message-1"},
        "input_event_type": "case_started",
        "input_events": [{"type": "case_started", "payload_ref": "fact_payload"}],
        "requested_route": "update_facts",
        "turn_contains_case_details": True,
        "messages": [HumanMessage(content=text)],
    }
    values.update(updates)
    return GuideState(**values)


def _run(state: GuideState, payload: dict) -> dict:
    return asyncio.run(node_update_facts(state, _deps(payload)))


def test_first_description_extracts_all_atomic_facts_with_sources():
    text = "我7月18日在闲鱼向个人卖家付款800元，对方没有发货，我想退款"
    result = _run(
        _state(text),
        _payload(
            {
                "key": "location.platform",
                "category": "location",
                "statement": "交易平台为闲鱼",
                "value": "闲鱼",
                "source_text": "闲鱼",
            },
            {
                "key": "actor.counterparty.identity",
                "category": "actor",
                "statement": "对方是个人卖家",
                "value": "个人卖家",
                "source_text": "个人卖家",
            },
            {
                "key": "transaction.payment.pay_01.date",
                "category": "time",
                "statement": "用户于7月18日付款",
                "value": "7月18日",
                "source_text": "7月18日",
            },
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "用户付款800元",
                "value": "800元",
                "source_text": "付款800元",
            },
            {
                "key": "performance.delivery.status",
                "category": "event",
                "statement": "对方没有发货",
                "value": "未发货",
                "source_text": "没有发货",
            },
            {
                "key": "claim.primary_request",
                "category": "claim",
                "statement": "用户希望退款",
                "value": "退款",
                "source_text": "想退款",
            },
        ),
    )

    assert len(result["fact_blackboard"]) == 6
    assert result["fact_blackboard_version"] == 1
    assert {item["status"] for item in result["fact_blackboard"]} == {"confirmed"}
    assert all(item["fact_id"].startswith("fact-") for item in result["fact_blackboard"])
    assert all(
        item["source_refs"][0]["message_id"] == "message-1"
        for item in result["fact_blackboard"]
    )
    assert result["downstream_invalidations"] == [
        "decision_sufficiency",
        "followup_plan",
        "legal_model",
        "evidence_plan",
    ]


def test_repeated_event_is_idempotent_and_does_not_increment_version():
    text = "我支付了800元"
    first = _run(
        _state(text),
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "用户支付800元",
                "value": "800元",
                "source_text": "支付了800元",
            }
        ),
    )
    repeated = _state(text).model_copy(update=first)

    second = _run(
        repeated,
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "用户支付800元",
                "value": "800元",
                "source_text": "支付了800元",
            }
        ),
    )

    assert second["fact_changes"] == []
    assert repeated.fact_blackboard_version == 1


def test_explicit_correction_supersedes_old_fact_without_deleting_history():
    first = _run(
        _state("我支付了800元"),
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "用户支付800元",
                "value": "800元",
                "source_text": "支付了800元",
            }
        ),
    )
    state = _state(
        "我之前说错了，实际支付900元",
        round=2,
        event_sequence=2,
        current_message_id="message-2",
        current_message_text="我之前说错了，实际支付900元",
        fact_payload={"text": "我之前说错了，实际支付900元"},
        input_event_type="fact_corrected",
        input_events=[{"type": "fact_corrected", "payload_ref": "fact_payload"}],
        messages=[HumanMessage(content="我之前说错了，实际支付900元")],
    ).model_copy(update=first)

    result = _run(
        state,
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "用户实际支付900元",
                "value": "900元",
                "source_text": "实际支付900元",
                "operation": "replace",
            }
        ),
    )

    assert [item["status"] for item in result["fact_blackboard"]] == [
        "superseded",
        "confirmed",
    ]
    assert result["fact_blackboard"][0]["superseded_by_fact_id"]
    assert result["fact_blackboard"][1]["supersedes_fact_id"]


def test_unmarked_changed_value_creates_a_conflict_group():
    first = _run(
        _state("付款金额是800元"),
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "付款金额为800元",
                "value": "800元",
                "source_text": "800元",
            }
        ),
    )
    state = _state(
        "付款金额是900元",
        round=2,
        event_sequence=2,
        current_message_id="message-2",
        current_message_text="付款金额是900元",
        fact_payload={"text": "付款金额是900元"},
        input_event_type="fact_added",
        input_events=[{"type": "fact_added", "payload_ref": "fact_payload"}],
        messages=[HumanMessage(content="付款金额是900元")],
    ).model_copy(update=first)

    result = _run(
        state,
        _payload(
            {
                "key": "transaction.payment.pay_01.amount",
                "category": "amount",
                "statement": "付款金额为900元",
                "value": "900元",
                "source_text": "900元",
            }
        ),
    )

    assert {item["status"] for item in result["fact_blackboard"]} == {"conflicted"}
    assert len(result["fact_conflict_groups"]) == 1
    assert len(next(iter(result["fact_conflict_groups"].values()))) == 2


def test_evidence_name_inventory_does_not_mark_claimed_material_as_assessed():
    text = "我有付款记录，但不知道是否还保存着聊天记录"
    result = _run(
        _state(text),
        _payload(
            {
                "key": "evidence.payment_record",
                "category": "evidence",
                "statement": "用户有付款记录",
                "value": "付款记录",
                "source_text": "有付款记录",
            },
            {
                "key": "evidence.chat_record",
                "category": "evidence",
                "statement": "用户不知道是否保存聊天记录",
                "value": "聊天记录",
                "source_text": "不知道是否还保存着聊天记录",
                "certainty": "unknown",
            },
        ),
    )

    statuses = {item["normalized_name"]: item["status"] for item in result["evidence_name_inventory"]}
    assert statuses["payment_record"] == "user_claimed_present"
    assert statuses["chat_record"] == "temporarily_unavailable"
    assert all(item["status"] != "assessed" for item in result["evidence_name_inventory"])


def test_attachment_only_creates_pending_material_observation_not_case_fact():
    state = _state(
        "",
        current_message_text="",
        fact_payload={},
        input_event_type="evidence_added",
        input_events=[{"type": "evidence_added", "payload_ref": "evidence_payload"}],
        current_attachments=[{"file_name": "聊天截图.png", "sha256": "abc"}],
        turn_contains_case_details=False,
        messages=[],
    )

    result = _run(state, _payload(issues=[]))

    assert result["fact_blackboard"] == []
    assert result["fact_blackboard_version"] == 0
    assert result["material_fact_observations"][0]["status"] == "pending_confirmation"
    assert result["evidence_name_inventory"][0]["status"] == "submitted"


def test_pure_control_command_never_enters_fact_blackboard():
    state = _state(
        "现在生成方案",
        input_event_type="control_conclude_now",
        input_events=[
            {"type": "control_conclude_now", "payload_ref": "control_payload"}
        ],
        turn_control_intent="conclude_now",
        turn_contains_case_details=False,
    )

    result = _run(state, _payload(issues=[]))

    assert result["fact_blackboard"] == []
    assert result["fact_changes"] == []
    assert result["fact_blackboard_version"] == 0
