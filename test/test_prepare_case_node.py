"""Contracts for the target prepare_case workflow entry node."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import src.agents.legal_guide.graph as guide_graph
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.prepare_case import (
    determine_requested_route,
    split_mixed_payload,
)
from src.agents.legal_guide.state import GuidePhase, GuideState


def _deps() -> MagicMock:
    return MagicMock(spec=GuideDeps)


def _prepare(state: GuideState) -> dict:
    with patch.object(
        guide_graph,
        "node_load_context",
        new=AsyncMock(return_value={}),
    ):
        return asyncio.run(guide_graph.node_prepare_case(state, _deps()))


def test_first_turn_builds_structured_case_started_event():
    state = GuideState(
        session_id="u:s",
        current_request_id="request-1",
        current_message_id="message-1",
        current_message_text=(
            "我在闲鱼向个人卖家买东西，支付800元后对方没有发货。"
            "我有订单截图和付款记录。"
        ),
        messages=[HumanMessage(content="首轮案情")],
    )

    updates = _prepare(state)

    assert updates["input_event_type"] == "case_started"
    assert {item["type"] for item in updates["input_events"]} >= {
        "fact_added",
        "evidence_named",
    }
    assert updates["fact_payload"]["source_message_id"] == "message-1"
    assert updates["evidence_payload"]["named_evidence"] == ["订单截图", "付款记录"]
    assert updates["requested_route"] == "update_facts"
    assert updates["round"] == 1
    assert updates["state_version"] == 1
    assert updates["event_sequence"] == 1
    assert updates["last_processed_request_id"] == "request-1"


def test_mixed_correction_upload_and_conclude_preserves_every_event():
    state = GuideState(
        round=2,
        total_rounds=2,
        state_version=5,
        event_sequence=4,
        current_request_id="request-2",
        current_message_id="message-2",
        current_message_text=(
            "我之前说错了，付款时间是7月18日，这是支付记录，现在生成方案。"
        ),
        current_attachments=[
            {
                "material_id": "material-1",
                "file_name": "支付账单.pdf",
                "sha256": "abc",
                "upload_status": "uploaded",
            }
        ],
        turn_control_intent="conclude_now",
    )

    updates = _prepare(state)
    event_types = {item["type"] for item in updates["input_events"]}

    assert updates["input_event_type"] == "mixed_update"
    assert {"fact_corrected", "evidence_added", "control_conclude_now"} <= event_types
    assert "现在生成方案" not in updates["fact_payload"]["text"]
    assert updates["requested_route"] == "update_facts"
    assert updates["route_after_guard"] == [
        "update_facts",
        "decide_facts",
        "plan_evidence",
        "assess_evidence",
        "generate_solution",
    ]
    assert updates["state_version"] == 6
    assert updates["event_sequence"] == 5


def test_pending_fact_batch_is_restored_and_answered_as_one_event():
    state = GuideState(
        round=1,
        total_rounds=1,
        workflow_stage="fact_gathering",
        pending_ask_type="facts",
        pending_ask_details=["对方身份？", "付款时间？"],
        pending_followup_ids=["actor.identity", "transaction.date"],
        current_message_text="对方是个人卖家，付款时间是2026年7月18日。",
    )

    updates = _prepare(state)

    assert updates["pause_state"] == {
        "type": "awaiting_fact_batch",
        "pending_followup_ids": ["actor.identity", "transaction.date"],
    }
    assert updates["input_event_type"] == "fact_batch_answered"
    assert updates["input_events"] == [
        {"type": "fact_batch_answered", "payload_ref": "fact_payload"}
    ]


def test_control_only_text_is_not_exposed_as_fact_payload():
    payloads = split_mixed_payload(
        "不要再问了，现在生成方案",
        control_intent="conclude_now",
        message_id="message-3",
    )

    assert payloads["fact_payload"]["text"] == ""
    assert payloads["control_payload"]["intent"] == "conclude_now"


def test_uncertain_boundary_can_only_route_to_read_only_guard():
    state = GuideState(
        awaiting_case_boundary=True,
        case_boundary_read_only=True,
    )

    requested, candidates = determine_requested_route(state, "unknown", [])

    assert requested == "guard_case_read_only"
    assert candidates == []
    assert guide_graph.route_after_urgency(state) == "pause_case_boundary"


def test_duplicate_request_replays_without_advancing_the_graph():
    state = GuideState(
        last_processed_request_id="request-retry",
        round=3,
        state_version=7,
        messages=[AIMessage(content="已经生成的回复")],
    )

    reply, replayed = asyncio.run(
        guide_graph.run_guide(
            "重复发送的内容",
            "u:s",
            _deps(),
            existing_state=state,
            request_context={"request_id": "request-retry"},
        )
    )

    assert reply == "已经生成的回复"
    assert replayed.round == 3
    assert replayed.state_version == 7


def test_uncertain_boundary_is_risk_checked_before_confirmation_prompt():
    pending_text = "有另一件事想问"
    state = GuideState(
        round=1,
        total_rounds=1,
        phase=GuidePhase.END,
        awaiting_case_boundary=True,
        case_boundary_read_only=True,
        pending_case_message=pending_text,
    )
    with patch.object(
        guide_graph,
        "node_check_urgency",
        new=AsyncMock(return_value={"urgency_level": "normal"}),
    ):
        reply, updated = asyncio.run(
            guide_graph.run_guide(
                pending_text,
                "u:s",
                _deps(),
                existing_state=state,
                request_context={"request_id": "boundary-1"},
            )
        )

    assert "继续" in reply and "新建" in reply
    assert updated.awaiting_case_boundary is True
    assert updated.pause_state == {"type": "awaiting_case_boundary"}
    assert not any(
        isinstance(message, HumanMessage) and message.content == pending_text
        for message in updated.messages
    )


def test_danger_in_unassigned_message_preempts_boundary_prompt():
    pending_text = "另一个人拿刀堵在门外，我现在有危险"
    state = GuideState(
        round=1,
        total_rounds=1,
        phase=GuidePhase.END,
        awaiting_case_boundary=True,
        case_boundary_read_only=True,
        pending_case_message=pending_text,
    )

    reply, updated = asyncio.run(
        guide_graph.run_guide(
            pending_text,
            "u:s",
            _deps(),
            existing_state=state,
            request_context={"request_id": "boundary-danger"},
        )
    )

    assert "110" in reply
    assert "新建一个独立案件" not in reply
    assert updated.safety_pause_active is True
    assert updated.current_safety_status == "danger"


def test_document_request_is_classified_and_guarded_before_handoff():
    state = GuideState(
        round=2,
        total_rounds=2,
        phase=GuidePhase.END,
        confirmed_issues=["网络购物未发货纠纷"],
    )
    with patch.object(
        guide_graph,
        "node_check_urgency",
        new=AsyncMock(return_value={"urgency_level": "normal"}),
    ):
        _reply, updated = asyncio.run(
            guide_graph.run_guide(
                "生成投诉信",
                "u:s",
                _deps(),
                existing_state=state,
                request_context={"request_id": "document-1"},
            )
        )

    assert updated.input_event_type == "document_requested"
    assert updated.requested_route == "document_service"
    assert updated.document_request_ready is True
    assert updated.state_version == 1
