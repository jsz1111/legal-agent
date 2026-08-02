"""Case-boundary and isolated-state contracts for the legal guide."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from fastapi import HTTPException

from src.agents.legal_guide.case_lifecycle import (
    CaseRelation,
    decide_case_boundary,
)
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.api.routers.chat import _prepare_case_turn
from src.agents.legal_guide.graph import (
    GuideDeps,
    node_parse_details,
    node_prepare_turn,
    route_after_parse,
    route_after_urgency,
)


class _Redis:
    def __init__(self):
        self.data: dict[str, object] = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True


def _llm(payload: dict) -> MagicMock:
    model = MagicMock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return model


def _completed_case() -> GuideState:
    return GuideState(
        case_id="case-old",
        case_generation=2,
        session_id="u:s",
        phase=GuidePhase.END,
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        collected_facts=["公司拖欠三个月工资"],
        case_facts=[{
            "key": "wage.arrears.duration",
            "category": "time",
            "statement": "公司拖欠三个月工资",
            "status": "asserted",
            "source_text": "公司拖欠三个月工资",
            "turn": 1,
        }],
        evidence_confirmed=["工资流水"],
        user_context={"user_id": "u", "long_term_memories": ["用户所在地区：上海"]},
    )


def test_boundary_classifier_uses_application_confidence_gate():
    state = _completed_case()
    decision = asyncio.run(decide_case_boundary(
        state,
        "补充一条情况",
        _llm({
            "relation": "continue",
            "confidence": 0.4,
            "reason": "连续性证据不足",
            "carries_case_detail": True,
        }),
    ))

    assert decision.relation == CaseRelation.UNCERTAIN
    assert decision.confidence == 0.4


def test_new_case_gets_clean_state_and_archives_previous_case():
    state = _completed_case()
    redis = _Redis()
    message, next_state, reply = asyncio.run(_prepare_case_turn(
        message="另一项独立纠纷的完整描述",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "new",
            "confidence": 0.96,
            "reason": "主体、事件和法律关系均独立",
            "carries_case_detail": True,
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert reply is None
    assert message == "另一项独立纠纷的完整描述"
    assert next_state.case_id != "case-old"
    assert next_state.case_generation == 3
    assert next_state.phase == GuidePhase.CLARIFY
    assert next_state.legal_domain == ""
    assert next_state.confirmed_issues == []
    assert next_state.case_facts == []
    assert next_state.evidence_confirmed == []
    assert next_state.user_context["long_term_memories"] == ["用户所在地区：上海"]
    assert "guide_case_archive:u:s:case-old" in redis.data
    archived = GuideState.model_validate_json(
        redis.data["guide_case_archive:u:s:case-old"]
    )
    assert archived.case_facts[0]["key"] == "wage.arrears.duration"


def test_continuation_reopens_completed_case_without_changing_identity():
    state = _completed_case()
    redis = _Redis()
    message, next_state, reply = asyncio.run(_prepare_case_turn(
        message="对原案件补充一项事实",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "continue",
            "confidence": 0.94,
            "reason": "仍是同一主体和欠薪事件",
            "carries_case_detail": True,
            "control_intent": "case_detail",
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert reply is None
    assert message == "对原案件补充一项事实"
    assert next_state.case_id == "case-old"
    assert next_state.phase == GuidePhase.ISSUE_SEARCH
    assert next_state.case_facts
    assert next_state.awaiting_case_boundary is False
    assert next_state.turn_control_intent == "case_detail"
    assert next_state.turn_contains_case_details is True
    assert next_state.case_boundary_audit[-1]["relation"] == "continue"


def test_uncertain_relation_pauses_before_mutating_case_state():
    state = _completed_case()
    redis = _Redis()
    message, waiting_state, reply = asyncio.run(_prepare_case_turn(
        message="有另一件事想问",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "uncertain",
            "confidence": 0.62,
            "reason": "缺少主体和事件信息",
            "carries_case_detail": False,
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert message == "有另一件事想问"
    assert waiting_state.phase == GuidePhase.END
    assert waiting_state.case_id == "case-old"
    assert waiting_state.awaiting_case_boundary is True
    assert waiting_state.pending_case_message == "有另一件事想问"
    assert "继续" in reply and "新建" in reply
    restored = GuideState.model_validate_json(redis.data["guide_state:u:s"])
    assert restored.case_facts == state.case_facts


def test_pending_message_is_recovered_after_semantic_confirmation():
    state = _completed_case().model_copy(update={
        "awaiting_case_boundary": True,
        "pending_case_message": "上一条尚未归属的案件描述",
    })
    redis = _Redis()
    message, next_state, reply = asyncio.run(_prepare_case_turn(
        message="这是一个独立案件，并补充了新的事实",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "new",
            "confidence": 0.98,
            "reason": "用户确认属于独立案件",
            "carries_case_detail": True,
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert reply is None
    assert message.startswith("上一条尚未归属的案件描述")
    assert "补充了新的事实" in message
    assert next_state.case_id != "case-old"


def test_natural_conclude_request_stops_questioning_without_becoming_a_fact():
    state = _completed_case().model_copy(update={
        "phase": GuidePhase.DETAIL_GATHER,
        "round": 2,
        "pending_ask_details": ["购买时间是什么时候？"],
        "pending_ask_type": "facts",
        "last_confirmed_count": 1,
    })
    redis = _Redis()
    user_message = "就按目前情况生成吧"
    message, state, reply = asyncio.run(_prepare_case_turn(
        message=user_message,
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "continue",
            "confidence": 0.97,
            "reason": "用户要求基于当前案件立即结束追问",
            "carries_case_detail": False,
            "control_intent": "conclude_now",
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))
    state.messages.append(HumanMessage(content=message))
    prepared = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))
    state = state.model_copy(update=prepared)

    assert reply is None
    assert state.wants_conclude is True
    assert state.turn_contains_case_details is False
    assert route_after_urgency(state) == "parse_details"

    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock()
    parsed = asyncio.run(node_parse_details(state, deps))
    deps.llm.ainvoke.assert_not_awaited()
    state = state.model_copy(update=parsed)
    assert state.pending_ask_details == []
    assert route_after_parse(state) == "assess_retrieve"


def test_post_conclusion_detail_reopens_the_same_case():
    state = _completed_case()
    redis = _Redis()
    _, reopened, reply = asyncio.run(_prepare_case_turn(
        message="我想继续补充刚才那个案件的事实",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "continue",
            "confidence": 0.98,
            "reason": "用户明确继续同一案件",
            "carries_case_detail": True,
            "control_intent": "case_detail",
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert reply is None
    assert reopened.case_id == "case-old"
    assert reopened.phase == GuidePhase.ISSUE_SEARCH
    assert reopened.turn_control_intent == "case_detail"
    assert reopened.turn_contains_case_details is True


def test_structured_evidence_action_bypasses_boundary_model_and_updates_same_case():
    state = _completed_case()
    redis = _Redis()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=AssertionError("boundary model must not run"))
    evidence_message = (
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：工资流水.txt\n来源形式：exported_file\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n2026年7月工资流水"
    )

    message, reopened, reply = asyncio.run(_prepare_case_turn(
        message=evidence_message,
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=llm,
        redis=redis,
        state_key="guide_state:u:s",
        action="submit_evidence",
        target_case_id="case-old",
        regenerate_solution=True,
    ))

    assert reply is None
    assert message == evidence_message
    assert reopened.case_id == "case-old"
    assert reopened.phase == GuidePhase.ISSUE_SEARCH
    assert reopened.turn_control_intent == "conclude_now"
    assert reopened.turn_contains_case_details is True
    assert reopened.case_boundary_audit[-1]["decision_source"] == "structured_case_action"
    llm.ainvoke.assert_not_awaited()


def test_structured_case_action_rejects_stale_case_id():
    state = _completed_case()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_prepare_case_turn(
            message="现在生成方案",
            existing_state=state,
            thread_id="u:s",
            user_id="u",
            llm=MagicMock(),
            redis=_Redis(),
            state_key="guide_state:u:s",
            action="regenerate_solution",
            target_case_id="another-case",
        ))

    assert exc.value.status_code == 409


def test_structured_evidence_action_rejects_stale_requirement_id():
    state = _completed_case().model_copy(update={
        "evidence_requirements": [{
            "id": "proof_target:payment",
            "active": True,
        }],
    })
    evidence_message = (
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：付款记录.txt\n来源形式：exported_file\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n付款记录"
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_prepare_case_turn(
            message=evidence_message,
            existing_state=state,
            thread_id="u:s",
            user_id="u",
            llm=MagicMock(),
            redis=_Redis(),
            state_key="guide_state:u:s",
            action="submit_evidence",
            target_case_id="case-old",
            evidence_requirement_ids=["proof_target:expired"],
        ))

    assert exc.value.status_code == 409


def test_safety_pause_resumes_same_state_and_preserves_conclusion_control():
    state = GuideState(
        case_id="safety-case",
        phase=GuidePhase.END,
        safety_pause_active=True,
        safety_pause_case_message="对方正在门外威胁我",
        current_safety_status="danger",
    )
    redis = _Redis()

    message, resumed, reply = asyncio.run(_prepare_case_turn(
        message="我现在安全了，请按现有情况生成方案",
        existing_state=state,
        thread_id="u:s",
        user_id="u",
        llm=_llm({
            "relation": "continue",
            "confidence": 0.95,
            "reason": "用户报告安全并要求继续处理同一事件",
            "carries_case_detail": True,
            "control_intent": "conclude_now",
        }),
        redis=redis,
        state_key="guide_state:u:s",
    ))

    assert reply is None
    assert message == "我现在安全了，请按现有情况生成方案"
    assert resumed.case_id == "safety-case"
    assert resumed.phase == GuidePhase.ISSUE_SEARCH
    assert resumed.turn_control_intent == "conclude_now"
    assert resumed.turn_contains_case_details is True
