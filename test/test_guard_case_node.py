"""Structured guard_case contracts across all supported risk families."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.graph import GuideDeps, node_guard_case
from src.agents.legal_guide.state import GuideState


def _deps(payload: dict | None = None, *, error: Exception | None = None):
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    if error is not None:
        deps.llm.ainvoke = AsyncMock(side_effect=error)
    else:
        deps.llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=json.dumps(
                    payload
                    or {
                        "risks": [],
                        "safety_relevant": False,
                        "current_safety_status": "not_applicable",
                        "time_clues": [],
                    },
                    ensure_ascii=False,
                )
            )
        )
    return deps


def _state(text: str, **updates) -> GuideState:
    values = {
        "case_id": "case-guard",
        "session_id": "u:s",
        "round": 2,
        "state_version": 3,
        "event_sequence": 3,
        "current_message_id": "message-guard",
        "current_message_text": text,
        "fact_payload": {"text": text, "source_message_id": "message-guard"},
        "input_event_type": "fact_added",
        "input_events": [{"type": "fact_added", "payload_ref": "fact_payload"}],
        "requested_route": "update_facts",
        "route_after_guard": ["update_facts", "decide_facts"],
        "messages": [HumanMessage(content=text)],
    }
    values.update(updates)
    return GuideState(**values)


def _run(state: GuideState, deps=None) -> dict:
    return asyncio.run(node_guard_case(state, deps or _deps()))


def test_past_violence_with_explicit_safety_does_not_pause():
    result = _run(_state("丈夫去年打过我，但我现在安全，想整理报警记录"))

    assert result["guard_status"] == "clear"
    assert result["current_safety_status"] == "safe"
    assert result["guard_pause_required"] is False
    assert result["guard_notice_markdown"] == ""


def test_past_violence_without_current_status_only_asks_safety():
    result = _run(_state("我昨天被人打了，留有医院记录"))

    assert result["guard_status"] == "unknown"
    assert result["current_safety_status"] == "unknown"
    assert result["guard_pause_required"] is True
    assert "是否已经脱离现场" in result["guard_notice_markdown"]
    assert "## 请补充" not in result["guard_notice_markdown"]


def test_threat_inside_attachment_does_not_become_current_danger():
    raw = (
        "【图片证据补充（视觉模型识别，需与原图核对）】\n"
        "文件：聊天截图.png\n对方说“今晚到你家伤害你”"
    )
    state = _state(
        raw,
        fact_payload={"text": "", "source_message_id": "message-guard"},
        evidence_payload={
            "attachments": [{"file_name": "聊天截图.png"}],
            "legacy_blocks": [{"file_name": "聊天截图.png"}],
        },
        input_event_type="evidence_added",
        input_events=[{"type": "evidence_added", "payload_ref": "evidence_payload"}],
    )

    result = _run(state)

    assert result["guard_status"] == "clear"
    assert result["current_safety_status"] == "not_applicable"
    assert result["guard_pause_required"] is False


def test_imminent_cctv_overwrite_creates_urgent_preservation_actions():
    result = _run(_state("商场监控明天就会覆盖，我还没有申请保留"))

    assert result["guard_status"] == "urgent"
    assert result["guard_pause_required"] is False
    assert {risk["risk_type"] for risk in result["guard_report"]["risks"]} == {
        "evidence_loss"
    }
    assert any("书面保留请求" in item["action"] for item in result["guard_report"]["immediate_actions"])
    assert "我会继续按您本轮提供的信息梳理案件" in result["guard_notice_markdown"]


def test_frozen_account_uses_asset_notice_not_personal_danger():
    result = _run(_state("我的银行卡被冻结了，钱暂时取不出来"))

    assert result["guard_status"] == "warning"
    assert result["current_safety_status"] == "not_applicable"
    assert result["guard_pause_required"] is False
    assert result["asset_emergency_risk"]["risk_type"] == "asset_emergency"
    assert "110" not in result["guard_notice_markdown"]


def test_unlawful_collection_is_blocked_with_legal_alternatives():
    result = _run(_state("能不能帮我破解对方账号，把聊天记录找出来"))

    assert result["guard_status"] == "warning"
    assert result["restricted_action_flags"][0]["risk_type"] == "unlawful_collection"
    actions = result["guard_report"]["immediate_actions"]
    assert any("不要入侵账号" in item["action"] for item in actions)
    assert any("平台导出" in item["action"] for item in actions)


def test_dangerous_confrontation_is_urgent_but_does_not_close_case():
    result = _run(_state("我准备去堵他并强行拿走他的东西"))

    assert result["guard_status"] == "urgent"
    assert result["guard_pause_required"] is False
    assert result["restricted_action_flags"][0]["risk_type"] == "dangerous_confrontation"
    assert any("不要单独上门堵人" in item["action"] for item in result["guard_report"]["immediate_actions"])


def test_deadline_warning_never_announces_fixed_one_or_three_year_period():
    result = _run(_state("我刚收到裁决书，但不知道上诉期限是否来得及"))

    assert result["guard_status"] == "warning"
    assert result["deadline_risk"]["risk_type"] == "deadline"
    assert result["risk_related_missing_facts"]
    assert "一年" not in result["guard_notice_markdown"]
    assert "三年" not in result["guard_notice_markdown"]
    assert "不会仅凭事情发生时间直接认定已经超过期限" in result["guard_notice_markdown"]
    assert result["guard_retrieval_trace"]["status"] == "required_but_deferred"


def test_deadline_and_evidence_loss_are_both_preserved():
    result = _run(_state("监控明天会覆盖，上诉期限也在明天截止"))

    assert result["guard_status"] == "urgent"
    assert {risk["risk_type"] for risk in result["guard_report"]["risks"]} == {
        "deadline",
        "evidence_loss",
    }


def test_model_failure_keeps_an_existing_safety_pause_closed():
    state = _state(
        "我不知道对方现在走了没有",
        safety_pause_active=True,
        current_safety_status="danger",
        safety_pause_case_message="对方拿刀堵在门外",
    )

    result = _run(state, _deps(error=TimeoutError("model timeout")))

    assert result["guard_status"] == "unknown"
    assert result["guard_pause_required"] is True
    assert result["safety_pause_active"] is True
    assert result["guard_report"]["degraded"] is True


def test_safe_reply_resolves_pause_and_keeps_original_resume_route():
    active_risk = {
        "risk_id": "risk-old",
        "risk_type": "personal_safety",
        "level": "critical",
        "status": "active",
        "trigger": "对方拿刀堵在门外",
    }
    state = _state(
        "我已经到朋友家，现在安全了，付款时间是7月18日",
        safety_pause_active=True,
        current_safety_status="danger",
        safety_pause_case_message="对方拿刀堵在门外",
        safety_resume_route=["update_facts", "decide_facts"],
        safety_resume_stage="fact_gathering",
        active_risk_flags=[active_risk],
    )

    result = _run(state)

    assert result["current_safety_status"] == "safe"
    assert result["safety_pause_active"] is False
    assert result["guard_pause_required"] is False
    assert result["workflow_stage"] == "fact_gathering"
    assert result["resolved_risk_flags"][-1]["risk_id"] == "risk-old"
    assert result["resolved_risk_flags"][-1]["resolution_source"] == "user_confirmed_safe"


def test_clear_turn_emits_no_redundant_notice():
    result = _run(_state("房东一直没有退还押金"))

    assert result["guard_status"] == "clear"
    assert result["guard_notice_markdown"] == ""
    assert result["guard_notice_pending"] is False
    assert result["guard_next_route"] == "update_facts"
