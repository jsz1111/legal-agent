"""Compatibility contracts for the grouped legal-guide state."""
from __future__ import annotations

import json

from src.agents.legal_guide.state import (
    ControlState,
    GuidePhase,
    GuideState,
)


def test_guide_state_exposes_only_grouped_top_level_channels():
    assert set(GuideState.model_fields) == {
        "messages",
        "phase",
        "case",
        "issues",
        "facts",
        "followup",
        "evidence",
        "retrieval",
        "control",
        "safety",
        "output",
        "strategy",
    }


def test_flat_constructor_and_attribute_access_remain_compatible():
    state = GuideState(
        case_id="case-1",
        legal_domain="consumer_market",
        case_facts=[{"key": "loss.amount", "statement": "损失3000元"}],
        followup_plan={"should_ask": True},
        force_conclude=True,
    )

    assert state.case.case_id == "case-1"
    assert state.issues.model_dump()["confirmed_issues"] == []
    assert state.facts.case_facts[0]["key"] == "loss.amount"
    assert state.followup.followup_plan["should_ask"] is True
    assert state.control.force_conclude is True

    state.force_conclude = False
    state.legal_domain = "contract_commercial"
    assert state.control.force_conclude is False
    assert state.case.legal_domain == "contract_commercial"


def test_legacy_flat_json_is_migrated_and_new_json_is_grouped():
    legacy = json.dumps({
        "phase": "DETAIL_GATHER",
        "session_id": "u:s",
        "confirmed_issues": ["合同违约"],
        "pending_ask_details": ["是否约定履行期限？"],
        "evidence_requirement_version": 3,
        "fraud_stop_loss_offered": True,
    }, ensure_ascii=False)

    state = GuideState.model_validate_json(legacy)
    assert state.phase == GuidePhase.DETAIL_GATHER
    assert state.session_id == "u:s"
    assert state.confirmed_issues == ["合同违约"]
    assert state.pending_ask_details == ["是否约定履行期限？"]
    assert state.evidence_requirement_version == 3
    assert state.fraud_stop_loss_offered is True

    persisted = json.loads(state.model_dump_json())
    assert persisted["case"]["session_id"] == "u:s"
    assert persisted["issues"]["confirmed_issues"] == ["合同违约"]
    assert "session_id" not in persisted
    assert "confirmed_issues" not in persisted


def test_flat_node_updates_are_grouped_without_losing_sibling_values():
    state = GuideState(
        session_id="u:s",
        legal_domain="labor_social_security",
        ask_rounds=2,
        confidence_tier="MEDIUM",
    )

    updates = state.group_updates({
        "legal_domain": "consumer_market",
        "force_conclude": True,
        "messages": [],
    })

    assert updates["case"].session_id == "u:s"
    assert updates["case"].legal_domain == "consumer_market"
    assert isinstance(updates["control"], ControlState)
    assert updates["control"].force_conclude is True
    assert updates["control"].ask_rounds == 2
    assert updates["control"].confidence_tier == "MEDIUM"
    assert updates["messages"] == []


def test_model_copy_accepts_legacy_flat_updates():
    original = GuideState(
        session_id="u:s",
        case_facts=[{"key": "occur_time", "statement": "2026年7月"}],
        solution_version=1,
    )

    copied = original.model_copy(update={
        "solution_version": 2,
        "wants_conclude": True,
    })

    assert copied.solution_version == 2
    assert copied.wants_conclude is True
    assert copied.session_id == "u:s"
    assert copied.case_facts == original.case_facts
    assert original.solution_version == 1
    assert original.wants_conclude is False
