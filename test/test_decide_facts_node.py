"""Contracts for the formal node-four fact decision checkpoint."""

from __future__ import annotations

import asyncio

from src.agents.legal_guide.decide_facts import (
    assess_fact_sufficiency,
    build_fact_question_candidates,
    run_decide_facts,
    update_internal_evidence_requirements,
)
from src.agents.legal_guide.graph import (
    build_guide_graph,
    route_after_decide_facts,
)
from src.agents.legal_guide.state import GuideState


def _fact(key: str, category: str, statement: str, status: str = "confirmed", group: str = ""):
    return {
        "fact_id": key,
        "semantic_key": key,
        "key": key,
        "category": category,
        "statement": statement,
        "status": status,
        "conflict_group_id": group,
    }


def _consumer_state(**changes):
    base = [
        _fact("location.platform", "location", "通过平台交易"),
        _fact("transaction.amount", "amount", "已经支付800元"),
        _fact("event.non_delivery", "event", "对方没有发货"),
        _fact("claim.refund", "claim", "希望退款"),
    ]
    values = {
        "fact_blackboard": base,
        "fact_blackboard_version": 1,
    }
    values.update(changes)
    return GuideState(
        case_id="node-four-test",
        legal_domain="consumer_market",
        **values,
    )


def test_first_pass_builds_a_batch_of_fact_questions_only():
    state = _consumer_state()
    candidates = build_fact_question_candidates(state)
    keys = {item["decision_key"] for item in candidates}
    assert "event.timeline" in keys
    assert "transaction.subject_and_counterparty" in keys
    assert all(item["kind"] == "facts" for item in candidates)


def test_confirmed_and_unknown_facts_are_not_reasked():
    state = _consumer_state(
        fact_blackboard=[
            _fact("location.platform", "location", "通过平台交易"),
            _fact("transaction.amount", "amount", "已经支付800元"),
            _fact("event.non_delivery", "event", "对方没有发货"),
            _fact("claim.refund", "claim", "希望退款"),
            _fact("relationship.seller", "relationship", "对方身份不清楚", "unknown"),
        ]
    )
    keys = {item["decision_key"] for item in build_fact_question_candidates(state)}
    assert "transaction.subject_and_counterparty" in keys
    assert "relationship.type" not in keys


def test_conflict_creates_a_confirmation_question():
    state = _consumer_state(
        fact_blackboard=[
            *_consumer_state().fact_blackboard,
            _fact("transaction.amount.old", "amount", "支付800元", "conflicted", "amount"),
            _fact("transaction.amount.new", "amount", "支付900元", "conflicted", "amount"),
        ]
    )
    result = asyncio.run(run_decide_facts(state))
    assert result["fact_sufficiency"]["status"] == "blocked_by_conflict"
    assert any(
        item["question_type"] == "conflict_confirmation"
        for item in result["question_batch"]["questions"]
    )


def test_missing_uploaded_evidence_does_not_block_fact_convergence():
    state = _consumer_state(
        fact_blackboard=[
            *_consumer_state().fact_blackboard,
            _fact("relationship.seller", "relationship", "对方是个人卖家"),
            _fact("transaction.product", "relationship", "购买二手手机"),
            _fact("event.payment_time", "time", "2026年7月18日付款"),
        ],
        evidence_name_inventory=[],
    )
    report = assess_fact_sufficiency(state)
    assert report.status == "sufficient"
    result = asyncio.run(run_decide_facts(state))
    assert result["decision_status"] == "await_snapshot_confirmation"
    assert result["fact_snapshot_draft"]


def test_conclude_now_produces_a_conditional_snapshot():
    state = _consumer_state(
        fact_blackboard=[
            *_consumer_state().fact_blackboard,
            _fact("relationship.seller", "relationship", "对方是个人卖家"),
            _fact("transaction.product", "relationship", "购买二手手机"),
            _fact("event.payment_time", "time", "2026年7月18日付款"),
        ],
        wants_conclude=True,
        turn_control_intent="conclude_now",
    )
    result = asyncio.run(run_decide_facts(state))
    assert result["decision_status"] == "proceed_to_evidence_planning"
    assert result["proceed_under_uncertainty"] is True
    assert result["fact_snapshot_draft"]["unknown_fact_ids"] == []


def test_internal_evidence_requirements_are_incremental():
    first = _consumer_state()
    requirements, changes = update_internal_evidence_requirements(first)
    assert {item["requirement_id"] for item in requirements} >= {
        "transaction.payment",
        "delivery.non_performance",
    }
    assert changes
    second = first.model_copy(update={
        "internal_evidence_requirements": requirements,
        "fact_blackboard": [
            *first.fact_blackboard,
            _fact("procedure.platform_complaint", "procedure", "已经向平台投诉"),
        ],
        "fact_blackboard_version": 2,
    })
    updated, changes = update_internal_evidence_requirements(second)
    assert "platform.complaint" in {item["requirement_id"] for item in updated}
    assert any(item["requirement_id"] == "platform.complaint" for item in changes)


def test_formal_graph_contains_decide_facts_and_routes_pause_to_end():
    compiled = build_guide_graph(object())
    nodes = set(compiled.get_graph().nodes)
    assert "decide_facts" in nodes
    assert route_after_decide_facts(
        GuideState(decision_status="ask_batch")
    ) == "__end__"
