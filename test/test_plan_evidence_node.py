"""Contracts for the formal node-five evidence planning checkpoint."""
from __future__ import annotations

import asyncio

from src.agents.legal_guide.graph import (
    build_guide_graph,
    route_after_decide_facts,
)
from src.agents.legal_guide.plan_evidence import (
    build_delivery_entries,
    run_plan_evidence,
    validate_fact_snapshot,
    validate_plan_citations,
    version_evidence_plan,
)
from src.agents.legal_guide.state import GuideState


def _fact(
    key: str,
    category: str,
    statement: str,
    status: str = "confirmed",
) -> dict:
    return {
        "fact_id": key,
        "semantic_key": key,
        "key": key,
        "category": category,
        "statement": statement,
        "status": status,
    }


def _state(**changes) -> GuideState:
    values = {
        "case_id": "plan-node-test",
        "legal_domain": "consumer_market",
        "fact_blackboard_version": 3,
        "fact_snapshot_version": 1,
        "fact_snapshot_confirmed": True,
        "fact_snapshot_draft": {
            "fact_snapshot_draft_id": "draft-1",
            "based_on_fact_blackboard_version": 3,
            "snapshot_hash": "sha256:test",
            "stale": False,
        },
        "fact_blackboard": [
            _fact("transaction.amount", "amount", "已经支付800元"),
            _fact("event.non_delivery", "event", "对方没有发货"),
            _fact("claim.refund", "claim", "希望退款"),
            _fact("relationship.seller", "relationship", "对方是个人卖家"),
        ],
        "internal_evidence_requirements": [
            {
                "requirement_id": "transaction.payment",
                "proof_target_id": "proof.transaction.payment",
                "label": "付款记录",
                "dependent_fact_keys": ["transaction.amount"],
                "recommended_material_classes": ["支付平台账单", "付款记录"],
                "alternative_material_classes": ["银行流水"],
                "status": "active_candidate",
            },
            {
                "requirement_id": "delivery.non_performance",
                "proof_target_id": "proof.delivery.non_performance",
                "label": "未发货或未履行记录",
                "dependent_fact_keys": ["event.non_delivery"],
                "recommended_material_classes": ["物流记录", "聊天记录"],
                "alternative_material_classes": ["平台工单"],
                "status": "active_candidate",
            },
        ],
        "evidence_name_inventory": [
            {
                "evidence_name_id": "ename-payment",
                "normalized_name": "payment_record",
                "display_name": "付款记录",
                "original_names": ["付款记录"],
                "status": "user_claimed_present",
                "source_refs": [],
            }
        ],
    }
    values.update(changes)
    return GuideState(**values)


def test_unconfirmed_snapshot_cannot_create_a_plan():
    state = _state(fact_snapshot_confirmed=False, proceed_under_uncertainty=False)
    result = validate_fact_snapshot(state)
    assert result["valid"] is False
    assert result["reason"] == "fact_snapshot_not_confirmed"


def test_stale_snapshot_is_rejected():
    state = _state(
        fact_snapshot_draft={
            "based_on_fact_blackboard_version": 2,
            "snapshot_hash": "sha256:old",
            "stale": False,
        }
    )
    result = validate_fact_snapshot(state)
    assert result["valid"] is False
    assert result["status"] == "stale"


def test_plan_creates_formal_requirements_and_delivery_entries_without_assessing_claimed_material():
    result = asyncio.run(run_plan_evidence(_state()))

    assert result["evidence_plan_version"] == 1
    assert result["evidence_collection_status"] == "open"
    assert result["next_route"] == "await_evidence_batch"
    requirements = {
        item["requirement_id"]: item
        for item in result["formal_evidence_requirements"]
    }
    assert {"transaction.payment", "delivery.non_performance"} <= set(requirements)
    assert requirements["transaction.payment"]["user_material_state"] == "user_claimed_present"
    assert requirements["transaction.payment"]["submitted_material_ids"] == []
    assert len(result["delivery_entries"]) == len(requirements)
    assert all(item["status"] == "open" for item in result["delivery_entries"])
    assert "真实性" not in requirements["transaction.payment"].get("assessment", "")


def test_needs_pinpoint_citation_is_not_user_visible_basis():
    valid, limitations = validate_plan_citations(
        [
            {
                "title": "未精确定位的法律",
                "law_id": "1",
                "article_no": "第一条",
                "version_key": "current",
                "review_status": "needs_pinpoint",
                "status": "active",
            },
            {
                "title": "已审校依据",
                "source_id": "official-1",
                "version_key": "2026",
                "locator": "第1条",
                "review_status": "approved",
                "status": "active",
            },
        ],
        return_limitations=True,
    )
    assert [item["title"] for item in valid] == ["已审校依据"]
    assert limitations


def test_repeated_same_plan_keeps_version_and_delivery_ids():
    state = _state(
        evidence_plan_version=2,
        evidence_plan_fingerprint="fingerprint",
        legal_model_version=4,
    )
    legal_model = {
        "legal_domain": "consumer_market",
        "relation_candidates": [],
        "request_models": [],
    }
    proof_targets = [
        {
            "proof_target_id": "proof.transaction.payment",
            "status": "active",
            "dependent_fact_keys": ["transaction.amount"],
        }
    ]
    requirements = [
        {
            "requirement_id": "transaction.payment",
            "proof_target_id": "proof.transaction.payment",
            "status": "active",
            "importance": "essential",
            "recommended_materials": ["付款记录"],
            "alternative_materials": [],
        }
    ]
    first = version_evidence_plan(
        state,
        legal_model=legal_model,
        requirements=requirements,
        proof_targets=proof_targets,
        fact_snapshot={
            "fact_snapshot_version": 1,
            "fact_snapshot_hash": "sha256:test",
        },
    )
    state = state.model_copy(
        update={
            "evidence_plan_version": first["evidence_plan_version"],
            "evidence_plan_fingerprint": first["fingerprint"],
        }
    )
    second = version_evidence_plan(
        state,
        legal_model=legal_model,
        requirements=requirements,
        proof_targets=proof_targets,
        fact_snapshot={
            "fact_snapshot_version": 1,
            "fact_snapshot_hash": "sha256:test",
        },
    )
    assert second["reused"] is True
    assert second["evidence_plan_version"] == first["evidence_plan_version"]
    assert first["delivery_entries"][0]["delivery_entry_id"] == build_delivery_entries(
        requirements,
        case_id=state.case_id,
        evidence_plan_version=first["evidence_plan_version"],
    )[0]["delivery_entry_id"]


def test_formal_graph_routes_converged_facts_to_plan_evidence():
    graph = build_guide_graph(object())
    assert "plan_evidence" in set(graph.get_graph().nodes)
    assert route_after_decide_facts(
        GuideState(decision_status="proceed_to_evidence_planning")
    ) == "plan_evidence"
