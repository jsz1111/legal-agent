"""Contracts for the formal node-six evidence assessment checkpoint."""
from __future__ import annotations

import asyncio

from src.agents.legal_guide.assess_evidence import (
    calculate_target_coverage,
    detect_new_fact_candidates,
    run_assess_evidence,
    validate_evidence_plan_version,
)
from src.agents.legal_guide.graph import (
    build_guide_graph,
    route_after_assess_evidence,
    route_after_guard_v2,
)
from src.agents.legal_guide.state import GuideState


def _state(**changes) -> GuideState:
    values = {
        "case_id": "assess-node-test",
        "legal_domain": "consumer_market",
        "fact_snapshot_version": 2,
        "evidence_plan_version": 3,
        "evidence_batch_id": "batch-3",
        "formal_evidence_requirements": [
            {
                "requirement_id": "transaction.payment",
                "proof_target_id": "proof.transaction.payment",
                "label": "付款记录",
                "purpose": "证明付款金额、时间和收款对象",
                "status": "active",
                "user_material_state": "not_submitted",
                "recommended_materials": ["平台账单"],
                "alternative_materials": ["银行流水"],
            }
        ],
        "proof_targets": [
            {
                "requirement_id": "transaction.payment",
                "proof_target_id": "proof.transaction.payment",
                "label": "付款记录",
                "purpose": "证明付款金额、时间和收款对象",
            }
        ],
        "fact_blackboard": [
            {
                "semantic_key": "transaction.amount",
                "statement": "用户已经支付800元",
                "status": "confirmed",
            }
        ],
    }
    values.update(changes)
    return GuideState(**values)


def _attachment(**changes) -> dict:
    value = {
        "material_id": "material-1",
        "file_name": "付款记录.txt",
        "file_type": "text/plain",
        "sha256": "a" * 64,
        "upload_status": "uploaded",
        "evidence_requirement_id": "transaction.payment",
        "evidence_batch_id": "batch-3",
    }
    value.update(changes)
    return value


def test_plan_version_mismatch_keeps_material_pending_remap():
    state = _state(
        base_fact_snapshot_version=1,
        current_attachments=[_attachment()],
    )
    result = validate_evidence_plan_version(state, state.current_attachments)
    assert result["valid"] is False
    assert result["status"] == "received_pending_remap"
    assert "fact_snapshot_version" in result["version_mismatches"]


def test_upload_without_batch_completion_is_staged_only():
    state = _state(
        current_attachments=[_attachment()],
        current_message_text="已先上传付款记录，稍后继续补充。",
        input_events=[{"type": "evidence_added"}],
    )
    result = asyncio.run(run_assess_evidence(state))
    assert result["evidence_review_status"] == "awaiting_batch"
    assert result["next_route"] == "await_evidence_batch"
    assert result["evidence_batch_completed"] is False
    assert result["pause_state"]["type"] == "awaiting_evidence_batch"


def test_completed_batch_creates_item_coverage_and_one_verification_pause():
    state = _state(
        current_attachments=[_attachment()],
        current_message_text="完成本批次并评估。",
        input_events=[{"type": "evidence_batch_completed"}],
    )
    result = asyncio.run(run_assess_evidence(state))
    assert result["evidence_review_status"] == "needs_verification"
    assert result["next_route"] == "assess_evidence"
    assert result["evidence_verification_pending"] is True
    assert result["pending_evidence_verification"]
    assert result["evidence_review_report"]["items"][0]["material_id"] == "material-1"
    assert result["evidence_review_report"]["coverage"][0]["status"] == "partially_covered"


def test_material_claim_different_from_confirmed_fact_is_not_auto_applied():
    state = _state()
    candidates = detect_new_fact_candidates(
        state,
        [
            {
                "material_id": "material-1",
                "source_locator": "付款记录.txt（解析文本）",
                "material_claims": [
                    {
                        "key": "transaction.amount",
                        "value": "1200",
                        "source_text": "付款金额：1200元",
                    }
                ],
            }
        ],
    )
    assert candidates
    assert candidates[0]["status"] == "pending_fact_confirmation"


def test_formal_graph_exposes_node_six_and_routes_to_it():
    graph = build_guide_graph(object())
    assert "assess_evidence" in set(graph.get_graph().nodes)
    state = GuideState(requested_route="assess_evidence")
    assert route_after_guard_v2(state) == "assess_evidence"
    assert route_after_assess_evidence(
        GuideState(next_route="conclude")
    ) == "generate_solution"


def test_not_submitted_is_not_explicitly_absent():
    state = _state()
    coverage = calculate_target_coverage(state, [])
    assert coverage[0]["status"] == "not_submitted"
