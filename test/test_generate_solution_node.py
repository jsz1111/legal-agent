"""Contracts for the formal node-seven solution draft."""
from __future__ import annotations

import asyncio

from src.agents.legal_guide.generate_solution import (
    LIKELIHOOD_TIERS,
    build_case_tasks,
    derive_qualitative_likelihood,
    load_reusable_action_basis,
    run_generate_solution,
    validate_solution_inputs,
)
from src.agents.legal_guide.graph import (
    build_guide_graph,
    node_conclude,
    route_after_assess_evidence,
    route_after_generate_solution,
    route_after_guard_v2,
    route_after_plan_evidence,
)
from src.agents.legal_guide.state import GuideState


def _fact(
    key: str,
    statement: str,
    status: str = "confirmed",
) -> dict:
    return {
        "fact_id": key,
        "semantic_key": key,
        "key": key,
        "statement": statement,
        "status": status,
    }


def _state(**changes) -> GuideState:
    values = {
        "case_id": "solution-node-test",
        "legal_domain": "consumer_market",
        "fact_blackboard_version": 4,
        "fact_snapshot_version": 2,
        "fact_snapshot_confirmed": True,
        "fact_snapshot_draft": {
            "based_on_fact_blackboard_version": 4,
            "snapshot_hash": "sha256:facts-v2",
            "stale": False,
        },
        "fact_blackboard": [
            _fact("transaction.amount", "用户已支付800元"),
            _fact("event.non_delivery", "个人卖家未发货并拉黑用户"),
            _fact("claim.refund", "用户要求退款"),
            _fact(
                "counterparty.identity",
                "卖家实名主体尚未确认",
                "unknown",
            ),
        ],
        "legal_model_version": 2,
        "legal_model_status": "candidate",
        "evidence_plan_version": 3,
        "evidence_plan_status": "active",
        "legal_model": {
            "legal_domain": "consumer_market",
            "relation_candidates": [
                {
                    "relation_id": "consumer_transaction",
                    "label": "网络买卖关系候选",
                }
            ],
            "request_models": [
                {
                    "request_id": "request.refund",
                    "request_type": "refund",
                    "label": "退款或返还款项",
                }
            ],
            "unknown_conditions": ["卖家实名主体"],
        },
        "relation_candidates": [
            {
                "relation_id": "consumer_transaction",
                "label": "网络买卖关系候选",
            }
        ],
        "request_models": [
            {
                "request_id": "request.refund",
                "request_type": "refund",
                "label": "退款或返还款项",
            }
        ],
        "formal_evidence_requirements": [
            {
                "requirement_id": "transaction.payment",
                "proof_target_id": "proof.transaction.payment",
                "label": "付款记录",
                "importance": "essential",
                "status": "active",
                "user_material_state": "not_submitted",
            },
            {
                "requirement_id": "platform.complaint",
                "proof_target_id": "proof.platform.complaint",
                "label": "平台投诉记录",
                "importance": "important",
                "status": "active",
                "user_material_state": "submitted",
            },
        ],
        "plan_basis_refs": [
            {
                "basis_type": "authority_rule",
                "source_id": "law-source-1",
                "source_version_id": "law-version-1",
                "title": "已审校的买卖合同规则",
                "issuing_authority": "权威机关",
                "locator": "相关条款",
                "review_status": "approved",
                "status": "active",
                "source_url": "https://example.test/law",
            }
        ],
        "evidence_review_version": 4,
        "evidence_review_status": "partial",
        "evidence_review_report": {
            "fact_snapshot_version": 2,
            "evidence_plan_version": 3,
            "assessment_status": "partial",
            "items": [
                {
                    "evidence_id": "evidence-platform",
                    "material_id": "material-platform",
                    "name": "平台投诉记录",
                    "assessment_status": "supports",
                }
            ],
            "coverage": [
                {
                    "target_id": "proof.transaction.payment",
                    "requirement_id": "transaction.payment",
                    "label": "付款记录",
                    "status": "not_submitted",
                    "next_action": "补充完整付款详情或银行流水",
                },
                {
                    "target_id": "proof.platform.complaint",
                    "requirement_id": "platform.complaint",
                    "label": "平台投诉记录",
                    "status": "covered",
                    "supporting_evidence_ids": ["evidence-platform"],
                    "next_action": "保存工单编号和正式回复",
                },
            ],
        },
        "relevant_channels": [
            {
                "name": "当前已检索平台争议处理入口",
                "url": "https://example.test/channel",
                "source_url": "https://example.test/channel",
                "source_org": "平台公开帮助中心",
                "route_stage": "平台处理",
                "recommendation_reason": "平台正在处理当前交易争议",
            }
        ],
    }
    values.update(changes)
    return GuideState(**values)


def test_missing_fact_snapshot_returns_to_fact_decision():
    result = validate_solution_inputs(
        _state(fact_snapshot_version=0, fact_snapshot_confirmed=False)
    )
    assert result["valid"] is False
    assert result["next_route"] == "decide_facts"


def test_stale_evidence_review_returns_to_node_six():
    state = _state(
        evidence_review_report={
            "fact_snapshot_version": 2,
            "evidence_plan_version": 2,
            "coverage": [],
        }
    )
    result = validate_solution_inputs(state)
    assert result["valid"] is False
    assert result["reason"] == "evidence_review_plan_version_stale"
    assert result["next_route"] == "assess_evidence"


def test_generate_solution_builds_five_dimensions_and_structured_draft():
    result = asyncio.run(run_generate_solution(_state()))
    dimensions = result["likelihood_assessment"]["dimensions"]
    assert result["solution_draft_status"] == "awaiting_audit"
    assert result["next_route"] == "audit_and_save"
    assert result["pending_solution_audit"] is True
    assert result["likelihood_tier"] in LIKELIHOOD_TIERS
    assert {item["dimension_id"] for item in dimensions} == {
        "rights_basis",
        "fact_clarity",
        "evidence_coverage",
        "procedural_feasibility",
        "performance_risk",
    }
    assert result["recommended_routes"]
    assert result["case_tasks"]
    assert result["document_suggestions"]
    assert "messages" not in result
    assert "## 当前维权可能性" in result["solution_draft_markdown"]
    assert "胜诉率" not in result["solution_draft_markdown"]
    assert "%" not in result["solution_draft_markdown"]


def test_no_review_still_creates_conditional_plan():
    state = _state(
        evidence_review_version=0,
        evidence_review_status="not_started",
        evidence_review_report={},
        evidence_items=[],
        evidence_observations=[],
        wants_conclude=True,
    )
    result = asyncio.run(run_generate_solution(state))
    assert result["conditional_plan"] is True
    assert result["likelihood_tier"] in {"条件性有利", "不确定"}
    assert "尚未完成材料评估" in result["solution_draft_markdown"]


def test_not_submitted_is_not_rewritten_as_explicitly_absent():
    result = asyncio.run(run_generate_solution(_state()))
    markdown = result["solution_draft_markdown"]
    assert "付款记录**：尚未提交" in markdown
    assert "付款记录**：当前明确缺失" not in markdown


def test_third_party_coverage_creates_a_collection_action():
    state = _state()
    state.evidence_review_report["coverage"][0]["status"] = (
        "third_party_available"
    )
    state.evidence_review_report["coverage"][0]["next_action"] = (
        "通过平台正式程序申请调取付款详情"
    )
    result = asyncio.run(run_generate_solution(state))
    assert any(
        "申请调取" in str(item.get("title") or "")
        for item in result["immediate_actions"]
    )


def test_unpinpointed_basis_is_not_used_as_action_authority():
    state = _state(
        plan_basis_refs=[
            {
                "title": "待定位规则",
                "review_status": "needs_pinpoint",
                "source_id": "candidate-1",
            }
        ],
        evidence_basis_refs=[],
        retrieved_law_refs=[],
        relevant_channels=[],
    )
    refs, gaps = load_reusable_action_basis(state)
    assert refs == []
    assert any("待定位规则" in item for item in gaps)


def test_likelihood_never_exposes_an_internal_score():
    dimensions = [
        {
            "dimension_id": key,
            "label": key,
            "status": "mixed",
            "positive_factors": [],
            "negative_factors": [],
            "unknown_factors": ["待确认"],
            "basis_refs": [],
            "limitations": [],
        }
        for key in (
            "rights_basis",
            "fact_clarity",
            "evidence_coverage",
            "procedural_feasibility",
            "performance_risk",
        )
    ]
    result = derive_qualitative_likelihood(dimensions)
    assert result["tier"] in LIKELIHOOD_TIERS
    assert "score" not in result
    assert "percentage" not in result


def test_task_status_is_carried_forward_without_fake_completion():
    state = _state(
        case_tasks=[
            {
                "task_id": "task-existing",
                "title": "旧任务",
                "status": "completed",
            }
        ]
    )
    action = {
        "action_id": "action-1",
        "title": "保存材料",
        "priority": 1,
        "reason": "需要回查",
        "completion_criteria": "材料已备份",
    }
    first = build_case_tasks(state, [action], [])
    generated_id = first[0]["task_id"]
    state.case_tasks = [
        {
            **first[0],
            "task_id": generated_id,
            "status": "in_progress",
        }
    ]
    second = build_case_tasks(state, [action], [])
    assert second[0]["status"] == "in_progress"


def test_repeated_same_inputs_reuse_candidate_version():
    first = asyncio.run(run_generate_solution(_state()))
    state = _state(
        solution_draft=first["solution_draft"],
        solution_draft_fingerprint=first["solution_draft_fingerprint"],
        plan_version_candidate=first["plan_version_candidate"],
        case_tasks=first["case_tasks"],
    )
    second = asyncio.run(run_generate_solution(state))
    assert second["plan_version_candidate"] == first["plan_version_candidate"]


def test_formal_graph_exposes_node_seven_and_routes_to_it():
    graph = build_guide_graph(object())
    assert "generate_solution" in set(graph.get_graph().nodes)
    assert "audit_and_save" in set(graph.get_graph().nodes)
    assert route_after_guard_v2(
        _state(requested_route="generate_solution")
    ) == "generate_solution"
    assert route_after_assess_evidence(
        GuideState(next_route="conclude")
    ) == "generate_solution"
    assert route_after_plan_evidence(
        _state(wants_conclude=True, next_route="await_evidence_batch")
    ) == "generate_solution"
    assert route_after_generate_solution(
        GuideState(
            next_route="audit_and_save",
            solution_draft={"case_id": "case"},
        )
    ) == "audit_and_save"


def test_legacy_conclude_presents_formal_draft_without_regenerating_it():
    state = _state(
        solution_draft_status="awaiting_audit",
        solution_draft_markdown=(
            "## 核心判断\n\n- 当前为条件式方案。\n\n"
            "## 当前维权可能性\n\n> **不确定**"
        ),
    )
    result = asyncio.run(node_conclude(state, object()))
    content = str(result["messages"][0].content)
    assert "## 核心判断" in content
    assert "## 当前维权可能性" in content
    assert result["solution_draft_status"] == "compatibility_presented"
