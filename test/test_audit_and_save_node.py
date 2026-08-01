"""Contracts for formal node eight plan audit, publication, and history."""
from __future__ import annotations

import asyncio
import copy
import json

from src.agents.legal_guide.audit_and_save import (
    audit_solution_draft,
    run_audit_and_save,
    validate_audit_inputs,
)
from src.agents.legal_guide.db_queries import save_solution_version
from src.agents.legal_guide.generate_solution import run_generate_solution
from src.agents.legal_guide.graph import (
    build_guide_graph,
    route_after_audit_and_save,
)
from src.agents.legal_guide.state import GuidePhase, GuideState


def _fact(key: str, statement: str, status: str = "confirmed") -> dict:
    return {
        "fact_id": key,
        "semantic_key": key,
        "key": key,
        "statement": statement,
        "status": status,
    }


def _state(**changes) -> GuideState:
    values = {
        "case_id": "audit-node-test",
        "session_id": "public-user:audit-session",
        "user_context": {"user_id": "public-user"},
        "legal_domain": "consumer_market",
        "confirmed_issues": ["网络买卖合同履行"],
        "fact_blackboard_version": 4,
        "fact_snapshot_version": 2,
        "fact_snapshot_confirmed": True,
        "fact_snapshot_draft": {
            "based_on_fact_blackboard_version": 4,
            "snapshot_hash": "sha256:facts-v2",
            "stale": False,
        },
        "fact_blackboard": [
            _fact("transaction.amount", "用户称已支付800元"),
            _fact("event.non_delivery", "用户称个人卖家未发货"),
            _fact("claim.refund", "用户希望退款"),
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
        "proof_targets": [
            {
                "target_id": "proof.transaction.payment",
                "label": "支付事实",
            }
        ],
        "delivery_entries": [
            {
                "entry_id": "delivery-payment",
                "requirement_id": "transaction.payment",
            }
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


async def _generated_state(**changes) -> GuideState:
    state = _state(**changes)
    generated = await run_generate_solution(state)
    return state.model_copy(update=generated)


def test_node_eight_publishes_audited_version_and_full_source_bundle():
    async def scenario():
        state = await _generated_state()
        assert validate_audit_inputs(state)["valid"] is True
        return await run_audit_and_save(state, object())

    result = asyncio.run(scenario())
    markdown = str(result["messages"][0].content)
    assert result["phase"] == GuidePhase.END
    assert result["workflow_stage"] == "plan_issued"
    assert result["solution_draft_status"] == "published"
    assert result["pending_solution_audit"] is False
    assert result["plan_version"] == 1
    assert result["solution_versions"][0]["fact_snapshot"]
    assert result["solution_versions"][0]["legal_model"]
    assert result["solution_versions"][0]["evidence_plan"]
    assert result["solution_versions"][0]["evidence_review"]
    assert result["solution_versions"][0]["case_tasks"]
    assert "正式方案版本：** 第 1 版" in markdown
    assert "plan-draft:" not in markdown
    assert "## 后续更新" in markdown


def test_stale_fact_snapshot_blocks_publication_and_returns_upstream():
    async def scenario():
        state = await _generated_state()
        return state.model_copy(
            update={
                "fact_snapshot_version": 3,
                "fact_snapshot_draft": {
                    "based_on_fact_blackboard_version": 4,
                    "snapshot_hash": "sha256:facts-v3",
                    "stale": False,
                },
            }
        )

    state = asyncio.run(scenario())
    result = asyncio.run(run_audit_and_save(state, object()))
    assert result["solution_audit_status"] == "blocked"
    assert result["solution_draft_status"] == "audit_blocked"
    assert result["next_route"] == "decide_facts"
    routed = state.model_copy(update=result)
    assert route_after_audit_and_save(routed) == "decide_facts"


def test_modified_draft_fingerprint_is_rejected():
    async def scenario():
        state = await _generated_state()
        tampered = copy.deepcopy(state.solution_draft)
        tampered["core_judgment"]["summary"] = "保证胜诉"
        return state.model_copy(update={"solution_draft": tampered})

    result = validate_audit_inputs(asyncio.run(scenario()))
    assert result["valid"] is False
    assert result["next_route"] == "generate_solution"
    assert any(
        item["code"] == "solution_draft_fingerprint_mismatch"
        for item in result["fatal_issues"]
    )


def test_correctable_fact_legal_and_expression_defects_are_repaired():
    async def scenario():
        state = await _generated_state()
        draft = copy.deepcopy(state.solution_draft)
        draft["confirmed_facts"].append(
            {
                "fact_id": "unknown.fact",
                "statement": "模型自行补造事实",
                "status": "confirmed",
            }
        )
        draft["action_basis_refs"].append(
            {
                "title": "不存在的具体法条",
                "article_no": "第一百条",
            }
        )
        draft["core_judgment"]["summary"] = "本案胜诉率为90%"
        return state, draft

    state, draft = asyncio.run(scenario())
    corrected, issues, _ = audit_solution_draft(state, draft)
    rendered = str(corrected)
    assert "模型自行补造事实" not in rendered
    assert "不存在的具体法条" not in rendered
    assert "90%" not in rendered
    assert any(item["code"] == "repair_fact_boundary" for item in issues)
    assert any(item["code"] == "repair_legal_boundary" for item in issues)
    assert any(
        item["code"] == "repair_forbidden_language_core_judgment"
        for item in issues
    )


def test_same_candidate_is_idempotent_and_does_not_create_new_version():
    async def scenario():
        generated_state = await _generated_state()
        first = await run_audit_and_save(generated_state, object())
        replay_state = generated_state.model_copy(
            update={
                "plan_version": first["plan_version"],
                "solution_versions": first["solution_versions"],
                "published_solution": first["published_solution"],
                "published_solution_markdown":
                    first["published_solution_markdown"],
                "published_solution_fingerprint":
                    first["published_solution_fingerprint"],
                "solution_audit_history":
                    first["solution_audit_history"],
            }
        )
        second = await run_audit_and_save(replay_state, object())
        return first, second

    first, second = asyncio.run(scenario())
    assert first["plan_version"] == 1
    assert second["plan_version"] == 1
    assert len(second["solution_versions"]) == 1
    assert second["decision_status"] == "solution_version_reused"


def test_formal_graph_exposes_node_eight():
    graph = build_guide_graph(object())
    assert "audit_and_save" in set(graph.get_graph().nodes)


def test_database_index_upserts_current_plan_and_version_history():
    class Result:
        def scalars(self):
            return self

        def first(self):
            return None

    class Database:
        def __init__(self):
            self.record = None
            self.committed = False

        async def execute(self, _statement):
            return Result()

        def add(self, record):
            self.record = record

        async def commit(self):
            self.committed = True

        async def refresh(self, record):
            record.id = 42

        async def rollback(self):
            raise AssertionError("rollback should not run")

    db = Database()
    version = {
        "case_id": "audit-node-test",
        "case_generation": 1,
        "plan_version": 1,
        "solution": {"published_markdown": "## 正式方案"},
    }
    result = asyncio.run(
        save_solution_version(
            user_id="12",
            session_id="12:session",
            domain="consumer_market",
            issues=["网络买卖合同履行"],
            version_record=version,
            version_history=[version],
            db=db,
        )
    )
    stored = json.loads(db.record.action_plan)
    assert db.committed is True
    assert result["status"] == "database_and_case_state"
    assert result["consultation_id"] == 42
    assert stored["current_plan_version"] == 1
    assert stored["versions"][0]["plan_version"] == 1
    assert db.record.legal_advice == "## 正式方案"
