from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide import graph as guide_graph
from src.agents.legal_guide.evidence_rules import (
    format_evidence_source,
    resolve_evidence_checklist,
    resolve_state_evidence_checklist,
)
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.state import GuideState


def _conclude_llm(final_reply: str) -> MagicMock:
    issue_payload = {
        "fact_tensions": [],
        "issues": [{
            "issue_id": "issue_1",
            "title": "核心争点",
            "importance": "core",
            "reason": "需要判断",
            "supporting_fact_keys": [],
            "retrieval_questions": [],
            "facts_that_change_result": [],
        }],
    }
    analysis_payload = {"analyses": [{
        "issue_id": "issue_1",
        "title": "核心争点",
        "current_view": "阶段性判断",
        "supporting_facts": [],
        "adverse_facts": [],
        "legal_basis_refs": [],
        "application_analysis": "适用分析",
        "conditional_branch": "条件分支",
        "facts_to_verify": [],
        "evidence_actions": [],
        "recommended_actions": [],
        "procedure_steps": [],
    }]}
    strategy_payload = {"strategy_plan": {
        "headline_assessment": {
            "position": "当前判断",
            "supporting_reason": "依据",
            "uncertainty": "未确认",
        },
        "priority_actions": [{
            "action": "保存材料",
            "object": "用户",
            "purpose": "固定证据",
            "why_now": "防止灭失",
            "risk": "影响举证",
        }],
        "procedure_path": [],
        "evidence_plan": [],
        "opponent_arguments": [],
        "institution_focus": [],
        "risk_boundaries": [],
        "conditions_that_change_result": [],
        "source_issue_ids": ["issue_1"],
        "source_law_refs": [],
    }}
    review_payload = {
        "adverse_points": [],
        "evidence_weaknesses": [],
        "unmet_legal_elements": [],
        "procedure_risks": [],
        "opponent_arguments": [],
        "premise_risks": [],
        "must_disclose": [],
        "current_procedure_stage": "待确认",
        "next_procedure_stage": "先补充材料",
        "next_stage_trigger": "材料齐全",
        "conditional_paths": [],
        "actionability_checks": [],
        "duplicate_actions": [],
    }
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps(issue_payload, ensure_ascii=False)),
        AIMessage(content=json.dumps(analysis_payload, ensure_ascii=False)),
        AIMessage(content=json.dumps({**strategy_payload, "adversarial_execution_review": review_payload}, ensure_ascii=False)),
        AIMessage(content=final_reply),
        AIMessage(content=json.dumps({"verdict": "acceptable", "issues": []}, ensure_ascii=False)),
    ])
    return llm


def test_labor_evidence_is_grounded_before_litigation_stage():
    checklist = resolve_evidence_checklist(
        "labor_social_security",
        "公司拖欠工资，尚未申请劳动仲裁",
    )

    assert checklist.is_officially_grounded
    assert checklist.source
    assert checklist.source["template_id"] == "spc_2025_labor_dispute_complaint"
    assert any("工资单" in item and "银行流水" in item for item in checklist.items)
    assert "法〔2025〕82号" in format_evidence_source(checklist)


def test_contract_evidence_uses_specific_case_type():
    lease = resolve_evidence_checklist(
        "contracts_property_housing",
        "房东拒绝退还租房押金",
    )
    lending = resolve_evidence_checklist(
        "contracts_property_housing",
        "朋友借钱后一直不还，有借条和转账",
    )

    assert lease.source and lease.source["case_type"] == "房屋租赁合同纠纷"
    assert any("押金" in item for item in lease.items)
    assert lending.source and lending.source["case_type"] == "民间借贷纠纷"
    assert any("款项实际交付" in item for item in lending.items)


def test_uncovered_domain_is_explicitly_system_guidance():
    checklist = resolve_evidence_checklist(
        "administrative_remedies",
        "对行政处罚决定不服",
    )

    assert not checklist.is_officially_grounded
    assert checklist.authority_level == "system_guidance"
    assert checklist.source is None
    assert "不是官方固定材料目录" in format_evidence_source(checklist)


def test_state_resolution_uses_accumulated_conversation():
    state = GuideState(
        legal_domain="contracts_property_housing",
        confirmed_issues=["返还租赁押金"],
        collected_facts=["租期届满后房东拒绝退押金"],
        messages=[HumanMessage(content="我租房交了5000元押金")],
    )

    checklist = resolve_state_evidence_checklist(state)

    assert checklist.source
    assert checklist.source["template_id"] == "spc_2025_house_lease_complaint"


def test_conclusion_prompt_and_reply_keep_official_evidence_source():
    llm = _conclude_llm("最终维权方案，依据法〔2025〕82号。")
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = GuideState(
        legal_domain="traffic_personal_injury",
        confirmed_issues=["机动车交通事故责任纠纷"],
        collected_facts=["交警已出具事故认定书"],
        evidence_confirmed=["事故认定书"],
        confidence_tier="MEDIUM",
    )

    with patch.object(
        guide_graph,
        "assess_user_situation",
        new=AsyncMock(return_value=type(
            "Verdict", (), {
                "own_risk_level": "none",
                "own_risk_kinds": [],
                "reasons": [],
                "counter_claim": False,
                "time_sensitive": False,
                "premise_risks": [],
            }
        )()),
    ), patch.object(
        guide_graph,
        "_supplement_strategy_law_retrieval",
        new=AsyncMock(return_value=([], "")),
    ):
        updates = asyncio.run(guide_graph.node_conclude(state, deps))
    prompt = next(
        call.args[0][0].content
        for call in llm.ainvoke.await_args_list
        if "请为用户生成一份实用的法律维权行动方案" in str(call.args[0][0].content)
    )
    reply = updates["messages"][0].content

    assert "医疗费凭证、费用清单和病历资料" in prompt
    assert "证据清单性质与来源" in prompt
    assert "法〔2025〕82号" in prompt
    assert "法〔2025〕82号" in reply
    assert reply == "最终维权方案，依据法〔2025〕82号。"
    assert "官方发布页" not in reply


def test_conclusion_llm_failure_propagates_without_fallback():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("slow model"))
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        collected_facts=["用户称购买的商品存在问题"],
        law_context_str=(
            "法条1【中华人民共和国消费者权益保护法 第五十五条】\n"
            "经营者提供商品或者服务有欺诈行为的，依法承担相应责任。"
        ),
        confidence_tier="LOW",
    )

    with pytest.raises(TimeoutError):
        asyncio.run(guide_graph.node_conclude(state, deps))
