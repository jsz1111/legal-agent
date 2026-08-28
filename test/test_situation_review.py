"""结论前"用户处境多角度审视"：自身被追责/反索赔/前提/时间敏感四维判定与守卫。

核心不变式：
- 无自身风险时保持"维权方"框架，不注入追责警示（避免误伤普通案件）；
- 命中自身风险时通过结论提示词切换"涉案当事人"框架；
- LLM 返回的内容原样作为最终结果，失败/解析失败直接抛错，不使用兜底。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.legal_guide import graph as guide_graph
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.prompts import CONCLUDE_PROMPT
from src.agents.legal_guide.situation_review import (
    UserSituationVerdict,
    _build_liability_warning_block,
    _ensure_risk_insights,
    assess_user_situation,
    situation_guidance,
)
from src.agents.legal_guide.state import GuideState


def _state(**kw: object) -> GuideState:
    defaults: dict = dict(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        collected_facts=["用户称购买的商品存在问题"],
    )
    defaults.update(kw)
    return GuideState(**defaults)


async def _assess(raw_content: str) -> UserSituationVerdict:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=raw_content))
    return await assess_user_situation(_state(), llm)


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


def test_parses_high_risk_verdict_from_llm_json():
    verdict = asyncio.run(_assess(
        '{"own_risk_level": "high", "own_risk_kinds": ["criminal", "civil_counter"], '
        '"reasons": ["用户还手致对方骨折"], "counter_claim": true, '
        '"time_sensitive": true, "premise_risks": ["对方不会追究"]}'
    ))

    assert verdict.own_risk_level == "high"
    assert "criminal" in verdict.own_risk_kinds
    assert verdict.counter_claim is True
    assert verdict.time_sensitive is True
    assert verdict.premise_risks == ["对方不会追究"]


def test_parses_code_fenced_json():
    verdict = asyncio.run(_assess(
        '```json\n{"own_risk_level": "warning", "own_risk_kinds": ["administrative"]}\n```'
    ))

    assert verdict.own_risk_level == "warning"
    assert verdict.own_risk_kinds == ["administrative"]


def test_invalid_json_raises_without_fallback():
    with pytest.raises(ValueError):
        asyncio.run(_assess("抱歉，我现在无法给出判定。"))


def test_llm_failure_propagates_without_fallback():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        asyncio.run(assess_user_situation(_state(), llm))


def test_conclude_prompt_has_situation_guidance_slot():
    assert "{situation_guidance}" in CONCLUDE_PROMPT
    assert "## 用户处境审视" in CONCLUDE_PROMPT


def test_normal_framework_guidance_when_no_own_risk():
    guidance = situation_guidance(UserSituationVerdict())

    assert "普通维权方框架" in guidance
    assert "涉案当事人框架" not in guidance
    assert "重要风险提示" not in guidance


def test_normal_framework_still_handles_side_angles():
    verdict = UserSituationVerdict(
        counter_claim=True,
        time_sensitive=True,
        premise_risks=["对方会赔"],
    )
    guidance = situation_guidance(verdict)

    assert "对方可能反过来向用户主张权利" in guidance
    assert "尽快调取/备份" in guidance
    assert "对方会赔" in guidance
    assert "不得把乐观预期当作既定事实" in guidance


def test_party_framework_guidance_when_own_risk_hit():
    verdict = UserSituationVerdict(
        own_risk_level="warning",
        own_risk_kinds=["criminal", "administrative"],
        reasons=["还手致对方骨折", "监控显示双方互殴"],
    )
    guidance = situation_guidance(verdict)

    assert "涉案当事人框架" in guidance
    assert "刑事追诉" in guidance
    assert "立即联系专业刑事/行政律师" in guidance
    assert "谨慎陈述" in guidance
    assert "书面形式" in guidance
    assert "调解" in guidance and "办案机关依程序决定" in guidance
    assert "禁止乐观承诺" in guidance
    assert "还手致对方骨折" in guidance


def test_ensure_risk_insights_noop_when_no_own_risk():
    reply = "**【优势与劣势】**\n**有利因素**：有购物凭证。"
    assert _ensure_risk_insights(reply, UserSituationVerdict()) == reply


def test_ensure_risk_insights_noop_when_reply_already_covers():
    verdict = UserSituationVerdict(own_risk_level="high", own_risk_kinds=["criminal"])
    covered = "您本人也可能被追责，建议立即联系刑事律师。"
    assert _ensure_risk_insights(covered, verdict) == covered


def test_ensure_risk_insights_prepends_warning_block():
    verdict = UserSituationVerdict(
        own_risk_level="high",
        own_risk_kinds=["criminal"],
        reasons=["还手致对方骨折"],
    )
    reply = "**【优势与劣势】**\n**有利因素**：有伤情照片。"
    out = _ensure_risk_insights(reply, verdict)

    assert out.startswith("> ⚠️ **重要风险提示：您本人也可能面临追责**")
    assert "立即联系专业刑事/行政律师" in out
    assert "谨慎陈述" in out
    assert "书面形式" in out
    assert "调解不是您可自主选择的路径" in out
    assert reply in out


def test_warning_block_lists_kinds_and_reasons():
    verdict = UserSituationVerdict(
        own_risk_level="warning",
        own_risk_kinds=["administrative"],
        reasons=["参与打架"],
    )
    block = _build_liability_warning_block(verdict)

    assert "行政处罚甚至刑事追诉" in block
    assert "参与打架" in block


def test_node_conclude_injects_framework_and_guard_for_high_risk():
    llm = _conclude_llm("最终维权方案，以下是行动步骤。")
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = _state(
        confirmed_issues=["互殴"],
        collected_facts=["对方先动手，我自卫还手，对方骨折"],
    )
    verdict = UserSituationVerdict(
        own_risk_level="high",
        own_risk_kinds=["criminal"],
        reasons=["还手致对方骨折"],
    )
    with patch.object(
        guide_graph, "assess_user_situation", new=AsyncMock(return_value=verdict)
    ), patch.object(
        guide_graph,
        "_supplement_strategy_law_retrieval",
        new=AsyncMock(return_value=([], "")),
    ):
        updates = asyncio.run(guide_graph.node_conclude(state, deps))

    conclusion_prompt = next(
        call.args[0][0].content
        for call in llm.ainvoke.await_args_list
        if "请为用户生成一份实用的法律维权行动方案" in str(call.args[0][0].content)
    )
    assert "涉案当事人框架" in conclusion_prompt
    assert "立即联系专业刑事/行政律师" in conclusion_prompt

    reply = updates["messages"][0].content
    assert reply == "最终维权方案，以下是行动步骤。"
    assert "重要风险提示" not in reply


def test_node_conclude_no_guard_for_normal_case():
    llm = _conclude_llm("最终维权方案，以下是行动步骤。")
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = _state(confirmed_issues=["拖欠工资"], collected_facts=["公司拖欠3个月工资"])
    with patch.object(
        guide_graph,
        "assess_user_situation",
        new=AsyncMock(return_value=UserSituationVerdict()),
    ), patch.object(
        guide_graph,
        "_supplement_strategy_law_retrieval",
        new=AsyncMock(return_value=([], "")),
    ):
        updates = asyncio.run(guide_graph.node_conclude(state, deps))

    reply = updates["messages"][0].content
    assert reply == "最终维权方案，以下是行动步骤。"
    assert "重要风险提示" not in reply
