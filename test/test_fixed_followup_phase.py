"""两阶段追问：固定剧本（领域题库）阶段的行为契约。

固定阶段只问事实（facts），evidence 走独立的证据中心流程；答案终结性保证
有/没有/不清楚 三种回答都会终结该题；安全类问题禁止模型改写。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agents.legal_guide import graph as guide_graph
from src.agents.legal_guide.followup_catalog import fact_followups
from src.agents.legal_guide.followup_planner import (
    _fixed_rule_answered,
    _slot_input_type,
    build_followup_candidates,
    plan_fixed_batch,
    plan_followup_batch,
    remaining_fixed_rules,
)
from src.agents.legal_guide.graph import (
    GuideDeps,
    _format_unknown_fact_notes,
    node_ask_followup,
    node_assess_retrieve,
)
from src.agents.legal_guide.state import GuideState


def _fact(key: str, category: str, statement: str, turn: int = 1) -> dict:
    return {
        "key": key,
        "category": category,
        "statement": statement,
        "status": "asserted",
        "source_text": statement,
        "turn": turn,
    }


def _llm(payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return llm


def _raising_llm() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
    return llm


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


CYBER_RULE_IDS = [
    "cyber_channel_amount",
    "cyber_emergency_action",
    "cyber_issue_type",
]


def _cyber_fields() -> list[dict]:
    return [
        {
            "field_id": "cyber_channel_amount",
            "question": "钱是通过什么平台转出去的，大概什么时候，转了多少？",
            "input_type": "short_text",
        },
        {
            "field_id": "cyber_emergency_action",
            "question": "您是否已经联系银行或平台办理止付？",
            "input_type": "single_choice",
            "options": ["已经办理", "还没有", "不清楚"],
        },
        {
            "field_id": "cyber_issue_type",
            "question": "这次主要是被骗转账还是账号被盗？",
            "input_type": "single_choice",
            "options": ["被骗转账", "账号被盗", "其他", "不清楚"],
        },
    ]


def test_plan_fixed_batch_returns_all_remaining_fixed_rules():
    """固定阶段一次性覆盖全部剩余必问事实题。"""
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    plan = asyncio.run(plan_fixed_batch(state, _llm({
        "should_ask": True,
        "fields": _cyber_fields(),
    })))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "fixed_catalog_batch"
    assert [item["candidate_id"] for item in plan["questions"]] == CYBER_RULE_IDS
    assert all(item["is_fixed_rule"] for item in plan["questions"])
    assert all(item["required"] is False for item in plan["questions"])


def test_rewrite_dropping_a_rule_falls_back_to_catalog_original():
    """模型漏掉某个规则时，该规则以目录原文补齐，覆盖不被破坏。"""
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    fields = _cyber_fields()[:2]  # 丢弃 cyber_issue_type
    plan = asyncio.run(plan_fixed_batch(state, _llm({
        "should_ask": True,
        "fields": fields,
    })))

    assert plan["planner_mode"] == "fixed_catalog_fallback"
    by_id = {item["candidate_id"]: item for item in plan["questions"]}
    assert set(by_id) == set(CYBER_RULE_IDS)
    assert by_id["cyber_issue_type"]["question"] == "主要是被骗转账、账号被盗、隐私泄露还是网络侵权？"


def test_llm_failure_falls_back_to_full_catalog_original():
    """LLM 整体异常时，全部使用目录原文，仍生成完整固定表单。"""
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    plan = asyncio.run(plan_fixed_batch(state, _raising_llm()))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "fixed_catalog_fallback"
    assert len(plan["questions"]) == 3
    by_id = {item["candidate_id"]: item for item in plan["questions"]}
    assert by_id["cyber_channel_amount"]["question"] == (
        "事情通过什么平台或账号发生，何时发生，涉及金额多少？"
    )
    assert by_id["cyber_emergency_action"]["input_type"] == "single_choice"
    assert by_id["cyber_issue_type"]["question"] == (
        "主要是被骗转账、账号被盗、隐私泄露还是网络侵权？"
    )


def test_unknown_answer_marks_rule_answered():
    """用户对某题回答“不清楚”后，该规则不再进入剩余固定表单。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        fact_records={"cyber_channel_amount": {"status": "unknown", "value": "不清楚"}},
    )
    rules = remaining_fixed_rules(state)
    assert "cyber_channel_amount" not in [rule.id for rule in rules]
    assert {rule.id for rule in rules} == {"cyber_emergency_action", "cyber_issue_type"}
    assert _fixed_rule_answered(fact_followups("cyber_data_fraud")[0], state) is True


def test_fixed_form_contains_no_evidence_rules():
    """固定阶段只问事实，evidence 规则不进入固定表单。"""
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    rules = remaining_fixed_rules(state)
    domain_rules = fact_followups("cyber_data_fraud")
    assert len(rules) == len(domain_rules)
    assert all(getattr(rule, "slot", "") for rule in rules)


def test_no_remaining_rules_reports_fixed_facts_done():
    """固定阶段全部覆盖后 should_ask=False，标记 fixed_facts_done。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        asked_followup_ids=list(CYBER_RULE_IDS),
    )
    plan = asyncio.run(plan_fixed_batch(state, _llm({"should_ask": True, "fields": []})))
    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fixed_facts_done"


def test_remaining_fixed_rules_filters_current_safety_when_not_relevant():
    """safety_relevant=False 时，current_safety 规则不进入固定表单。"""
    state = GuideState(legal_domain="criminal_public_security", safety_relevant=False)
    assert "criminal_event_safety" not in [rule.id for rule in remaining_fixed_rules(state)]

    state = GuideState(legal_domain="criminal_public_security", safety_relevant=True)
    assert "criminal_event_safety" in [rule.id for rule in remaining_fixed_rules(state)]


def test_fixed_form_skips_rules_covered_by_user_description():
    """用户首条消息已说清的事实，固定表单自动跳过（candidate_coverage 无 missing）。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        round=1,
        time_info="上周三晚上",
        case_facts=[_fact("event.scam", "event", "在二手平台被对方骗了")],
    )
    remaining = [rule.id for rule in remaining_fixed_rules(state)]
    # cyber_channel_amount（event_time）已被“时间 + 事件经过”覆盖，不再问。
    assert "cyber_channel_amount" not in remaining
    assert "cyber_emergency_action" in remaining
    assert "cyber_issue_type" in remaining


def test_terminality_removes_answered_rules_from_dynamic_candidates():
    """有/没有/不清楚 三种回答后，该规则不再作为动态候选被重问。"""
    for status in ("user_stated", "unknown"):
        state = GuideState(
            legal_domain="cyber_data_fraud",
            fact_records={"cyber_channel_amount": {"status": status, "value": "答过"}},
        )
        candidates, _ = build_followup_candidates(state)
        candidate_ids = [item["id"] for item in candidates]
        assert "cyber_channel_amount" not in candidate_ids, status
        assert "cyber_emergency_action" in candidate_ids, status


def test_unknown_only_dimension_generates_no_new_dynamic_questions():
    """用户已按“不清楚”答过的维度不生成新题，动态阶段自然收敛。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        fact_records={"cyber_channel_amount": {"status": "unknown"}},
        decision_sufficiency={
            "dimensions": [{
                "effect": "limitation",
                "satisfied": False,
                "unresolved_rule_ids": ["cyber_channel_amount"],
            }]
        },
    )
    plan = asyncio.run(plan_followup_batch(state, MagicMock()))
    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fact_dimensions_converged"


def test_gap_scan_asks_scenario_gap_when_all_dimensions_satisfied():
    """固定阶段全部覆盖、目录维度已满足后，动态阶段仍运行检索驱动缺口扫描。

    扫描基于检索法条发现固定问卷未覆盖的场景特有缺口（如人身伤害的伤情/就医），
    而不是因"维度已满足"直接收敛。
    """
    state = GuideState(
        legal_domain="criminal_public_security",
        fact_records={
            "criminal_event_safety": {"status": "user_stated", "value": "我现在安全"},
            "criminal_person_time": {"status": "user_stated", "value": "下午三点，北京海淀区"},
            "criminal_report_status": {"status": "user_stated", "value": "还没报警"},
        },
        decision_sufficiency={
            "dimensions": [
                {"effect": "limitation", "label": "关键时间节点", "satisfied": True},
                {"effect": "jurisdiction", "label": "受理机构管辖", "satisfied": True},
                {"effect": "procedure", "label": "处理路径", "satisfied": True},
            ]
        },
        followup_basis_refs=[{
            "source_type": "statute",
            "title": "中华人民共和国刑法",
            "article_no": "第二百三十四条",
            "text": "故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。",
        }],
    )
    plan = asyncio.run(plan_followup_batch(state, _llm({
        "should_ask": True,
        "fields": [{
            "field_id": "injury_treatment_status",
            "question": "您被殴打后是否受伤，是否已去医院检查或做伤情鉴定",
            "input_type": "single_choice",
            "options": ["已就医并检查", "还没去医院", "不清楚/无法确认"],
            "decision_effects": ["procedure"],
            "basis_indices": [0],
        }],
    })))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "dynamic_retrieval_batch"
    assert plan["questions"][0]["field_id"] == "injury_treatment_status"
    assert plan["questions"][0]["basis_refs"][0]["title"] == "中华人民共和国刑法"


def test_gap_scan_converges_when_llm_finds_no_new_gap():
    """缺口扫描 LLM 判定无更多高价值缺口 → 返回空 fields → 自然收敛。"""
    state = GuideState(
        legal_domain="criminal_public_security",
        fact_records={
            "criminal_event_safety": {"status": "user_stated", "value": "我现在安全"},
            "criminal_person_time": {"status": "user_stated", "value": "下午三点，北京海淀区"},
            "criminal_report_status": {"status": "user_stated", "value": "还没报警"},
        },
        decision_sufficiency={
            "dimensions": [
                {"effect": "limitation", "label": "关键时间节点", "satisfied": True},
                {"effect": "jurisdiction", "label": "受理机构管辖", "satisfied": True},
                {"effect": "procedure", "label": "处理路径", "satisfied": True},
            ]
        },
    )
    plan = asyncio.run(plan_followup_batch(state, _llm({
        "should_ask": False,
        "fields": [],
    })))
    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fact_dimensions_converged"


def test_assess_retrieve_enters_fixed_stage_and_marks_rules_asked():
    """端到端：node_assess_retrieve 进入固定阶段，展示后写入 asked_followup_ids。"""
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = _llm({"should_ask": True, "fields": _cyber_fields()})
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.5, "confidence_tier": "MID"}),
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    plan = result["followup_plan"]
    assert plan["should_ask"] is True
    assert plan["planner_mode"] in {"fixed_catalog_batch", "fixed_catalog_fallback"}
    assert len(plan["questions"]) == 3

    next_state = state.model_copy(update=result)
    updates = asyncio.run(node_ask_followup(next_state, deps))
    # 固定阶段展示即视为已问（与动态批次的“显示≠已回答”语义区分）。
    assert set(updates["asked_followup_ids"]) == set(CYBER_RULE_IDS)
    assert set(updates["pending_followup_ids"]) == set(CYBER_RULE_IDS)
    assert updates["pending_ask_type"] == "facts"


def test_assess_retrieve_falls_through_to_dynamic_when_fixed_done():
    """固定阶段全部覆盖后，node_assess_retrieve 自然进入动态补充路径。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        round=2,
        asked_followup_ids=list(CYBER_RULE_IDS),  # 固定阶段已完成
    )
    planner = AsyncMock(return_value={
        "should_ask": False,
        "planner_mode": "no_candidates",
    })
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.3, "confidence_tier": "LOW"}),
    ), patch(
        "src.agents.legal_guide.graph.plan_next_followup",
        new=planner,
    ), patch(
        "src.agents.legal_guide.graph.node_retrieve",
        new=AsyncMock(return_value={}),
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    planner.assert_awaited_once()
    assert result["followup_plan"]["planner_mode"] == "no_candidates"


def test_fixed_input_type_heuristics_and_illegal_type_fallback():
    """input_type 启发式：时间/金额→short_text，经过/损失→long_text，是/否→single_choice。

    模型提议的非法类型回退到 slot 启发式；选择类自动补上“不清楚”兜底选项。
    """
    state = GuideState(legal_domain="cyber_data_fraud", round=1)
    plan = asyncio.run(plan_fixed_batch(state, _llm({
        "should_ask": True,
        "fields": [
            {
                "field_id": "cyber_channel_amount",
                "question": "钱通过什么平台转出的？",
                "input_type": "bogus_type",  # 非法类型 → 回退 slot 启发式
            },
            {
                "field_id": "cyber_emergency_action",
                "question": "是否已经联系银行止付？",
                "input_type": "single_choice",
                "options": ["已经办理", "还没有"],  # 缺“不清楚”→ 自动补上
            },
        ],
    })))
    by_id = {item["candidate_id"]: item for item in plan["questions"]}
    # event_time（时间/金额）→ short_text
    assert by_id["cyber_channel_amount"]["input_type"] == "short_text"
    assert by_id["cyber_channel_amount"]["options"] == []
    # procedure（是/否类）→ single_choice 且含“不清楚”兜底
    assert by_id["cyber_emergency_action"]["input_type"] == "single_choice"
    assert any("不清楚" in item or "无法确认" in item for item in by_id["cyber_emergency_action"]["options"])

    # 启发式直接判定：经过/损失 → long_text
    assert _slot_input_type("event", "经过")[0] == "long_text"
    assert _slot_input_type("harm", "损失")[0] == "long_text"
    assert _slot_input_type("event_time", "何时")[0] == "short_text"
    assert _slot_input_type("procedure", "是否")[0] == "single_choice"
    assert _slot_input_type("current_safety", "是否安全")[0] == "single_choice"


def test_safety_rule_never_rewritten():
    """安全类问题禁止模型改写，永远使用目录原文（防止措辞被软化）。"""
    state = GuideState(legal_domain="criminal_public_security", safety_relevant=True)
    plan = asyncio.run(plan_fixed_batch(state, _llm({
        "should_ask": True,
        "fields": [{
            "field_id": "criminal_event_safety",
            "question": "你现在应该是安全的吧？",  # 软化措辞
            "input_type": "single_choice",
            "options": ["应该是安全的"],
        }],
    })))
    by_id = {item["candidate_id"]: item for item in plan["questions"]}
    safety = by_id["criminal_event_safety"]
    assert safety["question"] == "您现在是否安全？"
    assert safety["input_type"] == "single_choice"
    assert "应该是安全的吧" not in safety["question"]
    assert any("无法确认" in item for item in safety["options"])


def test_format_unknown_fact_notes_renders_only_unknown_records():
    """用户未确认的关键点只渲染 status=unknown 的目录事实题，已确认的不出现。"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        fact_records={
            "cyber_channel_amount": {
                "rule_id": "cyber_channel_amount",
                "slot": "event_time",
                "status": "unknown",
                "why": "用于定位账号、资金流和紧急止付时机",
            },
            "cyber_emergency_action": {
                "rule_id": "cyber_emergency_action",
                "slot": "procedure",
                "status": "user_stated",  # 已确认 → 不渲染
                "why": "用于优先减少继续损失",
            },
        },
    )
    section = _format_unknown_fact_notes(state)
    assert "用户未确认的关键点" in section
    assert "发生时间" in section
    assert "定位账号、资金流和紧急止付时机" in section
    assert "优先减少继续损失" not in section


def test_conclude_returns_ai_text_without_injected_unknown_section():
    """AI 返回的内容原样作为最终方案，不再注入额外关键点提示。"""
    llm = _conclude_llm("用户称被骗，以下给出维权方案。")
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = GuideState(
        legal_domain="cyber_data_fraud",
        confirmed_issues=["网络诈骗"],
        collected_facts=["用户称被对方骗走钱款"],
        confidence_tier="LOW",
        fact_records={
            "cyber_channel_amount": {
                "rule_id": "cyber_channel_amount",
                "slot": "event_time",
                "value": "不清楚",
                "status": "unknown",
                "why": "用于定位账号、资金流和紧急止付时机",
                "source": "user_statement",
            }
        },
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
    reply = updates["messages"][0].content
    assert reply == "用户称被骗，以下给出维权方案。"
    assert "用户未确认的关键点" not in reply
