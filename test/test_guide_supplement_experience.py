"""继续补充选择、含糊回答和追问可解释性的体验回归。"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.followup_catalog import (
    assess_initial_evidence,
    evidence_followups,
    fact_followups,
)
from src.agents.legal_guide.graph import (
    GuideDeps,
    _format_case_summary,
    _format_followup_reply,
    _with_memory_recall_preface,
    node_ask_followup,
    node_parse_details,
    node_prepare_turn,
    route_after_assess_retrieve,
    route_after_urgency,
)
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.core.config import get_settings


settings = get_settings()


def _deps_with_parse(payload: dict) -> GuideDeps:
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return deps


def _deps_with_plan(question: str = "还有哪个事实会明显影响下一步处理？") -> GuideDeps:
    return _deps_with_parse({
        "should_ask": True,
        "ask_type": "facts",
        "decision_key": "optional_material_fact",
        "candidate_id": "",
        "question": question,
        "reason": "判断是否需要调整处理路径",
        "answer_hint": "没有其他内容时可以直接说没有",
        "acknowledgement": "",
        "acknowledged_fact_keys": [],
        "basis_kind": "official_elements",
        "law_index": -1,
        "information_gain": 0.6,
        "user_burden": 0.2,
    })


def test_followup_explains_reason_and_authoritative_source_without_overclaiming():
    state = GuideState(
        legal_domain="contracts_property_housing",
        confirmed_issues=["房屋租赁押金返还纠纷"],
    )

    reply = _format_followup_reply(
        state,
        "押金是多少，对方为什么不退？",
        ask_type="facts",
        reason="确定请求金额和对方抗辩理由",
        answer_hint="金额说大概数也可以。",
    )

    assert "为什么" not in reply or "对方为什么不退" in reply
    assert "追问依据" in reply
    assert "最高人民法院" in reply
    assert "不是要求您必须提交的固定材料" in reply


def test_followup_reason_does_not_repeat_purpose_prepositions():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )

    reply = _format_followup_reply(
        state,
        "劳动关系现在还在继续，还是已经离职？",
        ask_type="facts",
        reason="用于判断仲裁时效和当前可走的处理路径",
    )

    assert "为了用于" not in reply
    assert "用于用于" not in reply
    assert "再确认这一点是为了判断仲裁时效" in reply


def test_high_value_plan_asks_a_specific_question_without_a_choice_menu():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        collected_facts=["拖欠3个月工资", "拖欠金额24000元"],
        evidence_confirmed=["劳动合同", "工资流水"],
        confidence_tier="HIGH",
        confidence_score=0.8,
    )

    updates = asyncio.run(node_ask_followup(state, _deps_with_plan()))

    assert updates["pending_ask_details"] == ["还有哪个事实会明显影响下一步处理？"]
    assert "继续补充" not in updates["messages"][0].content
    assert "现在生成方案" in updates["messages"][0].content
    assert "您是否还想继续补充" not in updates["messages"][0].content


def test_complete_state_routes_directly_to_conclusion_without_a_choice_menu():
    domain = "labor_social_security"
    state = GuideState(
        legal_domain=domain,
        confirmed_issues=["拖欠劳动报酬"],
        collected_facts=["还在职", "拖欠3个月工资", "拖欠金额24000元"],
        evidence_confirmed=["劳动合同", "工资流水", "考勤记录", "聊天记录"],
        asked_followup_ids=(
            [rule.id for rule in fact_followups(domain)]
            + [rule.id for rule in evidence_followups(domain)]
        ),
        confidence_tier="HIGH",
        confidence_score=0.9,
        followup_plan={"should_ask": False},
    )

    assert route_after_assess_retrieve(state) == "conclude"


def test_continue_choice_accepts_freeform_detail_when_catalog_is_exhausted():
    domain = "labor_social_security"
    state = GuideState(
        legal_domain=domain,
        confirmed_issues=["拖欠劳动报酬"],
        asked_followup_ids=(
            [rule.id for rule in fact_followups(domain)]
            + [rule.id for rule in evidence_followups(domain)]
        ),
        supplement_choice="continue",
        supplement_choice_offered=True,
        allow_extra_followups=True,
        confidence_tier="HIGH",
    )

    routed = state.model_copy(update={"followup_plan": {"should_ask": True}})
    assert route_after_assess_retrieve(routed) == "ask_followup"
    updates = asyncio.run(node_ask_followup(state, _deps_with_plan()))

    assert updates["pending_ask_type"] == "facts"
    assert updates["pending_followup_ids"] == []
    assert "还有哪个事实会明显影响下一步处理" in updates["messages"][0].content
    assert "现在生成方案" in updates["messages"][0].content
    assert "劳动争议调解仲裁法及劳动人事争议仲裁办事指南" in updates["messages"][0].content
    assert "https://rsj.beijing.gov.cn" in updates["messages"][0].content


def test_meaningful_optional_supplement_returns_control_to_user():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        messages=[HumanMessage(content="公司昨天承诺下个月补发，我保存了完整群聊录屏。")],
        pending_ask_details=["请补充一项您认为会影响处理结果的重要情况。"],
        pending_ask_type="facts",
        pending_followup_ids=[],
        supplement_choice="continue",
        supplement_choice_offered=True,
        allow_extra_followups=True,
        confidence_tier="HIGH",
    )
    deps = _deps_with_parse({
        "is_answer": True,
        "new_issues": [],
        "collected_facts": ["公司承诺下个月补发工资"],
        "evidence": ["完整群聊录屏"],
        "evidence_unavailable": [],
        "adverse_facts": [],
        "region": "",
        "time_info": "",
        "user_question": "",
    })

    updates = asyncio.run(node_parse_details(state, deps))

    assert updates["supplement_choice_offered"] is False
    assert updates["allow_extra_followups"] is False
    assert updates["supplement_choice"] == ""
    rescored = state.model_copy(update={**updates, "followup_plan": {"should_ask": False}})
    assert route_after_assess_retrieve(rescored) == "conclude"


def test_supplement_summary_prioritizes_core_dispute_facts_over_evidence_phrases():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        case_facts=[
            {
                "key": "wage.arrears.duration", "category": "time",
                "statement": "公司拖欠3个月工资", "value": "3个月",
                "status": "asserted", "source_text": "公司拖欠3个月工资", "turn": 1,
            },
            {
                "key": "wage.arrears.amount", "category": "amount",
                "statement": "拖欠工资24000元", "value": "24000元",
                "status": "asserted", "source_text": "拖欠工资24000元", "turn": 1,
            },
        ],
        evidence_confirmed=["劳动合同", "工资流水", "考勤记录"],
        confidence_tier="HIGH",
    )

    reply = _format_case_summary(state)

    assert "时间：3个月" in reply
    assert "金额：24000元" in reply
    assert "劳动合同" not in reply
    assert "工资流水" not in reply


def test_user_choice_continue_can_cross_soft_limit_but_keeps_hard_limit():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        ask_rounds=settings.GUIDE_SOFT_ASK_ROUNDS,
        facts_rounds=2,
        awaiting_supplement_choice=True,
        supplement_choice_offered=True,
        round=1,
        total_rounds=1,
        messages=[HumanMessage(content="继续补充")],
    )

    prepared = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))
    continued = state.model_copy(update=prepared)

    assert continued.supplement_choice == "continue"
    assert continued.allow_extra_followups is True
    assert continued.awaiting_supplement_choice is False
    assert route_after_urgency(continued) == "assess_retrieve"

    updates = asyncio.run(node_ask_followup(continued, _deps_with_plan()))
    assert updates["ask_rounds"] == settings.GUIDE_SOFT_ASK_ROUNDS + 1
    assert "直接回复“现在生成方案”" in updates["messages"][0].content


def test_legacy_unclear_choice_state_is_cleared_without_consuming_a_followup_round():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        awaiting_supplement_choice=True,
        supplement_choice_offered=True,
        ask_rounds=3,
        round=2,
        total_rounds=2,
        messages=[HumanMessage(content="好的")],
    )

    updates = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))

    assert updates["awaiting_supplement_choice"] is False
    assert updates["supplement_choice"] == ""
    assert "ask_rounds" not in updates


def test_freeform_details_implicitly_continue_after_supplement_choice():
    state = GuideState(
        legal_domain="consumer_market",
        awaiting_supplement_choice=True,
        supplement_choice_offered=True,
        round=2,
        total_rounds=2,
        messages=[HumanMessage(content="我有付款记录、会员卡和完整聊天记录。")],
    )

    prepared = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))
    continued = state.model_copy(update=prepared)

    assert continued.supplement_choice == "continue"
    assert continued.supplement_has_details is True
    assert continued.awaiting_supplement_choice is False
    assert route_after_urgency(continued) == "extract_issues"


def test_details_are_processed_before_a_simultaneous_conclusion_choice():
    state = GuideState(
        legal_domain="consumer_market",
        awaiting_supplement_choice=True,
        supplement_choice_offered=True,
        retrieval_completed=True,
        round=2,
        total_rounds=2,
        messages=[HumanMessage(content="我还拍了店铺关门的照片，现在生成方案。")],
    )

    prepared = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))
    continued = state.model_copy(update=prepared)

    assert continued.supplement_choice == "conclude"
    assert continued.supplement_has_details is True
    assert continued.wants_conclude is True
    assert route_after_urgency(continued) == "extract_issues"


def test_plain_conclusion_choice_does_not_look_like_case_details():
    state = GuideState(
        awaiting_supplement_choice=True,
        supplement_choice_offered=True,
        round=2,
        total_rounds=2,
        messages=[HumanMessage(content="现在生成方案")],
    )

    prepared = asyncio.run(node_prepare_turn(state, MagicMock(spec=GuideDeps)))
    continued = state.model_copy(update=prepared)

    assert continued.supplement_choice == "conclude"
    assert continued.supplement_has_details is False
    assert route_after_urgency(continued) == "assess_retrieve"


def test_pure_conclusion_command_is_not_parsed_or_saved_as_a_case_fact():
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["人身损害及报案处理"],
        wants_conclude=True,
        messages=[HumanMessage(content="现在生成方案")],
        pending_ask_details=["您是否认识对方？"],
        pending_ask_type="facts",
        pending_followup_ids=["criminal_person_time"],
        collected_facts=["用户昨天在百货大楼受伤"],
        draftable_facts=["用户昨天在百货大楼受伤"],
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock()

    updates = asyncio.run(node_parse_details(state, deps))

    deps.llm.ainvoke.assert_not_awaited()
    assert updates["pending_ask_details"] == []
    assert "collected_facts" not in updates
    assert "draftable_facts" not in updates
    assert "case_facts" not in updates


def test_unrelated_but_useful_detail_does_not_fake_resolution_of_pending_rule():
    state = GuideState(
        legal_domain="consumer_market",
        messages=[HumanMessage(content="我有付款记录和会员卡。")],
        pending_ask_details=["这家店的登记名称是什么？"],
        pending_ask_type="facts",
        pending_followup_ids=["consumer_transaction"],
    )
    deps = _deps_with_parse({
        "is_answer": True,
        "answers_asked_question": False,
        "user_question": "",
        "collected_facts": [],
        "case_updates": [{
            "key": "evidence.payment",
            "category": "evidence",
            "statement": "用户持有付款记录和会员卡",
            "value": "付款记录和会员卡",
            "certainty": "asserted",
            "operation": "add",
            "source_text": "我有付款记录和会员卡",
        }],
        "evidence": ["付款记录", "会员卡"],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(node_parse_details(state, deps))

    assert "consumer_transaction" not in updates["fact_records"]
    assert "付款记录" in updates["evidence_confirmed"]
    assert updates["pending_ask_details"] == []


def test_repeated_question_is_not_saved_as_fact_or_document_fact():
    question = "退房确认单上有房东或中介的签字、盖章吗？"
    state = GuideState(
        legal_domain="contracts_property_housing",
        messages=[HumanMessage(content=question)],
        pending_ask_details=[question],
        pending_ask_type="facts",
        pending_followup_ids=["contract_type_terms"],
    )
    deps = _deps_with_parse({
        "is_answer": True,
        "user_question": "",
        "new_issues": [],
        "collected_facts": ["退房确认单有房东或中介签字盖章"],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(node_parse_details(state, deps))

    assert updates["fact_records"]["contract_type_terms"]["status"] == "ambiguous"
    assert updates["pending_ask_details"] == [question]
    assert "collected_facts" not in updates
    assert "draftable_facts" not in updates
    assert "更像是把问题重复了一遍" in updates["messages"][0].content


def test_explicit_missing_evidence_is_kept_even_when_user_also_requests_conclusion():
    question = "您是否有劳动合同、工资条或工资流水？"
    state = GuideState(
        legal_domain="labor_social_security",
        messages=[HumanMessage(content="没有合同和工资条，不要再问了，请按现有信息给方案。")],
        pending_ask_details=[question],
        pending_ask_type="evidence",
        wants_conclude=True,
    )
    deps = _deps_with_parse({
        "is_answer": True,
        "user_question": "",
        "new_issues": [],
        "collected_facts": [],
        "evidence": [],
        "evidence_unavailable": ["劳动合同", "工资条"],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(node_parse_details(state, deps))

    assert "劳动合同" in updates["evidence_unavailable"]
    assert "工资条" in updates["evidence_unavailable"]
    assert updates["pending_ask_details"] == []
    assert updates["deferred_questions"] == []


def test_repair_quote_has_explicit_proof_limitations():
    assessments = assess_initial_evidence(["家具维修报价单"])
    record = next(iter(assessments.values()))
    limitations = "；".join(record["limitations"])

    assert "不能单独证明损坏原因" in limitations
    assert "实际维修已经发生" in limitations


def test_explicit_history_question_confirms_relevant_memory_before_followup():
    state = GuideState(
        user_context={
            "long_term_memories": [
                "用户所在地区：上海",
                "用户在上海，正面临劳动争议，老板已拖欠三个月工资。",
            ]
        }
    )

    reply = _with_memory_recall_preface(
        state,
        "我之前说的劳动争议是什么情况？",
        "为了进一步判断，请问您是否有劳动合同？",
    )

    assert "我记得您之前提到" in reply
    assert "拖欠三个月工资" in reply
    assert "以您这次说明为准" in reply
