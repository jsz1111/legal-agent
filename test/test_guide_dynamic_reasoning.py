"""Architecture contracts for context-grounded, dynamic legal follow-ups."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.case_model import (
    active_case_facts,
    evidence_from_case_facts,
    legacy_fact_updates,
    normalize_case_updates,
    reduce_case_facts,
)
from src.agents.legal_guide.followup_catalog import evidence_followups, fact_followups
from src.agents.legal_guide.followup_planner import plan_next_followup
from src.agents.legal_guide.graph import (
    GuideDeps,
    _format_case_summary,
    build_guide_graph,
    node_ask_followup,
    node_parse_details,
)
from src.agents.legal_guide.state import GuideState


def _fact(
    key: str,
    statement: str,
    source: str,
    *,
    turn: int = 1,
    category: str = "event",
) -> dict:
    return {
        "key": key,
        "category": category,
        "statement": statement,
        "subject": "",
        "relation": "",
        "value": "",
        "status": "asserted",
        "operation": "add",
        "source_text": source,
        "turn": turn,
        "verification": "user_stated",
    }


def _planner_payload(**updates) -> dict:
    payload = {
        "should_ask": True,
        "ask_type": "facts",
        "decision_key": "next_legal_decision",
        "candidate_id": "",
        "question": "接下来最影响处理路径的时间点是什么时候？",
        "reason": "判断时效和下一步程序",
        "contextual_reason": "确认这个时间点后，才能判断接下来应优先处理哪一步",
        "answer_hint": "记不清日期时说大概月份即可",
        "decision_effects": ["limitation", "procedure"],
        "acknowledgement": "这段自由文本不会直接展示",
        "acknowledged_fact_keys": ["event.latest"],
        "basis_kind": "official_elements",
        "law_index": -1,
        "information_gain": 0.8,
        "user_burden": 0.2,
    }
    payload.update(updates)
    return payload


def _llm(payload: dict | str):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def test_atomic_fact_requires_a_quote_from_the_current_user_message():
    updates = normalize_case_updates(
        [
            {
                "key": "transaction.amount",
                "category": "amount",
                "statement": "用户支付700元",
                "source_text": "一共充值了700元",
            },
            {
                "key": "counterparty.intent",
                "category": "event",
                "statement": "经营者故意诈骗",
                "source_text": "经营者故意诈骗",
            },
        ],
        user_text="一共充值了700元",
        turn=2,
    )

    assert [item["key"] for item in updates] == ["transaction.amount"]
    assert updates[0]["source_text"] == "一共充值了700元"


def test_denied_evidence_requires_an_explicit_negative_statement():
    hallucinated = normalize_case_updates(
        [{
            "key": "evidence.photo",
            "category": "evidence",
            "statement": "用户没有拍照",
            "value": "照片",
            "certainty": "denied",
            "source_text": "就是现场发现的",
        }],
        user_text="就是现场发现的，我吃到了苍蝇啊",
        turn=3,
    )
    grounded = normalize_case_updates(
        [{
            "key": "evidence.photo",
            "category": "evidence",
            "statement": "用户没有拍照",
            "value": "照片",
            "certainty": "denied",
            "source_text": "没有拍照",
        }],
        user_text="没有拍照，也没保留实物",
        turn=3,
    )

    assert hallucinated == []
    assert grounded[0]["status"] == "denied"


def test_legacy_model_fallback_persists_only_the_users_own_words():
    updates = legacy_fact_updates(
        ["模型擅自概括为经营者具有非法占有目的"],
        user_text="店关门了，我暂时联系不上老板",
    )

    assert len(updates) == 1
    assert updates[0]["statement"] == "店关门了，我暂时联系不上老板"
    assert "非法占有" not in updates[0]["statement"]


def test_enumerated_evidence_is_reduced_to_independent_materials():
    records = [_fact(
        "evidence.bundle",
        "用户持有多项材料",
        "我有付款记录、会员卡、余额截图和聊天记录",
        category="evidence",
    )]
    records[0]["value"] = "付款记录、会员卡、余额截图、聊天记录"

    present, unavailable = evidence_from_case_facts(records)

    assert present == ["付款记录", "会员卡", "余额截图", "聊天记录"]
    assert unavailable == []


def test_same_semantic_key_supports_replace_deny_and_unresolved_conflict():
    first = reduce_case_facts([], [{
        "key": "transaction.amount", "category": "amount",
        "statement": "支付700元", "source_text": "支付700元",
    }], user_text="支付700元", turn=1)
    corrected = reduce_case_facts(first, [{
        "key": "transaction.amount", "category": "amount",
        "statement": "支付500元", "source_text": "更正，是500元", "operation": "replace",
    }], user_text="更正，是500元", turn=2)
    denied = reduce_case_facts(corrected, [{
        "key": "transaction.amount", "category": "amount",
        "statement": "此前金额不准确", "source_text": "此前金额不准确",
        "certainty": "denied", "operation": "deny",
    }], user_text="此前金额不准确", turn=3)
    conflict = reduce_case_facts([], [{
        "key": "event.date", "category": "time",
        "statement": "事情发生在3月", "source_text": "事情发生在3月",
    }], user_text="事情发生在3月", turn=1)
    conflict = reduce_case_facts(conflict, [{
        "key": "event.date", "category": "time",
        "statement": "事情发生在4月", "source_text": "事情发生在4月",
    }], user_text="事情发生在4月", turn=2)

    assert [item["statement"] for item in active_case_facts(corrected)] == ["支付500元"]
    assert [item["status"] for item in active_case_facts(denied)] == ["denied"]
    assert {item["status"] for item in active_case_facts(conflict)} == {"conflicted"}


SCENES = [
    ("consumer_market", "预付消费", "我要求退款后，对方把我拉黑了"),
    ("labor_social_security", "欠薪", "公司还欠我三个月工资，我仍在职"),
    ("contracts_property_housing", "租赁", "房东说墙面损坏，所以不退押金"),
    ("traffic_personal_injury", "交通事故", "交警认定对方全责，我还在治疗"),
    ("family_vulnerable_groups", "离婚", "孩子六岁，一直由我照顾"),
    ("cyber_data_fraud", "网络转账", "我刚通过微信转了两万元"),
    ("administrative_remedies", "行政处罚", "昨天收到市场监管局的处罚决定"),
    ("intellectual_property", "著作权", "短视频账号用了我的摄影作品"),
    ("environment_pollution", "噪声", "楼下酒吧每天凌晨还在放音乐"),
    ("criminal_public_security", "人身伤害", "昨晚被打，现在已经安全"),
    ("medical_education_tax", "培训退费", "培训机构停课，还欠我八千元"),
    ("mediation_notary_arbitration", "仲裁", "我刚收到仲裁裁决书"),
    ("other", "其他纠纷", "对方是家公司，收款后一直没有履行"),
]


@pytest.mark.parametrize("domain,issue,latest", SCENES)
def test_all_scenes_use_the_same_grounded_acknowledgement_contract(
    domain: str,
    issue: str,
    latest: str,
):
    state = GuideState(
        round=1,
        legal_domain=domain,
        confirmed_issues=[issue],
        messages=[HumanMessage(content=latest)],
        case_facts=[_fact("event.latest", latest, latest)],
    )
    plan = asyncio.run(plan_next_followup(state, _llm(_planner_payload())))
    planned = state.model_copy(update={"followup_plan": plan})
    deps = MagicMock(spec=GuideDeps)

    updates = asyncio.run(node_ask_followup(planned, deps))
    reply = updates["messages"][0].content

    assert latest in reply
    assert reply.count("？") + reply.count("?") == 1
    assert "追问依据" in reply
    assert updates["asked_decision_keys"] == ["next_legal_decision"]


def test_planner_rejects_a_repeated_semantic_decision_even_with_new_wording():
    state = GuideState(
        round=1,
        legal_domain="consumer_market",
        asked_decision_keys=["counterparty_response"],
        case_facts=[_fact("event.latest", "对方没有回复", "对方没有回复")],
    )
    proposal = _planner_payload(
        decision_key="counterparty_response",
        question="经营者后来有没有再联系您？",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_fallback_invalid_expression"
    assert plan["decision_key"] == "consumer_transaction"
    assert plan["decision_trace"]["selected_candidate_id"] == "consumer_transaction"


def test_planner_rejects_catalog_dimension_already_covered_by_case_facts():
    state = GuideState(
        round=3,
        legal_domain="consumer_market",
        time_info="现场发现",
        case_facts=[
            _fact("transaction.merchant", "在新东方餐馆消费", "在新东方餐馆", category="relationship"),
            _fact("transaction.amount", "消费39元", "花了39", category="amount"),
            _fact("food.problem", "饺子中发现苍蝇", "吃的饺子，吃到了苍蝇", category="event"),
            _fact("event.discovery_time", "现场发现食品异物", "就是现场发现的", category="time"),
        ],
    )
    proposal = _planner_payload(
        candidate_id="consumer_problem_time",
        decision_key="problem_time",
        question="问题是什么时候发现的，商品或服务具体哪里不符合约定？",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_fallback_model_changed_candidate"
    assert plan["candidate_id"] == "consumer_negotiation_claim"
    assert plan["decision_trace"]["selected_candidate_id"] == "consumer_negotiation_claim"


def test_food_followup_renders_case_specific_question_and_reason():
    state = GuideState(
        round=2,
        legal_domain="consumer_market",
        confirmed_issues=["食品安全问题"],
        case_facts=[
            _fact(
                "transaction.summary", "在新东方餐馆花39元购买饺子",
                "在新东方餐馆，花了39，吃的饺子", turn=2, category="relationship",
            ),
            _fact(
                "food.problem", "饺子中发现苍蝇", "吃到了苍蝇",
                turn=2, category="event",
            ),
        ],
        followup_plan={
            **_planner_payload(
                ask_type="facts",
                candidate_id="consumer_problem_time",
                decision_key="discovery_time",
                question="您是当场发现，还是离开餐馆后才发现的？",
                contextual_reason="您已经说清了餐馆、39元饺子和苍蝇，现在只需要确认发现的时间点",
                answer_hint="说“当场”或“离店后”就可以",
            ),
        },
    )
    deps = MagicMock(spec=GuideDeps)

    updates = asyncio.run(node_ask_followup(state, deps))
    reply = updates["messages"][0].content

    assert "餐馆、39元饺子和苍蝇" in reply
    assert "当场发现，还是离开餐馆后" in reply
    assert "商品或服务具体哪里" not in reply
    assert "先确认一个关键点" not in reply
    assert reply.count("？") + reply.count("?") == 1


def test_evidence_plan_cannot_turn_material_question_into_payment_method_question():
    state = GuideState(
        round=2,
        legal_domain="consumer_market",
        case_facts=[
            _fact("transaction.amount", "消费39元", "花了39元", category="amount"),
            _fact("transaction.merchant", "在某餐馆消费", "在某餐馆", category="relationship"),
            _fact("food.problem", "饺子中发现苍蝇", "吃到了苍蝇", category="event"),
            _fact("event.discovery_time", "当场发现问题", "当场发现", category="time"),
            _fact("claim.refund", "用户希望退款", "我想退款", category="claim"),
        ],
        time_info="当场发现",
    )
    proposal = _planner_payload(
        ask_type="evidence",
        candidate_id="consumer_transaction_evidence",
        decision_key="proof_of_payment",
        question="您是用现金还是手机支付这39元的？",
        contextual_reason="如果没有付款记录，对方一定会否认这次消费",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert "订单、发票、收据或付款记录" in plan["question"]
    assert plan["contextual_reason"] == ""


def test_contextual_reason_cannot_make_unreviewed_compensation_claims():
    state = GuideState(
        round=1,
        legal_domain="consumer_market",
        case_facts=[
            _fact("food.problem", "在饭店碗里发现苍蝇", "碗里吃了苍蝇", category="event"),
        ],
    )
    proposal = _planner_payload(
        candidate_id="consumer_transaction",
        decision_key="transaction_amount",
        question="这顿饭大约花了多少钱？",
        contextual_reason="通常可以主张价款十倍赔偿，知道金额才能计算赔偿范围",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["contextual_reason"] == ""


def test_declarative_detail_is_not_misclassified_as_a_counter_question():
    user_text = "就是现场发现的，我吃到了苍蝇啊"
    state = GuideState(
        round=3,
        legal_domain="consumer_market",
        confirmed_issues=["食品安全问题"],
        messages=[HumanMessage(content=user_text)],
        pending_ask_details=["您有付款记录吗？"],
        pending_ask_type="evidence",
        pending_followup_ids=["consumer_transaction_evidence"],
    )
    payload = {
        "is_answer": False,
        "answers_asked_question": False,
        "user_question": "",
        "collected_facts": [],
        "case_updates": [{
            "key": "evidence.photo",
            "category": "evidence",
            "statement": "用户没有拍照",
            "value": "照片",
            "certainty": "denied",
            "source_text": "就是现场发现的",
        }],
        "evidence": [],
        "evidence_unavailable": ["照片"],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    }
    deps = MagicMock(spec=GuideDeps)
    deps.llm = _llm(payload)

    updates = asyncio.run(node_parse_details(state, deps))

    assert updates["deferred_questions"] == []
    assert updates["pending_ask_details"] == []
    assert updates["case_facts"][0]["statement"] == user_text
    assert updates["evidence_unavailable"] == []
    assert not updates["messages"]


def test_generic_planner_question_uses_scannable_markdown_without_repeating_context():
    state = GuideState(
        round=1,
        legal_domain="consumer_market",
        case_facts=[
            _fact(
                "food.problem", "在饭店吃饭时碗里发现苍蝇",
                "我在饭店里碗里吃了苍蝇", category="event",
            ),
        ],
        followup_plan={
            **_planner_payload(
                candidate_id="consumer_transaction",
                decision_key="transaction_identity_amount",
                question="这次是在哪家餐馆消费，大约花了多少钱？",
                contextual_reason="",
            ),
        },
    )
    deps = MagicMock(spec=GuideDeps)

    updates = asyncio.run(node_ask_followup(state, deps))
    reply = updates["messages"][0].content

    assert "碗里发现苍蝇" in reply
    assert reply.count("碗里发现苍蝇") == 1
    assert "### 已记录" in reply
    assert "### 请确认" in reply
    assert "### 为什么要问" in reply
    assert "> **这次是在哪家餐馆消费，大约花了多少钱？**" in reply
    assert "- **用途：**" in reply
    assert "- **追问依据：**" in reply
    assert "先确认一个关键点" not in reply


def test_case_summary_groups_atomic_facts_and_hides_repeated_paraphrases():
    root = _fact(
        "event.food_contamination", "您在饭店碗里吃到了苍蝇",
        "我在饭店里碗里吃了苍蝇", category="event",
    )
    root.update(subject="用户", relation="发现", value="苍蝇")
    location = _fact(
        "event.food_contamination.location", "您在新东方餐馆用餐",
        "在新东方餐馆", turn=2, category="location",
    )
    location["value"] = "新东方餐馆"
    amount = _fact(
        "event.food_contamination.amount", "您花费39元",
        "花了39元", turn=2, category="amount",
    )
    amount["value"] = "39元"
    repeated = _fact(
        "legacy.raw.repeat", "就是现场发现的，我吃到了苍蝇啊",
        "就是现场发现的，我吃到了苍蝇啊", turn=3, category="event",
    )
    state = GuideState(
        case_facts=[root, location, amount, repeated],
        evidence_confirmed=["付款记录"],
    )

    summary = _format_case_summary(state)
    assert "地点：新东方餐馆" in summary
    assert "金额：39元" in summary
    assert summary.count("苍蝇") == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"information_gain": 0.1},
        {"user_burden": 0.95},
    ],
)
def test_model_numeric_scores_cannot_change_application_policy(updates: dict):
    state = GuideState(
        round=1,
        legal_domain="other",
        case_facts=[_fact("event.latest", "我已经把主要情况说完了", "我已经把主要情况说完了")],
    )

    plan = asyncio.run(plan_next_followup(state, _llm(_planner_payload(**updates))))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_policy"
    assert plan["information_gain"] != updates.get("information_gain")
    assert plan["user_burden"] != updates.get("user_burden")


def test_null_optional_candidate_id_does_not_abort_a_valid_dynamic_plan():
    state = GuideState(
        round=1,
        legal_domain="other",
        case_facts=[_fact("event.latest", "对方没有履行约定", "对方没有履行约定")],
    )

    plan = asyncio.run(plan_next_followup(
        state,
        _llm(_planner_payload(candidate_id=None)),
    ))

    assert plan["should_ask"] is True
    assert plan["candidate_id"] == "other_event_party"
    assert plan["planner_mode"] == "deterministic_policy"


def test_unresolved_current_safety_is_a_mandatory_first_gate():
    state = GuideState(
        round=1,
        legal_domain="criminal_public_security",
        safety_relevant=True,
        current_safety_status="unknown",
        case_facts=[_fact("event.assault", "用户称被他人打伤", "我被人打了")],
    )
    llm = _llm(_planner_payload())

    plan = asyncio.run(plan_next_followup(state, llm))

    llm.ainvoke.assert_not_awaited()
    assert plan["candidate_id"] == "criminal_event_safety"
    assert plan["question"] == "您现在是否安全？"
    assert plan["planner_mode"] == "mandatory_safety_gate"


def test_explicit_current_safety_resolves_the_safety_gate():
    safe_fact = _fact(
        "status.current_safety", "用户目前安全", "我现在安全", category="event",
    )
    state = GuideState(
        round=1,
        legal_domain="criminal_public_security",
        safety_relevant=True,
        current_safety_status="safe",
        case_facts=[
            _fact("event.assault", "用户称被他人打伤", "我被人打了"),
            safe_fact,
        ],
    )
    llm = _llm(_planner_payload(candidate_id="criminal_person_time"))

    plan = asyncio.run(plan_next_followup(state, llm))

    llm.ainvoke.assert_awaited_once()
    assert plan["candidate_id"] == "criminal_person_time"


def test_non_safety_criminal_matter_does_not_trigger_the_safety_gate():
    state = GuideState(
        round=1,
        legal_domain="criminal_public_security",
        safety_relevant=False,
        current_safety_status="not_applicable",
        case_facts=[_fact("event.property_loss", "用户称财物遗失", "我的财物丢了")],
    )
    llm = _llm(_planner_payload(candidate_id="criminal_person_time"))

    plan = asyncio.run(plan_next_followup(state, llm))

    llm.ainvoke.assert_awaited_once()
    assert plan["candidate_id"] == "criminal_person_time"
    assert plan["planner_mode"] == "deterministic_policy"


@pytest.mark.parametrize(
    "contextual_reason",
    [
        "伤情等级会影响走治安报案还是刑事立案",
        "如果不知道对方身份，案件可能难以立案",
        "有监控的话警察可以更快锁定对方",
    ],
)
def test_contextual_reason_cannot_invent_procedural_outcomes(contextual_reason: str):
    state = GuideState(
        round=1,
        legal_domain="criminal_public_security",
        case_facts=[
            _fact("event.injury", "用户称被他人打伤", "我被人打伤了"),
            _fact("event.time", "事件发生在昨天", "昨天发生", category="time"),
            _fact("procedure.report", "用户已经报案", "我已经报案", category="procedure"),
        ],
        time_info="昨天",
    )
    payload = _planner_payload(
        candidate_id="criminal_original_clues",
        ask_type="evidence",
        question="现场监控或证人线索，您现在能找到哪一种？",
        contextual_reason=contextual_reason,
    )

    plan = asyncio.run(plan_next_followup(state, _llm(payload)))

    assert plan["should_ask"] is True
    assert plan["contextual_reason"] == ""
    assert "相关人员身份" in plan["reason"]


def test_planner_repairs_multiple_questions_to_one_center_question():
    state = GuideState(
        round=1,
        legal_domain="other",
        case_facts=[_fact("event.latest", "对方没有履行", "对方没有履行")],
    )

    plan = asyncio.run(plan_next_followup(
        state,
        _llm(_planner_payload(question="发生时间是什么时候？损失金额是多少？")),
    ))

    assert plan["should_ask"] is True
    assert plan["question"] == "您和对方是什么关系，例如买卖、借贷、劳动或租赁关系？"
    assert plan["question"].count("？") == 1


def test_invalid_law_reference_falls_back_to_official_elements_source():
    state = GuideState(
        round=1,
        legal_domain="consumer_market",
        retrieved_law_refs=[{
            "title": "中华人民共和国消费者权益保护法实施条例",
            "article_no": "第二十二条",
            "text": "经营者未按照约定提供服务的，应当退还预付款。",
        }],
        case_facts=[_fact("event.latest", "店已经停止营业", "店已经停止营业")],
    )
    proposal = _planner_payload(basis_kind="law", law_index=99)

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["basis_kind"] == "official_elements"
    assert plan["law_index"] == -1
    assert not plan["law_source"]


def test_planner_never_exposes_an_unsupported_model_citation_in_its_reason():
    state = GuideState(
        round=1,
        legal_domain="other",
        case_facts=[_fact("event.latest", "商品收到后出现故障", "商品收到后出现故障")],
    )
    proposal = _planner_payload(
        reason="适用七日无理由退货（消费者权益保护法第二十四条）",
        answer_hint="根据《消费者权益保护法》第二十四条回答即可",
        decision_effects=["claim_scope"],
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["reason"] == "识别法律关系和责任主体"
    assert plan["answer_hint"] == "简单说明您和对方是什么关系即可，不确定可以说“不清楚”。"
    assert "第二十四条" not in json.dumps(plan, ensure_ascii=False)


def test_planner_failure_uses_the_policy_selected_catalog_question():
    state = GuideState(legal_domain="consumer_market")

    plan = asyncio.run(plan_next_followup(state, _llm("not-json")))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_fallback_planner_error"
    assert plan["candidate_id"] == "consumer_transaction"
    assert plan["question"]
    assert plan["decision_trace"]["selected_candidate_id"] == "consumer_transaction"


def test_catalog_fallback_only_asks_the_uncovered_semantic_dimension():
    state = GuideState(
        legal_domain="other",
        case_facts=[
            _fact(
                "event.non_delivery",
                "用户已经付款但对方没有交付",
                "已经付款但没有交付",
                category="event",
            ),
        ],
    )

    plan = asyncio.run(plan_next_followup(state, _llm("not-json")))

    assert plan["candidate_id"] == "other_event_party"
    assert "关系" in plan["question"]
    assert "具体发生了什么" not in plan["question"]


def test_model_wording_cannot_reintroduce_an_already_covered_dimension():
    state = GuideState(
        legal_domain="other",
        case_facts=[
            _fact(
                "event.non_delivery",
                "用户已经付款但对方没有交付",
                "已经付款但没有交付",
                category="event",
            ),
        ],
    )
    proposal = _planner_payload(
        candidate_id="other_event_party",
        decision_key="other_event_party",
        question="对方是谁，事情具体又是怎么发生的？",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert "关系" in plan["question"]
    assert "怎么发生" not in plan["question"]
    assert "关系" in plan["answer_hint"]


def test_answered_event_and_counterparty_remove_repeated_party_event_candidate():
    state = GuideState(
        legal_domain="other",
        case_facts=[
            _fact(
                "event.non_delivery",
                "用户已经付款但对方没有交付",
                "已经付款但没有交付",
                category="event",
            ),
            _fact(
                "counterparty.identity",
                "对方是个人卖家",
                "对方是个人卖家",
                category="actor",
            ),
        ],
    )

    plan = asyncio.run(plan_next_followup(state, _llm("not-json")))

    assert plan["candidate_id"] != "other_event_party"
    assert "对方是个人、公司还是政府机构" not in plan["question"]


def test_parse_details_records_blacklist_through_generic_case_updates_only():
    user_text = "要求了，对方拉黑了我"
    state = GuideState(
        round=2,
        legal_domain="consumer_market",
        confirmed_issues=["预付式消费纠纷"],
        messages=[HumanMessage(content=user_text)],
        pending_ask_details=["您是否已经联系经营者要求退款，对方有没有回应？"],
        pending_ask_type="facts",
    )
    payload = {
        "is_answer": True,
        "user_question": "",
        "new_issues": ["可能涉嫌诈骗犯罪"],
        "collected_facts": ["用户要求退款后被经营者拉黑"],
        "case_updates": [{
            "key": "counterparty.response",
            "category": "procedure",
            "statement": "用户要求退款后被经营者拉黑",
            "subject": "经营者",
            "relation": "回应退款要求",
            "value": "将用户拉黑",
            "certainty": "asserted",
            "operation": "add",
            "source_text": user_text,
        }],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    }
    deps = MagicMock(spec=GuideDeps)
    deps.llm = _llm(payload)

    updates = asyncio.run(node_parse_details(state, deps))

    assert updates["case_facts"][0]["key"] == "counterparty.response"
    assert updates["case_facts"][0]["source_text"] == user_text
    assert "用户要求退款后被经营者拉黑" in updates["collected_facts"]
    assert updates["confirmed_issues"] == ["预付式消费纠纷"]


def test_graph_uses_prepare_case_as_the_entry_node():
    compiled = build_guide_graph(MagicMock())
    nodes = set(compiled.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {
        "prepare_case", "pause_case_boundary", "handoff_document",
            "guard_case", "update_facts", "clarify",
            "decide_facts", "plan_evidence",
            "assess_evidence",
            "generate_solution",
            "audit_and_save",
            "assess_retrieve", "ask_followup", "parse_details", "conclude",
            "save_record",
        }


def test_catalog_exhaustion_converges_without_an_open_ended_question():
    domain = "consumer_market"
    state = GuideState(
        legal_domain=domain,
        asked_followup_ids=(
            [item.id for item in fact_followups(domain)]
            + [item.id for item in evidence_followups(domain)]
        ),
    )

    plan = asyncio.run(plan_next_followup(state, _llm(_planner_payload())))

    assert plan == {"should_ask": False, "planner_mode": "no_candidates"}
