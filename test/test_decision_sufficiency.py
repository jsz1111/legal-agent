"""Decision-sufficiency contracts for legal-guide convergence."""
from __future__ import annotations

from src.agents.legal_guide.convergence import should_conclude
from src.agents.legal_guide.decision_sufficiency import assess_decision_sufficiency
from src.agents.legal_guide.graph import _ensure_decision_uncertainties
from src.agents.legal_guide.state import GuideState


def _fact(key: str, category: str, statement: str) -> dict:
    return {
        "key": key,
        "category": category,
        "statement": statement,
        "status": "asserted",
        "source_text": statement,
        "turn": 1,
    }


def test_high_confidence_and_one_evidence_no_longer_force_convergence():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在问题")],
        evidence_confirmed=["付款记录"],
        confidence_tier="HIGH",
        confidence_score=0.95,
    )

    report = assess_decision_sufficiency(state)
    should_stop, forced = should_conclude(state)

    assert report.sufficient_for_definitive_plan is False
    assert "responsibility" in report.blocking_gaps
    assert should_stop is False
    assert forced is False


def test_asked_but_unanswered_rule_is_not_treated_as_resolved():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        asked_followup_ids=[
            "consumer_transaction",
            "consumer_problem_time",
            "consumer_negotiation_claim",
        ],
        case_facts=[_fact("event.problem", "event", "商品存在问题")],
    )

    report = assess_decision_sufficiency(state)

    responsibility = next(
        item for item in report.dimensions if item.effect == "responsibility"
    )
    assert responsibility.satisfied is False
    assert "consumer_transaction" in responsibility.unresolved_rule_ids


def test_complete_decision_dimensions_allow_definitive_convergence():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        case_facts=[
            _fact("transaction.amount", "amount", "支付了399元"),
            _fact("transaction.merchant", "relationship", "向某平台商家购买商品"),
            _fact("event.problem", "event", "收到的商品存在质量问题"),
            _fact("event.discovery_time", "time", "收货当天发现问题"),
            _fact("claim.refund", "claim", "希望退款"),
        ],
        time_info="收货当天",
        evidence_confirmed=["平台订单和付款记录", "问题商品完整照片", "与商家的沟通/售后记录"],
        evidence_assessments={
            "transaction_material": {
                "rule_id": "consumer_order_payment",
                "evidence_key": "consumer_order",
                "canonical_item": "平台订单和付款记录",
                "availability": "user_claimed_present",
                "authenticity": "not_verified",
                "source_form": "exported_file",
                "completeness": "complete",
                    "identity_visibility": "clear",
                    "time_visibility": "clear",
                    "acquisition_method": "platform_or_institution_export",
                    "case_specificity": "case_specific",
                },
                "problem_material": {
                "rule_id": "consumer_product_photos",
                "evidence_key": "consumer_defect_photo",
                "canonical_item": "问题商品完整照片",
                "availability": "user_claimed_present",
                "authenticity": "not_verified",
                "source_form": "native_electronic",
                "completeness": "complete",
                    "identity_visibility": "not_applicable",
                    "time_visibility": "clear",
                    "acquisition_method": "user_created",
                    "case_specificity": "case_specific",
                },
                "negotiation_material": {
                "rule_id": "consumer_negotiation_record",
                "evidence_key": "consumer_negotiation",
                "canonical_item": "与商家的沟通/售后记录",
                "availability": "user_claimed_present",
                "authenticity": "not_verified",
                "source_form": "native_electronic",
                "completeness": "complete",
                    "identity_visibility": "clear",
                    "time_visibility": "clear",
                    "acquisition_method": "user_created",
                    "case_specificity": "case_specific",
                },
        },
        confidence_tier="LOW",
        confidence_score=0.2,
    )

    report = assess_decision_sufficiency(state)
    state = state.model_copy(update={"decision_sufficiency": report.model_dump()})
    should_stop, forced = should_conclude(state)

    assert report.sufficient_for_definitive_plan is True


def test_scenario_gap_is_added_only_before_user_confirmation():
    state = GuideState(
        legal_domain="cyber_data_fraud",
        confirmed_issues=["疑似网络诈骗"],
        scenario_analysis={
            "confidence": 0.4,
            "discriminating_facts": ["钱是直接转给个人还是平台支付"],
            "confirmation_options": ["平台下单后没收到货", "对方让您直接转账"],
        },
    )

    report = assess_decision_sufficiency(state)
    assert any(item.effect == "scenario" for item in report.dimensions)

    offered = state.model_copy(update={"scenario_confirmation_offered": True})
    report_after_offer = assess_decision_sufficiency(offered)
    assert not any(item.effect == "scenario" for item in report_after_offer.dimensions)


def test_conditional_plan_gets_deterministic_decision_limits():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在问题")],
    )
    report = assess_decision_sufficiency(state)
    state = state.model_copy(update={"decision_sufficiency": report.model_dump()})

    reply = _ensure_decision_uncertainties("## 行动清单\n\n- 保存现有材料。", state)

    assert "## 决策边界与条件" in reply
    assert "责任主体与责任范围" in reply
    assert "不宜把当前方案理解为责任已经成立或结果已经确定" in reply


def test_user_requested_conclusion_still_converges_without_claiming_sufficiency():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在问题")],
        wants_conclude=True,
    )

    report = assess_decision_sufficiency(state)
    should_stop, forced = should_conclude(state)

    assert report.sufficient_for_definitive_plan is False
    assert report.can_conclude_conditionally is True
    assert should_stop is True
    assert forced is False
