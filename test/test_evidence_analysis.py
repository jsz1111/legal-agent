"""Contracts for conservative, target-oriented evidence assessment."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.decision_sufficiency import (
    assess_decision_sufficiency,
)
from src.agents.legal_guide.evidence_analysis import (
    evaluate_evidence,
    format_evidence_coverage,
    inspect_uploaded_evidence_blocks,
    merge_evidence_observations,
    normalize_evidence_observations,
    split_uploaded_evidence_blocks,
)
from src.agents.legal_guide.followup_planner import build_followup_candidates
from src.agents.legal_guide.graph import (
    _ensure_evidence_coverage_section,
    node_parse_details,
)
from src.agents.legal_guide.state import GuideState


def _assessment(
    *,
    name: str,
    rule_id: str,
    evidence_key: str,
    availability: str = "user_claimed_present",
    **quality,
) -> dict:
    return {
        "rule_id": rule_id,
        "evidence_key": evidence_key,
        "canonical_item": name,
        "availability": availability,
        "authenticity": "not_verified",
        "relevance": "potentially_relevant",
        "legal_admissibility": "not_determined",
        "purpose": "由领域题库确定",
        "limitations": [],
        **quality,
    }


def _fact(key: str, category: str, statement: str) -> dict:
    return {
        "key": key,
        "category": category,
        "statement": statement,
        "status": "asserted",
        "source_text": statement,
        "turn": 1,
    }


def test_existing_payment_record_is_partial_not_automatically_sufficient():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "payment": _assessment(
                name="付款记录",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
            )
        },
        confirmed_items=["付款记录"],
        unavailable_items=[],
    )

    transaction = next(
        row for row in report.coverage
        if row.target_id == "proof_target:consumer_order_payment"
    )

    assert transaction.status == "partially_covered"
    assert "内容完整性" in transaction.quality_gaps
    assert "初步证明与经营者的交易关系和金额" in transaction.purpose
    assert report.preliminarily_covered_count == 0


def test_source_anchored_complete_material_can_preliminarily_cover_target():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "order": _assessment(
                name="平台导出的完整订单",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
                source_form="exported_file",
                completeness="complete",
                identity_visibility="clear",
                time_visibility="clear",
                acquisition_method="platform_or_institution_export",
                case_specificity="case_specific",
            )
        },
        confirmed_items=["平台导出的完整订单"],
        unavailable_items=[],
    )

    transaction = next(
        row for row in report.coverage
        if row.target_id == "proof_target:consumer_order_payment"
    )

    assert transaction.status == "preliminarily_covered"
    assert transaction.quality_gaps == []
    assert "可采性" in report.disclaimer


def test_weaker_duplicate_does_not_downgrade_a_complete_supporting_material():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "complete_order": _assessment(
                name="平台导出的完整订单",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
                source_form="exported_file",
                completeness="complete",
                identity_visibility="clear",
                time_visibility="clear",
                acquisition_method="platform_or_institution_export",
                case_specificity="case_specific",
            ),
            "payment_note": _assessment(
                name="付款记录",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
            ),
        },
        confirmed_items=["平台导出的完整订单", "付款记录"],
        unavailable_items=[],
    )

    transaction = next(
        row for row in report.coverage
        if row.target_id == "proof_target:consumer_order_payment"
    )

    assert transaction.status == "preliminarily_covered"
    assert transaction.quality_gaps == []


def test_unavailable_material_is_known_gap_not_proof_coverage():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={},
        confirmed_items=[],
        unavailable_items=["消费关系和付款材料"],
    )

    transaction = next(
        row for row in report.coverage
        if row.target_id == "proof_target:consumer_order_payment"
    )

    assert transaction.status == "known_missing"
    assert report.known_missing_count == 1
    assert "替代材料" in transaction.next_action


def test_cross_domain_proof_role_links_material_without_case_keyword_patch():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "problem_capture": _assessment(
                name="裁剪内容",
                rule_id="freeform_material",
                evidence_key="freeform_material",
                source_form="screenshot",
                completeness="partial",
                acquisition_method="user_created",
                proof_roles=["problem"],
            )
        },
        confirmed_items=["裁剪内容"],
        unavailable_items=[],
    )

    problem = next(
        row for row in report.coverage
        if row.target_id == "proof_target:consumer_product_photos"
    )
    matching_link = next(
        item for item in report.links
        if item.target_id == "proof_target:consumer_product_photos"
    )

    assert problem.status == "partially_covered"
    assert matching_link.basis == "proof_role_mapping"


def test_model_cannot_add_unanchored_quality_metadata():
    observations = normalize_evidence_observations(
        [{
            "name": "聊天记录",
            "source_form": "native_electronic",
            "completeness": "complete",
            "identity_visibility": "clear",
            "time_visibility": "clear",
            "acquisition_method": "user_created",
            "proof_roles": ["problem", "unsupported_role"],
            "source_text": "我保留了完整原始聊天",
        }],
        user_text="我有聊天截图",
    )

    assert observations == []


def test_source_anchored_proof_roles_are_allowlisted():
    source = "商品问题只有一张裁剪图片"
    observations = normalize_evidence_observations(
        [{
            "name": "裁剪图片",
            "source_form": "screenshot",
            "completeness": "partial",
            "identity_visibility": "unknown",
            "time_visibility": "unknown",
            "acquisition_method": "user_created",
            "proof_roles": ["problem", "made_up_role"],
            "source_text": source,
        }],
        user_text=source,
    )

    assert observations[0]["proof_roles"] == ["problem"]


def test_fingerprinted_attachment_is_recorded_as_uploaded_copy():
    source = "订单号123，付款800元，商品未发货。"
    user_text = (
        "请结合材料分析\n\n"
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：订单.txt\n来源形式：native_electronic\n"
        "原文件 SHA-256：abc123\n【提取文字】\n"
        f"{source}"
    )
    observations = normalize_evidence_observations(
        [{
            "name": "订单记录",
            "source_form": "native_electronic",
            "completeness": "complete",
            "identity_visibility": "clear",
            "time_visibility": "clear",
            "acquisition_method": "platform_or_institution_export",
            "proof_roles": ["transaction"],
            "source_text": source,
        }],
        user_text=user_text,
    )

    merged = merge_evidence_observations(
        {},
        observations,
        domain="consumer_market",
    )
    record = next(iter(merged.values()))

    assert observations[0]["uploaded_copy"] is True
    assert record["availability"] == "uploaded_copy"
    assert record["inspection_basis"] == "uploaded_copy"


def test_unrelated_uploaded_file_does_not_upgrade_user_claimed_material():
    claim = "我还有一份纸质合同"
    user_text = (
        f"{claim}\n\n"
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：付款记录.txt\n原文件 SHA-256：abc123\n"
        "【提取文字】\n付款800元"
    )
    observations = normalize_evidence_observations(
        [{
            "name": "纸质合同",
            "source_form": "paper_original",
            "completeness": "unknown",
            "identity_visibility": "unknown",
            "time_visibility": "unknown",
            "acquisition_method": "unknown",
            "proof_roles": ["relationship"],
            "source_text": claim,
        }],
        user_text=user_text,
    )

    merged = merge_evidence_observations(
        {},
        observations,
        domain="consumer_market",
    )
    record = next(iter(merged.values()))

    assert observations[0]["uploaded_copy"] is False
    assert record["availability"] == "user_claimed_present"


def test_uploaded_document_is_removed_from_case_narrative_but_kept_in_inventory():
    user_text = (
        "我付款后一直没有收到货。\n\n"
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：订单记录.txt\n来源形式：native_electronic\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n订单金额800元，状态为待发货。"
    )

    narrative, observations = split_uploaded_evidence_blocks(user_text)

    assert narrative == "我付款后一直没有收到货。"
    assert len(observations) == 1
    assert observations[0]["name"] == "订单记录.txt"
    assert observations[0]["uploaded_copy"] is True
    assert observations[0]["content_digest"] == "abcdef0123456789"


def test_uploaded_inventory_keeps_conservative_quality_when_model_is_unavailable():
    user_text = (
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：付款聊天转录.txt\n来源形式：native_electronic\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n"
        "测试案件编号：EVAL-01\n付款时间：2026-04-28 15:36\n"
        "付款金额：人民币1000元\n收款账户：S-01\n"
        "聊天转录：卖家承诺付款后发货。\n"
        "来源说明：依据公开案件事实重构，不是平台导出的原始凭证。"
    )

    _narrative, observations = split_uploaded_evidence_blocks(user_text)
    item = observations[0]

    assert item["source_form"] == "copy"
    assert item["completeness"] == "partial"
    assert item["identity_visibility"] == "clear"
    assert item["time_visibility"] == "clear"
    assert item["acquisition_method"] == "third_party"
    assert item["case_specificity"] == "case_specific"
    assert {"payment", "communication", "time"}.issubset(item["proof_roles"])


def test_blank_reference_material_does_not_cover_case_proof_target():
    report = evaluate_evidence(
        domain="labor_social_security",
        assessments={
            "blank_contract": _assessment(
                name="劳动合同示范文本.docx",
                rule_id="labor_contract_identity",
                evidence_key="labor_contract",
                availability="uploaded_copy",
                case_specificity="blank_or_reference",
                source_form="native_electronic",
                completeness="complete",
            ),
        },
        confirmed_items=["劳动合同示范文本.docx"],
        unavailable_items=[],
    )
    relationship = next(
        item for item in report.coverage
        if item.target_id == "proof_target:labor_contract_identity"
    )

    assert relationship.status == "unresolved"
    assert relationship.supporting_evidence_ids == []
    assert "空白模板或参考资料" in relationship.next_action


def test_uploaded_quality_inspection_is_source_anchored_and_marks_template():
    user_text = (
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：劳动合同模板.docx\n来源形式：native_electronic\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n劳动合同（通用）示范文本，甲方：____，乙方：____。"
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "items": [{
            "name": "劳动合同模板.docx",
            "source_form": "native_electronic",
            "completeness": "complete",
            "identity_visibility": "unclear",
            "time_visibility": "unclear",
            "acquisition_method": "unknown",
            "case_specificity": "blank_or_reference",
            "proof_roles": ["agreement"],
            "source_text": "劳动合同（通用）示范文本，甲方：____，乙方：____。",
        }],
    }, ensure_ascii=False)))

    observations = asyncio.run(
        inspect_uploaded_evidence_blocks(user_text, llm)
    )

    assert observations[0]["uploaded_copy"] is True
    assert observations[0]["case_specificity"] == "blank_or_reference"


def test_quality_source_anchor_accepts_joined_visible_fields_but_rejects_invention():
    user_text = (
        "【文档证据补充（程序提取，需与原文件核对）】\n"
        "文件：订单.txt\n来源形式：exported_file\n"
        "原文件 SHA-256：abcdef0123456789\n"
        "【提取文字】\n订单号：TEST-001\n成交金额：800.00元\n"
        "下单时间：2026-07-18 10:21"
    )
    grounded = normalize_evidence_observations([{
        "name": "订单.txt",
        "case_specificity": "case_specific",
        "source_text": (
            "订单号：TEST-001；成交金额：800.00元；"
            "下单时间：2026-07-18 10:21"
        ),
    }], user_text=user_text)
    invented = normalize_evidence_observations([{
        "name": "订单.txt",
        "case_specificity": "case_specific",
        "source_text": "订单号：TEST-001；订单状态：交易成功",
    }], user_text=user_text)

    assert grounded[0]["uploaded_copy"] is True
    assert invented == []


def test_uploaded_quality_inspection_isolates_each_file_and_keeps_successes():
    blocks = []
    for index in range(3):
        blocks.append(
            "【文档证据补充（程序提取，需与原文件核对）】\n"
            f"文件：材料{index}.txt\n来源形式：exported_file\n"
            f"原文件 SHA-256：{'a' * 15}{index}\n"
            "【提取文字】\n"
            f"记录编号：CASE-{index}，日期：2026年7月{index + 1}日。"
        )
    user_text = "\n\n".join(blocks)
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        TimeoutError("first file slow"),
        AIMessage(content=json.dumps({
            "items": [{
                "name": "材料1.txt",
                "source_form": "exported_file",
                "completeness": "complete",
                "identity_visibility": "unclear",
                "time_visibility": "clear",
                "acquisition_method": "platform_or_institution_export",
                "case_specificity": "case_specific",
                "proof_roles": ["time"],
                "source_text": "记录编号：CASE-1，日期：2026年7月2日。",
            }],
        }, ensure_ascii=False)),
        AIMessage(content=json.dumps({
            "items": [{
                "name": "材料2.txt",
                "source_form": "exported_file",
                "completeness": "complete",
                "identity_visibility": "unclear",
                "time_visibility": "clear",
                "acquisition_method": "platform_or_institution_export",
                "case_specificity": "case_specific",
                "proof_roles": ["time"],
                "source_text": "记录编号：CASE-2，日期：2026年7月3日。",
            }],
        }, ensure_ascii=False)),
        AIMessage(content=json.dumps({
            "items": [{
                "name": "材料0.txt",
                "source_form": "exported_file",
                "completeness": "complete",
                "identity_visibility": "unclear",
                "time_visibility": "clear",
                "acquisition_method": "platform_or_institution_export",
                "case_specificity": "case_specific",
                "proof_roles": ["time"],
                "source_text": "记录编号：CASE-0，日期：2026年7月1日。",
            }],
        }, ensure_ascii=False)),
    ])

    observations = asyncio.run(
        inspect_uploaded_evidence_blocks(user_text, llm)
    )

    assert llm.ainvoke.await_count == 4
    assert [item["name"] for item in observations] == [
        "材料1.txt", "材料2.txt", "材料0.txt",
    ]
    assert all(
        item["case_specificity"] == "case_specific"
        for item in observations
    )


def test_comparable_uploaded_record_fields_expose_amount_conflict():
    user_text = "\n\n".join([
        (
            "【文档证据补充（程序提取，需与原文件核对）】\n"
            "文件：订单.txt\n来源形式：native_electronic\n"
            "原文件 SHA-256：aaaaaaaaaaaaaaaa\n"
            "【提取文字】\n订单号：TEST-001\n成交金额：800.00元"
        ),
        (
            "【文档证据补充（程序提取，需与原文件核对）】\n"
            "文件：支付记录.txt\n来源形式：native_electronic\n"
            "原文件 SHA-256：bbbbbbbbbbbbbbbb\n"
            "【提取文字】\n订单备注：TEST-001\n金额：800.00元"
        ),
        (
            "【文档证据补充（程序提取，需与原文件核对）】\n"
            "文件：手工摘要.txt\n来源形式：user_statement\n"
            "原文件 SHA-256：cccccccccccccccc\n"
            "【提取文字】\n订单号：TEST-001\n付款金额：1,200.00元"
        ),
    ])
    _narrative, observations = split_uploaded_evidence_blocks(user_text)

    merged = merge_evidence_observations(
        {},
        observations,
        domain="consumer_market",
    )
    report = evaluate_evidence(
        domain="consumer_market",
        assessments=merged,
        confirmed_items=[item["name"] for item in observations],
        unavailable_items=[],
    )
    transaction = next(
        item for item in report.coverage
        if item.target_id == "proof_target:consumer_order_payment"
    )

    assert transaction.status == "conflicted"
    assert "数值或日期" in transaction.next_action
    assert all(
        record["content_conflicts"]
        for record in merged.values()
    )


def test_partial_evidence_creates_quality_followup_instead_of_marking_done():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "payment": _assessment(
                name="付款记录",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
            )
        },
        confirmed_items=["付款记录"],
        unavailable_items=[],
    )
    state = GuideState(
        legal_domain="consumer_market",
        evidence_confirmed=["付款记录"],
        evidence_assessments={
            "payment": _assessment(
                name="付款记录",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
            )
        },
        evidence_items=[item.model_dump() for item in report.items],
        evidence_coverage=report.model_dump(),
        asked_followup_ids=[
            "consumer_transaction",
            "consumer_problem_time",
            "consumer_negotiation_claim",
        ],
    )

    candidates, _ = build_followup_candidates(state)
    quality = next(
        item for item in candidates
        if item["id"] == "consumer_order_payment"
    )

    assert quality["evaluation_mode"] == "quality"
    assert "原始载体" in quality["seed_question"]
    assert "内容完整性" in quality["coverage"]["missing"]


def test_partial_evidence_keeps_plan_conditional_even_when_facts_are_complete():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费退款纠纷"],
        case_facts=[
            _fact("transaction.amount", "amount", "支付了399元"),
            _fact("transaction.merchant", "relationship", "向平台商家购买商品"),
            _fact("event.problem", "event", "商品存在质量问题"),
            _fact("event.discovery_time", "time", "收货当天发现问题"),
            _fact("claim.refund", "claim", "希望退款"),
        ],
        time_info="收货当天",
        evidence_confirmed=["付款记录", "问题商品照片"],
    )

    report = assess_decision_sufficiency(state)

    assert report.sufficient_for_definitive_plan is False
    assert "evidence_gap" in report.advisory_gaps
    assert any(
        "仍需核验" in item
        for dimension in report.dimensions
        if dimension.effect == "evidence_gap"
        for item in dimension.missing_information
    )


def test_user_facing_matrix_states_scope_and_non_admissibility_boundary():
    report = evaluate_evidence(
        domain="consumer_market",
        assessments={
            "payment": _assessment(
                name="付款记录",
                rule_id="consumer_order_payment",
                evidence_key="consumer_order",
            )
        },
        confirmed_items=["付款记录"],
        unavailable_items=[],
    )

    text = format_evidence_coverage(report)

    assert "可能用途" in text
    assert "初步证明与经营者的交易关系和金额" in text
    assert "不认定真实性、合法性、可采性或最终证明力" in text


def test_parse_details_persists_only_source_anchored_quality_attributes():
    user_text = (
        "我有平台导出的完整原始订单文件，里面能看到商家名称和下单时间。"
    )
    payload = {
        "is_answer": True,
        "answers_asked_question": True,
        "user_question": "",
        "collected_facts": [],
        "case_updates": [{
            "key": "evidence.order",
            "category": "evidence",
            "statement": "用户持有平台订单",
            "subject": "用户",
            "relation": "持有",
            "value": "平台订单",
            "certainty": "asserted",
            "operation": "add",
            "source_text": "我有平台导出的完整原始订单文件",
        }],
        "evidence": ["平台订单"],
        "evidence_unavailable": [],
        "evidence_details": [{
            "name": "平台订单",
            "source_form": "exported_file",
            "completeness": "complete",
            "identity_visibility": "clear",
            "time_visibility": "clear",
            "acquisition_method": "platform_or_institution_export",
            "proof_roles": ["transaction", "payment", "identity", "time"],
            "source_text": user_text,
        }],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    }
    deps = MagicMock()
    deps.llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    deps.fast_llm = deps.llm
    state = GuideState(
        messages=[HumanMessage(content=user_text)],
        legal_domain="consumer_market",
        round=2,
        pending_ask_details=["订单、发票或付款记录，您现在有吗？"],
        pending_ask_type="evidence",
        pending_followup_ids=["consumer_order_payment"],
    )

    updates = asyncio.run(node_parse_details(state, deps))
    records = list(updates["evidence_assessments"].values())

    assert any(item.get("source_form") == "exported_file" for item in records)
    assert any(item.get("completeness") == "complete" for item in records)
    assert any(item.get("identity_visibility") == "clear" for item in records)


def test_final_reply_gets_deterministic_evidence_scope_section():
    state = GuideState(
        legal_domain="consumer_market",
        evidence_confirmed=["付款记录"],
    )

    reply = _ensure_evidence_coverage_section(
        "**【行动清单】**\n1. 保存材料。",
        state,
    )

    assert "## 证据作用与缺口" in reply
    assert "订单/付款记录" in reply
    assert "可能用途" in reply
    assert "不认定真实性、合法性、可采性或最终证明力" in reply
