"""Contracts for retrieval-grounded batch follow-ups and evidence intake."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide.evidence_analysis import (
    evaluate_evidence,
    evaluate_state_evidence,
    merge_evidence_observations,
    merge_evidence_requirements,
    split_uploaded_evidence_blocks,
)
from src.agents.legal_guide.followup_planner import plan_followup_batch
from src.agents.legal_guide.graph import (
    GuideDeps,
    _fact_assessments_for_prompt,
    _format_followup_batch_reply,
    node_ask_followup,
    node_parse_details,
    node_retrieve_followup_basis,
    route_after_parse,
)
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.agents.legal_guide.debug_view import guide_debug_payload


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


def test_batch_form_is_generated_from_live_gaps_and_mixes_input_types():
    state = GuideState(
        legal_domain="consumer_market",
        case_facts=[_fact("transaction.item", "event", "用户购买了一台电脑")],
        decision_sufficiency={
            "dimensions": [
                {"effect": "claim_scope", "satisfied": False, "missing_information": ["金额和诉求"]},
                {"effect": "limitation", "satisfied": False, "missing_information": ["发现时间"]},
                {"effect": "procedure", "satisfied": False, "missing_information": ["商家回应"]},
            ]
        },
        followup_basis_refs=[{
            "title": "中华人民共和国消费者权益保护法",
            "article_no": "第二十四条",
            "text": "经营者提供的商品不符合质量要求时的处理规则",
        }],
    )
    llm = _llm({
        "should_ask": True,
        "fields": [
            {
                "field_id": "transaction_total",
                "question": "这台电脑实际支付了多少钱",
                "input_type": "short_text",
                "placeholder": "例如 5999 元",
                "decision_effects": ["claim_scope"],
                "basis_indices": [0],
            },
            {
                "field_id": "problem_discovery_time",
                "question": "电脑的问题大约在什么时候发现",
                "input_type": "short_text",
                "decision_effects": ["limitation"],
                "basis_indices": [0],
            },
            {
                "field_id": "merchant_response",
                "question": "商家对这台电脑的问题目前如何回应",
                "input_type": "single_choice",
                "options": ["同意处理", "拒绝处理", "尚未回复"],
                "decision_effects": ["procedure"],
                "basis_indices": [],
            },
        ],
    })

    plan = asyncio.run(plan_followup_batch(state, llm))

    assert plan["planner_mode"] == "dynamic_retrieval_batch"
    assert len(plan["questions"]) == 3
    assert {item["input_type"] for item in plan["questions"]} == {
        "short_text", "single_choice"
    }
    assert plan["questions"][0]["basis_refs"][0]["article_no"] == "第二十四条"
    assert "不清楚/无法确认" in plan["questions"][2]["options"]


def test_batch_form_drops_previously_asked_decision_key():
    state = GuideState(
        legal_domain="consumer_market",
        case_facts=[_fact("transaction.item", "event", "用户购买了一台电脑")],
        asked_decision_keys=["transaction_total"],
        decision_sufficiency={
            "dimensions": [{
                "effect": "claim_scope",
                "satisfied": False,
                "missing_information": ["金额"],
            }]
        },
    )
    plan = asyncio.run(plan_followup_batch(state, _llm({
        "should_ask": True,
        "fields": [{
            "field_id": "transaction_total",
            "question": "这台电脑实际支付了多少钱",
            "input_type": "short_text",
            "decision_effects": ["claim_scope"],
        }],
    })))

    assert all(
        item.get("field_id") != "transaction_total"
        for item in plan.get("questions", [])
    )


def test_ask_node_persists_all_batch_fields_in_one_round():
    questions = [
        {
            "field_id": "transaction_total",
            "question": "实际支付金额是多少？",
            "input_type": "short_text",
            "decision_effects": ["claim_scope"],
            "reason": "确定请求类型和范围",
        },
        {
            "field_id": "merchant_response",
            "question": "商家目前如何回应？",
            "input_type": "single_choice",
            "options": ["同意处理", "拒绝处理", "没有回复"],
            "decision_effects": ["procedure"],
            "reason": "判断下一步处理程序",
        },
    ]
    state = GuideState(followup_plan={
        "should_ask": True,
        "plan_kind": "followup_form",
        "questions": questions,
        "planner_mode": "dynamic_retrieval_batch",
    })
    deps = MagicMock(spec=GuideDeps)

    updates = asyncio.run(node_ask_followup(state, deps))

    assert updates["ask_rounds"] == 1
    assert updates["pending_ask_details"] == [item["question"] for item in questions]
    assert updates["asked_decision_keys"] == []
    assert "2 个问题" in updates["messages"][0].content


def test_displaying_fallback_batch_does_not_mark_unanswered_rules_as_asked():
    questions = [{
        "field_id": "consumer_purchase_amount",
        "candidate_id": "consumer_purchase_amount",
        "question": "这次实际支付金额是多少？",
        "input_type": "short_text",
        "decision_effects": ["claim_scope"],
        "reason": "确定请求范围",
    }]
    state = GuideState(followup_plan={
        "should_ask": True,
        "plan_kind": "followup_form",
        "questions": questions,
        "planner_mode": "catalog_fallback_batch",
    })

    updates = asyncio.run(node_ask_followup(state, MagicMock(spec=GuideDeps)))

    assert updates["asked_followup_ids"] == []
    assert updates["pending_followup_ids"] == ["consumer_purchase_amount"]


def test_followup_basis_retrieval_does_not_query_similar_cases():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[_fact("transaction.item", "event", "用户购买了一台电脑")],
    )
    deps = GuideDeps(
        llm=MagicMock(),
        neo4j_driver=MagicMock(),
        embedding_model=MagicMock(),
        milvus_client=MagicMock(),
        db_session=None,
    )
    law_hit = {"law_id": "law-1", "article_no": "第二十四条", "text": "质量处理规则"}
    case_search = AsyncMock()
    with patch(
        "src.agents.legal_knowledge.statute_rag.search_statutes_raw",
        new=AsyncMock(return_value=[law_hit]),
    ), patch(
        "src.agents.legal_guide.graph.query_laws_and_channels",
        new=AsyncMock(return_value={"laws": [], "channels": []}),
    ), patch(
        "src.agents.legal_knowledge.case_rag.search_cases_context",
        new=case_search,
    ):
        updates = asyncio.run(node_retrieve_followup_basis(state, deps))

    case_search.assert_not_awaited()
    assert updates["followup_basis_refs"][0]["article_no"] == "第二十四条"


def test_bound_upload_is_linked_to_the_selected_evidence_requirement():
    user_text = """【文档证据补充（程序提取，需与原文件核对）】
清单项ID：proof_target:consumer_transaction_evidence
清单项：消费关系和付款材料
文件：订单.pdf
来源形式：exported_file
原文件 SHA-256：aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
【提取文字】
订单金额5999元"""
    _narrative, observations = split_uploaded_evidence_blocks(user_text)
    merged = merge_evidence_observations({}, observations, domain="consumer_market")

    record = merged["consumer_transaction_evidence"]
    assert record["availability"] == "uploaded_copy"
    assert record["requirement_id"] == "proof_target:consumer_transaction_evidence"
    report = evaluate_evidence(
        domain="consumer_market",
        assessments=merged,
        confirmed_items=[],
        unavailable_items=[],
    )
    linked_targets = {
        item.target_id for item in report.links
        if item.evidence_id == "consumer_transaction_evidence"
    }
    assert linked_targets == {"proof_target:consumer_transaction_evidence"}


def test_evidence_only_upload_never_enters_the_case_fact_store():
    user_text = """【文档证据补充（程序提取，需与原文件核对）】
清单项ID：proof_target:consumer_transaction_evidence
清单项：消费关系和付款材料
文件：订单.txt
来源形式：native_electronic
原文件 SHA-256：aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
【提取文字】
订单金额5999元"""
    state = GuideState(
        legal_domain="consumer_market",
        round=2,
        evidence_evaluation_version=2,
        messages=[HumanMessage(content=user_text)],
        pending_ask_details=["请按证据清单提交材料"],
        pending_ask_type="evidence_collection",
    )
    llm = _llm({
        "is_answer": True,
        "answers_asked_question": True,
        "answered_question_ids": [],
        "collected_facts": [user_text],
        "case_updates": [{
            "key": "other.upload_wrapper",
            "category": "other",
            "statement": user_text,
            "operation": "add",
            "certainty": "asserted",
            "source_text": user_text,
        }],
        "evidence": [],
        "evidence_details": [],
        "evidence_unavailable": [],
        "adverse_facts": [],
    })
    deps = MagicMock(spec=GuideDeps)
    deps.llm = llm

    updates = asyncio.run(node_parse_details(state, deps))

    assert updates["case_facts"] == []
    assert updates["collected_facts"] == []
    assert updates["evidence_assessments"]["consumer_transaction_evidence"]["requirement_id"] == (
        "proof_target:consumer_transaction_evidence"
    )
    assert updates["evidence_evaluation_version"] == 3


def test_post_solution_debug_keeps_evidence_revision_channel_visible():
    state = GuideState(
        phase=GuidePhase.END,
        evidence_requirements=[{
            "id": "proof_target:payment",
            "label": "付款记录",
            "active": True,
        }],
        evidence_requirement_version=4,
        evidence_evaluation_version=3,
        solution_version=2,
        solution_evidence_version=1,
    )

    debug = guide_debug_payload(state)

    assert debug["followup_form"]["kind"] == "evidence_collection"
    assert debug["followup_form"]["planner_mode"] == "post_solution_evidence_revision"
    assert debug["evidence_evaluation_version"] == 3
    assert debug["solution_version"] == 2
    assert debug["solution_evidence_version"] == 1


def test_partial_structured_form_answer_is_not_mistaken_for_question_repetition():
    questions = [
        {
            "field_id": "purchase_amount",
            "candidate_id": "",
            "question": "您购买这台电脑的总价大约是多少？",
            "input_type": "short_text",
        },
        {
            "field_id": "problem_time",
            "candidate_id": "",
            "question": "您是什么时候发现电脑存在质量问题的？",
            "input_type": "short_text",
        },
    ]
    answer = """【动态追问表单回答】
1. [purchase_amount] 您购买这台电脑的总价大约是多少？
回答：5999元"""
    state = GuideState(
        legal_domain="consumer_market",
        round=2,
        messages=[HumanMessage(content=answer)],
        pending_ask_details=[item["question"] for item in questions],
        pending_ask_type="facts",
        followup_plan={"plan_kind": "followup_form", "questions": questions},
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = _llm({
        "is_answer": True,
        "answers_asked_question": True,
        "answered_question_ids": ["purchase_amount"],
        "collected_facts": ["购买电脑支付5999元"],
        "case_updates": [{
            "key": "transaction.amount",
            "category": "amount",
            "statement": "购买电脑支付5999元",
            "operation": "add",
            "certainty": "asserted",
            "source_text": "5999元",
        }, {
            "key": "other.form_wrapper",
            "category": "other",
            "statement": answer,
            "operation": "add",
            "certainty": "asserted",
            "source_text": answer,
        }],
        "evidence": [],
        "evidence_details": [],
        "evidence_unavailable": [],
        "adverse_facts": [],
    })

    updates = asyncio.run(node_parse_details(state, deps))

    assert any(item["statement"] == "购买电脑支付5999元" for item in updates["case_facts"])
    assert all("【动态追问表单回答】" not in item["statement"] for item in updates["case_facts"])
    assert updates["asked_decision_keys"] == ["purchase_amount"]
    assert "problem_time" not in updates["asked_decision_keys"]
    assert updates["pending_ask_details"] == []


def test_conclude_control_bypasses_issue_extraction_with_stale_issue_counter():
    state = GuideState(
        wants_conclude=True,
        turn_control_intent="conclude_now",
        turn_contains_case_details=False,
        confirmed_issues=["消费者权益保护"],
        last_confirmed_count=0,
    )

    assert route_after_parse(state) == "assess_retrieve"


def test_evidence_requirement_version_only_changes_when_checklist_changes():
    basis = {
        "title": "消费者权益保护法",
        "article_no": "第二十四条",
        "source_type": "statute",
        "text": "经营者提供的商品不符合质量要求时，消费者可以依法主张相应责任。",
        "issuer": "全国人民代表大会常务委员会",
        "url": "https://example.test/law/24",
    }
    state = GuideState(
        legal_domain="consumer_market",
        round=2,
        case_facts=[_fact("transaction.item", "event", "用户购买了一台电脑")],
        followup_basis_refs=[basis],
    )
    report = evaluate_state_evidence(state)
    rows, version = merge_evidence_requirements(
        state, report, basis_refs=state.followup_basis_refs
    )
    unchanged = state.model_copy(update={
        "evidence_requirements": rows,
        "evidence_requirement_version": version,
        "round": 3,
    })
    rows2, version2 = merge_evidence_requirements(
        unchanged, report, basis_refs=unchanged.followup_basis_refs
    )

    assert version == 1
    assert version2 == 1
    assert all(item["trigger_fact_keys"] == ["transaction.item"] for item in rows2)
    assert all(item["basis_refs"][0]["text"] == basis["text"] for item in rows2)
    assert all(item["basis_refs"][0]["issuer"] == basis["issuer"] for item in rows2)
    assert all(item["basis_refs"][0]["url"] == basis["url"] for item in rows2)


def test_evidence_requirement_prioritizes_official_source_with_body_and_url():
    statute_refs = [
        {
            "title": f"测试法条{i}",
            "article_no": f"第{i}条",
            "source_type": "statute",
            "text": f"法条正文{i}",
        }
        for i in range(1, 4)
    ]
    official_ref = {
        "title": "最高人民法院关于民事诉讼证据的若干规定",
        "article_no": "",
        "source_type": "official_evidence_guidance",
        "text": "电子数据应结合生成、存储、传输环境等因素审查真实性。",
        "issuer": "最高人民法院",
        "url": "https://www.court.gov.cn/zixun/xiangqing/212721.html",
    }
    state = GuideState(
        legal_domain="consumer_market",
        case_facts=[_fact("transaction.item", "event", "用户购买了一台手机")],
    )

    rows, _ = merge_evidence_requirements(
        state,
        evaluate_state_evidence(state),
        basis_refs=[*statute_refs, official_ref],
    )

    assert rows
    for row in rows:
        basis = row["basis_refs"][0]
        assert basis["source_type"] == "official_evidence_guidance"
        assert basis["text"] == official_ref["text"]
        assert basis["issuer"] == official_ref["issuer"]
        assert basis["url"] == official_ref["url"]


def test_debug_backfills_basis_body_for_persisted_evidence_checklist():
    basis = {
        "title": "中华人民共和国刑法",
        "article_no": "第二百六十六条",
        "source_type": "statute",
        "text": "诈骗公私财物，依照数额和情节承担相应刑事责任。",
        "issuer": "全国人民代表大会",
    }
    state = GuideState(
        followup_basis_refs=[basis],
        evidence_requirements=[{
            "id": "proof_target:payment",
            "label": "付款记录",
            "active": True,
            "basis_refs": [{
                "title": basis["title"],
                "article_no": basis["article_no"],
                "source_type": "statute",
            }],
        }],
    )

    debug = guide_debug_payload(state)
    enriched = debug["evidence_checklist"][0]["basis_refs"][0]

    assert enriched["text"] == basis["text"]
    assert enriched["issuer"] == basis["issuer"]


def test_structured_form_assesses_only_text_after_answer_marker():
    question = {
        "field_id": "desired_outcome",
        "candidate_id": "other_claim_outcome",
        "question": "您现在最希望实现什么结果？",
        "input_type": "long_text",
        "decision_effects": ["claim_scope"],
    }
    answer = """【动态追问表单回答】
1. [desired_outcome] 您现在最希望实现什么结果？
回答：我希望找回我的钱"""
    state = GuideState(
        legal_domain="other",
        round=2,
        messages=[HumanMessage(content=answer)],
        pending_ask_details=[question["question"]],
        pending_ask_type="facts",
        pending_followup_ids=["other_claim_outcome"],
        followup_plan={"plan_kind": "followup_form", "questions": [question]},
        fact_records={
            "other_claim_outcome": {
                "value": "旧表单题目？",
                "status": "ambiguous",
            }
        },
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = _llm({
        "is_answer": True,
        "answers_asked_question": True,
        "answered_question_ids": ["desired_outcome"],
        "collected_facts": [],
        "case_updates": [],
        "evidence": [],
        "evidence_details": [],
        "evidence_unavailable": [],
        "adverse_facts": [],
    })

    updates = asyncio.run(node_parse_details(state, deps))

    record = updates["fact_records"]["other_claim_outcome"]
    assert record["value"] == "我希望找回我的钱"
    assert record["status"] == "user_stated"
    assert any(item["statement"] == "我希望找回我的钱" for item in updates["case_facts"])
    assert updates["pending_ask_details"] == []


def test_structured_form_keeps_each_answer_atomic_when_parser_json_is_broken():
    questions = [
        {
            "field_id": "channel_amount",
            "candidate_id": "cyber_channel_amount",
            "question": "平台、时间和金额是什么？",
            "decision_effects": ["limitation"],
        },
        {
            "field_id": "issue_type",
            "candidate_id": "cyber_issue_type",
            "question": "主要是哪类问题？",
            "decision_effects": ["responsibility"],
        },
        {
            "field_id": "prior_action",
            "candidate_id": "cyber_emergency_action",
            "question": "是否已经止付或报警？",
            "decision_effects": ["procedure"],
        },
    ]
    answer = """【动态追问表单回答】
1. [channel_amount] 平台、时间和金额是什么？
回答：闲鱼，2026年7月21日，3000元
2. [issue_type] 主要是哪类问题？
回答：被骗转账
3. [prior_action] 是否已经止付或报警？
回答：还没有报警"""
    state = GuideState(
        legal_domain="cyber_data_fraud",
        round=2,
        messages=[HumanMessage(content=answer)],
        pending_ask_details=[item["question"] for item in questions],
        pending_ask_type="facts",
        pending_followup_ids=[item["candidate_id"] for item in questions],
        followup_plan={"plan_kind": "followup_form", "questions": questions},
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"is_answer": true, "collected_facts": ["broken'))

    updates = asyncio.run(node_parse_details(state, deps))
    current = [item for item in updates["case_facts"] if item.get("turn") == 2]

    assert {item["statement"] for item in current} == {
        "闲鱼，2026年7月21日，3000元",
        "被骗转账",
        "还没有报警",
    }
    assert all("动态追问表单" not in item["statement"] for item in current)


def test_followup_reply_and_debug_include_retrieved_basis_body():
    basis = {
        "source_type": "statute",
        "title": "中华人民共和国刑法",
        "article_no": "第二百六十六条",
        "text": "诈骗公私财物，依照数额和情节承担相应刑事责任。",
    }
    state = GuideState(
        followup_basis_refs=[basis],
        followup_basis_error="",
    )
    reply = _format_followup_batch_reply(state, [{
        "field_id": "loss_amount",
        "question": "损失金额是多少。",
        "input_type": "short_text",
        "reason": "确定请求范围",
        "basis_refs": [basis],
    }])
    debug = guide_debug_payload(state)

    assert "损失金额是多少？" in reply
    assert "。？" not in reply
    assert "条文内容" in reply
    assert basis["text"] in reply
    assert debug["followup_basis_refs"][0]["text"] == basis["text"]


def test_batch_planner_rejects_field_combining_claim_and_prior_process():
    state = GuideState(
        legal_domain="other",
        decision_sufficiency={
            "dimensions": [
                {"effect": "claim_scope", "satisfied": False},
                {"effect": "procedure", "satisfied": False},
            ]
        },
    )
    plan = asyncio.run(plan_followup_batch(state, _llm({
        "should_ask": True,
        "fields": [{
            "field_id": "claim_and_process",
            "question": "您想要什么结果，之前又投诉过吗",
            "input_type": "long_text",
            "decision_effects": ["claim_scope", "procedure"],
        }],
    })))

    assert all(
        not {"claim_scope", "procedure"}.issubset(set(item["decision_effects"]))
        for item in plan.get("questions", [])
    )


def test_primary_detail_store_suppresses_stale_conflicted_auxiliary_record():
    state = GuideState(
        legal_domain="other",
        case_facts=[_fact("claim.outcome", "claim", "用户希望追回款项")],
        fact_records={
            "other_claim_outcome": {
                "value": "旧表单协议文本",
                "status": "conflicted",
                "verification": "not_independently_verified",
            }
        },
    )

    rendered = _fact_assessments_for_prompt(state)

    assert "旧表单协议文本" not in rendered
