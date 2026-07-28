"""法律指引多轮场景、状态黑板与记忆契约测试。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import src.agents.legal_guide.graph as guide_graph
import src.agents.workers.guide_agent as guide_worker
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.db_queries import load_user_context
from src.agents.legal_guide.issue_normalizer import extract_legal_issues
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.agents.legal_knowledge.statute_rag import _expand_pg_keywords, format_statute_context


def _deps_with_json(payload: dict) -> GuideDeps:
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return deps


def test_prepare_turn_merges_context_without_losing_identity_or_memory():
    state = GuideState(
        session_id="12:s1",
        user_context={
            "user_id": "12",
            "long_term_memories": ["用户此前在上海工作", "已保存劳动合同"],
        },
    )
    deps = MagicMock(spec=GuideDeps)
    deps.db_session = MagicMock()

    with patch.object(
        guide_graph,
        "load_user_context",
        new=AsyncMock(return_value={"prior_domains": ["labor_social_security"], "region": "上海"}),
    ):
        updates = asyncio.run(guide_graph.node_prepare_turn(state, deps))

    context = updates["user_context"]
    assert context["user_id"] == "12"
    assert context["long_term_memories"] == ["用户此前在上海工作", "已保存劳动合同"]
    assert context["prior_domains"] == ["labor_social_security"]
    assert updates["region"] == "上海"
    assert updates["round"] == 1


def test_non_numeric_user_id_does_not_open_postgres_query():
    db = MagicMock()
    db.execute = AsyncMock()

    result = asyncio.run(load_user_context("external-user-id", db))

    assert result == {}
    db.execute.assert_not_awaited()


def test_prepare_turn_detects_user_requested_and_hard_limit_convergence():
    deps = MagicMock(spec=GuideDeps)
    requested = GuideState(
        round=3,
        total_rounds=3,
        messages=[HumanMessage(content="不要再问了，请按现有信息给我方案")],
    )
    request_updates = asyncio.run(guide_graph.node_prepare_turn(requested, deps))
    assert request_updates["wants_conclude"] is True
    assert request_updates["force_conclude"] is False

    concise_request = GuideState(
        round=4,
        total_rounds=4,
        messages=[HumanMessage(content="没有其他材料，请给方案。")],
    )
    concise_updates = asyncio.run(guide_graph.node_prepare_turn(concise_request, deps))
    assert concise_updates["wants_conclude"] is True

    limited = GuideState(
        round=11,
        total_rounds=11,
        messages=[HumanMessage(content="什么是劳动仲裁？")],
    )
    limit_updates = asyncio.run(guide_graph.node_prepare_turn(limited, deps))
    assert limit_updates["total_rounds"] == 12
    assert limit_updates["force_conclude"] is True


def test_followup_reply_has_at_most_two_questions_without_llm_expansion():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        confidence_tier="LOW",
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock()

    updates = asyncio.run(guide_graph.node_ask_facts(state, deps))
    reply = updates["messages"][0].content

    assert reply.count("？") + reply.count("?") <= 2
    assert len(updates["pending_ask_details"]) <= 2
    deps.llm.ainvoke.assert_not_awaited()


def test_evidence_followup_skips_resolved_aliases_and_conditional_materials():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        collected_facts=["公司拖欠工资两个月"],
        evidence_confirmed=["书面劳动合同", "银行流水"],
        confidence_tier="MEDIUM",
    )

    remaining = guide_graph._remaining_evidence_questions(state)

    assert not any("劳动合同、录用通知" in item for item in remaining)
    assert not any("工资标准和支付材料" in item for item in remaining)
    assert not any("进入诉讼阶段时" in item for item in remaining)
    assert not any("工伤认定" in item for item in remaining)
    assert remaining == []


def test_dine_in_food_followup_does_not_ask_for_identity_or_logistics():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["食品不符合食品安全标准"],
        collected_facts=["在餐馆就餐时饭里发现玻璃渣"],
        confidence_tier="LOW",
    )

    remaining = guide_graph._remaining_evidence_questions(state)

    assert remaining == ["消费关系和付款材料", "商品或服务问题材料"]
    assert not any("身份证明" in item for item in remaining)
    assert not any("物流" in item or "签收" in item for item in remaining)


def test_fact_followup_does_not_repeat_known_time_contract_or_bank_record():
    state = GuideState(
        legal_domain="labor_social_security",
        time_info="已拖欠工资两个月",
        evidence_confirmed=["书面劳动合同", "银行工资流水"],
    )

    remaining = guide_graph._remaining_fact_questions(state)

    assert "劳动关系现在还在继续，还是已经离职？" in remaining
    assert all("劳动合同" not in question and "银行流水" not in question for question in remaining)


def test_statute_citation_whitelist_expands_supported_alias_and_drops_unretrieved_article():
    context = (
        "法条1【中华人民共和国劳动合同法 第三十条】\n按时足额支付劳动报酬\n\n"
        "法条2【中华人民共和国劳动合同法 第八十五条】\n责令限期支付"
    )
    reply = (
        "- 《劳动合同法》第30条支持追索工资。\n"
        "- 《劳动合同法》第82条规定未签合同可主张双倍工资。\n"
        "- 请先固定证据。"
    )

    sanitized = guide_graph._sanitize_statute_citations(reply, context)

    assert "《中华人民共和国劳动合同法》第30条" in sanitized
    assert "第82条" not in sanitized
    assert "未在本轮检索结果中出现" in sanitized
    assert "请先固定证据" in sanitized


def test_statute_citation_filter_renumbers_remaining_items():
    context = "法条1【中华人民共和国食品安全法 第一百四十八条】\n惩罚性赔偿"
    reply = (
        "**【法律依据】**\n"
        "1. 《中华人民共和国食品安全法》第三十四条：禁止混有异物。\n"
        "2. 《中华人民共和国食品安全法》第一百四十八条：惩罚性赔偿。"
    )

    sanitized = guide_graph._sanitize_statute_citations(reply, context)

    assert "第三十四条" not in sanitized
    assert "1. 《中华人民共和国食品安全法》第一百四十八条" in sanitized
    assert "2. 《中华人民共和国食品安全法》第一百四十八条" not in sanitized


def test_pg_keyword_expansion_and_article_number_normalization():
    assert _expand_pg_keywords(["拖欠劳动报酬"]) == ["拖欠劳动报酬", "拖欠", "劳动报酬"]
    context = format_statute_context(
        [{"law_id": "1", "article_no": "三十", "text": "支付劳动报酬"}],
        {"1": "中华人民共和国劳动合同法"},
    )
    assert "【中华人民共和国劳动合同法 第三十条】" in context
    assert guide_graph._source_statute_refs(context)["中华人民共和国劳动合同法"] == {(30, None)}


def test_statute_citation_whitelist_does_not_treat_document_title_as_law():
    context = "法条1【中华人民共和国劳动合同法 第八十五条】\n责令限期支付"
    reply = "监察部门可以下达《责令改正通知书》，再依据第85条处理。"

    sanitized = guide_graph._sanitize_statute_citations(reply, context)

    assert sanitized == reply


def test_evaluator_only_reads_titles_inside_law_section():
    from scripts.evaluate_guide_scenarios import _reply_law_names

    reply = (
        "**【法律依据】**\n"
        "《中华人民共和国劳动合同法》第三十条。\n"
        "**【行动清单】**\n"
        "证据清单参考《部分案件起诉状答辩状示范文本》。"
    )

    assert _reply_law_names(reply) == {"中华人民共和国劳动合同法"}


def test_accessible_mode_is_enabled_for_user_who_says_they_are_elderly_and_unclear():
    state = GuideState(messages=[HumanMessage(content="我年纪大了，说不清楚。")])

    guidance = guide_graph._audience_guidance(state)

    assert guide_graph._uses_accessible_language(state) is True
    assert "1800字" in guidance
    assert "行动清单只保留最重要的3步" in guidance


def test_forced_conclusion_uses_concise_answer_guidance():
    state = GuideState(force_conclude=True)

    guidance = guide_graph._audience_guidance(state)

    assert "禁止继续追问" in guidance
    assert "2200字" in guidance


def test_user_facing_tone_replaces_harsh_evidence_wording():
    reply = "没有合同是非常致命的，微信证明力严重不足，还要看证据够不够硬。"

    sanitized = guide_graph._sanitize_user_facing_tone(reply)

    assert "致命" not in sanitized
    assert "严重不足" not in sanitized
    assert "够不够硬" not in sanitized
    assert "增加举证难度" in sanitized


def test_labor_sanitizer_keeps_additional_compensation_conditional():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "| 申请劳动仲裁 | 可以主张工资、经济补偿金甚至加付赔偿金（50%-100%）。 |\n"
        "《劳动合同法》第八十二条规定二倍工资。"
        "（注：这项权益您可以去主张，但有1年的仲裁时效，从您知道权利被侵害之日起算。）"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "不能作为劳动仲裁当然支持的请求" in sanitized
    assert "从您知道权利被侵害之日起算" not in sanitized
    assert "需结合用工状态" in sanitized


def test_labor_sanitizer_handles_indirect_and_application_request_variants():
    state = GuideState(legal_domain="labor_social_security")
    reply = (
        "**【法律依据】**\n"
        "《劳动合同法》第八十五条：逾期不支付的，加付百分之五十以上百分之一百以下的赔偿金。\n"
        "**【维权路径比较】**\n"
        "申请劳动仲裁，除了要回工资，还有可能要到刚才说的那笔额外赔偿金。\n"
        "**【行动清单】**\n"
        "请求公司加付50%-100%的赔偿金。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "第八十五条" in sanitized
    assert "其他请求要结合事实和受理规则核验" in sanitized
    assert "先向劳动监察部门核实是否已具备法定前提" in sanitized


def test_labor_sanitizer_corrects_wage_arbitration_limitation_start():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = "劳动仲裁的时效是**一年**，从您知道或应当知道权利被侵害之日起计算。"

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "从您知道或应当知道权利被侵害之日起" not in sanitized
    assert "劳动关系终止的应自终止之日起一年内提出" in sanitized


def test_compact_accessible_reply_removes_optional_sections_not_required_sections():
    reply = (
        "这是重复开场。" * 100
        + "\n**【理解您的情况】**\n工资被拖欠。"
        + "\n**【法律依据】**\n法条。"
        + "\n**【维权路径比较】**\n投诉或仲裁。"
        + "\n**【维权胜算评估】**\n较低。"
        + "\n**【行动清单】**\n1. 保存材料。2. 打12333。3. 请家人协助。"
        + "\n**【常见误区】**\n" + "重复说明。" * 200
    )

    compacted = guide_graph._compact_final_reply(reply, accessible=True)

    assert "重复开场" not in compacted
    assert "常见误区" not in compacted
    for section in ("理解您的情况", "法律依据", "维权路径比较", "维权胜算评估", "行动清单"):
        assert section in compacted


def test_output_section_normalizer_uses_stable_path_heading():
    reply = "**【初步方向建议】**\n先投诉，再仲裁。"

    normalized = guide_graph._normalize_required_sections(reply)

    assert "【维权路径比较】" in normalized
    assert "【初步方向建议】" not in normalized


def test_statute_citation_whitelist_keeps_numbered_list_reference():
    context = "法条1【中华人民共和国劳动合同法 第三十条】\n支付劳动报酬"
    reply = "请先确认上面第3条所列的证据情况。"

    sanitized = guide_graph._sanitize_statute_citations(reply, context)

    assert sanitized == reply


def test_forced_conclusion_removes_request_for_another_information_round():
    reply = (
        "**【行动清单】**\n先保存证据。\n\n"
        "**强烈建议**\n请补充以下关键信息，我将为您生成更精准的方案：\n"
        "- 有合同吗？\n- 在哪里工作？\n\n---\n📄 需要参考文书？"
    )

    sanitized = guide_graph._sanitize_forced_followups(reply)

    assert "请补充以下" not in sanitized
    assert "我将为您生成" not in sanitized
    assert "您无需继续回答" in sanitized
    assert "需要参考文书" in sanitized


def test_requested_conclusion_removes_single_numbered_followup_without_losing_action_list():
    reply = (
        "**【行动清单】**\n"
        "1. 请务必先回答上面缺失的信息。\n"
        "2. 保存聊天记录。\n"
        "3. 拨打12348。"
    )

    sanitized = guide_graph._sanitize_forced_followups(reply)

    assert "请务必先回答" not in sanitized
    assert "保存聊天记录" in sanitized
    assert "拨打12348" in sanitized


def test_structured_case_is_added_when_model_omits_case_section():
    reply = "**【法律依据】**\n法条内容\n\n**【行动清单】**\n先保存证据。"
    cases = [{"title": "刘某追索劳动报酬案", "gist": "法院支持支付拖欠工资。", "text": ""}]

    completed = guide_graph._ensure_case_reference(reply, cases)

    assert "【类似案例参考】" in completed
    assert "刘某追索劳动报酬案" in completed
    assert "法院支持支付拖欠工资" in completed


def test_case_context_is_used_when_structured_case_list_is_missing():
    reply = "**【法律依据】**\n法条内容\n\n**【行动清单】**\n先保存证据。"
    context = (
        "案例1【何某产品销售者责任纠纷案】\n"
        "基本信息：（2021）鲁01民终8915号｜产品销售者责任纠纷｜某法院\n"
        "案情摘要：消费者主张食品不符合安全标准，法院审查了购买关系与证据。\n"
        "裁判结果：驳回部分请求。"
    )

    completed = guide_graph._ensure_case_reference(reply, [], context)

    assert "【类似案例参考】" in completed
    assert "（2021）鲁01民终8915号" in completed
    assert "不能替代" in completed


def test_generated_case_section_is_removed_when_retrieval_has_no_case():
    reply = (
        "**【法律依据】**\n法条。\n"
        "**【类似案例参考】**\n上海此类案件胜诉率很高。\n"
        "**【维权路径比较】**\n可以投诉。"
    )

    sanitized = guide_graph._ensure_case_reference(reply, [], "")

    assert "类似案例" not in sanitized
    assert "胜诉率很高" not in sanitized
    assert "维权路径比较" in sanitized


def test_force_conclude_breaks_out_of_repeated_counter_question_loop():
    state = GuideState(
        total_rounds=12,
        force_conclude=True,
        messages=[HumanMessage(content="这个流程为什么这么做？")],
        pending_ask_details=["是否签有书面劳动合同？"],
        pending_ask_type="facts",
    )
    deps = _deps_with_json({
        "is_answer": False,
        "user_question": "这个流程为什么这么做？",
        "new_issues": [],
        "collected_facts": [],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(guide_graph.node_parse_details(state, deps))

    assert updates["pending_ask_details"] == []
    assert updates["phase"] == GuidePhase.ISSUE_SEARCH
    assert updates["deferred_questions"] == ["这个流程为什么这么做？"]


def test_three_consecutive_counter_questions_trigger_early_convergence():
    state = GuideState(
        total_rounds=4,
        consecutive_counter_questions=2,
        messages=[HumanMessage(content="劳动仲裁收费吗？")],
        pending_ask_details=["是否签有书面劳动合同？"],
        pending_ask_type="facts",
    )
    deps = _deps_with_json({
        "is_answer": False,
        "user_question": "劳动仲裁收费吗？",
        "new_issues": [],
        "collected_facts": [],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(guide_graph.node_parse_details(state, deps))

    assert updates["force_conclude"] is True
    assert updates["pending_ask_details"] == []
    assert updates["consecutive_counter_questions"] == 3


def test_issue_extraction_reuses_same_llm_call_for_initial_fact_blackboard():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "issues": ["拖欠劳动报酬"],
        "domain": "labor_social_security",
        "facts": ["拖欠3个月工资", "拖欠金额24000元"],
        "region": "上海",
        "time_info": "2025年4月开始",
    }, ensure_ascii=False)))

    result = asyncio.run(extract_legal_issues("完整案情", llm))
    assert result.facts == ["拖欠3个月工资", "拖欠金额24000元"]
    assert result.region == "上海"
    assert result.time_info == "2025年4月开始"


def test_issue_extraction_uses_high_precision_wage_fallback_on_invalid_json():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="not-json"))

    result = asyncio.run(extract_legal_issues("公司欠我工资。", llm))

    assert result.issues == ["拖欠劳动报酬"]
    assert result.domain == "labor_social_security"
    assert result.facts == ["公司欠我工资。"]


def test_parse_details_accumulates_fact_blackboard_across_turns():
    first = GuideState(
        messages=[HumanMessage(content="我2022年3月入职，月薪8000元")],
        pending_ask_details=["何时入职？", "工资是多少？"],
        pending_ask_type="facts",
    )
    first_deps = _deps_with_json({
        "is_answer": True,
        "user_question": "",
        "new_issues": [],
        "collected_facts": ["2022年3月入职", "月薪8000元"],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "2022年3月",
        "adverse_facts": [],
    })
    first_updates = asyncio.run(guide_graph.node_parse_details(first, first_deps))
    assert first_updates["collected_facts"] == ["2022年3月入职", "月薪8000元"]

    second = first.model_copy(update={
        **first_updates,
        "messages": [HumanMessage(content="公司拖欠了3个月，一共24000元")],
        "pending_ask_details": ["拖欠多久、金额多少？"],
        "pending_ask_type": "facts",
    })
    second_deps = _deps_with_json({
        "is_answer": True,
        "user_question": "",
        "new_issues": [],
        "collected_facts": ["拖欠3个月工资", "拖欠金额24000元"],
        "evidence": [],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })
    second_updates = asyncio.run(guide_graph.node_parse_details(second, second_deps))

    assert second_updates["collected_facts"] == [
        "2022年3月入职",
        "月薪8000元",
        "拖欠3个月工资",
        "拖欠金额24000元",
    ]


def test_parse_details_does_not_count_evidence_mentioned_only_inside_image():
    state = GuideState(
        confirmed_issues=["食品安全"],
        messages=[HumanMessage(content=(
            "【图片证据补充（视觉模型识别，需与原图核对）】\n"
            "【证据类型】聊天记录截图（语音转文字记录）\n"
            "语音转写称消费者另有实物和现场照片"
        ))],
        pending_ask_details=["图片能证明什么？"],
        pending_ask_type="evidence",
    )
    deps = _deps_with_json({
        "is_answer": True,
        "user_question": "",
        "new_issues": ["可能涉及系统性食品安全隐患"],
        "collected_facts": ["截图中消费者表示另有玻璃渣实物和现场照片"],
        "evidence": [
            "聊天记录截图（语音转文字记录）",
            "玻璃渣实物（消费者声称持有）",
            "现场照片（消费者声称持有）",
        ],
        "evidence_unavailable": [],
        "region": "",
        "time_info": "",
        "adverse_facts": [],
    })

    updates = asyncio.run(guide_graph.node_parse_details(state, deps))

    assert updates["evidence_confirmed"] == ["已上传图片：聊天记录截图（语音转文字记录）"]
    assert updates["evidence_unverified"] == [
        "玻璃渣实物（消费者声称持有）",
        "现场照片（消费者声称持有）",
    ]
    assert updates["confirmed_issues"] == ["食品安全"]
    assert updates["collected_facts"][0].startswith("待核验线索")
    assert "玻璃渣实物" in updates["collected_facts"][0]


def test_evidence_overconfidence_is_deterministically_softened():
    reply = (
        "现场照片是直接铁证，已经足够证明异物来自商家，胜诉希望很大。"
        "获得一千元赔偿的可能性非常大。"
    )

    sanitized = guide_graph._sanitize_evidence_overconfidence(reply)

    assert "铁证" not in sanitized
    assert "已经足够证明" not in sanitized
    assert "胜诉希望很大" not in sanitized
    assert "可能性非常大" not in sanitized
    assert "仍需核验" in sanitized


def test_unverified_image_leads_cannot_be_presented_as_evidence_in_hand():
    reply = (
        "玻璃渣实物是强有力的客观证据。"
        "您手中持有现场照片，可以直接起诉。"
    )

    sanitized = guide_graph._sanitize_unverified_evidence_assertions(
        reply,
        ["玻璃渣实物", "现场照片"],
    )

    assert "强有力的客观证据" not in sanitized
    assert "您手中持有现场照片" not in sanitized
    assert sanitized.count("本次未直接展示") == 2


def test_food_minimum_additional_compensation_remains_conditional():
    state = GuideState(
        confirmed_issues=["食品安全问题"],
        collected_facts=["饭菜中疑似有玻璃渣"],
    )
    reply = "18元的十倍不足一千元，所以您最低可以索赔 **1000元**。"

    sanitized = guide_graph._sanitize_food_compensation_certainty(reply, state)

    assert "最低可以索赔" not in sanitized
    assert "若经核验符合" in sanitized
    assert "结合证据判断" in sanitized


def test_retrieval_query_uses_accumulated_facts_and_long_term_memory():
    search_statutes = AsyncMock(return_value=[])
    search_cases = AsyncMock(return_value={"context": "", "cases": [], "fallback_guide": None})
    graph_query = AsyncMock(return_value={"laws": [], "channels": []})
    state = GuideState(
        legal_domain="other",
        confirmed_issues=["拖欠工资"],
        collected_facts=["月薪8000元", "拖欠3个月工资"],
        time_info="2025年4月",
        region="上海",
        user_context={"long_term_memories": ["用户此前保存了劳动合同"]},
    )
    deps = MagicMock(spec=GuideDeps)
    deps.embedding_model = MagicMock()
    deps.milvus_client = MagicMock()
    deps.llm = MagicMock()
    deps.db_session = None
    deps.neo4j_driver = MagicMock()

    with patch(
        "src.agents.legal_knowledge.statute_rag.search_statutes_raw",
        new=search_statutes,
    ), patch(
        "src.agents.legal_knowledge.statute_rag.format_statute_context",
        return_value="",
    ), patch(
        "src.agents.legal_knowledge.case_rag.search_cases_context",
        new=search_cases,
    ), patch.object(guide_graph, "query_laws_and_channels", new=graph_query):
        asyncio.run(guide_graph.node_retrieve(state, deps))

    query = search_statutes.await_args.kwargs["question"]
    assert "月薪8000元" in query
    assert "拖欠3个月工资" in query
    assert "2025年4月" in query
    assert "上海" in query
    assert "此前保存了劳动合同" in query


def test_conclusion_prompt_uses_retrieval_and_complete_state_blackboard():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="最终方案含《测试法》第一条"))
    deps = MagicMock(spec=GuideDeps)
    deps.llm = llm
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠工资"],
        collected_facts=["月薪8000元", "拖欠3个月工资"],
        evidence_confirmed=["劳动合同"],
        evidence_unavailable=["工资条"],
        region="上海",
        time_info="2025年4月",
        law_context_str="GROUNDING_LAW\n法条1【测试法 第一条】\n测试法条原文",
        case_context_str="GROUNDING_CASE：相似案例裁判摘要",
        relevant_channels=[{"name": "劳动仲裁委员会", "phone": "12333"}],
        user_context={"long_term_memories": ["GROUNDING_MEMORY：此前已向公司催告"]},
        confidence_tier="HIGH",
    )

    updates = asyncio.run(guide_graph.node_conclude(state, deps))
    prompt = llm.ainvoke.await_args.args[0][0].content

    assert "GROUNDING_LAW" in prompt
    assert "GROUNDING_CASE" in prompt
    assert "劳动仲裁委员会" in prompt
    assert "月薪8000元" in prompt
    assert "拖欠3个月工资" in prompt
    assert "2025年4月" in prompt
    assert "GROUNDING_MEMORY" in prompt
    assert "禁止使用“稳赢”" in prompt
    assert "最终方案含《测试法》第一条" in updates["messages"][0].content


class _FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True


class _SessionContext:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_worker_persists_complete_short_term_blackboard():
    redis = _FakeRedis()
    state = GuideState(
        session_id="12:s1",
        phase=GuidePhase.DETAIL_GATHER,
        round=3,
        total_rounds=3,
        confirmed_issues=["拖欠工资"],
        collected_facts=["月薪8000元"],
        evidence_confirmed=["劳动合同"],
        pending_ask_details=["拖欠多久？"],
        pending_ask_type="facts",
    )
    with patch.object(
        guide_worker,
        "AsyncSessionLocal",
        new=MagicMock(return_value=_SessionContext()),
    ), patch.object(
        guide_worker,
        "build_guide_deps",
        return_value=MagicMock(),
    ), patch.object(
        guide_worker,
        "run_guide",
        new=AsyncMock(return_value=("请继续补充", state)),
    ), patch.object(
        guide_worker,
        "get_checkpointer_redis",
        return_value=redis,
    ):
        reply = asyncio.run(guide_worker.call_guide_agent_impl("欠薪", "12", "s1"))

    restored = GuideState.model_validate_json(redis.data["guide_state:12:s1"])
    assert reply == "请继续补充"
    assert restored.collected_facts == ["月薪8000元"]
    assert restored.evidence_confirmed == ["劳动合同"]
    assert restored.pending_ask_type == "facts"
    assert redis.data["guide_active:12:s1"] == "1"


def test_worker_tool_retrieves_long_term_memory_deterministically():
    from src.agents.tools.worker_tools import _search_relevant_memories

    store = MagicMock()
    store.asearch = AsyncMock(return_value=[
        SimpleNamespace(value={"content": "用户此前保存了劳动合同"}),
        SimpleNamespace(value={"text": "用户所在地区：上海"}),
    ])
    runtime = SimpleNamespace(
        store=store,
        context=SimpleNamespace(user_id="12", session_id="s1"),
    )
    memories = asyncio.run(_search_relevant_memories("公司拖欠工资", runtime))

    assert memories == ["用户此前保存了劳动合同", "用户所在地区：上海"]
    store.asearch.assert_awaited_once_with(
        ("users", "12", "memories"),
        query="公司拖欠工资",
        limit=5,
    )


def test_conclusion_writes_region_and_case_summary_to_long_term_memory():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="最终维权方案"))
    deps = MagicMock(spec=GuideDeps)
    deps.llm = llm
    store = MagicMock()
    store.aput = AsyncMock()
    state = GuideState(
        session_id="12:s1",
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠工资"],
        collected_facts=["拖欠3个月工资"],
        evidence_confirmed=["劳动合同"],
        region="上海",
        time_info="2025年4月",
        confidence_tier="MEDIUM",
        user_context={"user_id": "12"},
    )

    with patch("src.infra.milvus_store.get_milvus_store", return_value=store):
        asyncio.run(guide_graph.node_conclude(state, deps))

    assert store.aput.await_count == 2
    values = [call.kwargs["value"] for call in store.aput.await_args_list]
    assert any("用户所在地区：上海" in value["content"] for value in values)
    assert any("拖欠3个月工资" in value["content"] for value in values)
    assert any("劳动合同" in value["content"] for value in values)
