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


def test_state_preserves_non_pilot_region_for_documents_but_channels_can_fallback():
    assert guide_graph._state_region_name("杭州") == "杭州"
    assert guide_graph._extract_case_region("公司在杭州，我也在这里工作") == "杭州"
    assert guide_graph._state_region_name("在饭馆吃东西") == ""


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
        round=guide_graph.settings.GUIDE_MAX_TOTAL_ROUNDS - 1,
        total_rounds=guide_graph.settings.GUIDE_MAX_TOTAL_ROUNDS - 1,
        messages=[HumanMessage(content="什么是劳动仲裁？")],
    )
    limit_updates = asyncio.run(guide_graph.node_prepare_turn(limited, deps))
    assert limit_updates["total_rounds"] == guide_graph.settings.GUIDE_MAX_TOTAL_ROUNDS
    assert limit_updates["force_conclude"] is True


def test_followup_reply_has_at_most_two_questions_without_llm_expansion():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        confidence_tier="LOW",
        followup_plan={
            "should_ask": True,
            "ask_type": "facts",
            "decision_key": "employment_status",
            "candidate_id": "",
            "question": "您目前还在这家公司工作吗？",
            "reason": "判断当前可走的程序和时效",
            "answer_hint": "在职或已离职都可以直接说",
            "basis_kind": "official_elements",
            "official_source": {
                "authority_level": "official_basis_derived",
                "issuer": "全国人大常委会",
                "title": "劳动争议调解仲裁法",
                "url": "https://flk.npc.gov.cn/",
                "usage_note": "",
            },
            "information_gain": 0.8,
            "user_burden": 0.2,
        },
    )
    deps = MagicMock(spec=GuideDeps)

    updates = asyncio.run(guide_graph.node_ask_facts(state, deps))
    reply = updates["messages"][0].content

    assert reply.count("？") + reply.count("?") == 1
    assert updates["pending_ask_details"] == ["您目前还在这家公司工作吗？"]


def legacy_contract_evidence_followup_skips_resolved_aliases_and_conditional_materials():
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


def legacy_contract_dine_in_food_followup_does_not_ask_for_identity_or_logistics():
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


def legacy_contract_fact_followup_does_not_repeat_known_time_contract_or_bank_record():
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


def legacy_patch_consumer_sanitizer_corrects_small_claim_and_burden_overclaims():
    state = GuideState(legal_domain="consumer_market")
    reply = (
        "300元的案子，法院可能认为金额太小不立案。"
        "商家需要证明它没有跑路或提供了对等价值服务。"
        "集体诉讼可以分摊诉讼成本，通常7个工作日内有答复。\n"
        "**耗时**：一般15-45天有初步结果。\n"
        "路径二：联合报案，如果涉嫌诈骗就去派出所。"
    )

    sanitized = guide_graph._sanitize_consumer_procedure_claims(reply, state)

    assert "金额太小不立案" not in sanitized
    assert "小额本身不是法院不予立案的理由" in sanitized
    assert "商家需要证明它没有跑路" not in sanitized
    assert "您需先证明消费关系" in sanitized
    assert "需先核对是否符合法定共同诉讼条件" in sanitized
    assert "以12315平台及承办部门反馈为准" in sanitized
    assert "15-45天" not in sanitized
    assert "联合报案" not in sanitized
    assert "诈骗" not in sanitized
    assert "派出所" not in sanitized


def legacy_patch_consumer_sanitizer_removes_unsupported_deadlines_costs_and_certainty():
    state = GuideState(legal_domain="consumer_market")
    reply = (
        "通常7-15个工作日内给出是否受理的答复。"
        "流程较长（通常1-3个月）。"
        "[需预交50元左右的诉讼费，胜诉后由败诉方承担]。\n"
        "第一步（今日必做）：投诉。\n"
        "花50元诉讼费去当地法院起诉（小额诉讼），胜诉概率很高。\n"
        "建议放弃诉讼，转为向12315举报其欺诈行为，以警示他人。"
    )

    sanitized = guide_graph._sanitize_consumer_procedure_claims(reply, state)

    for unsafe in (
        "7-15个工作日", "1-3个月", "预交50元", "今日必做",
        "花50元", "胜诉概率很高", "建议放弃诉讼", "欺诈行为",
    ):
        assert unsafe not in sanitized
    assert "是否适用小额诉讼程序由法院依法确定" in sanitized
    assert "最终负担以法院通知和裁判为准" in sanitized


def legacy_patch_prepaid_plan_understanding_uses_accumulated_context():
    state = GuideState(
        legal_domain="consumer_market",
        messages=[
            HumanMessage(content="在理发店充值后一周店就关门了，卡里还有300元"),
            HumanMessage(content="我要求退款，对方把我拉黑了"),
            HumanMessage(content="听说还有很多人也有损失"),
        ],
        collected_facts=["充值金额700元", "卡内余额300元"],
    )
    reply = (
        "**【理解您的情况】**\n预付卡的钱没了，确实让人生气。\n\n"
        "**【法律依据】**\n已检索。"
    )

    contextual = guide_graph._ensure_contextual_understanding(reply, state)

    assert "最初充值700元" in contextual
    assert "尚余300元" in contextual
    assert "要求退款后被拉黑" in contextual
    assert "待核实线索" in contextual
    assert "确实让人生气" not in contextual


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
    from test.evaluate_guide_scenarios import _reply_law_names

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


def legacy_patch_user_facing_tone_replaces_harsh_evidence_wording():
    reply = "没有合同是非常致命的，微信证明力严重不足，还要看证据够不够硬。"

    sanitized = guide_graph._sanitize_user_facing_tone(reply)

    assert "致命" not in sanitized
    assert "严重不足" not in sanitized
    assert "够不够硬" not in sanitized
    assert "增加举证难度" in sanitized


def legacy_patch_labor_sanitizer_keeps_additional_compensation_conditional():
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


def legacy_patch_labor_sanitizer_handles_indirect_and_application_request_variants():
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


def legacy_patch_labor_sanitizer_corrects_wage_arbitration_limitation_start():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = "劳动仲裁的时效是**一年**，从您知道或应当知道权利被侵害之日起计算。"

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "从您知道或应当知道权利被侵害之日起" not in sanitized
    assert "劳动关系终止的应自终止之日起一年内提出" in sanitized


def legacy_patch_labor_sanitizer_corrects_false_arbitration_fee_claim():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "**方案二：申请劳动仲裁（更全面）**\n"
        "- **[收费]：**劳动争议仲裁通常预收受理费，最终由败诉方承担。"
        "具体费用以受理通知为准。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "[免费]" in sanitized
    assert "预收受理费" not in sanitized
    assert "败诉方承担" not in sanitized
    assert "劳动争议仲裁不收费" in sanitized


def legacy_patch_labor_sanitizer_removes_demo_risk_wording_and_stale_compensation_math():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "注意：这条是说，您可以要求公司除了补发工资，再额外赔偿您拖欠金额的50%-100%。"
        "但这通常需要先经过劳动监察部门或仲裁机构的程序才能主张。\n"
        "**方案二：申请劳动仲裁（“撕破脸”但最彻底）**\n"
        "一旦申请仲裁，通常意味着和公司关系破裂，不适合想继续工作的您。\n"
        "如果公司因此有违法行为（比如未缴社保），还可以主张经济补偿。\n"
        "您2022年3月入职，至今2年多，大约可拿到2-3个月的工资作为补偿。\n"
        "这是最划算的选择，可从企查查、天眼查上查到公司地址。\n"
        "12345（可以投诉政府部门办事不力）"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "提出劳动仲裁就当然支持" in sanitized
    assert "撕破脸" not in sanitized
    assert "关系破裂" not in sanitized
    assert "未缴社保就当然获得经济补偿" in sanitized
    assert "至今2年多" not in sanitized
    assert "不能仅按入职年份估算" in sanitized
    assert "最划算" not in sanitized
    assert "国家企业信用信息公示系统" in sanitized
    assert "政务服务咨询和事项转办" in sanitized


def legacy_patch_labor_sanitizer_handles_long_reply_variants_from_real_conversation():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "劳动监察只负责追讨工资，不能处理经济补偿金，通常15-30个工作日。\n"
        "申请仲裁可以一次性解决所有问题。如果不想干了，可以主动提出离职并要求经济补偿。\n"
        "按工作年限算，2022年3月入职，工作2年多，能拿到约2-2.5个月工资的经济补偿。\n"
        "向杭州市劳动人事争议仲裁委员会申请仲裁。\n"
        "一旦申请仲裁，通常意味着和公司关系破裂。\n"
        "你的证据很好，追回工资的可能性很大。你现在手上的证据已经能说明这一点。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "只负责追讨工资" not in sanitized
    assert "15-30个工作日" not in sanitized
    assert "一次性解决所有问题" not in sanitized
    assert "主动提出离职并要求经济补偿" not in sanitized
    assert "工作2年多" not in sanitized
    assert "有管辖权的劳动人事争议仲裁委员会" in sanitized
    assert "关系破裂" not in sanitized
    assert "可能性很大" not in sanitized
    assert "你的" not in sanitized


def legacy_patch_labor_sanitizer_handles_markdown_split_and_incorrect_in_service_claims():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "**缺点**：劳动监察**只负责追讨工资，不能处理经济补偿金**，一般15-60天会有结果。\n"
        "**优点**：可以**一次性解决所有问题**，同时，可以申请加付50%-100%的赔偿金。\n"
        "如果您不离职，仲裁这条路就走不通。一旦申请仲裁，基本意味着和公司撕破脸。\n"
        "材料已经构成了完整的证据链，胜算的基础非常扎实。\n"
        "前往公司注册地所在的区劳动人事争议仲裁委员会申请立案。\n"
        "直接告诉对方：“不解决就走仲裁，到时您还得赔我经济补偿金。”"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    assert "**" not in sanitized
    assert "只负责追讨工资" not in sanitized
    assert "15-60天" not in sanitized
    assert "一次性解决所有问题" not in sanitized
    assert "申请加付50%-100%" not in sanitized
    assert "走不通" not in sanitized
    assert "撕破脸" not in sanitized
    assert "完整的证据链" not in sanitized
    assert "公司注册地所在的区" not in sanitized
    assert "还得赔我经济补偿金" not in sanitized


def legacy_patch_labor_sanitizer_handles_real_long_wage_plan_variants():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "预计时长：通常1-2个月能有初步结果（责令支付的通知）。\n"
        "不直接跟老板“撕破脸”打官司。由政府部门出面施压，很多公司会优先解决。\n"
        "公司账上没钱或老板跑路，监察部门也只能发“空头支票”责令支付。\n"
        "预计时长：从申请到裁决通常需要2-4个月，甚至更长。\n"
        "有国家强制力，可以一并主张解除劳动合同的经济补偿金等其他诉求。\n"
        "一旦正式提起仲裁，基本意味着跟公司关系彻底破裂，通常不再适合继续工作。\n"
        "光有聊天记录就是有力证据。用银行流水证明没有进账就行。"
        "举证责任主要在您这边，但您现有的证据已经足够了。\n"
        "追回工资是大概率事件。\n"
        "直接去公司注册地所属的杭州市区级劳动人事争议仲裁委员会递交申请。\n"
        "如果因此被迫离职，可以要求公司支付经济补偿金（每工作满一年赔一个月工资）。\n"
        "如果公司提出异议，支付令就会失效，程序转到普通诉讼。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(reply, state)

    for unsafe in (
        "1-2个月", "2-4个月", "撕破脸", "政府部门出面施压", "老板跑路",
        "空头支票", "有国家强制力", "可以一并主张解除劳动合同",
        "关系彻底破裂", "聊天记录就是有力证据", "没有进账就行",
        "证据已经足够", "是大概率事件", "公司注册地所属",
        "被迫离职，可以要求", "程序转到普通诉讼",
    ):
        assert unsafe not in sanitized
    assert "45日" in sanitized
    assert "申请仲裁不以离职为前提" in sanitized
    assert "劳动合同履行地或用人单位所在地" in sanitized
    assert "第三十八条" in sanitized
    assert "最终仍需核验原件和完整内容" in sanitized


def legacy_patch_labor_sanitizer_handles_second_real_long_plan_variants():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "预计时长：15-45个工作日。公司铁了心不给，适用于公司有支付能力但“赖账”的情况。\n"
        "证据非常有利，这套证据组合非常扎实，成功率很高。您的情况并不复杂，放平心态。\n"
        "可在“天眼查”或“国家企业信用信息公示系统”查询。\n"
        "如果因为公司拖欠工资导致您被迫离职，还可以要求经济补偿金。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(
        guide_graph._sanitize_evidence_overconfidence(reply),
        state,
    )

    for unsafe in (
        "15-45个工作日", "铁了心", "赖账", "非常扎实", "成功率很高",
        "并不复杂", "放平心态", "天眼查", "被迫离职，还可以要求",
    ):
        assert unsafe not in sanitized
    assert "第三十八条" in sanitized
    assert "国家企业信用信息公示系统" in sanitized


def legacy_patch_labor_sanitizer_handles_final_real_plan_procedure_variants():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
    )
    reply = (
        "时长：通常较快，一般在15-30个工作日内有初步结果。\n"
        "从申请到开庭再到出结果，通常需要2-3个月。\n"
        "能一揽子解决拖欠工资、加班费、经济补偿等所有劳动争议。"
        "只能处理“拖欠工资”本身，不能处理其他争议（如未缴社保）。\n"
        "且通常要先经过劳动监察调解不成后才建议走这条路。\n"
        "您的情况完全符合《中华人民共和国劳动合同法》第三十条和第八十五条的规定，拖欠的还要加付赔偿金。\n"
        "这份证据对您非常有利，直接证明了公司承认欠薪。您已经完成了最基本的举证责任。\n"
        "按兵不动，立刻行动：在公司承诺的“下个月”期限到来前一周，如果还没动静，就拨打12333。\n"
        "去公司注册地的杭州市劳动监察大队。如果公司准时发工资：万事大吉。\n"
        "支付令通常需要时间较长，且程序复杂，不推荐作为首选。\n"
        "法院会支持劳动者拿回自己的血汗钱。"
    )

    sanitized = guide_graph._sanitize_labor_procedure_claims(
        guide_graph._sanitize_evidence_overconfidence(reply),
        state,
    )

    for unsafe in (
        "15-30个工作日", "2-3个月", "一揽子解决", "只能处理",
        "先经过劳动监察", "完全符合", "拖欠的还要加付", "非常有利",
        "直接证明", "完成了最基本的举证责任", "按兵不动", "期限到来前一周",
        "公司注册地的杭州市", "万事大吉", "不推荐作为首选", "血汗钱",
    ):
        assert unsafe not in sanitized
    assert "不以先经劳动监察调解为前提" in sanitized
    assert "您现在即可拨打12333" in sanitized
    assert "第八十五条的加付赔偿还需满足" in sanitized


def test_required_sections_fill_empty_recommendation_block():
    state = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="MEDIUM",
        relevant_channels=[{"name": "本地劳动争议受理机构", "phone": "12333"}],
    )
    reply = (
        "【法律依据】\n《劳动合同法》第三十条。\n"
        "【维权路径比较】\n- 劳动监察\n- 劳动仲裁\n"
        "【推荐方案】\n"
        "【维权胜算评估】\n- 综合判断：中等\n"
        "【行动清单】\n1. 保存证据"
    )

    normalized = guide_graph._ensure_required_plan_sections(reply, state)

    assert "本地劳动争议受理机构" in normalized
    assert "具体受理范围和材料以该机构答复为准" in normalized


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


def test_compact_reply_enforces_budget_and_preserves_document_offer():
    reply = (
        "**【理解您的情况】**\n" + "案情。" * 100
        + "\n**【法律依据】**\n" + "法条原文。" * 300
        + "\n**【类似案例参考】**\n" + "类案内容。" * 300
        + "\n**【维权路径比较】**\n" + "路径说明。" * 300
        + "\n**【维权情况分析】**\n" + "重复分析。" * 300
        + "\n**【行动清单】**\n" + "行动步骤。" * 300
        + "\n**【维权胜算评估】**\n一般。"
        + "\n\n---\n📄 **需要参考文书？** 请回复「生成文书」。"
    )

    compacted = guide_graph._compact_final_reply(reply, accessible=False)
    concluded = guide_graph._compact_final_reply(reply, accessible=False, compact=True)

    assert len(compacted) <= 3000
    assert len(concluded) <= 2600
    assert "维权情况分析" not in compacted
    assert "需要参考文书" in compacted
    for section in ("理解您的情况", "法律依据", "维权路径比较", "维权胜算评估", "行动清单"):
        assert section in compacted


def test_forced_conclusion_restores_action_checklist_if_cleanup_removed_it():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        force_conclude=True,
        relevant_channels=[{"name": "劳动保障服务渠道", "phone": "12333"}],
    )
    reply = (
        "**【法律依据】**\n法条。\n"
        "**【维权路径比较】**\n投诉或仲裁。\n"
        "**【维权胜算评估】**\n较低。"
    )

    restored = guide_graph._ensure_action_checklist(reply, state)

    assert "【行动清单】" in restored
    assert "12333" in restored
    assert "12348" in restored


def test_output_section_normalizer_uses_stable_path_heading():
    reply = "**【初步方向建议】**\n先投诉，再仲裁。"

    normalized = guide_graph._normalize_required_sections(reply)

    assert "【维权路径比较】" in normalized
    assert "【初步方向建议】" not in normalized


def test_output_section_normalizer_accepts_plain_retrieved_law_heading():
    reply = "### 检索到的相关法律依据\n《中华人民共和国劳动合同法》第三十条。"

    normalized = guide_graph._normalize_required_sections(reply)

    assert "【法律依据】" in normalized


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


def test_conclusion_sanitizer_removes_batch_questionnaire_without_force_flag():
    reply = (
        "**【理解您的情况】**\n已记录。\n"
        "**【关键缺失信息清单】**\n"
        "1. 什么时候发生？\n2. 在哪里发生？\n3. 是否报警？\n"
        "**【行动清单】**\n1. 保存现有材料。"
    )

    sanitized = guide_graph._sanitize_forced_followups(reply)

    assert "关键缺失信息清单" not in sanitized
    assert "什么时候发生" not in sanitized
    assert "行动清单" in sanitized


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


def test_forced_cleanup_does_not_delete_required_sections_after_supplement_sentence():
    reply = (
        "**【维权路径比较】**\n先投诉。\n"
        "**【维权胜算评估】**\n需补充证据后才能准确评估。\n"
        "**【行动清单】**\n1. 保存材料。\n2. 拨打12348。"
    )

    sanitized = guide_graph._sanitize_forced_followups(reply)

    assert "【维权路径比较】" in sanitized
    assert "【维权胜算评估】" in sanitized
    assert "【行动清单】" in sanitized


def test_required_plan_sections_are_restored_before_document_offer():
    state = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="MEDIUM",
    )
    reply = "**【法律依据】**\n法条。\n\n---\n📄 **需要参考文书？**"

    restored = guide_graph._ensure_required_plan_sections(reply, state)

    assert "【维权路径比较】" in restored
    assert "【维权胜算评估】" in restored
    assert "【行动清单】" in restored
    assert restored.index("【行动清单】") < restored.index("需要参考文书")


def test_structured_case_is_added_when_model_omits_case_section():
    reply = "**【法律依据】**\n法条内容\n\n**【行动清单】**\n先保存证据。"
    cases = [{"title": "刘某追索劳动报酬案", "gist": "法院支持支付拖欠工资。", "text": ""}]

    completed = guide_graph._ensure_case_reference(reply, cases)

    assert "【类似案例参考】" in completed
    assert "刘某追索劳动报酬案" in completed
    assert "法院支持支付拖欠工资" in completed


def test_generated_case_claims_are_replaced_only_by_structured_retrieval_results():
    reply = (
        "**【法律依据】**\n法条内容\n\n"
        "**【类似案例参考】**\n"
        "甘肃（2013）某案证明法院大概率支持。\n\n"
        "**【维权路径比较】**\n先投诉。"
    )
    cases = [
        {
            "title": "白某诉李某服务合同纠纷案",
            "gist": "经营者停业后，法院结合剩余履行期限认定应返还的预付款。",
            "text": "",
        },
        {
            "title": "郑某等诈骗案",
            "gist": "经查明存在欺诈目的并依法追究刑事责任。",
            "text": "",
        },
    ]
    state = GuideState(
        legal_domain="consumer_market",
        messages=[HumanMessage(content="理发店会员卡充值后关门")],
    )

    completed = guide_graph._ensure_case_reference(reply, cases, state=state)

    assert "甘肃（2013）" not in completed
    assert "法院大概率" not in completed
    assert "白某诉李某服务合同纠纷案" in completed
    assert "郑某等诈骗案" in completed


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


def test_issue_extraction_uses_contract_nonperformance_fallback_on_invalid_json():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="not-json"))

    result = asyncio.run(extract_legal_issues(
        "我委托对方代购门票并支付4000元，双方约定未交付就退款，但对方没有退款。",
        llm,
    ))

    assert result.issues == ["合同履行与退款争议"]
    assert result.domain == "contracts_property_housing"
    assert result.degraded is True


def test_structured_intake_uses_form_facts_and_compact_domain_classification():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "issues": ["网络买卖合同履行"],
        "domain": "consumer_market",
        "region": "",
        "time_info": "2026年7月18日",
    }, ensure_ascii=False)))
    intake = (
        "【首次案件材料包】\n"
        "【事情经过】\n我在平台付款后卖家没有发货。\n"
        "【对方及双方关系】\n平台个人卖家\n"
        "【时间、地点和金额】\n2026年7月18日，800元\n"
        "【希望解决的结果】\n取消交易并退款\n"
        "【已经沟通或处理的情况】\n已经向平台投诉\n\n"
        "[本轮附件清单]\n- 订单.txt"
    )

    result = asyncio.run(extract_legal_issues(intake, llm))

    assert result.domain == "consumer_market"
    assert result.issues == ["网络买卖合同履行"]
    assert [item.key for item in result.case_updates] == [
        "event.summary",
        "relationship.counterparty",
        "case.time_place_amount",
        "claim.requested_outcome",
        "procedure.current_status",
    ]
    prompt = llm.ainvoke.await_args.args[0][0].content
    assert "订单.txt" not in prompt
    assert "evidence_details" not in prompt


def test_issue_extraction_timeout_preserves_recent_dialogue_as_semantic_seed():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("slow model"))
    recent_dialogue = "我在网上向个人卖家付款购买商品，对方没有交付"

    result = asyncio.run(
        extract_legal_issues(
            "包含内部格式要求的长提示词",
            llm,
            fallback_text=recent_dialogue,
        )
    )

    assert result.issues == [recent_dialogue]
    assert result.domain == "other"
    assert result.degraded is True


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
        "case_updates": [
            {
                "key": "employment.start_date", "category": "time",
                "statement": "2022年3月入职", "source_text": "2022年3月入职",
                "certainty": "asserted", "operation": "add",
            },
            {
                "key": "employment.monthly_wage", "category": "amount",
                "statement": "月薪8000元", "source_text": "月薪8000元",
                "certainty": "asserted", "operation": "add",
            },
        ],
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
        "case_updates": [
            {
                "key": "wage.unpaid_duration", "category": "time",
                "statement": "拖欠3个月工资", "source_text": "拖欠了3个月",
                "certainty": "asserted", "operation": "add",
            },
            {
                "key": "wage.unpaid_amount", "category": "amount",
                "statement": "拖欠金额24000元", "source_text": "一共24000元",
                "certainty": "asserted", "operation": "add",
            },
        ],
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
        "case_updates": [{
            "key": "evidence.unverified_claim", "category": "uncertainty",
            "statement": "待核验线索（图片文字转述，本次未直接展示）：消费者称另有玻璃渣实物和现场照片",
            "source_text": "语音转写称消费者另有实物和现场照片",
            "certainty": "uncertain", "operation": "add",
        }],
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


def legacy_patch_evidence_overconfidence_is_deterministically_softened():
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


def legacy_patch_food_minimum_additional_compensation_remains_conditional():
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
    prompt = llm.ainvoke.await_args_list[0].args[0][0].content

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
        self.expirations: dict[str, int | None] = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.expirations[key] = ex
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
    expected_ttl = guide_worker.settings.GUIDE_SESSION_TTL or None
    assert redis.expirations["guide_state:12:s1"] == expected_ttl


def test_worker_persists_end_state_without_matching_invitation_copy():
    redis = _FakeRedis()
    state = GuideState(
        session_id="12:s1",
        phase=GuidePhase.END,
        confirmed_issues=["拖欠工资"],
        legal_domain="labor_social_security",
        collected_facts=["拖欠3个月工资"],
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
        new=AsyncMock(return_value=("最终维权方案", state)),
    ), patch.object(
        guide_worker,
        "get_checkpointer_redis",
        return_value=redis,
    ):
        reply = asyncio.run(guide_worker.call_guide_agent_impl("现在生成方案", "12", "s1"))

    restored = GuideState.model_validate_json(redis.data["guide_state:12:s1"])
    assert reply == "最终维权方案"
    assert restored.phase == GuidePhase.END
    assert redis.data["guide_active:12:s1"] == "1"
    expected_ttl = guide_worker.settings.GUIDE_SESSION_TTL or None
    assert redis.expirations["guide_state:12:s1"] == expected_ttl


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
