"""结论前分析子流程的最小回归测试。"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.agents.legal_guide.graph import (
    _build_case_analysis_packet,
    _build_final_legal_basis,
    _clean_dialogue_message,
    _critique_plan,
    _deterministic_conclusion_draft,
    _deterministic_draft_problems,
    _ensure_case_reconstruction,
    _ensure_evidence_strategy,
    _ensure_fact_tensions,
    _ensure_issue_analysis_section,
    _enforce_final_output_contract,
    _ensure_optimal_procedure_path,
    _ensure_priority_actions,
    _format_long_dialogue_memory,
    _format_legal_element_matrix,
    _format_adversarial_review,
    _ensure_adversarial_review,
    _ensure_legal_element_review,
    _force_generic_boilerplate_revision,
    _humanized_followup_answers,
    _merge_law_refs,
    _revise_plan,
    _select_grounded_statute_entries,
    _strip_generic_boilerplate,
    _fallback_issue_map,
    _validate_analysis_grounding,
    GuideDeps,
)
from src.agents.legal_guide.state import GuideState
from src.agents.legal_guide.case_model import (
    evidence_from_case_facts,
    normalize_case_updates,
)
from src.agents.legal_guide.prompts import (
    CONCLUDE_PROMPT,
    ISSUE_APPLICATION_PROMPT,
    PLAN_CRITIQUE_PROMPT,
    PLAN_REVISION_PROMPT,
    STRATEGY_SYNTHESIS_PROMPT,
)


def test_conclusion_prompts_require_element_level_reasoning():
    assert "legal_element_matrix" in ISSUE_APPLICATION_PROMPT
    assert "opponent_counterarguments" in ISSUE_APPLICATION_PROMPT
    assert "要件拆解规则" in CONCLUDE_PROMPT
    assert "legal_element_matrix" in STRATEGY_SYNTHESIS_PROMPT
    assert "issue_authorities" in ISSUE_APPLICATION_PROMPT
    assert "issue_authorities" in STRATEGY_SYNTHESIS_PROMPT
    assert "issue_authorities" in CONCLUDE_PROMPT
    assert "禁止输出 user_" in ISSUE_APPLICATION_PROMPT
    assert "自然语言证据名称" in ISSUE_APPLICATION_PROMPT
    assert "最终输出禁令" in CONCLUDE_PROMPT
    assert "禁止输出【关键缺失信息清单】" in CONCLUDE_PROMPT
    assert "同一事实只写一次" in CONCLUDE_PROMPT
    assert "仍需核实】只列真正未知或冲突的信息" in CONCLUDE_PROMPT
    assert "长对话事实沉淀" in CONCLUDE_PROMPT
    assert "adversarial_execution_review" in CONCLUDE_PROMPT
    assert "legal_element_review" in CONCLUDE_PROMPT
    assert "adversarial_review_block" in CONCLUDE_PROMPT
    assert "法律方案批判员" in PLAN_CRITIQUE_PROMPT
    assert "法律方案修订员" in PLAN_REVISION_PROMPT
    assert "generic_boilerplate" in PLAN_CRITIQUE_PROMPT
    assert "必输栏目" in CONCLUDE_PROMPT
    assert "missing_section" in PLAN_CRITIQUE_PROMPT
    assert "missing_section" in PLAN_REVISION_PROMPT


def test_critique_and_revision_stage_contract():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({"verdict": "revise", "issues": [{"type": "redundancy"}]})),
        AIMessage(content="修订后的最终方案"),
    ])
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = llm
    deps.fast_llm = llm
    state = GuideState()

    critique = asyncio.run(_critique_plan(
        state, deps, {}, [], [], {}, {}, "初稿方案"
    ))
    revised = asyncio.run(_revise_plan(
        state, deps, {}, [], [], {}, {}, "初稿方案", critique
    ))

    assert critique["verdict"] == "revise"
    assert revised == "修订后的最终方案"


def test_enforce_final_output_contract_removes_forbidden_sections():
    reply = (
        "**【关键缺失信息清单】**\n- 缺少报警记录。\n\n"
        "**【强烈建议】**\n请补充上述信息。\n\n"
        "**【行动清单】**\n1. 保存现有证据。"
    )

    cleaned = _enforce_final_output_contract(reply)

    assert "关键缺失信息清单" not in cleaned
    assert "强烈建议" not in cleaned
    assert "请补充上述信息" not in cleaned
    assert "行动清单" in cleaned


def test_generic_boilerplate_forces_revision():
    draft = "当前判断：需要结合完整事实、证据和办案机关认定。"
    critique = {"verdict": "acceptable", "issues": []}

    forced = _force_generic_boilerplate_revision(critique, draft)

    assert forced["verdict"] == "revise"
    assert any(item["type"] == "generic_boilerplate" for item in forced["issues"])


def test_strip_generic_boilerplate_removes_template_lines():
    reply = (
        "当前判断：需要结合完整事实、证据和办案机关认定。\n\n"
        "仍需核实：对方的具体行为细节。"
    )

    cleaned = _strip_generic_boilerplate(reply)

    assert "需要结合完整事实、证据和办案机关认定" not in cleaned
    assert "仍需核实：对方的具体行为细节" in cleaned


def test_deterministic_draft_problems_detects_contract_violations():
    generic = "当前判断：需要结合完整事实、证据和办案机关认定。"
    forbidden = "**【强烈建议】**\n请补充上述信息。"
    internal_key = "当前事实：followup.fraudster_identification"

    generic_issues = _deterministic_draft_problems(generic)
    forbidden_issues = _deterministic_draft_problems(forbidden)
    key_issues = _deterministic_draft_problems(internal_key)

    assert any(item["type"] == "generic_boilerplate" for item in generic_issues)
    assert any(item["type"] == "forbidden_output" for item in forbidden_issues)
    assert any(item["type"] == "internal_key_leak" for item in key_issues)


def test_legal_element_matrix_renders_joint_analysis():
    analyses = [{
        "title": "淘宝平台责任",
        "legal_element_matrix": [{
            "legal_basis_ref": "《民法典》第一千一百九十七条",
            "element": "知道或者应当知道",
            "supporting_facts": ["用户通过淘宝平台交易"],
            "evidence_items": ["聊天记录"],
            "status": "unknown",
            "why": "缺少平台投诉记录",
            "what_would_change": "同类投诉记录可证明平台知情",
        }],
    }]

    block = _format_legal_element_matrix(analyses)

    assert "淘宝平台责任" in block
    assert "知道或者应当知道" in block
    assert "用户通过淘宝平台交易" in block
    assert "聊天记录" in block
    assert "同类投诉记录可证明平台知情" in block


def test_legal_element_matrix_translates_internal_keys():
    analyses = [{
        "title": "淘宝平台责任",
        "legal_element_matrix": [{
            "legal_basis_ref": "《民法典》第一千一百九十七条",
            "element": "知道或者应当知道",
            "supporting_facts": ["followup.fraudster_identification"],
            "evidence_items": ["user_complaint_to_platform"],
            "status": "unknown",
            "why": "待核实",
            "what_would_change": "平台投诉记录",
        }],
    }]

    block = _format_legal_element_matrix(analyses)

    assert "诈骗者身份识别信息" in block
    assert "用户向平台投诉情况" in block
    assert "followup.fraudster_identification" not in block


def test_ensure_legal_element_review_inserts_block():
    reply = "**【维权路径比较】**\n先报警。"
    block = "**【法条要件核对】**\n- 要件：知道或者应当知道"

    out = _ensure_legal_element_review(reply, block)

    assert "法条要件核对" in out
    assert "维权路径比较" in out
    assert out.index("法条要件核对") < out.index("维权路径比较")


def test_adversarial_review_renders_opponent_arguments():
    review = {
        "opponent_arguments": [{
            "argument": "平台不知道这是诈骗",
            "response": "提供聊天记录证明是通过平台联系",
            "evidence_needed": "平台投诉记录",
        }],
        "adverse_points": [{
            "point": "用户未报警",
            "impact": "影响损失认定",
            "countermeasure": "尽快报警",
        }],
        "current_procedure_stage": "已投诉未报警",
        "next_procedure_stage": "报警立案",
        "next_stage_trigger": "材料齐全",
    }

    block = _format_adversarial_review(review)

    assert "对方/平台可能反驳" in block
    assert "平台不知道这是诈骗" in block
    assert "提供聊天记录证明是通过平台联系" in block
    assert "平台投诉记录" in block
    assert "用户未报警" in block


def test_ensure_adversarial_review_inserts_block():
    reply = "**【维权路径比较】**\n先报警。"
    block = "**【反方压力测试】**\n- 对方可能反驳：平台不知情。"

    out = _ensure_adversarial_review(reply, block)

    assert "反方压力测试" in out
    assert out.index("反方压力测试") < out.index("维权路径比较")


def test_merge_law_refs_keeps_followup_and_issue_links():
    merged = _merge_law_refs([
        [{"law_id": "1", "article_no": "第一条", "title": "测试法", "text": "原文A"}],
        [{"law_id": "2", "article_no": "第二条", "title": "测试法二", "text": "原文B", "source_issue_ids": ["issue_1"]}],
        [{"law_id": "1", "article_no": "第一条", "title": "测试法", "text": "原文A2", "source_issue_ids": ["issue_2"]}],
    ])

    assert len(merged) == 2
    by_id = {item["law_id"]: item for item in merged}
    assert by_id["1"]["source_issue_ids"] == ["issue_2"]
    assert by_id["2"]["source_issue_ids"] == ["issue_1"]


def test_final_legal_basis_groups_authorities_by_issue():
    state = GuideState(retrieved_law_refs=[
        {
            "law_id": "1",
            "article_no": "第一条",
            "title": "测试法A",
            "text": "原文",
            "source_issue_ids": ["issue_2"],
        },
        {
            "law_id": "2",
            "article_no": "第二条",
            "title": "测试法B",
            "text": "原文",
            "source_issue_ids": ["issue_1"],
        },
        {
            "law_id": "3",
            "article_no": "第三条",
            "title": "测试法C",
            "text": "原文",
        },
    ])
    issue_map = [
        {"issue_id": "issue_1", "title": "争点一", "retrieval_questions": []},
        {"issue_id": "issue_2", "title": "争点二", "retrieval_questions": []},
    ]

    packet = _build_final_legal_basis(state, issue_map)

    assert packet["primary_authorities"][0]["law_id"] == "2"
    assert packet["issue_authorities"][0]["issue_id"] == "issue_1"
    assert packet["issue_authorities"][0]["authorities"][0]["law_id"] == "2"
    assert packet["issue_authorities"][1]["authorities"][0]["law_id"] == "1"


def test_humanized_followup_answers_remove_internal_form_envelope():
    message = (
        "【动态追问表单回答】\n"
        "1. [transaction_method] 通过什么平台交易？\n"
        "回答: 淘宝平台\n"
        "2. [event_time] 大概什么时间发生？\n"
        "回答: 没有\n"
    )
    questions = [
        {"field_id": "transaction_method", "question": "通过什么平台交易？"},
        {"field_id": "event_time", "question": "大概什么时间发生？"},
    ]

    out = _humanized_followup_answers(message, questions)

    assert "【动态追问表单回答】" not in out
    assert "[transaction_method]" not in out
    assert "用户补充：淘宝平台；大概什么时间发生：没有" in out


def test_clean_dialogue_message_strips_evidence_wrapper():
    message = (
        "我在淘宝被诈骗了。\n\n"
        "【图片证据补充（视觉模型识别，需与原图核对）】\n"
        "文件：聊天记录.png\n"
        "原图 SHA-256：abcdefabcdef0123\n"
        "【提取文字】\n"
        "对方让我转账500元"
    )

    out = _clean_dialogue_message(message)

    assert "【图片证据补充" not in out
    assert "对方让我转账500元" not in out
    assert "我在淘宝被诈骗了" in out


def test_clean_dialogue_message_hides_control_only_text():
    assert _clean_dialogue_message("现在生成方案") == "用户：流程控制语（非案件事实）"


def test_long_dialogue_memory_keeps_old_turn_facts():
    state = GuideState(case_facts=[
        {
            "key": f"event.old.{index}",
            "statement": f"第{index}轮事实",
            "status": "asserted",
            "turn": index,
            "source_text": f"第{index}轮事实",
        }
        for index in range(1, 11)
    ])

    memory = _format_long_dialogue_memory(state)

    assert "第1轮：第1轮事实（已确认）" in memory
    assert "第10轮：第10轮事实（已确认）" in memory


def test_case_analysis_packet_keeps_earlier_facts_and_unknown_status():
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["抢包"],
        case_facts=[
            {"key": "event.time", "statement": "昨天晚上发生", "status": "asserted", "turn": 1},
            {"key": "event.weapon", "statement": "对方拿刀威胁", "status": "asserted", "turn": 2},
            {"key": "event.location", "statement": "具体地点记不清", "status": "uncertain", "turn": 3},
        ],
    )

    packet = _build_case_analysis_packet(state)

    assert len(packet["facts"]) == 3
    assert "昨天晚上发生" in packet["fact_context"]
    assert "对方拿刀威胁" in packet["fact_context"]
    assert "具体地点记不清" in packet["unknown_facts"]


def test_fallback_issue_map_and_draft_are_case_specific():
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["抢劫行为定性", "反击是否属于正当防卫"],
        collected_facts=["对方持刀威胁", "用户骨折"],
    )
    packet = _build_case_analysis_packet(state)
    issues = _fallback_issue_map(state, packet)
    state = state.model_copy(update={
        "issue_analyses": [
            {
                "title": "反击是否属于正当防卫",
                "current_view": "持刀威胁仍在进行时具有正当防卫基础",
                "application_analysis": "需要结合冲突先后顺序判断",
                "conditional_branch": "危险解除后继续追打可能改变评价",
            }
        ],
    })

    draft = _deterministic_conclusion_draft(state)

    assert len(issues) == 2
    assert "核心争点分析" in draft
    assert "危险解除后继续追打可能改变评价" in draft


def test_analysis_grounding_reports_unknown_fact_references():
    packet = {"facts": [{"key": "event.weapon", "statement": "持刀"}]}
    report = _validate_analysis_grounding(
        packet,
        [{"issue_id": "i1", "title": "争点"}],
        [{"issue_id": "i1", "supporting_facts": ["missing.fact"], "legal_basis_refs": []}],
        {"primary_authorities": []},
    )

    assert report["status"] == "needs_review"
    assert "missing.fact" in report["unknown_fact_keys"]


def test_issue_analysis_section_is_injected_when_renderer_omits_it():
    state = GuideState(issue_analyses=[{
        "title": "抢夺与抢劫的区分",
        "current_view": "是否存在暴力或胁迫会影响定性",
        "application_analysis": "用户称手机被抢走，但具体行为仍需核实",
        "conditional_branch": "如存在当场暴力或胁迫，评价可能变化",
        "facts_to_verify": ["具体暴力或威胁方式"],
        "recommended_actions": ["向警方补充经过并申请保存监控"],
    }])
    reply = "**【理解您的情况】**\n手机被抢走。\n\n**【法律依据】**\n《测试法》第一条。"

    out = _ensure_issue_analysis_section(reply, state)

    assert "【核心争点分析】" in out
    assert "具体暴力或威胁方式" in out
    assert out.index("【核心争点分析】") < out.index("【法律依据】")


def test_priority_actions_and_case_reconstruction_use_analysis_and_full_facts():
    state = GuideState(
        legal_domain="criminal_public_security",
        case_facts=[
            {"key": "event", "category": "event", "statement": "手机被抢走", "status": "asserted"},
            {"key": "place", "category": "location", "statement": "发生在清河小营桥", "status": "asserted"},
            {"key": "report", "category": "procedure", "statement": "已经报警但未取得回执", "status": "asserted"},
            {"key": "witness", "category": "evidence", "statement": "能够联系目击者", "status": "asserted"},
        ],
        issue_analyses=[{
            "recommended_actions": ["向受理公安机关补充监控和目击者线索"],
            "evidence_actions": ["联系监控管理方保全原始录像"],
        }],
    )
    reply = "**【理解您的情况】**\n已记录。\n\n**【核心争点分析】**\n分析。\n\n**【法律依据】**\n法条。"

    out = _ensure_priority_actions(reply, state)
    out = _ensure_case_reconstruction(out, state)

    assert "【现在最优先行动】" in out
    assert "向受理公安机关补充监控和目击者线索" in out
    assert "【案件完整还原】" in out
    assert "发生在清河小营桥" in out
    assert "已经报警但未取得回执" in out


def test_final_legal_basis_is_issue_linked_snapshot():
    state = GuideState(
        retrieved_law_refs=[{"title": "测试法", "article_no": "第一条", "text": "原文"}],
        case_context_str="类案摘要",
        relevant_channels=[{"name": "报警机关", "phone": "110"}],
    )

    packet = _build_final_legal_basis(
        state,
        [{"title": "行为定性", "retrieval_questions": ["持刀取得财物"]}],
    )

    assert packet["retrieval_mode"] == "latest_snapshot_issue_linking"
    assert packet["issue_queries"] == ["持刀取得财物"]
    assert packet["primary_authorities"][0]["article_no"] == "第一条"


def test_procedure_and_evidence_sections_follow_issue_analysis():
    state = GuideState(
        legal_domain="criminal_public_security",
        case_facts=[
            {"key": "report", "category": "procedure", "statement": "police report already filed", "status": "asserted"},
            {"key": "witness", "category": "evidence", "statement": "witness contact is available", "status": "asserted"},
        ],
        issue_analyses=[{
            "recommended_actions": ["submit the witness contact to police"],
            "evidence_actions": ["ask the shop to preserve original surveillance video"],
        }],
    )
    reply = "**【核心争点分析】**\nanalysis\n\n**【法律依据】**\nlaw"

    out = _ensure_optimal_procedure_path(reply, state)
    out = _ensure_evidence_strategy(out, state)

    assert "【最优程序路径】" in out
    assert "【证据作战图】" in out
    assert "submit the witness contact to police" in out
    assert "ask the shop to preserve original surveillance video" in out
    assert out.index("【最优程序路径】") < out.index("【证据作战图】") < out.index("【法律依据】")


def test_control_message_is_not_persisted_and_evidence_lead_is_not_confirmed():
    control = normalize_case_updates([
        {"key": "control", "statement": "现在生成方案", "source_text": "现在生成方案"},
    ], user_text="现在生成方案", turn=3)
    evidence = normalize_case_updates([
        {
            "key": "evidence.camera", "category": "evidence",
            "statement": "事发地点附近有监控，可以申请调取",
            "source_text": "事发地点附近有监控，可以申请调取",
        },
    ], user_text="事发地点附近有监控，可以申请调取", turn=3)

    present, unavailable = evidence_from_case_facts(evidence)

    assert control == []
    assert evidence[0]["evidence_status"] == "lead"
    assert present == []
    assert unavailable == []


def test_fact_tension_is_rendered_as_a_boundary_not_a_verdict():
    state = GuideState(case_analysis_packet={
        "facts": [
            {"key": "event.force.yes", "statement": "对方使用暴力", "status": "asserted"},
            {"key": "event.force.no", "statement": "对方没有使用具体暴力", "status": "asserted"},
        ],
        "fact_tensions": [{
            "title": "是否使用暴力存在矛盾",
            "side_a_fact_keys": ["event.force.yes"],
            "side_b_fact_keys": ["event.force.no"],
            "why_it_matters": "会影响行为的法律评价。",
            "resolution_action": "优先调取监控并联系目击者。",
        }],
    })

    out = _ensure_fact_tensions("**【法律依据】**\nlaw", state)

    assert "【关键矛盾与待核实】" in out
    assert "对方使用暴力" in out
    assert "对方没有使用具体暴力" in out
    assert "会影响行为的法律评价" in out


def test_evidence_lead_is_rendered_as_lead_not_held_material():
    state = GuideState(case_facts=[{
        "key": "evidence.camera", "category": "evidence",
        "statement": "附近商户可能保存监控录像", "status": "asserted",
        "evidence_status": "lead",
    }])

    out = _ensure_evidence_strategy("**【法律依据】**\nlaw", state)

    assert "【证据作战图】" in out
    assert "目前只是线索、尚未取得" in out
    assert "目前已确认" not in out


def test_grounded_statute_selection_prefers_issue_linked_article():
    entries = [
        ("甲法", "第一条", "background"),
        ("乙法", "第二条", "directly relevant"),
    ]
    state = GuideState(issue_analyses=[{"legal_basis_refs": ["乙法第二条"]}])

    selected = _select_grounded_statute_entries(entries, state, limit=2)

    assert selected == [("乙法", "第二条", "directly relevant")]
