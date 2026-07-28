"""法律指引九节点工作流的路由回归测试。"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.agents.legal_guide.graph import (
    GuideDeps,
    _needs_clarify,
    _next_ask_type,
    _remaining_fact_questions,
    build_guide_graph,
    node_assess_retrieve,
    node_parse_details,
    route_after_assess_retrieve,
    route_after_extract,
    route_after_parse,
    run_guide,
)
from src.agents.legal_guide.followup_catalog import evidence_followups, fact_followups
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.core.config import get_settings


settings = get_settings()


def test_needs_clarify_2_round_cap():
    assert _needs_clarify(GuideState(clarify_rounds=0)) is True
    assert _needs_clarify(GuideState(clarify_rounds=1)) is True
    assert _needs_clarify(GuideState(clarify_rounds=2)) is False
    assert _needs_clarify(GuideState(confirmed_issues=["拖欠工资"])) is False
    assert _needs_clarify(GuideState(unmatched_issues=["老板不给钱"])) is False


def test_route_after_extract_branches():
    assert route_after_extract(GuideState(clarify_rounds=0)) == "clarify"
    assert route_after_extract(GuideState(clarify_rounds=2)) == "assess_retrieve"
    assert route_after_extract(
        GuideState(confirmed_issues=["拖欠工资"])
    ) == "assess_retrieve"


def test_route_after_parse_branches():
    waiting = GuideState(pending_ask_details=["是否签订合同？"])
    assert route_after_parse(waiting) == END

    new_issue = GuideState(
        confirmed_issues=["拖欠工资", "违法解除劳动合同"],
        last_confirmed_count=1,
    )
    assert route_after_parse(new_issue) == "extract_issues"

    answered = GuideState(
        confirmed_issues=["拖欠工资"],
        last_confirmed_count=1,
    )
    assert route_after_parse(answered) == "assess_retrieve"


def test_followup_type_uses_explicit_fact_and_evidence_limits():
    domain = "labor_social_security"
    low = GuideState(legal_domain=domain, confidence_tier="LOW")
    assert _next_ask_type(low) == "facts"

    medium = GuideState(legal_domain=domain, confidence_tier="MEDIUM")
    assert _next_ask_type(medium) == "evidence"

    facts_exhausted = GuideState(
        legal_domain=domain,
        confidence_tier="LOW",
        asked_followup_ids=[rule.id for rule in fact_followups(domain)],
    )
    assert _next_ask_type(facts_exhausted) == "evidence"

    all_exhausted = GuideState(
        legal_domain=domain,
        confidence_tier="LOW",
        asked_followup_ids=(
            [rule.id for rule in fact_followups(domain)]
            + [rule.id for rule in evidence_followups(domain)]
        ),
    )
    assert _next_ask_type(all_exhausted) == ""

    total_limit = GuideState(
        legal_domain=domain,
        confidence_tier="LOW",
        ask_rounds=settings.GUIDE_MAX_ASK_ROUNDS,
    )
    assert _next_ask_type(total_limit) == ""


def test_fact_followup_does_not_repeat_merchant_response_from_user_message():
    state = GuideState(
        legal_domain="consumer_market",
        messages=[HumanMessage(content="店里说给我退钱，但我不知道该怎么办")],
    )

    remaining = _remaining_fact_questions(state)

    assert "您在哪里买了什么商品或服务，大约花了多少钱？" in remaining
    assert not any("商家沟通后" in question for question in remaining)


def test_route_after_assess_retrieve_converges_or_asks_once():
    low = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="LOW",
        total_rounds=1,
    )
    assert route_after_assess_retrieve(low) == "ask_followup"

    high = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="HIGH",
        confidence_score=0.8,
        evidence_confirmed=["劳动合同"],
    )
    assert route_after_assess_retrieve(high) == "conclude"

    forced = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="LOW",
        force_conclude=True,
    )
    assert route_after_assess_retrieve(forced) == "conclude"

    round_limit = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="LOW",
        total_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS,
    )
    assert route_after_assess_retrieve(round_limit) == "conclude"


def test_parse_details_clears_explicit_ask_type_and_saves_time():
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "is_answer": True,
        "new_issues": [],
        "evidence": ["劳动合同"],
        "evidence_unavailable": [],
        "adverse_facts": [],
        "region": "北京",
        "time_info": "2025年3月",
        "user_question": "",
    }, ensure_ascii=False)))
    state = GuideState(
        messages=[HumanMessage(content="2025年3月入职，有劳动合同")],
        pending_ask_details=["何时入职？", "是否有劳动合同？"],
        pending_ask_type="facts",
    )

    result = asyncio.run(node_parse_details(state, deps))
    assert result["pending_ask_details"] == []
    assert result["pending_ask_type"] == ""
    assert result["time_info"] == "2025年3月"
    assert result["evidence_confirmed"] == ["劳动合同"]


def test_assess_retrieve_sets_force_conclude_at_total_limit():
    state = GuideState(total_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS)
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.1, "confidence_tier": "LOW"}),
    ), patch(
        "src.agents.legal_guide.graph.node_retrieve",
        new=AsyncMock(return_value={"law_context_str": "示例法条"}),
    ):
        result = asyncio.run(node_assess_retrieve(state, MagicMock(spec=GuideDeps)))
    assert result["confidence_tier"] == "LOW"
    assert result["law_context_str"] == "示例法条"
    assert result["force_conclude"] is True


def test_assess_retrieve_reuses_snapshot_when_user_requests_conclusion():
    state = GuideState(
        wants_conclude=True,
        retrieval_completed=True,
        law_context_str="上一轮法条",
        case_context_str="上一轮类案",
    )
    retrieve = AsyncMock(return_value={"law_context_str": "不应重新检索"})
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.2, "confidence_tier": "LOW"}),
    ), patch(
        "src.agents.legal_guide.graph.node_retrieve",
        new=retrieve,
    ):
        result = asyncio.run(node_assess_retrieve(state, MagicMock(spec=GuideDeps)))

    retrieve.assert_not_awaited()
    assert result["confidence_tier"] == "LOW"
    assert result["force_conclude"] is False


def test_compiled_graph_has_exactly_nine_business_nodes():
    compiled = build_guide_graph(MagicMock())
    nodes = set(compiled.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {
        "prepare_turn",
        "check_urgency",
        "extract_issues",
        "clarify",
        "assess_retrieve",
        "ask_followup",
        "parse_details",
        "conclude",
        "save_record",
    }


def test_end_to_end_route_returns_question_then_final_plan():
    deps = MagicMock(spec=GuideDeps)
    common = [
        patch("src.agents.legal_guide.graph.node_load_context", new=AsyncMock(return_value={})),
        patch("src.agents.legal_guide.graph.node_check_urgency", new=AsyncMock(return_value={"urgency_level": "normal"})),
        patch("src.agents.legal_guide.graph.node_extract_issues", new=AsyncMock(return_value={
            "confirmed_issues": ["拖欠工资"],
            "legal_domain": "labor_social_security",
            "phase": GuidePhase.ISSUE_SEARCH,
        })),
        patch("src.agents.legal_guide.graph.node_assess_retrieve", new=AsyncMock(return_value={
            "confidence_score": 0.3,
            "confidence_tier": "LOW",
            "law_context_str": "《劳动法》相关条文",
            "last_confirmed_count": 1,
        })),
        patch("src.agents.legal_guide.graph.node_ask_followup", new=AsyncMock(return_value={
            "phase": GuidePhase.DETAIL_GATHER,
            "ask_rounds": 1,
            "facts_rounds": 1,
            "pending_ask_details": ["拖欠了几个月工资？"],
            "pending_ask_type": "facts",
            "messages": [AIMessage(content="先确认一下：拖欠了几个月工资？")],
        })),
    ]
    for mocked in common:
        mocked.start()
    try:
        reply, state = asyncio.run(run_guide("公司拖欠工资", "u:s", deps))
    finally:
        for mocked in reversed(common):
            mocked.stop()

    assert reply == "先确认一下：拖欠了几个月工资？"
    assert state.pending_ask_type == "facts"
    assert state.phase == GuidePhase.DETAIL_GATHER

    second_turn = [
        patch("src.agents.legal_guide.graph.node_check_urgency", new=AsyncMock(return_value={"urgency_level": "normal"})),
        patch("src.agents.legal_guide.graph.node_parse_details", new=AsyncMock(return_value={
            "pending_ask_details": [],
            "pending_ask_type": "",
            "evidence_confirmed": ["工资流水"],
            "phase": GuidePhase.ISSUE_SEARCH,
        })),
        patch("src.agents.legal_guide.graph.node_assess_retrieve", new=AsyncMock(return_value={
            "confidence_score": 0.8,
            "confidence_tier": "HIGH",
            "last_confirmed_count": 1,
        })),
        patch("src.agents.legal_guide.graph.node_conclude", new=AsyncMock(return_value={
            "phase": GuidePhase.CONCLUDE,
            "messages": [AIMessage(content="最终维权方案")],
        })),
        patch("src.agents.legal_guide.graph.node_save_record", new=AsyncMock(return_value={"phase": GuidePhase.END})),
    ]
    for mocked in second_turn:
        mocked.start()
    try:
        reply, state = asyncio.run(run_guide("拖欠两个月，我有工资流水", "u:s", deps, existing_state=state))
    finally:
        for mocked in reversed(second_turn):
            mocked.stop()

    assert reply == "最终维权方案"
    assert state.phase == GuidePhase.END


if __name__ == "__main__":
    test_needs_clarify_2_round_cap()
    test_route_after_extract_branches()
    test_route_after_parse_branches()
    test_followup_type_uses_explicit_fact_and_evidence_limits()
    test_route_after_assess_retrieve_converges_or_asks_once()
    test_parse_details_clears_explicit_ask_type_and_saves_time()
    test_assess_retrieve_sets_force_conclude_at_total_limit()
    test_compiled_graph_has_exactly_nine_business_nodes()
    test_end_to_end_route_returns_question_then_final_plan()
    print("ALL PASS")
