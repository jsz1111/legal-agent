"""法律指引九节点工作流的路由回归测试。"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.agents.legal_guide.graph import (
    GuideDeps,
    _needs_clarify,
    _recall_relevant_memories,
    build_guide_graph,
    node_assess_retrieve,
    node_clarify,
    node_extract_issues,
    node_parse_details,
    route_after_assess_retrieve,
    route_after_extract,
    route_after_parse,
    run_guide,
)
from src.agents.legal_guide.decision_sufficiency import DecisionSufficiencyReport
from src.agents.legal_guide.scenario_assessment import ScenarioAssessment
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.core.config import get_settings
from src.api.routers.chat import _has_guide_session, _should_keep_guide_state


settings = get_settings()


class _SessionRedis:
    def __init__(self, keys: set[str]):
        self.keys = keys

    async def exists(self, key: str):
        return int(key in self.keys)


def test_guide_session_recovers_when_only_structured_state_survives():
    redis = _SessionRedis({"guide_state:u:s"})
    assert asyncio.run(
        _has_guide_session(redis, "guide_active:u:s", "guide_state:u:s")
    ) is True


def test_end_state_persistence_uses_structured_case_readiness():
    ready = GuideState(
        phase=GuidePhase.END,
        confirmed_issues=["拖欠工资"],
        legal_domain="labor_social_security",
    )
    urgent_without_case = GuideState(phase=GuidePhase.END)
    safety_pause = GuideState(
        phase=GuidePhase.END,
        safety_pause_active=True,
        current_safety_status="danger",
    )
    degraded_but_substantive = GuideState(
        phase=GuidePhase.END,
        unmatched_issues=["用户尚未被标准化的纠纷描述"],
        case_facts=[{
            "key": "legacy.raw.case",
            "category": "event",
            "statement": "用户已经描述具体纠纷",
            "source_text": "用户已经描述具体纠纷",
            "status": "asserted",
            "turn": 1,
        }],
    )

    assert _should_keep_guide_state(ready) is True
    assert _should_keep_guide_state(safety_pause) is True
    assert _should_keep_guide_state(degraded_but_substantive) is True
    assert _should_keep_guide_state(urgent_without_case) is False


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


def test_low_information_message_clarifies_even_with_prior_domain_and_facts():
    state = GuideState(
        legal_domain="cyber_data_fraud",
        confirmed_issues=["疑似网络诈骗"],
        collected_facts=["用户此前被网络诈骗"],
        messages=[HumanMessage(content="我")],
    )

    assert route_after_extract(state) == "clarify"


def test_extract_low_information_message_skips_memory_and_llm():
    state = GuideState(
        user_context={"long_term_memories": ["用户此前被网络诈骗"]},
        messages=[HumanMessage(content="我")],
    )
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    normalizer = AsyncMock()
    with patch(
        "src.agents.legal_guide.graph.normalize_legal_issues",
        new=normalizer,
    ):
        result = asyncio.run(node_extract_issues(state, deps))

    normalizer.assert_not_awaited()
    assert result["phase"] == GuidePhase.CLARIFY
    assert result["issue_refresh_needed"] is False


def test_recall_relevant_memories_skips_low_information_query():
    with patch("src.infra.milvus_store.get_milvus_store") as store:
        result = asyncio.run(_recall_relevant_memories("user_1", "我"))

    assert result == []
    store.assert_not_called()


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

    off_target_supplement = GuideState(issue_refresh_needed=True)
    assert route_after_parse(off_target_supplement) == "extract_issues"


def test_route_after_assess_retrieve_converges_or_asks_once():
    low = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="LOW",
        total_rounds=1,
        followup_plan={"should_ask": True},
    )
    assert route_after_assess_retrieve(low) == "ask_followup"

    high = GuideState(
        legal_domain="labor_social_security",
        confidence_tier="HIGH",
        confidence_score=0.8,
        evidence_confirmed=["劳动合同"],
        followup_plan={"should_ask": False},
    )
    assert route_after_assess_retrieve(high) == "conclude"

    high_after_choice = high.model_copy(update={
        "supplement_choice_offered": True,
    })
    assert route_after_assess_retrieve(high_after_choice) == "conclude"

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


def test_unresolved_safety_plan_routes_to_one_followup_not_conclusion():
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["故意伤害"],
        safety_relevant=True,
        current_safety_status="unknown",
        confidence_tier="LOW",
        followup_plan={
            "should_ask": True,
            "candidate_id": "criminal_event_safety",
            "decision_key": "current_safety",
        },
    )

    assert route_after_assess_retrieve(state) == "ask_followup"


def test_parse_details_clears_explicit_ask_type_and_saves_time():
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
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


def test_clarify_uses_recent_context_and_registers_a_pending_question():
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="对方是什么身份，例如个人还是公司？")
    )
    state = GuideState(
        messages=[
            HumanMessage(content="我遇到了一笔有问题的交易"),
            AIMessage(content="这笔交易大概是什么情况？"),
            HumanMessage(content="我已经付款但对方没有交付"),
        ],
    )

    result = asyncio.run(node_clarify(state, deps))
    prompt = deps.llm.ainvoke.await_args.args[0][0].content

    assert "我遇到了一笔有问题的交易" in prompt
    assert "我已经付款但对方没有交付" in prompt
    assert result["pending_ask_details"] == ["对方是什么身份，例如个人还是公司？"]
    assert result["pending_ask_type"] == "facts"


def test_more_specific_grounded_facts_can_revise_an_early_low_confidence_domain():
    state = GuideState(
        round=3,
        legal_domain="cyber_data_fraud",
        confidence_tier="LOW",
        confirmed_issues=["疑似网络诈骗"],
        messages=[
            HumanMessage(content="我是在网络平台向个人卖家购买商品，付款后没有收到货")
        ],
    )
    normalized = {
        "standard": ["网络购物付款后未收到货物"],
        "colloquial": [],
        "term_map": {},
        "domain": "consumer_market",
        "collected_facts": ["付款后没有收到货"],
        "case_updates": [{
            "key": "transaction.non_delivery",
            "category": "event",
            "statement": "用户付款后没有收到货物",
            "source_text": "付款后没有收到货",
            "certainty": "asserted",
            "operation": "add",
        }],
        "evidence_details": [],
        "region": "",
        "time_info": "",
    }
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    deps.neo4j_driver = MagicMock()
    deps.embedding_model = MagicMock()
    deps.milvus_client = MagicMock()
    with patch(
        "src.agents.legal_guide.graph.normalize_legal_issues",
        new=AsyncMock(return_value=normalized),
    ):
        result = asyncio.run(node_extract_issues(state, deps))

    assert result["legal_domain"] == "consumer_market"
    assert "网络购物付款后未收到货物" in result["confirmed_issues"]


def test_parse_details_timeout_preserves_declarative_user_answer():
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(side_effect=TimeoutError("slow model"))
    state = GuideState(
        round=2,
        messages=[HumanMessage(content="我有付款记录，但没有发票")],
        pending_ask_details=["您是否有交易凭证？"],
        pending_ask_type="evidence",
    )

    result = asyncio.run(node_parse_details(state, deps))

    assert "我有付款记录，但没有发票" in result["collected_facts"]
    assert result["pending_ask_details"] == []
    assert result["pending_ask_type"] == ""


def test_off_target_but_substantive_answer_requests_issue_and_domain_refresh():
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "is_answer": True,
        "answers_asked_question": False,
        "user_question": "",
        "new_issues": [],
        "collected_facts": ["用户补充了另一项具体交易经过"],
        "case_updates": [{
            "key": "event.supplement",
            "category": "event",
            "statement": "用户补充了另一项具体交易经过",
            "source_text": "我补充另一项具体交易经过",
            "certainty": "asserted",
            "operation": "add",
        }],
        "evidence": [],
        "evidence_details": [],
        "evidence_unavailable": [],
        "adverse_facts": [],
        "region": "",
        "time_info": "",
    }, ensure_ascii=False)))
    state = GuideState(
        round=3,
        messages=[HumanMessage(content="我补充另一项具体交易经过")],
        pending_ask_details=["事情是什么时候发生的？"],
        pending_ask_type="facts",
        pending_followup_ids=["other_time_harm"],
    )

    result = asyncio.run(node_parse_details(state, deps))

    assert result["issue_refresh_needed"] is True
    assert route_after_parse(state.model_copy(update=result)) == "extract_issues"


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


def test_assess_retrieve_asks_scenario_confirmation_before_domain_retrieval():
    state = GuideState(
        legal_domain="cyber_data_fraud",
        confirmed_issues=["疑似网络诈骗"],
    )
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    report = ScenarioAssessment(
        primary_scenario="平台下单付款后没收到货",
        primary_domain="consumer_market",
        primary_frame="contract_service",
        confidence=0.4,
        competing_scenarios=["对方诱导直接转账后被拉黑"],
        discriminating_facts=["钱是直接转给个人还是平台支付"],
        confirmation_question="您的情况更接近哪一种？",
        confirmation_options=[
            "平台下单付款后没收到货",
            "对方诱导直接转账后被拉黑",
        ],
    )
    with patch(
        "src.agents.legal_guide.graph.assess_scenario",
        new=AsyncMock(return_value=report),
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    assert result["followup_plan"]["planner_mode"] == "scenario_confirmation"
    assert result["followup_plan"]["should_ask"] is True
    assert result["scenario_confirmation_offered"] is True
    assert result["scenario_analysis"]["primary_domain"] == "consumer_market"


def test_assess_retrieve_explicit_continue_overrides_soft_sufficiency_stop():
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["故意伤害"],
        collected_facts=["用户被他人殴打"],
        ask_rounds=settings.GUIDE_SOFT_ASK_ROUNDS,
        total_rounds=5,
        retrieval_completed=True,
        supplement_choice="continue",
        allow_extra_followups=True,
    )
    followup = {
        "should_ask": True,
        "candidate_id": "criminal_cctv_recording",
        "planner_mode": "deterministic_policy",
    }
    planner = AsyncMock(return_value=followup)
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.8, "confidence_tier": "HIGH"}),
    ), patch(
        "src.agents.legal_guide.graph.assess_decision_sufficiency",
        return_value=DecisionSufficiencyReport(
            sufficient_for_definitive_plan=True,
            recommended_action="conclude_definitive",
            reason="自动判断已经充分",
        ),
    ), patch(
        "src.agents.legal_guide.graph.plan_fixed_batch",
        new=AsyncMock(return_value={"should_ask": False}),
    ), patch(
        "src.agents.legal_guide.graph.plan_next_followup",
        new=planner,
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    planner.assert_awaited_once()
    assert result["followup_plan"]["should_ask"] is True
    assert result["force_conclude"] is False


def test_assess_retrieve_skips_all_knowledge_retrieval_while_following_up():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[{
            "id": "fact-1",
            "key": "event.problem",
            "category": "event",
            "statement": "商品存在问题",
            "source_text": "商品有问题",
            "active": True,
            "turn": 1,
        }],
    )
    retrieve = AsyncMock(return_value={"law_context_str": "不应出现"})
    planner = AsyncMock(return_value={
        "should_ask": True,
        "candidate_id": "consumer_transaction",
        "planner_mode": "deterministic_policy",
    })
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    with patch(
        "src.agents.legal_guide.graph.node_score",
        new=AsyncMock(return_value={"confidence_score": 0.3, "confidence_tier": "LOW"}),
    ), patch(
        "src.agents.legal_guide.graph.plan_fixed_batch",
        new=AsyncMock(return_value={"should_ask": False}),
    ), patch(
        "src.agents.legal_guide.graph.plan_next_followup",
        new=planner,
    ), patch(
        "src.agents.legal_guide.graph.node_retrieve",
        new=retrieve,
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    retrieve.assert_not_awaited()
    assert result["followup_plan"]["should_ask"] is True
    assert "law_context_str" not in result


def test_assess_retrieve_runs_knowledge_retrieval_when_planner_converges():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
    )
    retrieve = AsyncMock(return_value={
        "law_context_str": "最终方案所需法条",
        "retrieval_completed": True,
        "retrieval_fingerprint": "snapshot",
    })
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
        "src.agents.legal_guide.graph.plan_fixed_batch",
        new=AsyncMock(return_value={"should_ask": False}),
    ), patch(
        "src.agents.legal_guide.graph.plan_next_followup",
        new=planner,
    ), patch(
        "src.agents.legal_guide.graph.node_retrieve",
        new=retrieve,
    ):
        result = asyncio.run(node_assess_retrieve(state, deps))

    retrieve.assert_awaited_once()
    assert result["law_context_str"] == "最终方案所需法条"
    assert result["followup_plan"]["should_ask"] is False


def test_explicit_continue_can_use_bounded_extra_rounds_but_not_exceed_absolute_limit():
    deps = MagicMock(spec=GuideDeps)
    deps.fast_llm = None
    deps.llm = MagicMock()
    followup = {
        "should_ask": True,
        "candidate_id": "remaining_evidence",
        "planner_mode": "deterministic_policy",
    }
    base = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["故意伤害"],
        collected_facts=["用户称被他人殴打"],
        retrieval_completed=True,
        supplement_choice="continue",
        allow_extra_followups=True,
    )
    planner = AsyncMock(return_value=followup)
    patches = (
        patch(
            "src.agents.legal_guide.graph.node_score",
            new=AsyncMock(return_value={"confidence_score": 0.4, "confidence_tier": "LOW"}),
        ),
        patch(
            "src.agents.legal_guide.graph.plan_fixed_batch",
            new=AsyncMock(return_value={"should_ask": False}),
        ),
        patch(
            "src.agents.legal_guide.graph.plan_next_followup",
            new=planner,
        ),
    )
    with patches[0], patches[1]:
        within_extra = asyncio.run(node_assess_retrieve(
            base.model_copy(update={"ask_rounds": settings.GUIDE_MAX_ASK_ROUNDS}),
            deps,
        ))
        at_absolute = asyncio.run(node_assess_retrieve(
            base.model_copy(update={"ask_rounds": settings.GUIDE_MAX_OPT_IN_ASK_ROUNDS}),
            deps,
        ))

    assert within_extra["followup_plan"]["should_ask"] is True
    assert at_absolute["followup_plan"]["should_ask"] is False


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
    deps.fast_llm = None
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
            "followup_plan": {"should_ask": True},
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
            "followup_plan": {"should_ask": False},
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
    test_route_after_assess_retrieve_converges_or_asks_once()
    test_parse_details_clears_explicit_ask_type_and_saves_time()
    test_assess_retrieve_sets_force_conclude_at_total_limit()
    test_compiled_graph_has_exactly_nine_business_nodes()
    test_end_to_end_route_returns_question_then_final_plan()
    print("ALL PASS")
