"""Application-owned scoring contracts for legal-guide follow-ups."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.agents.legal_guide.followup_planner import (
    build_followup_candidates,
    plan_next_followup,
)
from src.agents.legal_guide.followup_policy import (
    rank_followup_candidates,
    score_followup_candidate,
)
from src.agents.legal_guide.state import GuideState
from src.core.config import get_settings


def _fact(key: str, category: str, statement: str) -> dict:
    return {
        "key": key,
        "category": category,
        "statement": statement,
        "status": "asserted",
        "source_text": statement,
        "turn": 1,
    }


def _llm(payload: dict) -> MagicMock:
    model = MagicMock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps(payload, ensure_ascii=False))
    )
    return model


def _proposal(**updates) -> dict:
    payload = {
        "should_ask": True,
        "ask_type": "facts",
        "decision_key": "transaction_identity",
        "candidate_id": "consumer_transaction",
        "question": "您是向哪位经营者购买了什么，大约支付了多少钱？",
        "reason": "模型提供的理由不会替代题库依据",
        "contextual_reason": "",
        "answer_hint": "金额说大概数即可",
        "decision_effects": ["procedure"],
        "acknowledgement": "",
        "acknowledged_fact_keys": [],
        "basis_kind": "official_elements",
        "law_index": -1,
        "information_gain": 0.01,
        "user_burden": 0.99,
    }
    payload.update(updates)
    return payload


def test_policy_ranks_candidates_before_the_model_is_called():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在质量问题")],
    )
    candidates, _ = build_followup_candidates(state)
    scores = rank_followup_candidates(candidates, state)

    assert scores[0].candidate_id == "consumer_transaction"
    assert scores[0].decision_effects == [
        "responsibility",
        "claim_scope",
        "jurisdiction",
    ]
    assert scores[0].eligible is True


def test_model_numeric_self_scores_are_ignored():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在质量问题")],
    )

    plan = asyncio.run(plan_next_followup(state, _llm(_proposal())))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_policy"
    assert plan["information_gain"] > 0.01
    assert plan["user_burden"] < 0.99
    assert plan["decision_effects"] == [
        "responsibility",
        "claim_scope",
        "jurisdiction",
    ]


def test_model_cannot_replace_the_policy_selected_candidate():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在质量问题")],
    )
    proposal = _proposal(
        candidate_id="consumer_problem_time",
        decision_key="problem_time",
        question="问题是什么时候发现的？",
    )

    plan = asyncio.run(plan_next_followup(state, _llm(proposal)))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "deterministic_fallback_model_changed_candidate"
    assert plan["candidate_id"] == "consumer_transaction"
    assert plan["decision_trace"]["selected_candidate_id"] == "consumer_transaction"


def test_decision_trace_records_scores_and_rejection_reasons():
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        case_facts=[_fact("event.problem", "event", "商品存在质量问题")],
    )

    plan = asyncio.run(plan_next_followup(state, _llm(_proposal())))
    trace = plan["decision_trace"]

    assert trace["mode"] == "deterministic_policy"
    assert trace["selected_candidate_id"] == "consumer_transaction"
    assert trace["candidates"]
    assert {
        "information_gain",
        "user_burden",
        "net_score",
        "eligible",
        "rejection_reasons",
    }.issubset(trace["candidates"][0])


def test_explicit_continue_restores_normal_value_threshold_after_soft_limit():
    candidate = {
        "id": "criminal_original_clues",
        "kind": "evidence",
        "decision_dimension": "original_clues",
        "coverage": {"missing": ["原始电子记录", "实物线索"]},
        "priority": 1,
    }
    passive_state = GuideState(
        ask_rounds=get_settings().GUIDE_SOFT_ASK_ROUNDS,
    )
    opted_in_state = passive_state.model_copy(update={
        "supplement_choice": "continue",
        "allow_extra_followups": True,
    })

    passive_score = score_followup_candidate(candidate, passive_state)
    opted_in_score = score_followup_candidate(candidate, opted_in_state)

    assert passive_score.net_score == opted_in_score.net_score
    assert passive_score.eligible is False
    assert "value_below_policy_threshold" in passive_score.rejection_reasons
    assert opted_in_score.eligible is True
