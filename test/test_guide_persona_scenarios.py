"""通过完整九节点图验证典型用户画像与确定性流转。"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import src.agents.legal_guide.graph as guide_graph
from src.agents.legal_guide.graph import GuideDeps, run_guide
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.core.config import get_settings


settings = get_settings()


def _merge(old: list[str], new: list[str]) -> list[str]:
    return list(dict.fromkeys(old + new))


async def _fake_urgency(state: GuideState, deps: GuideDeps) -> dict:
    last = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if "上门打我" in last:
        return {
            "urgency_level": "critical",
            "phase": GuidePhase.END,
            "messages": [AIMessage(content="请立即确保安全并拨打110，也可联系12348。")],
        }
    return {"urgency_level": "normal"}


async def _fake_extract(state: GuideState, deps: GuideDeps) -> dict:
    last = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if "说不清" in last and state.clarify_rounds < settings.GUIDE_MAX_CLARIFY_ROUNDS:
        return {"confirmed_issues": [], "unmatched_issues": [], "phase": GuidePhase.CLARIFY}

    complete = "材料都在" in last
    updates = {
        "confirmed_issues": _merge(state.confirmed_issues, ["拖欠劳动报酬"]),
        "legal_domain": "labor_social_security",
        "phase": GuidePhase.ISSUE_SEARCH,
    }
    if complete:
        updates.update({
            "collected_facts": ["拖欠3个月工资", "拖欠金额24000元", "月薪8000元"],
            "evidence_confirmed": ["劳动合同", "工资流水", "考勤记录"],
            "region": "上海",
            "time_info": "2025年4月",
        })
    return updates


async def _fake_clarify(state: GuideState, deps: GuideDeps) -> dict:
    return {
        "clarify_rounds": state.clarify_rounds + 1,
        "phase": GuidePhase.CLARIFY,
        "messages": [AIMessage(content="我先帮您慢慢理清：是谁欠了什么钱，大概多久了？")],
    }


async def _fake_assess(state: GuideState, deps: GuideDeps) -> dict:
    is_complete = len(state.evidence_confirmed) >= 2 and len(state.collected_facts) >= 2
    force = (
        state.force_conclude
        or state.total_rounds >= settings.GUIDE_MAX_TOTAL_ROUNDS
        or state.ask_rounds >= settings.GUIDE_MAX_ASK_ROUNDS
    )
    return {
        "confidence_score": 0.85 if is_complete else 0.25,
        "confidence_tier": "HIGH" if is_complete else "LOW",
        "law_context_str": "RETRIEVED_LAW：《劳动合同法》第三十条：应及时足额支付劳动报酬",
        "case_context_str": "RETRIEVED_CASE：欠薪案件通常需证明劳动关系及欠薪金额",
        "relevant_channels": [{"name": "劳动保障监察", "phone": "12333"}],
        "last_confirmed_count": len(state.confirmed_issues),
        "force_conclude": force,
        "followup_plan": {
            "should_ask": not is_complete and not force,
            "ask_type": "facts",
            "decision_key": "wage_duration_and_amount",
            "candidate_id": "",
            "question": "公司拖欠工资大约多久、金额多少？",
            "reason": "确认请求范围",
            "information_gain": 0.8,
            "user_burden": 0.2,
        },
    }


async def _fake_ask(state: GuideState, deps: GuideDeps) -> dict:
    plan = state.followup_plan
    if plan.get("should_ask"):
        item = plan["question"]
        return {
            "phase": GuidePhase.DETAIL_GATHER,
            "ask_rounds": state.ask_rounds + 1,
            "facts_rounds": state.facts_rounds + 1,
            "asked_details": state.asked_details + [item],
            "pending_ask_details": [item],
            "pending_ask_type": plan.get("ask_type", "facts"),
            "messages": [AIMessage(content=(
                f"相关法律主要看具体事实。您还记得{item}吗？\n"
                "如果不方便补充，直接回复“现在生成方案”。"
            ))],
        }
    return {}


async def _fake_parse(state: GuideState, deps: GuideDeps) -> dict:
    last = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    pending = state.pending_ask_details
    unavailable = state.evidence_unavailable
    facts = state.collected_facts
    evidence = state.evidence_confirmed
    low_info = state.consecutive_low_info_answers
    if "没有" in last:
        unavailable = _merge(unavailable, pending)
        facts = _merge(facts, [f"用户无法提供：{pending[0]}"] if pending else [])
        low_info += 1
    if "图片证据" in last:
        evidence = _merge(evidence, ["工资转账截图"])
    return {
        "pending_ask_details": [],
        "pending_ask_type": "",
        "evidence_unavailable": unavailable,
        "evidence_confirmed": evidence,
        "collected_facts": facts,
        "consecutive_low_info_answers": low_info,
        "force_conclude": low_info >= settings.GUIDE_MAX_LOW_INFO_ANSWERS,
        "phase": GuidePhase.ISSUE_SEARCH,
    }


async def _fake_conclude(state: GuideState, deps: GuideDeps) -> dict:
    reply = (
        "**【法律依据】** RETRIEVED_LAW\n"
        "**【维权路径比较】** 可先投诉，再申请劳动仲裁。\n"
        "**【维权胜算评估】** 综合胜算：中等。现有证据越完整越有利。\n"
        "**【行动清单】** 保存现有材料，拨打12333咨询。"
    )
    return {"phase": GuidePhase.CONCLUDE, "messages": [AIMessage(content=reply)]}


@contextmanager
def _scripted_graph():
    patchers = [
        patch.object(guide_graph, "node_load_context", new=AsyncMock(return_value={})),
        patch.object(guide_graph, "node_check_urgency", new=AsyncMock(side_effect=_fake_urgency)),
        patch.object(guide_graph, "node_extract_issues", new=AsyncMock(side_effect=_fake_extract)),
        patch.object(guide_graph, "node_clarify", new=AsyncMock(side_effect=_fake_clarify)),
        patch.object(guide_graph, "node_assess_retrieve", new=AsyncMock(side_effect=_fake_assess)),
        patch.object(guide_graph, "node_ask_followup", new=AsyncMock(side_effect=_fake_ask)),
        patch.object(guide_graph, "node_parse_details", new=AsyncMock(side_effect=_fake_parse)),
        patch.object(guide_graph, "node_conclude", new=AsyncMock(side_effect=_fake_conclude)),
        patch.object(
            guide_graph,
            "node_save_record",
            new=AsyncMock(return_value={"phase": GuidePhase.END}),
        ),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        yield
    finally:
        for patcher in reversed(patchers):
            patcher.stop()


def _run_messages(messages: list[str]) -> tuple[list[dict], GuideState]:
    deps = MagicMock(spec=GuideDeps)
    state = None
    trace: list[dict] = []
    for message in messages:
        reply, state = asyncio.run(
            run_guide(message, "persona:user", deps, existing_state=state)
        )
        trace.append({
            "reply": reply,
            "phase": state.phase,
            "round": state.round,
            "tier": state.confidence_tier,
            "facts": list(state.collected_facts),
            "unavailable": list(state.evidence_unavailable),
        })
        if state.phase == GuidePhase.END:
            break
    return trace, state


def test_elderly_unclear_and_evidence_poor_user_converges_by_ninth_turn():
    messages = ["老板那个钱，我说不清"] * 2 + ["还是说不清"] + ["没有，我都没有"] * 8
    with _scripted_graph():
        trace, state = _run_messages(messages)

    assert trace[0]["phase"] == GuidePhase.CLARIFY
    assert trace[1]["phase"] == GuidePhase.CLARIFY
    assert state.phase == GuidePhase.END
    assert state.round <= 9
    assert 0 < state.ask_rounds <= settings.GUIDE_MAX_ASK_ROUNDS
    assert state.evidence_unavailable
    assert "维权胜算评估" in trace[-1]["reply"]
    assert all(item["reply"] for item in trace)


def test_knowledgeable_adult_gets_grounded_plan_without_an_extra_menu():
    message = "公司拖欠3个月工资24000元，上海，劳动合同、流水、考勤材料都在"
    with _scripted_graph():
        trace, state = _run_messages([message])

    assert state.phase == GuidePhase.END
    assert state.round == 1
    assert state.confidence_tier == "HIGH"
    assert "RETRIEVED_LAW" in trace[-1]["reply"]
    assert "维权胜算评估" in trace[-1]["reply"]
    assert "继续补充" not in trace[0]["reply"]


def test_long_dialogue_can_continue_beyond_soft_limit_and_stop_at_will():
    messages = [
        "公司拖欠工资",
        "2024年入职",
        "每月8000元",
        "一共欠24000元",
        "【图片证据补充】这是工资转账截图，属于图片证据",
        "现在生成方案",
    ]
    with _scripted_graph():
        trace, state = _run_messages(messages)

    assert state.phase == GuidePhase.END
    assert state.ask_rounds > settings.GUIDE_SOFT_ASK_ROUNDS
    assert state.ask_rounds <= settings.GUIDE_MAX_ASK_ROUNDS
    assert "维权胜算评估" in trace[-1]["reply"]
    assert all("继续补充" not in item["reply"] for item in trace)
    assert all(item["reply"] for item in trace)


def test_multimodal_evidence_is_accumulated_before_next_retrieval():
    with _scripted_graph():
        trace, state = _run_messages([
            "公司拖欠工资",
            "【图片证据补充】这是银行工资转账的截图，属于图片证据",
        ])

    assert "工资转账截图" in state.evidence_confirmed
    assert state.round == 2
    assert trace[1]["reply"]


def test_later_round_violence_interrupts_normal_evidence_flow():
    with _scripted_graph():
        trace, state = _run_messages(["公司拖欠工资", "老板现在上门打我"])

    assert state.phase == GuidePhase.END
    assert state.urgency_level == "critical"
    assert "110" in trace[-1]["reply"]
    assert state.round == 2


def test_same_scenario_has_deterministic_state_transition_trace():
    messages = ["公司拖欠工资", "没有，我都没有", "没有，我都没有"]
    with _scripted_graph():
        first_trace, _ = _run_messages(messages)
    with _scripted_graph():
        second_trace, _ = _run_messages(messages)

    first = [(x["phase"], x["round"], x["tier"], x["facts"], x["unavailable"]) for x in first_trace]
    second = [(x["phase"], x["round"], x["tier"], x["facts"], x["unavailable"]) for x in second_trace]
    assert first == second
