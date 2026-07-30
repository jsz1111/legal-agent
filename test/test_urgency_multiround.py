"""验证高危检测在多轮对话中每一轮都会触发（而非仅首轮）。

场景：用户先聊普通租房纠纷，第二轮才追加"对方上门殴打我"这类高危案情。
修复前 check_urgency 仅在 round==0 执行、且旧分发节点在后续轮绕过该节点，
导致中途追加的高危内容不会熔断。本测试验证修复后每轮都会重跑高危检测。
"""
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import src.agents.legal_guide.graph as g
from src.agents.legal_guide.state import GuideState, GuidePhase
from src.agents.legal_guide.graph import (
    node_check_urgency, node_prepare_turn, route_after_urgency,
    URGENCY_CRITICAL_RESPONSE, GuideDeps,
)
from langgraph.graph import END


def _make_deps(urgency_json: dict) -> GuideDeps:
    """构造一个 LLM 只返回指定 urgency JSON 的 deps。"""
    deps = MagicMock(spec=GuideDeps)
    resp = AIMessage(content=json.dumps(urgency_json, ensure_ascii=False))
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(return_value=resp)
    return deps


def test_check_urgency_runs_on_later_rounds():
    """核心回归：round>0 时 check_urgency 仍会调用 LLM 并能返回 CRITICAL。"""
    deps = _make_deps({"urgency": "CRITICAL", "time_clue": ""})
    state = GuideState(
        session_id="u1:s1",
        round=2,  # 非首轮
        legal_domain="housing",
        confirmed_issues=["房东拖延退还押金"],
        messages=[HumanMessage(content="对方今天上门把我打伤了")],
    )
    result = asyncio.run(node_check_urgency(state, deps))
    assert deps.llm.ainvoke.await_count == 1, "非首轮必须实际执行高危检测"
    assert result["urgency_level"] == "critical"
    assert result["phase"] == GuidePhase.END
    assert result["messages"][0].content == URGENCY_CRITICAL_RESPONSE


def test_past_domestic_violence_with_explicit_current_safety_continues_guidance():
    deps = _make_deps({"urgency": "CRITICAL", "time_clue": ""})
    state = GuideState(
        messages=[HumanMessage(content="丈夫打过我，但今天暂时安全，我有报警记录")],
    )

    result = asyncio.run(node_check_urgency(state, deps))

    assert result["urgency_level"] == "normal"
    assert result["safety_relevant"] is True
    assert result["current_safety_status"] == "safe"


def test_current_danger_overrides_safety_phrase():
    deps = _make_deps({"urgency": "CRITICAL", "time_clue": ""})
    state = GuideState(
        messages=[HumanMessage(content="刚才暂时安全，但他现在拿刀就在门外")],
    )

    result = asyncio.run(node_check_urgency(state, deps))

    assert result["urgency_level"] == "critical"
    assert result["phase"] == GuidePhase.END


def test_later_evidence_detail_inherits_recent_explicit_safety():
    deps = _make_deps({
        "urgency": "CRITICAL",
        "current_danger": False,
        "time_clue": "",
    })
    state = GuideState(
        round=2,
        messages=[
            HumanMessage(content="我昨天受伤了，但现在已经安全"),
            AIMessage(content="请说明是否认识对方"),
            HumanMessage(content="医院记录有，对方我不认识"),
        ],
    )

    result = asyncio.run(node_check_urgency(state, deps))

    assert result["urgency_level"] == "normal"
    assert result["safety_relevant"] is True
    assert result["current_safety_status"] == "safe"


def test_past_violence_without_current_status_is_marked_unknown_not_critical():
    deps = _make_deps({
        "urgency": "NORMAL",
        "safety_relevant": True,
        "safety_status": "unknown",
        "time_clue": "",
    })
    state = GuideState(messages=[HumanMessage(content="我被人打了")])

    result = asyncio.run(node_check_urgency(state, deps))

    assert result["urgency_level"] == "normal"
    assert result["safety_relevant"] is True
    assert result["current_safety_status"] == "unknown"


def test_prepare_turn_increments_user_round_once():
    """每条用户消息只由 prepare_turn 推进一次轮次。"""
    deps = MagicMock(spec=GuideDeps)
    state = GuideState(round=2, total_rounds=2)
    result = asyncio.run(node_prepare_turn(state, deps))
    assert result["round"] == 3
    assert result["total_rounds"] == 3


def test_route_after_urgency_branches():
    """check_urgency 之后：END(熔断) / parse_details(有待解析追问) / extract_issues。"""
    critical = GuideState(phase=GuidePhase.END)
    assert route_after_urgency(critical) == END

    waiting = GuideState(round=2, pending_ask_details=["有无合同？"])
    assert route_after_urgency(waiting) == "parse_details"

    normal = GuideState(round=2)
    assert route_after_urgency(normal) == "extract_issues"


def test_normal_input_not_flagged():
    """普通纠纷不应被误判为 critical。"""
    deps = _make_deps({"urgency": "NORMAL", "time_clue": ""})
    state = GuideState(
        session_id="u1:s1", round=1,
        messages=[HumanMessage(content="房东一直不退押金")],
    )
    result = asyncio.run(node_check_urgency(state, deps))
    assert result.get("urgency_level") == "normal"
    assert result.get("phase") != GuidePhase.END


if __name__ == "__main__":
    test_check_urgency_runs_on_later_rounds()
    test_prepare_turn_increments_user_round_once()
    test_route_after_urgency_branches()
    test_normal_input_not_flagged()
    print("ALL PASS")
