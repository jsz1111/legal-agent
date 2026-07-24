"""验证高危检测在多轮对话中每一轮都会触发（而非仅首轮）。

场景：用户先聊普通租房纠纷，第二轮才追加"对方上门殴打我"这类高危案情。
修复前 check_urgency 仅在 round==0 执行、且 dispatcher 在 round>0 时绕过该节点，
导致中途追加的高危内容不会熔断。本测试验证修复后每轮都会重跑高危检测。
"""
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import src.agents.legal_guide.graph as g
from src.agents.legal_guide.state import GuideState, GuidePhase
from src.agents.legal_guide.graph import (
    node_check_urgency, route_dispatcher, route_after_urgency,
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


def test_dispatcher_always_routes_through_urgency():
    """dispatcher 在任何轮次都要把流程导向 check_urgency（首轮经 load_context）。"""
    s0 = GuideState(round=0)
    assert route_dispatcher(s0) == "load_context"

    s1 = GuideState(round=1)
    assert route_dispatcher(s1) == "check_urgency"

    s2 = GuideState(round=3, pending_ask_details=["有无合同？"])
    assert route_dispatcher(s2) == "check_urgency"


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
    test_dispatcher_always_routes_through_urgency()
    test_route_after_urgency_branches()
    test_normal_input_not_flagged()
    print("ALL PASS")
