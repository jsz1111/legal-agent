"""Context continuity and source-grounding regressions."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.legal_guide.graph import (
    _active_long_term_memories,
    _ensure_contextual_understanding,
    _ensure_grounded_legal_basis,
    _sanitize_statute_citations,
)
from src.agents.legal_guide.state import GuideState


def _atom(key: str, statement: str, source: str, *, status: str = "asserted") -> dict:
    return {
        "key": key,
        "category": "event",
        "statement": statement,
        "status": status,
        "operation": "add",
        "source_text": source,
        "turn": 1,
        "verification": "user_stated",
    }


def test_final_understanding_uses_generic_case_atoms_instead_of_scene_templates():
    state = GuideState(case_facts=[
        _atom("transaction.total", "用户称最初支付700元", "一共充值了700元"),
        _atom("counterparty.response", "用户要求退款后被经营者拉黑", "要求了，对方拉黑了我"),
        _atom(
            "other_people.report",
            "用户听说其他人也有类似损失",
            "听说还有很多人也有损失",
            status="uncertain",
        ),
    ])
    reply = (
        "**【理解您的情况】**\n旧的通用开场。\n\n"
        "**【法律依据】**\n已检索。"
    )

    contextual = _ensure_contextual_understanding(reply, state)

    assert "您提到最初支付700元" in contextual
    assert "您要求退款后被经营者拉黑" in contextual
    assert "仍需核对：您听说其他人也有类似损失" in contextual
    assert "旧的通用开场" not in contextual


def test_grounded_legal_basis_is_restored_after_unsupported_lines_are_removed():
    context = (
        "## 核心法条（高度相关，优先引用）\n\n---\n\n"
        "法条1【中华人民共和国消费者权益保护法实施条例 第二十二条】\n"
        "经营者以收取预付款方式提供商品或者服务的，应当与消费者订立书面合同。"
        "经营者未按照约定提供商品或者服务的，应当按照消费者的要求履行约定或者退还预付款。"
    )
    reply = (
        "**【理解您的情况】**\n预付款余额无法使用。\n\n"
        "**【法律依据】**\n《中华人民共和国刑法》第二百六十六条：涉嫌诈骗。\n\n"
        "**【维权路径比较】**\n先核对受理渠道。"
    )

    sanitized = _sanitize_statute_citations(reply, context)
    restored = _ensure_grounded_legal_basis(sanitized, context)

    assert "《中华人民共和国消费者权益保护法实施条例》第二十二条" in restored
    assert "经营者未按照约定提供商品或者服务" in restored
    assert "《中华人民共和国刑法》" not in restored


def test_grounded_law_rendering_preserves_retrieval_order_without_scene_priorities():
    context = "\n\n---\n\n".join([
        "法条1【甲法 第一条】\n第一条检索原文。",
        "法条2【乙法 第二条】\n第二条检索原文。",
        "法条3【丙法 第三条】\n第三条检索原文。",
        "法条4【丁法 第四条】\n第四条检索原文。",
    ])
    reply = "**【法律依据】**\n模型自由改写。\n\n**【维权路径比较】**\n待核对。"

    restored = _ensure_grounded_legal_basis(reply, context)

    assert restored.index("《甲法》第一条") < restored.index("《乙法》第二条")
    assert "《丙法》第三条" not in restored
    assert "《丁法》第四条" not in restored


def test_grounded_legal_basis_includes_cited_laws_beyond_first_two():
    context = "\n\n---\n\n".join([
        "法条1【甲法 第一条】\n第一条检索原文。",
        "法条2【乙法 第二条】\n第二条检索原文。",
        "法条3【丙法 第三条】\n第三条检索原文。",
        "法条4【丁法 第四条】\n第四条检索原文。",
    ])
    reply = (
        "**【法律依据】**\n模型先写的一段。\n\n"
        "**【核心争点分析】**\n根据《丙法》第三条，需要继续分析。"
    )

    restored = _ensure_grounded_legal_basis(reply, context)

    assert "《丙法》第三条" in restored
    assert "《甲法》第一条" in restored
    assert "模型先写的一段" not in restored


def test_long_term_case_memory_is_available_without_explicit_recall():
    memory = "法律咨询摘要：案情事实：支付700元；经营者没有回应"
    new_case = GuideState(
        messages=[HumanMessage(content="我今天遇到另一件事")],
        user_context={"long_term_memories": [memory]},
    )
    recall = new_case.model_copy(update={
        "messages": [HumanMessage(content="我之前说的纠纷是什么情况？")],
    })

    assert _active_long_term_memories(new_case) == [memory]
    assert _active_long_term_memories(recall) == [memory]
