"""结论方案【优势与劣势】：无胜算等级，只列有利/不利因素。

用户决定不再输出"综合胜算：较高/中等/较低"等级。验证：
- CONCLUDE_PROMPT 与三档置信度引导语都不再出现胜算等级/百分比指令；
- _ensure_pros_cons 在 LLM 漏写不利因素时用既有结构化数据补齐（不重复注入）；
- _ensure_required_plan_sections 兜底渲染【优势与劣势】。
"""
from __future__ import annotations

from src.agents.legal_guide.confidence import tier_guidance
from src.agents.legal_guide.graph import (
    _ensure_pros_cons,
    _ensure_required_plan_sections,
)
from src.agents.legal_guide.prompts import CONCLUDE_PROMPT
from src.agents.legal_guide.state import GuideState


def _state(**kw: object) -> GuideState:
    defaults: dict = dict(
        legal_domain="traffic_personal_injury",
        adverse_facts=["仅凭车牌号无法直接锁定行为人，车牌可能套牌或非本人驾驶"],
        evidence_unavailable=["现场监控录像", "伤情鉴定报告"],
    )
    defaults.update(kw)
    return GuideState(**defaults)


def _reply_with_pros_cons(only_pros: bool = True) -> str:
    if only_pros:
        return (
            "**【优势与劣势】**\n"
            "**有利因素**：《道路交通安全法》相关条文支持您主张赔偿，"
            "您已提供医疗费票据。\n\n"
            "**【行动清单】**\n"
            "□ 立即保存的证据：医疗费票据。"
        )
    return (
        "**【优势与劣势】**\n"
        "**有利因素**：《道路交通安全法》相关条文支持您主张赔偿。\n"
        "**不利因素**：仅凭车牌号可能无法锁定行为人。\n\n"
        "**【行动清单】**\n"
        "□ 立即保存的证据：医疗费票据。"
    )


def test_conclude_prompt_has_no_win_verdict():
    """CONCLUDE_PROMPT 不再要求"综合胜算"等级，且含【优势与劣势】规则。"""
    assert "综合胜算" not in CONCLUDE_PROMPT
    assert "维权胜算评估" not in CONCLUDE_PROMPT
    assert "【优势与劣势】" in CONCLUDE_PROMPT
    assert "不估计胜算等级" in CONCLUDE_PROMPT


def test_tier_guidance_has_no_win_verdict():
    """三档置信度引导语不再要求胜算等级，仍保留置信度档位作用。"""
    guidances = [tier_guidance(tier) for tier in ("HIGH", "MEDIUM", "LOW")]
    for guidance in guidances:
        assert "综合胜算" not in guidance
        assert "维权胜算评估" not in guidance
        assert "【优势与劣势】" in guidance
    assert len(set(guidances)) == 3


def test_ensure_pros_cons_injects_missing_cons():
    """【优势与劣势】只有有利因素时，用 adverse_facts + evidence_unavailable 补齐。"""
    state = _state()
    reply = _ensure_pros_cons(_reply_with_pros_cons(only_pros=True), state)

    assert "**不利因素**" in reply
    assert "仅凭车牌号无法直接锁定行为人" in reply
    assert "缺少「现场监控录像」" in reply
    assert "缺少「伤情鉴定报告」" in reply


def test_ensure_pros_cons_keeps_existing_cons():
    """已含"不利"内容时不再重复注入。"""
    state = _state()
    original = _reply_with_pros_cons(only_pros=False)
    assert _ensure_pros_cons(original, state) == original


def test_ensure_pros_cons_noop_without_section():
    """回复中没有【优势与劣势】段时不注入（交给 _ensure_required_plan_sections 兜底）。"""
    reply = "**【法律依据】**\n《刑法》第234条。\n\n**【行动清单】**\n□ 保存证据。"
    state = _state()
    assert _ensure_pros_cons(reply, state) == reply


def test_ensure_pros_cons_injects_at_section_end_when_no_following_heading():
    """【优势与劣势】位于回复末尾时，不利因素追加到段尾而非丢失。"""
    reply = "**【优势与劣势】**\n**有利因素**：有医疗费票据。"
    state = _state(adverse_facts=["对方否认殴打"], evidence_unavailable=[])
    out = _ensure_pros_cons(reply, state)

    assert "**不利因素**" in out
    assert out.rstrip().endswith("对方否认殴打")


def test_ensure_required_plan_sections_fallback_renders_pros_cons():
    """方案整体漏掉【优势与劣势】时，兜底渲染有利/不利因素。"""
    reply = "**【法律依据】**\n《刑法》第234条。\n\n**【行动清单】**\n□ 保存证据。"
    state = _state()
    out = _ensure_required_plan_sections(reply, state)

    assert "【优势与劣势】" in out
    assert "**有利因素**" in out
    assert "**不利因素**" in out
    assert "仅凭车牌号无法直接锁定行为人" in out
