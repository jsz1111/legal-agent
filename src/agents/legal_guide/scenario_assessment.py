"""Re-evaluate the dominant legal scenario after follow-up and retrieval."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.legal_guide.case_model import format_case_context
from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage
from src.agents.legal_guide.prompts import DOMAIN_LABELS
from src.core.config import get_settings


SCENARIO_ASSESSMENT_PROMPT = """你是法律维权工作流中的场景再判断器。

你的任务不是重新整理事实，而是基于完整案情判断“用户当前情况最接近哪一类实际场景”。
场景用日常语言表达，不要直接用法律领域代码当用户选项。

结构化案情（必须只使用这些已陈述内容，不得补写、推测或认定违法责任）：
{case_context}

当前法律问题：
{issues}

当前领域：
{domain}

用户称持有的材料：
{evidence}

上一次场景判断（供参考，本轮以新事实为准）：
{previous_analysis}

规则：
1. 使用全部已积累事实，而不只看最后一句话；若新事实足以改变判断，允许调整 primary_domain。
2. confidence 只表示“对 primary_scenario 是当前最可能场景”的信心，不表示信息已经充分，
   也不表示证据充分或结论确定。
3. 只有当两个及以上实际场景都明显可能时，才填写 competing_scenarios 和 discriminating_facts。
4. discriminating_facts 只写用户尚未说明、且能区分竞争场景的具体事实；没有则留空数组。
5. confirmation_question 和 confirmation_options 必须用用户能理解的日常情境，
   不得把“诈骗/合同纠纷/侵权”等法律结论直接当选项；场景明确时 question 返回“无需确认”，options 返回空数组。
6. 不得把用户未陈述的内容写成已确认事实；不得仅因领域代码相似就改动 primary_domain。

只输出 JSON：
{{
  "primary_scenario": "最接近的日常场景描述",
  "primary_domain": "领域代码",
  "primary_frame": "case_frame 代码",
  "confidence": 0.0,
  "competing_scenarios": ["竞争场景的日常描述"],
  "discriminating_facts": ["仍能区分场景的关键事实"],
  "confirmation_question": "是否需要用户确认时的问题，否则为“无需确认”",
  "confirmation_options": ["日常情境选项"],
  "reason": "一句为什么这样判断"
}}
"""


class ScenarioAssessment(BaseModel):
    primary_scenario: str = ""
    primary_domain: str = ""
    primary_frame: str = "other"
    confidence: float = 0.0
    competing_scenarios: list[str] = Field(default_factory=list)
    discriminating_facts: list[str] = Field(default_factory=list)
    confirmation_question: str = ""
    confirmation_options: list[str] = Field(default_factory=list)
    reason: str = ""


def _fallback_scenario(state: Any, previous: dict | None = None) -> ScenarioAssessment:
    domain = str(getattr(state, "legal_domain", "") or "other")
    frame = str(getattr(state, "case_frame", "") or "other")
    if previous and previous.get("primary_domain") == domain:
        try:
            return ScenarioAssessment.model_validate(previous)
        except Exception:
            pass
    return ScenarioAssessment(
        primary_scenario=DOMAIN_LABELS.get(domain, domain or "其他"),
        primary_domain=domain,
        primary_frame=frame,
        confidence=0.5,
        reason="模型场景判断不可用，暂时沿用当前领域；继续由决策充分度决定是否需要补充事实。",
    )


async def assess_scenario(state: Any, llm: Any) -> ScenarioAssessment:
    """Run a model re-judgement over the complete current case snapshot."""
    previous = getattr(state, "scenario_analysis", None) or {}
    prompt = SCENARIO_ASSESSMENT_PROMPT.format(
        case_context=format_case_context(getattr(state, "case_facts", []) or [])
        or "（暂无结构化事实）",
        issues="、".join(getattr(state, "confirmed_issues", []) or []) or "未明确",
        domain=getattr(state, "legal_domain", "") or "other",
        evidence="、".join(getattr(state, "evidence_confirmed", []) or []) or "无",
        previous_analysis=json.dumps(previous, ensure_ascii=False),
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=900),
            [SystemMessage(content=prompt)],
            timeout=get_settings().GUIDE_LLM_TIMEOUT_FOLLOWUP,
            stage="scenario_assessment",
        )
        content = str(response.content or "").strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        data = json.loads(content)
        return ScenarioAssessment.model_validate(data)
    except Exception as exc:
        logger.warning("场景再判断失败，沿用当前领域: {}", exc)
        return _fallback_scenario(state, previous)
