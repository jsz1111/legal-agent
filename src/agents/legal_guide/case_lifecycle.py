"""Semantic case-boundary decisions and isolated legal-guide case creation."""
from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any

from langchain_core.messages import SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.agents.legal_guide.case_model import format_case_context
from src.agents.legal_guide.state import GuidePhase, GuideState
from src.core.config import get_settings


class CaseRelation(str, Enum):
    """How a new user message relates to the currently loaded case."""

    CONTINUE = "continue"
    NEW = "new"
    UNCERTAIN = "uncertain"


class TurnControlIntent(str, Enum):
    """User control over the current questioning lifecycle."""

    CONCLUDE_NOW = "conclude_now"
    CONTINUE_GATHERING = "continue_gathering"
    CASE_DETAIL = "case_detail"
    OTHER = "other"


class CaseBoundaryProposal(BaseModel):
    """Untrusted semantic classification returned by the language model."""

    relation: CaseRelation = CaseRelation.UNCERTAIN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    carries_case_detail: bool = False
    control_intent: TurnControlIntent = TurnControlIntent.OTHER


class CaseBoundaryDecision(BaseModel):
    """Application-approved case-boundary decision with an audit trail."""

    relation: CaseRelation
    confidence: float
    reason: str
    carries_case_detail: bool = False
    control_intent: TurnControlIntent = TurnControlIntent.OTHER
    decision_source: str = "semantic_classifier"


_CASE_BOUNDARY_PROMPT = """你负责判断一条新消息与当前维权案件的关系。

这不是法律问题分类，也不是关键词匹配。请比较双方主体、争议事件、法律关系、
标的、时间线和用户目标之间是否存在连续性。

当前案件：
- 案件编号：{case_id}
- 法律领域：{domain}
- 已确认问题：{issues}
- 案情事实：
{case_context}
- 当前等待用户回答的问题：{pending_question}
- 当前流程状态：{phase}

待判断消息：
{message}

判断标准：
1. continue：补充、纠正、否认或确认当前案件事实；回答当前追问；请求解释当前方案；
   为当前案件补充证据；或者请求基于当前案件生成文书。
2. new：消息描述的是另一组主体、另一争议事件或另一独立法律关系，继续沿用当前
   案件事实会造成串案。仅仅出现不同法律领域不能单独作为 new 的依据。
3. uncertain：信息不足，无法可靠判断应当继承还是隔离当前案件。
4. 不得因为措辞简短、口语化或没有重复旧案件名词就直接判定 new。
5. 同时识别用户对当前追问流程的控制意图：
   - conclude_now：用户要求停止追问，直接按当前信息生成方案；
   - continue_gathering：用户明确愿意继续回答或希望系统继续追问；
   - case_detail：用户在回答问题、补充或修正实质案情；
   - other：不属于以上三类。
6. carries_case_detail 只表示消息是否含有需要写入案件的实质事实或诉求。单纯要求
   按目前情况生成不属于案情事实；如果一条消息既补充事实又要求立即生成，则
   control_intent=conclude_now 且 carries_case_detail=true。

只输出 JSON：
{{
  "relation": "continue|new|uncertain",
  "confidence": 0.0,
  "reason": "一句简短的语义判断依据",
  "carries_case_detail": true,
  "control_intent": "conclude_now|continue_gathering|case_detail|other"
}}"""


_PENDING_BOUNDARY_PROMPT = """用户上一条消息与当前案件的关系不明确，系统询问其要继续
原案件还是建立新案件。请根据用户的确认答复完成案件边界判断，不做法律分析。

当前案件摘要：
- 法律领域：{domain}
- 已确认问题：{issues}
- 案情事实：
{case_context}

上一条待归属消息：
{pending_message}

用户本次确认：
{message}

relation 只能表示“上一条待归属消息”应归入原案件还是新案件；如果用户仍未明确，
返回 uncertain。carries_case_detail 表示本次确认本身是否还增加了实质案情。
control_intent 按本次确认是否同时要求立即生成、继续追问或补充案情填写。

只输出 JSON：
{{
  "relation": "continue|new|uncertain",
  "confidence": 0.0,
  "reason": "一句简短的判断依据",
  "carries_case_detail": false,
  "control_intent": "conclude_now|continue_gathering|case_detail|other"
}}"""


def _json_payload(content: Any) -> dict[str, Any]:
    value = str(content or "").strip()
    if "```" in value:
        value = value.split("```", 2)[1].lstrip("json").strip()
    return json.loads(value)


def _case_context(state: GuideState) -> str:
    return format_case_context(state.case_facts) or "暂无结构化事实"


def _pending_question(state: GuideState) -> str:
    return "；".join(state.pending_ask_details) or "无"


async def decide_case_boundary(
    state: GuideState,
    message: str,
    llm: Any,
) -> CaseBoundaryDecision:
    """Classify a message against the active case and apply a confidence gate."""

    settings = get_settings()
    prompt = _CASE_BOUNDARY_PROMPT.format(
        case_id=state.case_id,
        domain=state.legal_domain or "尚未确定",
        issues="；".join(state.confirmed_issues) or "尚未确认",
        case_context=_case_context(state),
        pending_question=_pending_question(state),
        phase=state.phase.value,
        message=message,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        proposal = CaseBoundaryProposal.model_validate(_json_payload(response.content))
    except (AttributeError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.warning("案件边界判断失败，转人工确认式澄清 | error={}", exc)
        return CaseBoundaryDecision(
            relation=CaseRelation.UNCERTAIN,
            confidence=0.0,
            reason="语义判断不可用，不能安全继承旧案件状态",
            decision_source="safe_fallback",
        )

    relation = proposal.relation
    if proposal.confidence < settings.GUIDE_CASE_BOUNDARY_CONFIDENCE:
        relation = CaseRelation.UNCERTAIN
    return CaseBoundaryDecision(
        relation=relation,
        confidence=proposal.confidence,
        reason=proposal.reason[:300],
        carries_case_detail=proposal.carries_case_detail,
        control_intent=proposal.control_intent,
    )


async def resolve_pending_boundary(
    state: GuideState,
    message: str,
    llm: Any,
) -> CaseBoundaryDecision:
    """Resolve a previously deferred case-boundary decision semantically."""

    settings = get_settings()
    prompt = _PENDING_BOUNDARY_PROMPT.format(
        domain=state.legal_domain or "尚未确定",
        issues="；".join(state.confirmed_issues) or "尚未确认",
        case_context=_case_context(state),
        pending_message=state.pending_case_message,
        message=message,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        proposal = CaseBoundaryProposal.model_validate(_json_payload(response.content))
    except (AttributeError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.warning("案件边界确认解析失败，继续保持待确认状态 | error={}", exc)
        return CaseBoundaryDecision(
            relation=CaseRelation.UNCERTAIN,
            confidence=0.0,
            reason="用户确认尚未被可靠解析",
            decision_source="safe_fallback",
        )

    relation = proposal.relation
    if proposal.confidence < settings.GUIDE_CASE_BOUNDARY_CONFIDENCE:
        relation = CaseRelation.UNCERTAIN
    return CaseBoundaryDecision(
        relation=relation,
        confidence=proposal.confidence,
        reason=proposal.reason[:300],
        carries_case_detail=proposal.carries_case_detail,
        control_intent=proposal.control_intent,
    )


def boundary_audit_entry(
    state: GuideState,
    message: str,
    decision: CaseBoundaryDecision,
) -> dict[str, Any]:
    """Build a compact, state-persisted explanation of a boundary decision."""

    return {
        "case_id": state.case_id,
        "message_excerpt": " ".join(message.split())[:200],
        "relation": decision.relation.value,
        "confidence": round(decision.confidence, 4),
        "reason": decision.reason,
        "control_intent": decision.control_intent.value,
        "decision_source": decision.decision_source,
        "at_round": state.round,
    }


def start_isolated_case(
    previous: GuideState,
    *,
    thread_id: str,
    user_id: str | None,
    transition: dict[str, Any],
) -> GuideState:
    """Create a clean case state while preserving only non-case user context."""

    user_context = {
        "user_id": user_id,
        "long_term_memories": list(
            (previous.user_context or {}).get("long_term_memories") or []
        ),
    }
    return GuideState(
        case_id=uuid.uuid4().hex,
        case_generation=previous.case_generation + 1,
        session_id=thread_id,
        user_context=user_context,
        case_boundary_audit=[transition],
        phase=GuidePhase.CLARIFY,
    )


def boundary_confirmation_reply(state: GuideState) -> str:
    """Return a stable UX prompt without trying to infer the answer from keywords."""

    issue = "、".join(state.confirmed_issues[:2]) or "刚才的维权事项"
    return (
        f"为了避免把两个案件的事实混在一起，我需要确认一下："
        f"您这条消息是在继续“{issue}”，还是要新建一个独立案件？\n\n"
        "请直接说明它与原案件的关系；确认后我会继续处理，刚才的信息不会丢失。"
    )
