"""节点四：事实决策、批量追问和事实阶段收敛。

这个模块只读取节点三的事实黑板。事实写入仍由 ``update_facts`` 负责，
正式证据清单和证据效力评估分别由后续节点负责。为了支持渐进迁移，
这里保留了旧追问目录和旧状态字段的投影，但不再依赖旧的单题流程。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.legal_guide.case_model import active_case_facts
from src.agents.legal_guide.retrieval_query import build_case_retrieval_inputs
from src.agents.legal_guide.state import GuidePhase
from src.agents.legal_guide.followup_catalog import (
    FactFollowup,
    fact_rule_resolved,
    get_domain_followups,
)
from src.agents.legal_guide.followup_planner import (
    build_followup_candidates,
    candidate_coverage,
)
from src.agents.legal_guide.followup_policy import (
    DECISION_EFFECT_WEIGHTS,
    SLOT_DECISION_EFFECTS,
)
from src.core.config import get_settings


_CHINA_TZ = timezone(timedelta(hours=8))
_FACT_STATUSES = {
    "confirmed",
    "denied",
    "unknown",
    "unclear",
    "conflicted",
    "superseded",
    "not_provided",
    # legacy case_facts projection
    "asserted",
    "uncertain",
}
_RESOLVED_STATUSES = {"confirmed", "denied", "unknown", "asserted"}
_IGNORED_STATUSES = {"superseded", "not_provided"}
_UNCLEAR_STATUSES = {"unclear", "uncertain", "ambiguous"}
_CONFLICT_STATUSES = {"conflicted"}

EFFECT_LABELS = {
    "responsibility": "责任主体与责任范围",
    "claim_scope": "请求类型、范围或金额",
    "limitation": "期限与关键时间节点",
    "jurisdiction": "受理机构与管辖",
    "procedure": "下一步程序路径",
    "harm_loss": "损失与影响",
    "conflict_resolution": "重大事实冲突",
    "safety": "当前安全状态",
}

_BLOCKING_EFFECTS = {
    "responsibility",
    "claim_scope",
    "limitation",
    "conflict_resolution",
    "safety",
}

_SLOT_KEY_MAP = {
    "current_safety": "safety.current_status",
    "event_time": "event.timeline",
    "legal_relationship": "relationship.type",
    "transaction": "transaction.subject_and_counterparty",
    "claim": "claim.scope",
    "procedure": "procedure.history",
    "event": "event.core_behavior",
    "harm": "harm.loss",
    "administrative_action": "procedure.administrative_action",
    "event_and_liability": "event.liability",
    "insurance_and_claim": "claim.insurance_scope",
    "agreement": "agreement.terms",
    "right_type": "right.type",
    "infringement": "infringement.behavior",
    "source_and_harm": "harm.source",
    "property_and_safety": "property.safety",
    "children": "family.children",
}

_TOPIC_LABELS = {
    "actor": "双方和主体",
    "counterparty": "双方和主体",
    "relationship": "双方和主体",
    "transaction": "交易和标的",
    "subject_matter": "交易和标的",
    "event": "经过和约定",
    "time": "时间、金额和地点",
    "amount": "时间、金额和地点",
    "location": "时间、金额和地点",
    "claim": "诉求和损失",
    "harm": "诉求和损失",
    "procedure": "处理经过",
    "safety": "安全状态",
}

_SLOT_DIMENSIONS = {
    "current_safety": {"safety"},
    "event_time": {"time"},
    "legal_relationship": {"relationship", "counterparty", "actor"},
    "transaction": {"transaction", "amount", "counterparty", "relationship", "location"},
    "claim": {"claim"},
    "procedure": {"procedure"},
    "event": {"event"},
    "harm": {"harm"},
    "administrative_action": {"event", "procedure"},
    "event_and_liability": {"event", "location", "time"},
    "insurance_and_claim": {"relationship", "claim"},
    "agreement": {"relationship"},
    "right_type": {"relationship", "subject_matter"},
    "infringement": {"event", "location", "time"},
    "source_and_harm": {"actor", "harm"},
    "property_and_safety": {"event", "harm"},
    "children": {"relationship"},
}

_SLOT_REQUIREMENTS = {
    # Each tuple is an OR-group; every group needs one known dimension.
    "current_safety": (("safety",),),
    "event_time": (("event", "harm"), ("time",)),
    "legal_relationship": (("event",), ("relationship", "counterparty", "actor")),
    "transaction": (
        ("amount",),
        ("counterparty", "relationship"),
        ("subject_matter", "transaction"),
    ),
    "claim": (("claim",),),
    "procedure": (("procedure",),),
    "event": (("event",),),
    "harm": (("harm",),),
    "administrative_action": (("event",), ("procedure",)),
    "event_and_liability": (("event",), ("time", "location")),
    "insurance_and_claim": (("relationship", "counterparty"), ("claim",)),
    "agreement": (("relationship",),),
    "right_type": (("relationship", "subject_matter"),),
    "infringement": (("event",), ("time", "location")),
    "source_and_harm": (("actor", "counterparty"), ("harm",)),
    "property_and_safety": (("event", "harm"),),
    "children": (("relationship",),),
}

_DIMENSION_QUESTIONS = {
    "actor": "这件事主要涉及哪些人或单位？",
    "counterparty": "对方是个人、店铺、公司还是其他机构？",
    "relationship": "您和对方是什么关系？",
    "transaction": "具体购买或约定的商品、服务是什么？",
    "subject_matter": "争议涉及的商品、服务或其他标的是什么？",
    "event": "对方具体做了什么，争议结果是什么？",
    "time": "事情或关键问题大约是什么时候发生或发现的？",
    "amount": "涉及金额是多少，已经支付或损失了多少？",
    "location": "事情发生在哪里，或者是通过哪个平台办理的？",
    "claim": "您现在最希望对方或有关机构怎么处理？",
    "procedure": "您之前是否已经联系、投诉、报警或申请处理，结果怎样？",
    "harm": "这件事目前造成了哪些实际损失或影响？",
    "safety": "您现在是否已经处于安全位置？",
}

_FACT_EFFECT_BY_CATEGORY = {
    "actor": {"responsibility"},
    "relationship": {"responsibility", "jurisdiction"},
    "event": {"responsibility", "procedure"},
    "claim": {"claim_scope"},
    "amount": {"claim_scope", "limitation"},
    "time": {"limitation", "procedure"},
    "location": {"jurisdiction"},
    "procedure": {"procedure", "jurisdiction"},
    "harm": {"claim_scope", "harm_loss"},
    "uncertainty": set(),
}

_FACT_DIMENSIONS_BY_CATEGORY = {
    "actor": {"actor", "counterparty"},
    "relationship": {"relationship", "counterparty"},
    "event": {"event"},
    "claim": {"claim"},
    "amount": {"amount"},
    "time": {"time"},
    "location": {"location"},
    "evidence": {"evidence"},
    "procedure": {"procedure"},
    "harm": {"harm"},
    "uncertainty": set(),
}

_MATERIAL_TOKENS = (
    "actor", "counterparty", "relationship", "transaction", "payment", "paid",
    "amount", "price", "delivery", "deliver", "breach", "non_delivery",
    "problem", "claim", "request", "time", "date", "deadline", "location",
    "jurisdiction", "procedure", "complaint", "report", "harm", "loss",
    "safety", "liability", "agreement", "contract",
)


class FactQuestionCandidate(BaseModel):
    question_id: str
    decision_key: str
    fact_slot_keys: list[str] = Field(default_factory=list)
    question_type: str = "missing_fact"
    topic: str = "events_and_agreements"
    prompt: str
    answer_hint: str = "不清楚时可以直接写“不清楚”，没有时写“没有”。"
    decision_effects: list[str] = Field(default_factory=list)
    information_gain: float = 0.0
    user_burden: float = 0.0
    priority_score: float = 0.0
    source_rule_ids: list[str] = Field(default_factory=list)
    basis_refs: list[dict[str, Any]] = Field(default_factory=list)
    contextual_reason: str = ""
    unknown_allowed: bool = True
    reopened_reason: str = ""


class FactQuestionBatch(BaseModel):
    batch_id: str
    questions: list[FactQuestionCandidate] = Field(default_factory=list)
    markdown: str = ""
    context_summary: list[str] = Field(default_factory=list)
    retrieval_basis: list[dict[str, Any]] = Field(default_factory=list)
    fact_blackboard_version: int = 0
    created_at: str = ""


class FactSufficiencyReport(BaseModel):
    status: str = "insufficient"
    can_proceed_conditionally: bool = False
    blocking_gaps: list[str] = Field(default_factory=list)
    advisory_gaps: list[str] = Field(default_factory=list)
    conflict_groups: list[str] = Field(default_factory=list)
    dimensions: list[dict] = Field(default_factory=list)
    reason: str = ""


class InternalEvidenceRequirement(BaseModel):
    requirement_id: str
    proof_target_candidate_id: str
    label: str
    dependent_fact_keys: list[str] = Field(default_factory=list)
    recommended_material_classes: list[str] = Field(default_factory=list)
    alternative_material_classes: list[str] = Field(default_factory=list)
    provisional_importance: str = "candidate"
    status: str = "active_candidate"
    generation_round: int = 0
    last_updated_round: int = 0
    change_reason: str = "carried_forward"
    source_type: str = "fact_rule"
    basis_candidate_refs: list[str] = Field(default_factory=list)
    retrieval_trace_id: str = ""
    matched_evidence_name_ids: list[str] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(_CHINA_TZ).isoformat()


def _stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def _compact(value: object, limit: int = 360) -> str:
    return " ".join(str(value or "").split())[:limit]


def _fact_status(item: dict[str, Any]) -> str:
    raw = str(item.get("status") or item.get("fact_status") or "unclear").strip()
    if raw == "asserted":
        return "confirmed"
    if raw in {"uncertain", "ambiguous"}:
        return "unclear"
    return raw if raw in _FACT_STATUSES else "unclear"


def _fact_rows(state: Any) -> list[dict[str, Any]]:
    """Prefer the canonical blackboard, while accepting legacy test states."""

    source = getattr(state, "fact_blackboard", None) or getattr(state, "case_facts", None) or []
    rows: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict) or _fact_status(item) in _IGNORED_STATUSES:
            continue
        row = dict(item)
        row["status"] = _fact_status(row)
        row["key"] = str(row.get("semantic_key") or row.get("key") or "")
        row["statement"] = _compact(row.get("statement") or row.get("value"))
        rows.append(row)
    return rows


def _active_rows(state: Any) -> list[dict[str, Any]]:
    return [item for item in _fact_rows(state) if item.get("status") != "superseded"]


def _dimensions_for_fact(item: dict[str, Any]) -> set[str]:
    category = str(item.get("category") or "")
    result = set(_FACT_DIMENSIONS_BY_CATEGORY.get(category, set()))
    tokens = re.split(r"[._:-]+", str(item.get("key") or "").lower())
    for token in tokens:
        if token in {"counterparty", "merchant", "operator", "seller", "employer"}:
            result.add("counterparty")
        elif token in {"relationship", "contract", "agreement"}:
            result.add("relationship")
        elif token in {"payment", "paid", "amount", "price", "total", "loss"}:
            result.add("amount")
        elif token in {"time", "date", "deadline", "discovery"}:
            result.add("time")
        elif token in {"location", "place", "platform", "region", "address"}:
            result.add("location")
        elif token in {"item", "product", "service", "subject", "goods"}:
            result.add("subject_matter")
        elif token in {"claim", "request", "remedy", "refund", "compensation"}:
            result.add("claim")
        elif token in {"procedure", "complaint", "report", "negotiation", "appeal"}:
            result.add("procedure")
        elif token in {"harm", "damage", "injury", "loss"}:
            result.add("harm")
        elif token in {"event", "problem", "breach", "delivery", "infringement"}:
            result.add("event")
    return result


def _has_dimension(state: Any, *names: str) -> bool:
    return any(bool(_dimensions_for_fact(item) & set(names)) for item in _active_rows(state))


def _status_rows(state: Any, dimensions: set[str]) -> list[dict[str, Any]]:
    return [
        item for item in _active_rows(state)
        if _dimensions_for_fact(item) & dimensions
    ]


def _rule_key(rule: FactFollowup) -> str:
    return _SLOT_KEY_MAP.get(rule.slot, f"fact.{rule.slot}")


def _rule_topic(rule: FactFollowup) -> str:
    dimensions = _SLOT_DIMENSIONS.get(rule.slot, {"event"})
    dimension = next(iter(dimensions), "event")
    return _TOPIC_LABELS.get(dimension, "经过和约定")


def _coverage_for_rule(rule: FactFollowup, state: Any) -> dict[str, Any]:
    """Coverage over the canonical blackboard, including ``confirmed`` facts.

    The legacy planner reads ``case_facts`` with the old ``asserted`` status.
    Node four must also work when only the new canonical ``fact_blackboard`` is
    present, so its coverage calculation is intentionally local and explicit.
    """

    known_dimensions: set[str] = set()
    for item in _active_rows(state):
        known_dimensions.update(_dimensions_for_fact(item))
    requirements = _SLOT_REQUIREMENTS.get(rule.slot, (("event",),))
    missing_keys: list[str] = []
    known_keys: list[str] = []
    for alternatives in requirements:
        matched = next((value for value in alternatives if value in known_dimensions), "")
        if matched:
            known_keys.append(matched)
        else:
            missing_keys.append(alternatives[0])
    known_statements = [
        str(item.get("statement") or "")
        for item in _active_rows(state)
        if _dimensions_for_fact(item) & known_dimensions and item.get("statement")
    ][-3:]
    return {
        "known": list(dict.fromkeys(known_statements)),
        "missing": [
            _DIMENSION_QUESTIONS.get(value, value)
            for value in missing_keys
        ],
        "known_dimension_keys": known_keys,
        "missing_dimension_keys": missing_keys,
    }


def _rule_related_rows(rule: FactFollowup, state: Any) -> list[dict[str, Any]]:
    dimensions = {
        value for alternatives in _SLOT_REQUIREMENTS.get(rule.slot, (("event",),))
        for value in alternatives
    }
    rows = _active_rows(state)
    related: list[dict[str, Any]] = []
    for item in rows:
        item_dimensions = _dimensions_for_fact(item)
        if item_dimensions & dimensions:
            related.append(item)
    return related


def activate_fact_dependencies(state: Any) -> list[dict[str, Any]]:
    """Return currently activated fact rules without mutating the blackboard."""

    domain_rules = get_domain_followups(getattr(state, "legal_domain", "") or "other")
    active: list[dict[str, Any]] = []
    for rule in domain_rules.facts:
        if rule.slot == "current_safety" and not getattr(state, "safety_relevant", False):
            continue
        coverage = _coverage_for_rule(rule, state)
        active.append({
            "rule_id": rule.id,
            "slot": rule.slot,
            "decision_key": _rule_key(rule),
            "depends_on": coverage.get("known_dimension_keys") or [],
            "missing_dependencies": coverage.get("missing_dimension_keys") or [],
            "decision_effects": list(SLOT_DECISION_EFFECTS.get(rule.slot, ("procedure",))),
            "priority": rule.priority,
            "activated": True,
        })
    return active


def _dimension_status(
    effect: str,
    *,
    satisfied: bool,
    missing: list[str],
    required: bool = True,
) -> dict[str, Any]:
    return {
        "effect": effect,
        "label": EFFECT_LABELS.get(effect, effect),
        "required": required,
        "satisfied": satisfied,
        "missing_fact_keys": list(dict.fromkeys(missing)),
    }


def assess_fact_sufficiency(state: Any, candidates: list[dict[str, Any]] | None = None) -> FactSufficiencyReport:
    """Assess fact sufficiency without treating evidence availability as a blocker."""

    rows = _active_rows(state)
    core_event = _has_dimension(state, "event", "transaction", "subject_matter")
    subject_and_relation = _has_dimension(state, "relationship", "counterparty", "actor")
    claim_known = _has_dimension(state, "claim")
    time_known = bool(getattr(state, "time_info", "")) or _has_dimension(state, "time")
    location_known = bool(getattr(state, "region", "")) or _has_dimension(state, "location")
    procedure_known = _has_dimension(state, "procedure")
    harm_known = _has_dimension(state, "harm", "amount")
    safety_relevant = bool(getattr(state, "safety_relevant", False))
    safety_known = (
        not safety_relevant
        or str(getattr(state, "current_safety_status", "not_applicable"))
        in {"safe", "danger"}
        or bool(getattr(state, "guard_status", "") in {"clear", "warning"})
    )

    dimensions = [
        _dimension_status(
            "responsibility",
            satisfied=bool(core_event and subject_and_relation),
            missing=[
                key for key, ok in (
                    ("event.core_behavior", core_event),
                    ("relationship.type", subject_and_relation),
                ) if not ok
            ],
        ),
        _dimension_status(
            "claim_scope",
            satisfied=claim_known,
            missing=[] if claim_known else ["claim.request"],
        ),
        _dimension_status(
            "limitation",
            satisfied=time_known,
            missing=[] if time_known else ["event.timeline"],
        ),
        _dimension_status(
            "jurisdiction",
            satisfied=location_known,
            missing=[] if location_known else ["location.platform_or_place"],
            required=False,
        ),
        _dimension_status(
            "procedure",
            satisfied=procedure_known,
            missing=[] if procedure_known else ["procedure.history"],
            required=False,
        ),
        _dimension_status(
            "harm_loss",
            satisfied=harm_known,
            missing=[] if harm_known else ["harm.loss"],
            required=False,
        ),
        _dimension_status(
            "safety",
            satisfied=safety_known,
            missing=[] if safety_known else ["safety.current_status"],
            required=safety_relevant,
        ),
    ]

    conflicts: list[str] = []
    for item in rows:
        if item.get("status") in _CONFLICT_STATUSES:
            group = str(item.get("conflict_group_id") or item.get("key") or "")
            if group and group not in conflicts:
                conflicts.append(group)
    critical_conflicts = [
        item for item in rows
        if item.get("status") in _CONFLICT_STATUSES
        and _dimensions_for_fact(item) & {"relationship", "event", "claim", "time", "amount"}
    ]
    if critical_conflicts:
        dimensions.append(_dimension_status(
            "conflict_resolution",
            satisfied=False,
            missing=[
                str(item.get("key") or item.get("statement") or "conflict")
                for item in critical_conflicts[:6]
            ],
        ))
    blocking_gaps = [
        item["effect"] for item in dimensions
        if item.get("required") and not item.get("satisfied")
        and item["effect"] in _BLOCKING_EFFECTS
    ]
    advisory_gaps = [
        item["effect"] for item in dimensions
        if not item.get("satisfied") and item["effect"] not in blocking_gaps
    ]
    minimum_actionable = bool(core_event and subject_and_relation)
    candidate_rows = candidates if candidates is not None else build_fact_question_candidates(state)
    has_high_value = any(
        float(item.get("information_gain") or 0) >= _settings().FACT_QUESTION_MIN_INFORMATION_GAIN
        for item in candidate_rows
    )
    if getattr(state, "guard_pause_required", False) or getattr(state, "safety_pause_active", False):
        status = "paused_by_guard"
        reason = "安全或即时风险尚未解除，事实决策暂时暂停"
    elif critical_conflicts:
        status = "blocked_by_conflict"
        reason = "存在可能改变责任、诉求或期限判断的事实冲突，需要先核对"
    elif blocking_gaps or not minimum_actionable or has_high_value:
        status = "insufficient"
        if not minimum_actionable:
            reason = "还没有形成责任主体、核心事件和用户诉求所需的最低案件事实"
        elif blocking_gaps:
            reason = "仍有会改变法律路径的关键事实缺口"
        else:
            reason = "当前仍有能够明显改变方案的高价值事实缺口"
    elif advisory_gaps:
        status = "conditionally_sufficient"
        reason = "最低案件事实已经具备，但仍有不阻断分析的补充信息缺口"
    else:
        status = "sufficient"
        reason = "当前高价值事实已经覆盖，继续追问不会明显改变下一阶段方案"
    return FactSufficiencyReport(
        status=status,
        can_proceed_conditionally=minimum_actionable,
        blocking_gaps=list(dict.fromkeys(blocking_gaps)),
        advisory_gaps=list(dict.fromkeys(advisory_gaps)),
        conflict_groups=conflicts,
        dimensions=dimensions,
        reason=reason,
    )


def _candidate_status(rule: FactFollowup, state: Any) -> str:
    records = getattr(state, "fact_records", {}) or {}
    record = records.get(rule.id) or {}
    if record.get("status"):
        status = str(record["status"])
        if status in {"unknown", "conflicted", "ambiguous", "user_stated", "corrected"}:
            return status
    related = _rule_related_rows(rule, state)
    key_match = [
        item for item in related
        if str(item.get("key") or "").startswith(rule.slot.split("_", 1)[0])
    ]
    statuses = [item.get("status") for item in (key_match or related)]
    if "conflicted" in statuses:
        return "conflicted"
    if any(item in _UNCLEAR_STATUSES for item in statuses):
        return "unclear"
    if "unknown" in statuses:
        return "unknown"
    return ""


def _ensure_one_question_mark(text: str) -> str:
    value = _compact(text, 300).rstrip("。；; ")
    if value.endswith(("？", "?")):
        return value[:-1] + "？"
    return value + "？"


def _conflict_prompt(rule: FactFollowup, state: Any) -> str:
    rows = [
        item for item in _rule_related_rows(rule, state)
        if item.get("status") == "conflicted"
    ]
    statements = list(dict.fromkeys(
        _compact(item.get("statement") or item.get("value"), 100)
        for item in rows if item.get("statement") or item.get("value")
    ))
    if len(statements) >= 2:
        return _ensure_one_question_mark(
            f"关于“{'”和“'.join(statements[:2])}”，请确认哪一种说法更准确"
        )
    return _ensure_one_question_mark(rule.question)


def _focused_prompt(rule: FactFollowup, state: Any, missing_keys: list[str]) -> str:
    """Ask only the uncovered part of a catalog rule."""

    if not missing_keys:
        return _ensure_one_question_mark(rule.question)
    if len(missing_keys) == 1:
        return _ensure_one_question_mark(_DIMENSION_QUESTIONS.get(
            missing_keys[0], rule.question
        ))
    # Two missing dimensions are usually a natural pair in the catalog
    # (subject/amount, event/time, or relationship/actor).
    questions = [
        _DIMENSION_QUESTIONS.get(key, "")
        for key in missing_keys[:2]
        if _DIMENSION_QUESTIONS.get(key)
    ]
    if questions:
        merged = "；".join(item.rstrip("？?。；;") for item in questions)
        return _ensure_one_question_mark(merged)
    return _ensure_one_question_mark(rule.question)


def _build_candidate(rule: FactFollowup, state: Any) -> dict[str, Any] | None:
    status = _candidate_status(rule, state)
    if status in {"unknown", "denied", "user_stated", "corrected"}:
        return None
    coverage = _coverage_for_rule(rule, state)
    if not coverage.get("missing") and status not in {"unclear", "conflicted"}:
        return None
    decision_key = _rule_key(rule)
    effects = list(SLOT_DECISION_EFFECTS.get(rule.slot, ("procedure",)))
    gain = max(
        (DECISION_EFFECT_WEIGHTS.get(item, 0.5) for item in effects),
        default=0.5,
    )
    priority = max(1, int(rule.priority or 100))
    burden = min(0.8, 0.14 + 0.035 * max(0, len(coverage.get("missing") or []) - 1))
    score = max(0.0, min(1.0, gain + max(0.0, 0.08 - priority * 0.02) - burden * 0.4))
    question_type = (
        "conflict_confirmation"
        if status == "conflicted"
        else "clarification"
        if status in {"unclear", "ambiguous"}
        else "missing_fact"
    )
    prompt = (
        _conflict_prompt(rule, state)
        if question_type == "conflict_confirmation"
        else _focused_prompt(rule, state, list(coverage.get("missing_dimension_keys") or []))
    )
    return {
        "id": rule.id,
        "candidate_id": rule.id,
        "kind": "facts",
        "decision_dimension": rule.slot,
        "decision_key": decision_key,
        "fact_slot_keys": [decision_key],
        "question_type": question_type,
        "topic": _rule_topic(rule),
        "prompt": prompt,
        "seed_question": rule.question,
        "answer_hint": rule.answer_hint or "不清楚时可以直接说“不清楚”。",
        "decision_effects": effects,
        "legal_effect": rule.why,
        "coverage": coverage,
        "information_gain": round(gain, 4),
        "user_burden": round(burden, 4),
        "priority_score": round(score, 4),
        "priority": rule.priority,
        "source_rule_ids": [rule.id],
        "unknown_allowed": True,
    }


def build_fact_question_candidates(state: Any) -> list[dict[str, Any]]:
    """Build only fact candidates; evidence quality questions stay in node six."""

    domain_rules = get_domain_followups(getattr(state, "legal_domain", "") or "other")
    asked = set(getattr(state, "asked_decision_keys", []) or [])
    asked.update(getattr(state, "answered_decision_keys", []) or [])
    asked.update(getattr(state, "unknown_decision_keys", []) or [])
    asked.update(getattr(state, "waived_decision_keys", []) or [])
    pending = set(getattr(state, "pending_decision_keys", []) or [])
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in domain_rules.facts:
        if rule.slot == "current_safety" and not getattr(state, "safety_relevant", False):
            continue
        item = _build_candidate(rule, state)
        if not item:
            continue
        key = item["decision_key"]
        if key in asked or key in pending or key in seen:
            continue
        seen.add(key)
        result.append(item)
    result.sort(key=lambda item: (
        -float(item.get("priority_score") or 0),
        int(item.get("priority") or 100),
        str(item.get("decision_key") or ""),
    ))
    return result


def select_question_batch(
    state: Any,
    candidates: list[dict[str, Any]] | None = None,
    *,
    max_questions: int | None = None,
) -> list[dict[str, Any]]:
    rows = list(candidates if candidates is not None else build_fact_question_candidates(state))
    max_questions = max_questions or _settings().FACT_BATCH_MAX_QUESTIONS
    threshold = _settings().FACT_QUESTION_MIN_INFORMATION_GAIN
    selected: list[dict[str, Any]] = []
    for item in rows:
        is_blocking = bool(set(item.get("decision_effects") or []) & _BLOCKING_EFFECTS)
        if float(item.get("information_gain") or 0) < threshold and not is_blocking:
            continue
        selected.append(item)
        if len(selected) >= max(1, int(max_questions)):
            break
    return selected


def build_question_batch(
    state: Any,
    candidates: list[dict[str, Any]] | None = None,
    *,
    max_questions: int | None = None,
    retrieval_basis: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = select_question_batch(state, candidates, max_questions=max_questions)
    basis = [
        dict(item)
        for item in (
            retrieval_basis
            if retrieval_basis is not None
            else getattr(state, "retrieval_basis_candidates", []) or []
        )
        if isinstance(item, dict)
    ][:8]
    questions: list[FactQuestionCandidate] = []
    for item in selected:
        qid = _stable_id("question", state.case_id, item["decision_key"])
        questions.append(FactQuestionCandidate(
            question_id=qid,
            decision_key=item["decision_key"],
            fact_slot_keys=list(item.get("fact_slot_keys") or [item["decision_key"]]),
            question_type=item.get("question_type") or "missing_fact",
            topic=item.get("topic") or "events_and_agreements",
            prompt=item["prompt"],
            answer_hint=item.get("answer_hint") or "不清楚时可以直接说“不清楚”。",
            decision_effects=list(item.get("decision_effects") or []),
            information_gain=float(item.get("information_gain") or 0),
            user_burden=float(item.get("user_burden") or 0),
            priority_score=float(item.get("priority_score") or 0),
            source_rule_ids=list(item.get("source_rule_ids") or []),
            basis_refs=[
                {
                    "basis_candidate_id": ref.get("basis_candidate_id")
                    or ref.get("rule_id")
                    or ref.get("law_id"),
                    "basis_type": ref.get("basis_type") or "retrieved_basis",
                    "title": ref.get("title") or "",
                    "locator": ref.get("locator") or ref.get("article_no") or "",
                }
                for ref in basis[:3]
            ],
            contextual_reason=str(item.get("contextual_reason") or ""),
            unknown_allowed=True,
        ))
    batch_id = _stable_id(
        "fact-batch",
        state.case_id,
        getattr(state, "fact_blackboard_version", 0),
        ",".join(item.question_id for item in questions),
    )
    batch = FactQuestionBatch(
        batch_id=batch_id,
        questions=questions,
        context_summary=_question_context_summary(state),
        retrieval_basis=basis,
        fact_blackboard_version=int(getattr(state, "fact_blackboard_version", 0) or 0),
        created_at=_now(),
    )
    batch.markdown = render_question_batch_markdown(batch)
    return batch.model_dump()


def render_question_batch_markdown(batch: dict[str, Any] | FactQuestionBatch) -> str:
    data = batch if isinstance(batch, FactQuestionBatch) else FactQuestionBatch.model_validate(batch)
    if not data.questions:
        return ""
    lines: list[str] = []
    if data.context_summary:
        lines.extend(["## 已记录", ""])
        lines.extend(f"- {item}" for item in data.context_summary[:6])
        lines.append("")
    lines.extend(["## 请补充目前尚未明确的信息", ""])
    grouped: dict[str, list[FactQuestionCandidate]] = {}
    for item in data.questions:
        grouped.setdefault(item.topic or "经过和约定", []).append(item)
    number = 1
    for topic, questions in grouped.items():
        lines.extend([f"### {topic}", ""])
        for item in questions:
            label = {
                "conflict_confirmation": "需要核对",
                "clarification": "需要澄清",
                "missing_fact": "请说明",
            }.get(item.question_type, "请说明")
            lines.append(f"{number}. **{label}：** {item.prompt}")
            number += 1
        lines.append("")
    basis_labels: list[str] = []
    for item in data.retrieval_basis:
        title = _compact(item.get("title") or "", 80)
        locator = _compact(
            item.get("locator") or item.get("article_no") or "", 50
        )
        label = " ".join(value for value in (title, locator) if value).strip()
        if label and label not in basis_labels:
            basis_labels.append(label)
    if basis_labels:
        lines.extend([
            "### 本轮追问依据",
            "",
            *[f"- {item}" for item in basis_labels[:4]],
            "",
            "> 检索结果只用于识别可能影响责任、期限、诉求或程序路径的事实条件，不会被写成本案已经发生的事实。",
            "",
        ])
    lines.extend([
        "> 可以一次回答多个问题；不清楚的项目写“不清楚”，没有的写“没有”。",
        "> 暂时不方便补充时，直接回复“现在生成方案”，我会保留未知项并按当前信息继续。",
    ])
    rendered = "\n".join(lines).strip()
    max_length = int(getattr(_settings(), "FACT_BATCH_MAX_RENDERED_LENGTH", 4000) or 4000)
    return rendered if len(rendered) <= max_length else rendered[: max_length - 1].rstrip() + "…"


def _question_context_summary(state: Any) -> list[str]:
    changed_ids = {
        str(item.get("fact_id") or "")
        for item in (getattr(state, "fact_changes", []) or [])
        if isinstance(item, dict) and item.get("fact_id")
    }
    rows = [
        item
        for item in _active_rows(state)
        if not changed_ids or str(item.get("fact_id") or "") in changed_ids
    ]
    if not rows:
        rows = _active_rows(state)[-6:]
    result: list[str] = []
    for item in rows:
        statement = _compact(item.get("statement") or item.get("value"), 140)
        if statement and statement not in result:
            result.append(statement)
    return result[:6]


def _requirement_profile(item: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    key = str(item.get("key") or "").lower()
    category = str(item.get("category") or "")
    text = f"{key} {item.get('statement') or ''}".lower()
    if category == "amount" or any(token in text for token in ("payment", "paid", "付款", "转账", "价格", "金额")):
        return "transaction.payment", "证明付款、金额和支付时间", ["支付记录", "平台账单", "银行流水"], ["订单详情", "收款方确认"]
    if any(token in text for token in ("non_delivery", "not_delivered", "delivery", "发货", "未发", "交付")):
        return "delivery.non_performance", "证明约定履行内容和未交付经过", ["订单记录", "物流记录", "催告或聊天记录"], ["平台售后记录", "对方承认未履行的消息"]
    if category == "procedure" or any(token in text for token in ("complaint", "report", "platform", "投诉", "举报", "平台")):
        return "platform.complaint", "证明已经采取的投诉、退款或平台处理经过", ["投诉记录", "平台工单", "处理结果或回执"], ["客服聊天记录", "订单状态截图"]
    if category in {"relationship", "actor"} or any(token in text for token in ("contract", "agreement", "seller", "merchant", "合同", "约定", "卖家", "商家")):
        return "relationship.and.terms", "证明双方身份、关系和关键约定", ["订单或合同", "完整聊天记录", "对方账号信息"], ["收据", "平台实名或店铺信息"]
    if category == "time" or any(token in text for token in ("time", "date", "deadline", "时间", "日期", "期限")):
        return "event.timeline", "证明关键时间、约定和通知节点", ["带时间的聊天记录", "订单时间线", "通知或回执"], ["平台日志", "证人说明"]
    if category == "harm" or any(token in text for token in ("harm", "damage", "loss", "损失", "影响", "伤")):
        return "harm.loss", "证明实际损失、影响及其计算基础", ["损失凭证", "维修或医疗材料", "费用记录"], ["照片视频", "证人说明"]
    if category == "claim":
        return "claim.request", "证明用户主张、协商内容和请求范围", ["退款或赔偿请求记录", "协商聊天", "投诉内容"], ["邮件或短信", "平台申请记录"]
    return "", "", [], []


def match_evidence_names_to_requirements(
    requirements: Iterable[dict[str, Any]],
    inventory: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = [item for item in inventory if isinstance(item, dict)]
    output: list[dict[str, Any]] = []
    for raw in requirements:
        item = dict(raw)
        classes = [
            str(value).lower()
            for value in (
                item.get("recommended_material_classes") or []
            ) + list(item.get("alternative_material_classes") or [])
        ]
        matches = []
        for name in names:
            haystack = " ".join(
                str(name.get(field) or "")
                for field in ("display_name", "normalized_name", "original_names")
            ).lower()
            if any(material and (material in haystack or haystack in material) for material in classes):
                matches.append(str(name.get("evidence_name_id") or ""))
        item["matched_evidence_name_ids"] = list(dict.fromkeys(item.get("matched_evidence_name_ids") or []))
        item["matched_evidence_name_ids"].extend(
            value for value in matches if value and value not in item["matched_evidence_name_ids"]
        )
        output.append(item)
    return output


def update_internal_evidence_requirements(state: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    previous = {
        str(item.get("requirement_id")): dict(item)
        for item in (getattr(state, "internal_evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    requirements = dict(previous)
    changes: list[dict[str, Any]] = []
    rows = _active_rows(state)
    for item in rows:
        if item.get("category") == "evidence":
            continue
        requirement_id, label, materials, alternatives = _requirement_profile(item)
        if not requirement_id:
            continue
        fact_key = str(item.get("key") or item.get("fact_id") or "")
        if requirement_id not in requirements:
            initial_status = (
                "pending_fact_confirmation"
                if item.get("status") in _CONFLICT_STATUSES | _UNCLEAR_STATUSES
                else "not_applicable"
                if item.get("status") == "denied"
                else "active_candidate"
            )
            requirements[requirement_id] = InternalEvidenceRequirement(
                requirement_id=requirement_id,
                proof_target_candidate_id=f"proof.{requirement_id}",
                label=label,
                dependent_fact_keys=[fact_key] if fact_key else [],
                recommended_material_classes=materials,
                alternative_material_classes=alternatives,
                generation_round=int(getattr(state, "round", 0) or 0),
                last_updated_round=int(getattr(state, "round", 0) or 0),
                change_reason="added_fact",
                status=initial_status,
            ).model_dump()
            changes.append({
                "requirement_id": requirement_id,
                "change_type": "added",
                "change_reason": "added_fact",
                "status": initial_status,
            })
            continue
        current = requirements[requirement_id]
        deps = list(current.get("dependent_fact_keys") or [])
        if fact_key and fact_key not in deps:
            deps.append(fact_key)
        status = "active_candidate"
        if item.get("status") in _CONFLICT_STATUSES | _UNCLEAR_STATUSES:
            status = "pending_fact_confirmation"
        elif item.get("status") == "denied":
            status = "not_applicable"
        old_status = current.get("status")
        changed = deps != current.get("dependent_fact_keys") or status != old_status
        current.update({
            "dependent_fact_keys": deps,
            "last_updated_round": int(getattr(state, "round", 0) or 0),
            "change_reason": "refined_fact" if changed else "carried_forward",
            "status": status,
        })
        if changed:
            changes.append({
                "requirement_id": requirement_id,
                "change_type": "updated" if old_status == status else "status_changed",
                "change_reason": current["change_reason"],
                "status": status,
            })
    current_keys = {str(item.get("key") or "") for item in rows}
    for requirement_id, item in requirements.items():
        deps = set(item.get("dependent_fact_keys") or [])
        if deps and not deps & current_keys and item.get("status") not in {"superseded", "not_applicable"}:
            item["status"] = "superseded"
            item["change_reason"] = "dependent_fact_replaced"
            changes.append({
                "requirement_id": requirement_id,
                "change_type": "superseded",
                "change_reason": "dependent_fact_replaced",
            })
    merged = match_evidence_names_to_requirements(requirements.values(), getattr(state, "evidence_name_inventory", []) or [])
    return merged, changes


def calculate_fact_change_materiality(state: Any) -> str:
    changes = list(getattr(state, "fact_changes", []) or [])
    if not changes:
        return "none"
    if any(str(item.get("change_type") or "") not in {"source_added", "unchanged"} for item in changes):
        keys = " ".join(
            str(item.get("semantic_key") or item.get("key") or "").lower()
            for item in changes
        )
        if any(token in keys for token in _MATERIAL_TOKENS):
            return "material"
        return "possibly_material"
    return "non_material"


def build_targeted_retrieval_query(state: Any) -> dict[str, Any]:
    changes = [
        item for item in (getattr(state, "fact_changes", []) or [])
        if str(item.get("change_type") or "") != "source_added"
    ]
    changed_keys = list(dict.fromkeys(
        str(item.get("semantic_key") or item.get("key") or "")
        for item in changes
        if item.get("semantic_key") or item.get("key")
    ))
    effects: set[str] = set()
    for item in _active_rows(state):
        if str(item.get("key") or "") in changed_keys:
            effects.update(_FACT_EFFECT_BY_CATEGORY.get(str(item.get("category") or ""), set()))
    if any(token in " ".join(changed_keys).lower() for token in ("time", "date", "deadline")):
        effects.add("limitation")
    if any(token in " ".join(changed_keys).lower() for token in ("platform", "procedure", "complaint")):
        effects.add("procedure")
    mode = "targeted" if effects & {
        "responsibility", "claim_scope", "limitation", "jurisdiction", "procedure"
    } else "none"
    return {
        "changed_fact_keys": changed_keys,
        "legal_domain_candidate": getattr(state, "legal_domain", "") or getattr(state, "domain_candidate", "") or "other",
        "decision_effects": sorted(effects),
        "mode": mode,
        "triggered": mode == "targeted",
    }


def record_retrieval_trace(state: Any, query: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[str]]:
    trace_id = _stable_id(
        "retrieval",
        state.case_id,
        getattr(state, "fact_blackboard_version", 0),
        json.dumps(query, ensure_ascii=False, sort_keys=True),
    )
    basis = [
        dict(item) for item in (
            getattr(state, "retrieval_basis_candidates", []) or []
        )
        if isinstance(item, dict)
    ]
    # 节点四不代替节点五做完整检索。已有且未受事实变化影响的候选只做
    # 内部依据复用；没有可复用依据时显式记录缺口，不编造法条条件。
    gaps = [] if basis or query.get("mode") == "none" else ["targeted_retrieval_basis_unavailable"]
    summary = {
        "mode": query.get("mode", "none"),
        "triggered": bool(query.get("triggered")),
        "reused_basis_ids": [
            str(item.get("basis_candidate_id") or item.get("id") or "")
            for item in basis
        ],
        "new_basis_candidate_ids": [],
        "retrieval_gaps": gaps,
        "changed_fact_keys": list(query.get("changed_fact_keys") or []),
        "decision_effects": list(query.get("decision_effects") or []),
    }
    return summary, trace_id, basis, gaps


def _issue_seeds(state: Any) -> list[str]:
    seeds: list[str] = []
    for item in [
        *(getattr(state, "confirmed_issues", []) or []),
        *(getattr(state, "issue_candidates", []) or []),
    ]:
        value = _compact(item, 180)
        if value and value not in seeds:
            seeds.append(value)
    if seeds:
        return seeds[:8]
    for item in _active_rows(state):
        if item.get("status") not in _RESOLVED_STATUSES:
            continue
        if item.get("category") not in {
            "relationship", "event", "claim", "procedure", "harm",
        }:
            continue
        value = _compact(item.get("statement"), 180)
        if value and value not in seeds:
            seeds.append(value)
    return seeds[:8]


async def normalize_issue_projection(
    state: Any,
    deps: Any = None,
) -> dict[str, Any]:
    """Reuse the legacy exact/semantic terminology pipeline without re-extracting facts."""

    seeds = _issue_seeds(state)
    if not seeds or deps is None:
        return {}
    embedding_model = getattr(deps, "embedding_model", None)
    milvus_client = getattr(deps, "milvus_client", None)
    neo4j_driver = getattr(deps, "neo4j_driver", None)
    if embedding_model is None or milvus_client is None or neo4j_driver is None:
        return {}
    settings = _settings()
    timeout = float(
        getattr(settings, "GUIDE_RETRIEVE_TIMEOUT_AUX", 5.0) or 5.0
    )
    try:
        from src.agents.legal_guide.issue_normalizer import (
            confirm_domain_in_neo4j,
            match_issues_in_neo4j,
            semantic_fallback,
        )

        current_domain = str(
            getattr(state, "legal_domain", "")
            or getattr(state, "domain_candidate", "")
            or "other"
        )
        matched, unmatched = await asyncio.wait_for(
            match_issues_in_neo4j(seeds, neo4j_driver),
            timeout=timeout,
        )
        term_map, still_unmatched, mapped_domains = await asyncio.wait_for(
            semantic_fallback(
                unmatched,
                embedding_model,
                milvus_client,
                domain="" if current_domain == "other" else current_domain,
            ),
            timeout=timeout,
        )
        normalized_domain = current_domain
        if current_domain == "other" and len(mapped_domains) == 1:
            normalized_domain = await asyncio.wait_for(
                confirm_domain_in_neo4j(mapped_domains[0], neo4j_driver),
                timeout=timeout,
            )
        standard = list(dict.fromkeys([*matched, *term_map.values()]))
        colloquial = list(dict.fromkeys(still_unmatched))
        return {
            "confirmed_issues": standard
            or list(getattr(state, "confirmed_issues", []) or []),
            "unmatched_issues": colloquial,
            "legal_domain": normalized_domain,
            "issue_term_map": term_map,
            "issue_normalization_trace": {
                "fact_blackboard_version": int(
                    getattr(state, "fact_blackboard_version", 0) or 0
                ),
                "input_seeds": seeds,
                "exact_matches": matched,
                "semantic_mappings": term_map,
                "unmatched": colloquial,
                "domain_before": current_domain,
                "domain_after": normalized_domain,
                "pipeline": [
                    "fact_blackboard",
                    "neo4j_exact",
                    "milvus_legal_term",
                ],
                "status": "completed",
                "created_at": _now(),
            },
        }
    except Exception as exc:
        logger.warning("节点④法律术语标准化降级 | error={}", exc)
        return {
            "issue_normalization_trace": {
                "fact_blackboard_version": int(
                    getattr(state, "fact_blackboard_version", 0) or 0
                ),
                "input_seeds": seeds,
                "status": "degraded",
                "error_type": type(exc).__name__,
                "created_at": _now(),
            }
        }


def _targeted_retrieval_text(state: Any, query: dict[str, Any]) -> tuple[str, str]:
    inputs = build_case_retrieval_inputs(
        getattr(state, "confirmed_issues", []) or [],
        active_case_facts(getattr(state, "case_facts", []) or []),
    )
    facts = [
        _compact(item.get("statement"), 220)
        for item in _active_rows(state)
        if item.get("status") in _RESOLVED_STATUSES
        and item.get("statement")
    ][-12:]
    effects = "、".join(query.get("decision_effects") or [])
    question = "；".join(
        [
            *list(inputs.get("semantic_phrases") or [])[-10:],
            *facts,
            f"需要识别会影响{effects}的事实条件" if effects else "",
        ]
    ).strip("；")
    return question or "案件事实追问所需法律条件", str(
        inputs.get("sparse_query") or ""
    )


async def retrieve_targeted_fact_basis(
    state: Any,
    deps: Any,
    query: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Retrieve small, auditable law/authority sets for the current fact delta."""

    if not query.get("triggered"):
        summary, trace_id, basis, gaps = record_retrieval_trace(state, query)
        return summary, trace_id, basis, gaps, []
    embedding_model = getattr(deps, "embedding_model", None) if deps else None
    milvus_client = getattr(deps, "milvus_client", None) if deps else None
    if embedding_model is None or milvus_client is None:
        summary, trace_id, basis, gaps = record_retrieval_trace(state, query)
        return summary, trace_id, basis, gaps, []

    settings = _settings()
    domain = str(query.get("legal_domain_candidate") or "")
    domain_filter = "" if domain == "other" else domain
    question, sparse_query = _targeted_retrieval_text(state, query)
    trace_id = _stable_id(
        "retrieval",
        state.case_id,
        getattr(state, "fact_blackboard_version", 0),
        json.dumps(query, ensure_ascii=False, sort_keys=True),
    )
    gaps: list[str] = []
    law_hits: list[dict[str, Any]] = []
    authority_hits: list[dict[str, Any]] = []
    try:
        from src.agents.legal_knowledge.statute_rag import search_statutes_raw
        from src.agents.legal_guide.authority_rag import (
            search_authority_basis_raw,
        )

        statute_task = asyncio.wait_for(
            search_statutes_raw(
                question=question,
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                domain=domain_filter,
                llm=getattr(deps, "llm", None),
                use_hyde=False,
                use_rrf=bool(sparse_query),
                sparse_query=sparse_query,
                top_k=12,
                rerank_top_k=6,
                skip_rerank=True,
            ),
            timeout=float(
                getattr(settings, "GUIDE_RETRIEVE_TIMEOUT_STATUTE", 8.0)
                or 8.0
            ),
        )
        authority_task = asyncio.wait_for(
            search_authority_basis_raw(
                question=question,
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                domain=domain_filter,
                top_k=6,
            ),
            timeout=float(
                getattr(settings, "GUIDE_RETRIEVE_TIMEOUT_AUX", 5.0)
                or 5.0
            ),
        )
        raw_laws, raw_authorities = await asyncio.gather(
            statute_task,
            authority_task,
            return_exceptions=True,
        )
        if isinstance(raw_laws, Exception):
            gaps.append(
                "targeted_statute_retrieval_timeout"
                if isinstance(raw_laws, asyncio.TimeoutError)
                else "targeted_statute_retrieval_failed"
            )
        else:
            law_hits = [dict(item) for item in raw_laws or []]
        if isinstance(raw_authorities, Exception):
            gaps.append("targeted_authority_retrieval_failed")
        else:
            authority_hits = [
                dict(item) for item in raw_authorities or []
            ]
    except Exception as exc:
        logger.warning("节点④定向检索降级 | error={}", exc)
        gaps.append("targeted_retrieval_failed")

    titles: dict[str, str] = {}
    db_session = getattr(deps, "db_session", None)
    if law_hits and db_session is not None:
        try:
            from src.agents.legal_knowledge.statute_rag import _fetch_law_titles

            titles = await asyncio.wait_for(
                _fetch_law_titles(law_hits, db_session),
                timeout=float(
                    getattr(settings, "GUIDE_RETRIEVE_TIMEOUT_AUX", 5.0)
                    or 5.0
                ),
            )
        except Exception:
            gaps.append("targeted_law_title_lookup_failed")

    law_refs: list[dict[str, Any]] = []
    for hit in law_hits[:8]:
        law_id = str(hit.get("law_id") or "")
        article = str(hit.get("article_no") or "")
        title = titles.get(law_id, "")
        law_refs.append({
            "basis_candidate_id": _stable_id(
                "fact-law", law_id, article, hit.get("text")
            ),
            "basis_type": "statute",
            "law_id": law_id,
            "source_id": law_id,
            "title": title,
            "article_no": article,
            "locator": " ".join(
                item for item in (title, article) if item
            ),
            "text": _compact(hit.get("text"), 1400),
            "domain": str(hit.get("domain") or domain),
            "score": float(hit.get("score") or 0.0),
            "status": "active",
            "version_key": "runtime-index-current",
            "review_status": "retrieved_candidate",
            "decision_effects": list(query.get("decision_effects") or []),
            "retrieval_trace_id": trace_id,
        })
    authority_refs: list[dict[str, Any]] = []
    for hit in authority_hits[:6]:
        rule_id = str(hit.get("rule_id") or hit.get("id") or "")
        authority_refs.append({
            **hit,
            "basis_candidate_id": rule_id
            or _stable_id("fact-authority", hit.get("title"), hit.get("locator")),
            "basis_type": "authority_rule",
            "retrieval_trace_id": trace_id,
            "decision_effects": list(query.get("decision_effects") or []),
        })
    basis = [*law_refs, *authority_refs]
    if not basis:
        gaps.append("targeted_retrieval_basis_unavailable")
    summary = {
        "mode": "targeted",
        "triggered": True,
        "query_text": question[:1200],
        "sparse_query": sparse_query,
        "reused_basis_ids": [],
        "new_basis_candidate_ids": [
            str(item.get("basis_candidate_id") or "")
            for item in basis
            if item.get("basis_candidate_id")
        ],
        "retrieval_gaps": list(dict.fromkeys(gaps)),
        "changed_fact_keys": list(query.get("changed_fact_keys") or []),
        "decision_effects": list(query.get("decision_effects") or []),
        "source_types": list(dict.fromkeys(
            str(item.get("basis_type") or "") for item in basis
        )),
        "fact_blackboard_version": int(
            getattr(state, "fact_blackboard_version", 0) or 0
        ),
    }
    return summary, trace_id, basis, summary["retrieval_gaps"], law_refs


def build_fact_snapshot_draft(
    state: Any,
    report: FactSufficiencyReport | dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_obj = report if isinstance(report, FactSufficiencyReport) else FactSufficiencyReport.model_validate(
        report or assess_fact_sufficiency(state)
    )
    rows = _active_rows(state)
    confirmed = [item for item in rows if item.get("status") == "confirmed"]
    denied = [item for item in rows if item.get("status") == "denied"]
    unknown = [item for item in rows if item.get("status") == "unknown"]
    unclear = [
        item for item in rows
        if item.get("status") in _UNCLEAR_STATUSES | _CONFLICT_STATUSES
    ]
    conflict_ids = list(dict.fromkeys(
        str(item.get("conflict_group_id") or item.get("key") or "")
        for item in unclear if item.get("status") == "conflicted"
    ))
    payload = {
        "case_id": state.case_id,
        "case_generation": state.case_generation,
        "fact_blackboard_version": state.fact_blackboard_version,
        "confirmed_fact_ids": [str(item.get("fact_id") or item.get("key") or "") for item in confirmed],
        "denied_fact_ids": [str(item.get("fact_id") or item.get("key") or "") for item in denied],
        "unknown_fact_ids": [str(item.get("fact_id") or item.get("key") or "") for item in unknown],
        "conflict_group_ids": conflict_ids,
        "legal_relation_candidates": list(dict.fromkeys(
            value for value in (
                getattr(state, "legal_domain", ""),
                getattr(state, "domain_candidate", ""),
            ) if value and value != "other"
        )),
        "proceed_under_uncertainty": bool(
            report_obj.status == "conditionally_sufficient"
            or getattr(state, "wants_conclude", False)
            or getattr(state, "force_conclude", False)
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    snapshot_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    draft_id = _stable_id("snapshot-draft", state.case_id, state.fact_blackboard_version, snapshot_hash)
    confirmed_lines = [
        f"- {_compact(item.get('statement') or item.get('value'), 240)}"
        for item in confirmed if item.get("statement") or item.get("value")
    ]
    unknown_lines = [
        f"- {_compact(item.get('statement') or item.get('key'), 180)}"
        for item in unknown + unclear if item.get("statement") or item.get("key")
    ]
    evidence_lines = []
    for item in getattr(state, "evidence_name_inventory", []) or []:
        if not isinstance(item, dict):
            continue
        status = {
            "submitted": "已提交副本，尚未完成证据评估",
            "user_claimed_present": "用户称持有，尚未提交评估",
            "temporarily_unavailable": "用户暂时找不到",
            "explicitly_absent": "用户明确没有",
        }.get(str(item.get("status") or ""), "状态待确认")
        evidence_lines.append(f"- {item.get('display_name') or item.get('normalized_name')}: {status}")
    markdown_parts = ["## 请确认案件事实", "", "### 已确认", ""]
    markdown_parts.extend(confirmed_lines or ["- （目前还没有可列为已确认的事实）"])
    markdown_parts.extend(["", "### 仍不明确", ""])
    markdown_parts.extend(unknown_lines or ["- （暂无需要说明的未知或冲突事实）"])
    markdown_parts.extend(["", "### 已提到的材料", ""])
    markdown_parts.extend(evidence_lines or ["- （暂无已登记的材料名称）"])
    markdown_parts.extend([
        "",
        "> 可以回复“确认并继续”，也可以补充或更正具体内容。",
        "> 如果希望按当前信息继续，请回复“现在生成方案”；未知项不会被系统补全。",
    ])
    return {
        **payload,
        "fact_snapshot_draft_id": draft_id,
        "based_on_fact_blackboard_version": int(state.fact_blackboard_version or 0),
        "snapshot_hash": f"sha256:{snapshot_hash}",
        "created_at": _now(),
        "status": "draft",
        "stale": False,
        "sufficiency_status": report_obj.status,
        "markdown": "\n".join(markdown_parts),
    }


def _settings():
    return get_settings()


def _compat_sufficiency(report: FactSufficiencyReport) -> dict[str, Any]:
    """Keep old ``node_assess_retrieve`` consumers readable during migration."""

    definitive = report.status == "sufficient"
    conditional = report.status in {"conditionally_sufficient", "sufficient"}
    return {
        **report.model_dump(),
        "sufficient_for_definitive_plan": definitive,
        "can_conclude_conditionally": conditional,
        "recommended_action": "ask" if report.status in {"insufficient", "blocked_by_conflict"} else "conclude_conditional",
    }


async def run_decide_facts(state: Any, deps: Any = None) -> dict[str, Any]:
    """Execute node four and return a checkpoint suitable for the graph/API."""

    settings = _settings()
    normalization_updates = await normalize_issue_projection(state, deps)
    decision_state = (
        state.model_copy(update=normalization_updates)
        if normalization_updates
        else state
    )
    prior_pending = set(getattr(state, "pending_decision_keys", []) or [])
    if not prior_pending:
        prior_pending = {
            str(item.get("decision_key") or "")
            for item in (getattr(state, "question_batch", {}) or {}).get("questions", [])
            if isinstance(item, dict) and item.get("decision_key")
        }
    planning_state = decision_state
    # ``update_facts`` deliberately owns fact writes and clears the old
    # compatibility pending text. Re-open the previous batch keys for this
    # planning pass so partially answered questions can be selected again;
    # confirmed/denied/unknown facts are filtered by the status rules below.
    if (
        prior_pending
        and getattr(state, "input_event_type", "") in {
            "fact_batch_answered", "fact_added", "fact_corrected", "fact_denied", "mixed_update",
        }
    ):
        planning_state = decision_state.model_copy(update={
            "asked_decision_keys": [
                item for item in (getattr(state, "asked_decision_keys", []) or [])
                if item not in prior_pending
            ],
            "pending_decision_keys": [],
        })
    activated = activate_fact_dependencies(planning_state)
    candidates = build_fact_question_candidates(planning_state)
    report = assess_fact_sufficiency(decision_state, candidates)
    requirements, requirement_changes = update_internal_evidence_requirements(
        decision_state
    )
    materiality = calculate_fact_change_materiality(decision_state)
    query = build_targeted_retrieval_query(decision_state)
    (
        retrieval_summary,
        retrieval_trace_id,
        basis,
        retrieval_gaps,
        targeted_law_refs,
    ) = await retrieve_targeted_fact_basis(decision_state, deps, query)

    previous_no_progress = int(getattr(state, "no_progress_rounds", 0) or 0)
    no_progress = (
        previous_no_progress + 1
        if materiality in {"none", "non_material"}
        and getattr(state, "decision_status", "") == "ask_batch"
        else 0
    )
    technical_limit = int(getattr(settings, "FACT_TECHNICAL_MAX_ROUNDS", 12) or 12)
    hit_no_progress = no_progress >= int(getattr(settings, "FACT_MAX_NO_PROGRESS_ROUNDS", 3) or 3)
    hit_technical_limit = int(getattr(state, "facts_rounds", 0) or 0) >= technical_limit

    has_minimum = report.can_proceed_conditionally
    wants_conclude = bool(
        getattr(state, "wants_conclude", False)
        or getattr(state, "turn_control_intent", "") == "conclude_now"
        or getattr(state, "supplement_choice", "") == "conclude"
        or getattr(state, "force_conclude", False)
    )
    if report.status == "paused_by_guard":
        decision_status = "paused_by_guard"
        next_route = "__end__"
        pause_state = {
            "type": "safety",
            "pause_type": "paused_by_guard",
            "case_id": state.case_id,
        }
        batch = {}
        snapshot = None
        reply = ""
    elif (
        report.status in {"blocked_by_conflict", "insufficient"}
        and not (wants_conclude and has_minimum)
        and not (hit_no_progress and has_minimum)
        and not (hit_technical_limit and has_minimum)
    ):
        selected = select_question_batch(planning_state, candidates)
        if selected:
            batch = build_question_batch(
                decision_state,
                selected,
                retrieval_basis=basis,
            )
            decision_status = "ask_batch"
            next_route = "__interrupt__"
            pause_state = {
                "type": "awaiting_fact_batch",
                "pause_type": "awaiting_fact_batch",
                "batch_id": batch["batch_id"],
                "question_ids": [item["question_id"] for item in batch["questions"]],
                "decision_keys": [item["decision_key"] for item in batch["questions"]],
                "fact_blackboard_version": state.fact_blackboard_version,
                "created_at": batch["created_at"],
            }
            snapshot = None
            reply = batch["markdown"]
        elif has_minimum:
            report.status = "conditionally_sufficient"
            decision_status = "proceed_to_evidence_planning"
            next_route = "plan_evidence"
            pause_state = None
            batch = {}
            snapshot = build_fact_snapshot_draft(decision_state, report)
            reply = snapshot["markdown"]
        else:
            decision_status = "unable_to_decide"
            next_route = "__interrupt__"
            pause_state = {"type": "awaiting_minimum_facts", "pause_type": "awaiting_fact_batch"}
            batch = {}
            snapshot = None
            reply = "## 还需要最少的案件信息\n\n请先说明：对方是谁、发生了什么，以及您希望解决什么。"
    elif wants_conclude or hit_no_progress or hit_technical_limit or report.status == "conditionally_sufficient":
        report.status = "conditionally_sufficient"
        decision_status = "proceed_to_evidence_planning"
        next_route = "plan_evidence"
        pause_state = None
        batch = {}
        snapshot = build_fact_snapshot_draft(decision_state, report)
        reply = snapshot["markdown"]
    else:
        decision_status = "await_snapshot_confirmation"
        next_route = "__interrupt__"
        pause_state = {
            "type": "awaiting_fact_snapshot_confirmation",
            "pause_type": "awaiting_fact_snapshot_confirmation",
            "fact_snapshot_draft_id": "",
            "fact_blackboard_version": state.fact_blackboard_version,
            "created_at": _now(),
        }
        batch = {}
        snapshot = build_fact_snapshot_draft(decision_state, report)
        pause_state["fact_snapshot_draft_id"] = snapshot["fact_snapshot_draft_id"]
        reply = snapshot["markdown"]

    question_history = list(getattr(state, "question_batch_history", []) or [])
    new_batch_issued = bool(
        batch
        and not any(
            item.get("batch_id") == batch["batch_id"]
            for item in question_history
        )
    )
    if new_batch_issued:
        question_history = [*question_history, batch][-50:]
    question_ids = [item["question_id"] for item in batch.get("questions", [])]
    decision_keys = [item["decision_key"] for item in batch.get("questions", [])]
    asked_keys = list(dict.fromkeys([
        *(getattr(state, "asked_decision_keys", []) or []),
        *decision_keys,
    ]))
    trace_id = _stable_id(
        "decision",
        state.case_id,
        state.case_generation,
        state.fact_blackboard_version,
        decision_status,
        ",".join(question_ids),
    )
    trace = {
        "decision_trace_id": trace_id,
        "case_id": state.case_id,
        "case_generation": state.case_generation,
        "fact_blackboard_version": state.fact_blackboard_version,
        "activated_rule_ids": [item["rule_id"] for item in activated],
        "sufficiency_report": report.model_dump(),
        "candidate_question_ids": [
            _stable_id("question", state.case_id, item["decision_key"])
            for item in candidates
        ],
        "selected_question_ids": question_ids,
        "question_scores": [
            {
                "decision_key": item["decision_key"],
                "information_gain": item["information_gain"],
                "user_burden": item["user_burden"],
                "priority_score": item["priority_score"],
            }
            for item in candidates
        ],
        "selected_batch_id": batch.get("batch_id", ""),
        "evidence_requirement_changes": requirement_changes,
        "retrieval_summary": retrieval_summary,
        "retrieval_trace_id": retrieval_trace_id,
        "fact_snapshot_draft_id": (snapshot or {}).get("fact_snapshot_draft_id", ""),
        "decision_status": decision_status,
        "next_route": next_route,
        "planner_version": "decide_facts.v1",
        "prompt_version": "deterministic",
        "created_at": _now(),
    }
    config_snapshot = {
        "question_min_information_gain": settings.FACT_QUESTION_MIN_INFORMATION_GAIN,
        "batch_max_questions": settings.FACT_BATCH_MAX_QUESTIONS,
        "batch_max_rendered_length": settings.FACT_BATCH_MAX_RENDERED_LENGTH,
        "max_no_progress_rounds": settings.FACT_MAX_NO_PROGRESS_ROUNDS,
        "technical_max_rounds": settings.FACT_TECHNICAL_MAX_ROUNDS,
    }
    compat = _compat_sufficiency(report)
    updates = {
        **normalization_updates,
        "fact_sufficiency": report.model_dump(),
        "sufficiency_report": report.model_dump(),
        "decision_sufficiency": compat,
        "convergence_reason": report.reason,
        "no_progress_rounds": no_progress,
        "convergence_config_snapshot": config_snapshot,
        "active_fact_schema": [item["decision_key"] for item in activated],
        "active_fact_schema_version": len(activated),
        "activated_fact_slots": [item["decision_key"] for item in activated],
        "question_batch": batch,
        "question_batch_history": question_history,
        "asked_question_batches": question_history,
        "pending_fact_batch_id": batch.get("batch_id", ""),
        "pending_question_ids": question_ids,
        "pending_decision_keys": decision_keys,
        "asked_decision_keys": asked_keys,
        "pending_ask_details": [item["prompt"] for item in batch.get("questions", [])],
        "pending_ask_type": "facts" if batch else "",
        "pending_followup_ids": question_ids,
        "internal_evidence_requirements": requirements,
        "evidence_requirement_changes": requirement_changes,
        "fact_snapshot_draft": snapshot,
        "proceed_under_uncertainty": bool(
            snapshot and snapshot.get("proceed_under_uncertainty")
        ),
        "fact_change_materiality": materiality,
        "decision_trace": trace,
        "decision_trace_id": trace_id,
        "retrieval_summary": retrieval_summary,
        "retrieval_trace_id": retrieval_trace_id,
        "retrieval_basis_candidates": basis,
        "retrieval_gaps": retrieval_gaps,
        "retrieved_law_refs": (
            targeted_law_refs
            if targeted_law_refs
            else getattr(state, "retrieved_law_refs", []) or []
        ),
        "targeted_retrieval_cache": [
            *(
                getattr(state, "targeted_retrieval_cache", []) or []
            ),
            {
                "retrieval_trace_id": retrieval_trace_id,
                "fact_blackboard_version": int(
                    getattr(state, "fact_blackboard_version", 0) or 0
                ),
                "summary": retrieval_summary,
                "basis_candidate_ids": [
                    item.get("basis_candidate_id")
                    for item in basis
                    if item.get("basis_candidate_id")
                ],
                "created_at": _now(),
            },
        ][-30:],
        "ask_rounds": int(getattr(state, "ask_rounds", 0) or 0)
        + (1 if new_batch_issued else 0),
        "facts_rounds": int(getattr(state, "facts_rounds", 0) or 0)
        + (1 if new_batch_issued else 0),
        "decision_status": decision_status,
        "next_route": next_route,
        "pause_state": pause_state,
        "workflow_stage": (
            "fact_clarification"
            if decision_status == "ask_batch"
            else "fact_snapshot"
            if decision_status == "await_snapshot_confirmation"
            else "evidence_planning"
            if decision_status == "proceed_to_evidence_planning"
            else getattr(state, "workflow_stage", "case_intake")
        ),
        "messages": [AIMessage(content=reply)] if reply else [],
        "phase": (
            GuidePhase.DETAIL_GATHER
            if decision_status == "ask_batch"
            else GuidePhase.ISSUE_SEARCH
        ),
    }
    logger.info(
        "节点④事实决策 | case={} version={} status={} decision={} questions={}",
        state.case_id,
        state.fact_blackboard_version,
        report.status,
        decision_status,
        len(question_ids),
    )
    return updates


__all__ = [
    "FactQuestionCandidate",
    "FactQuestionBatch",
    "FactSufficiencyReport",
    "InternalEvidenceRequirement",
    "activate_fact_dependencies",
    "assess_fact_sufficiency",
    "build_fact_question_candidates",
    "select_question_batch",
    "build_question_batch",
    "render_question_batch_markdown",
    "update_internal_evidence_requirements",
    "match_evidence_names_to_requirements",
    "build_targeted_retrieval_query",
    "record_retrieval_trace",
    "calculate_fact_change_materiality",
    "build_fact_snapshot_draft",
    "run_decide_facts",
]
