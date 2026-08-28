"""Decision-oriented sufficiency assessment for legal-guide convergence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from src.agents.legal_guide.followup_catalog import (
    evidence_rule_resolved,
    fact_rule_resolved,
    get_domain_followups,
)
from src.agents.legal_guide.followup_planner import (
    build_followup_candidates,
    candidate_coverage,
)
from src.agents.legal_guide.followup_policy import (
    SLOT_DECISION_EFFECTS,
)
from src.agents.legal_guide.evidence_analysis import (
    case_context_text,
    coverage_for_rule,
    evaluate_state_evidence,
    evidence_rule_relevant,
)


DECISION_EFFECT_LABELS = {
    "responsibility": "责任主体与责任范围",
    "claim_scope": "请求类型、范围或金额",
    "limitation": "期限与关键时间节点",
    "jurisdiction": "受理机构与管辖",
    "procedure": "下一步程序路径",
    "evidence_gap": "关键证据缺口",
    "safety": "当前安全措施",
    "scenario": "场景与领域确认",
}


class DecisionDimensionStatus(BaseModel):
    effect: str
    label: str
    required: bool = True
    satisfied: bool = False
    resolved_rule_ids: list[str] = Field(default_factory=list)
    unresolved_rule_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class DecisionSufficiencyReport(BaseModel):
    sufficient_for_definitive_plan: bool = False
    can_conclude_conditionally: bool = False
    recommended_action: str = "ask"
    blocking_gaps: list[str] = Field(default_factory=list)
    advisory_gaps: list[str] = Field(default_factory=list)
    dimensions: list[DecisionDimensionStatus] = Field(default_factory=list)
    reason: str = ""


def _applicable_rules(
    state: Any,
) -> list[tuple[str, list[str], bool, list[str]]]:
    rules = get_domain_followups(getattr(state, "legal_domain", ""))
    context_text = case_context_text(state)
    result: list[tuple[str, list[str], bool, list[str]]] = []
    for rule in rules.facts:
        if rule.slot == "current_safety" and not getattr(state, "safety_relevant", False):
            continue
        effects = list(SLOT_DECISION_EFFECTS.get(rule.slot, ("procedure",)))
        coverage = candidate_coverage(rule.slot, state)
        resolved_by_dimensions = bool(
            coverage.get("known") and not coverage.get("missing")
        )
        resolved = fact_rule_resolved(rule, state) or resolved_by_dimensions
        result.append((
            rule.id,
            effects,
            resolved,
            list(coverage.get("missing") or []),
        ))
    raw_evidence_report = getattr(state, "evidence_coverage", {}) or {}
    evidence_report = raw_evidence_report or evaluate_state_evidence(state)
    for rule in rules.evidence:
        # 与本案细节库不相关的证据点不参与收敛：陌生互殴案不会被
        # “聊天/转账记录缺失”阻塞。
        if not evidence_rule_relevant(rule, context_text):
            continue
        coverage = coverage_for_rule(evidence_report, rule.id)
        if coverage:
            resolved = coverage.status == "preliminarily_covered"
            # E1：统一消费覆盖行的缺失文案，不在此处各自硬编码。
            missing = [coverage.missing_text] if coverage.missing_text else []
        else:
            known_evidence = list(
                getattr(state, "evidence_confirmed", []) or []
            )
            resolved = evidence_rule_resolved(rule, known_evidence)
            missing = [] if resolved else [rule.item]
        result.append((
            rule.id,
            ["evidence_gap"],
            resolved,
            missing,
        ))
    return result


def assess_decision_sufficiency(state: Any) -> DecisionSufficiencyReport:
    """Assess whether minimum legal decisions are supported by current state."""

    unresolved_candidates, _ = build_followup_candidates(state)
    effect_rules: dict[str, list[str]] = defaultdict(list)
    resolved_by_id: dict[str, bool] = {}
    missing_by_id: dict[str, list[str]] = {}
    for rule_id, effects, resolved, missing in _applicable_rules(state):
        resolved_by_id[rule_id] = resolved
        missing_by_id[rule_id] = missing
        for effect in effects:
            effect_rules[effect].append(rule_id)

    dimensions: list[DecisionDimensionStatus] = []
    for effect in DECISION_EFFECT_LABELS:
        rule_ids = effect_rules.get(effect, [])
        if not rule_ids:
            continue
        unresolved_ids = [item for item in rule_ids if not resolved_by_id.get(item, False)]
        resolved_ids = [item for item in rule_ids if resolved_by_id.get(item, False)]
        missing: list[str] = []
        for rule_id in unresolved_ids:
            for label in missing_by_id.get(rule_id, []):
                if label and label not in missing:
                    missing.append(str(label))
        dimensions.append(DecisionDimensionStatus(
            effect=effect,
            label=DECISION_EFFECT_LABELS[effect],
            satisfied=not unresolved_ids,
            resolved_rule_ids=resolved_ids,
            unresolved_rule_ids=unresolved_ids,
            missing_information=missing,
        ))

    scenario_analysis = getattr(state, "scenario_analysis", None) or {}
    try:
        scenario_confidence = float(scenario_analysis.get("confidence") or 0.0)
    except (TypeError, ValueError):
        scenario_confidence = 0.0
    scenario_ambiguous = (
        scenario_confidence < 0.65
        and not getattr(state, "scenario_confirmation_offered", False)
        and bool(scenario_analysis.get("discriminating_facts"))
        and bool(scenario_analysis.get("confirmation_options"))
    )
    if scenario_ambiguous:
        dimensions.append(DecisionDimensionStatus(
            effect="scenario",
            label=DECISION_EFFECT_LABELS["scenario"],
            required=True,
            satisfied=False,
            unresolved_rule_ids=["scenario_disambiguation"],
            missing_information=list(scenario_analysis.get("discriminating_facts") or [])[:3],
        ))

    blocking_effects = {
        "responsibility",
        "claim_scope",
        "limitation",
        "safety",
    }
    blocking_gaps = [
        item.effect
        for item in dimensions
        if not item.satisfied and item.effect in blocking_effects
    ]
    advisory_gaps = [
        item.effect
        for item in dimensions
        if not item.satisfied and item.effect not in blocking_effects
    ]
    unresolved_any = any(not item.satisfied for item in dimensions)
    has_issue = bool(
        getattr(state, "confirmed_issues", [])
        or (
            getattr(state, "legal_domain", "")
            and getattr(state, "legal_domain", "") != "other"
        )
    )
    has_case_facts = bool(
        getattr(state, "case_facts", [])
        or getattr(state, "collected_facts", [])
    )
    definitive = has_issue and has_case_facts and not unresolved_any
    conditional = has_issue and has_case_facts and not definitive

    if definitive:
        action = "conclude_definitive"
        reason = "最低法律决策维度已经覆盖，可以生成明确行动方案"
    elif conditional:
        action = "ask" if unresolved_candidates else "conclude_conditional"
        reason = "可以先给条件式行动方案，但仍存在会影响判断的信息缺口"
    else:
        action = "ask"
        reason = "尚未形成可供行动的最低案件事实"
    return DecisionSufficiencyReport(
        sufficient_for_definitive_plan=definitive,
        can_conclude_conditionally=conditional,
        recommended_action=action,
        blocking_gaps=blocking_gaps,
        advisory_gaps=advisory_gaps,
        dimensions=dimensions,
        reason=reason,
    )


def unresolved_decision_summary(report: DecisionSufficiencyReport) -> list[str]:
    """Render deterministic, user-facing uncertainty statements."""

    result: list[str] = []
    for item in report.dimensions:
        if item.satisfied:
            continue
        missing = "、".join(item.missing_information) or "相关关键事实"
        if item.effect == "evidence_gap":
            result.append(f"{item.label}：{missing}")
        else:
            result.append(f"{item.label}：仍缺少{missing}")
    return result
