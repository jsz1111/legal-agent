"""Deterministic policy scoring for legal-guide follow-up candidates."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core.config import get_settings


DECISION_EFFECT_WEIGHTS: dict[str, float] = {
    "responsibility": 0.88,
    "claim_scope": 0.82,
    "limitation": 0.92,
    "jurisdiction": 0.72,
    "procedure": 0.68,
    "evidence_gap": 0.62,
    "safety": 1.0,
}


# These are legal decision dimensions, not case-specific words or scenarios.
SLOT_DECISION_EFFECTS: dict[str, tuple[str, ...]] = {
    "current_safety": ("safety",),
    "event_time": ("limitation", "procedure"),
    "employment_status": ("limitation", "procedure", "claim_scope"),
    "legal_relationship": ("responsibility", "jurisdiction", "procedure"),
    "transaction": ("responsibility", "claim_scope", "jurisdiction"),
    "claim": ("claim_scope",),
    "procedure": ("procedure", "jurisdiction"),
    "event_and_liability": ("responsibility", "procedure"),
    "insurance_and_claim": ("responsibility", "claim_scope", "procedure"),
    "administrative_action": ("responsibility", "procedure", "jurisdiction"),
    "agreement": ("responsibility", "procedure", "jurisdiction"),
    "right_type": ("responsibility", "claim_scope", "jurisdiction"),
    "infringement": ("responsibility", "claim_scope"),
    "source_and_harm": ("responsibility", "claim_scope", "evidence_gap"),
    "property_and_safety": ("claim_scope", "safety"),
    "event": ("responsibility", "procedure"),
    "harm": ("claim_scope", "evidence_gap"),
    "children": ("claim_scope", "procedure"),
}


class FollowupPolicyScore(BaseModel):
    candidate_id: str
    ask_type: str
    decision_dimension: str
    decision_effects: list[str] = Field(default_factory=list)
    information_gain: float
    user_burden: float
    priority_bonus: float
    net_score: float
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)


def candidate_decision_effects(candidate: dict[str, Any]) -> list[str]:
    """Map a catalog decision dimension to application-owned legal effects."""

    if candidate.get("kind") == "evidence":
        return ["evidence_gap"]
    slot = str(candidate.get("decision_dimension") or "")
    return list(SLOT_DECISION_EFFECTS.get(slot, ("procedure",)))


def _information_gain(effects: list[str]) -> float:
    weights = sorted(
        (DECISION_EFFECT_WEIGHTS[item] for item in set(effects) if item in DECISION_EFFECT_WEIGHTS),
        reverse=True,
    )
    if not weights:
        return 0.0
    # The strongest affected decision dominates; additional independent effects
    # add bounded value instead of allowing arbitrary score inflation.
    return min(1.0, weights[0] + 0.04 * max(0, len(weights) - 1))


def _user_burden(candidate: dict[str, Any], state: Any) -> float:
    ask_type = str(candidate.get("kind") or "facts")
    base = 0.30 if ask_type == "evidence" else 0.16
    missing = list((candidate.get("coverage") or {}).get("missing") or [])
    base += 0.04 * max(0, len(missing) - 1)
    if candidate.get("low_burden_hint") or candidate.get("alternatives"):
        base -= 0.04
    base += 0.04 * min(int(getattr(state, "consecutive_low_info_answers", 0) or 0), 3)
    soft_rounds = get_settings().GUIDE_SOFT_ASK_ROUNDS
    rounds_over_soft = max(0, int(getattr(state, "ask_rounds", 0) or 0) - soft_rounds + 1)
    base += 0.08 * rounds_over_soft
    return max(0.05, min(1.0, base))


def _minimum_value_threshold(state: Any) -> float:
    """Keep the patience threshold soft when the user explicitly opts back in."""
    if bool(getattr(state, "allow_extra_followups", False)):
        return 0.45
    soft_rounds = get_settings().GUIDE_SOFT_ASK_ROUNDS
    return (
        0.62
        if int(getattr(state, "ask_rounds", 0) or 0) >= soft_rounds
        else 0.45
    )


def score_followup_candidate(
    candidate: dict[str, Any],
    state: Any,
) -> FollowupPolicyScore:
    """Score one candidate without accepting model-generated numeric scores."""

    effects = candidate_decision_effects(candidate)
    gain = _information_gain(effects)
    burden = _user_burden(candidate, state)
    priority = max(1, int(candidate.get("priority") or 100))
    priority_bonus = max(0.0, 0.08 - 0.02 * (priority - 1))
    net_score = max(0.0, min(1.0, gain + priority_bonus - 0.45 * burden))

    minimum = _minimum_value_threshold(state)
    reasons: list[str] = []
    if not (candidate.get("coverage") or {}).get("missing"):
        reasons.append("decision_dimension_already_covered")
    if net_score < minimum:
        reasons.append("value_below_policy_threshold")
    return FollowupPolicyScore(
        candidate_id=str(candidate.get("id") or ""),
        ask_type=str(candidate.get("kind") or "facts"),
        decision_dimension=str(candidate.get("decision_dimension") or ""),
        decision_effects=effects,
        information_gain=round(gain, 4),
        user_burden=round(burden, 4),
        priority_bonus=round(priority_bonus, 4),
        net_score=round(net_score, 4),
        eligible=not reasons,
        rejection_reasons=reasons,
    )


def rank_followup_candidates(
    candidates: list[dict[str, Any]],
    state: Any,
) -> list[FollowupPolicyScore]:
    """Return a stable, auditable ordering for all unresolved candidates."""

    scores = [score_followup_candidate(candidate, state) for candidate in candidates]
    return sorted(
        scores,
        key=lambda item: (
            not item.eligible,
            -item.net_score,
            item.candidate_id,
        ),
    )


def score_dynamic_proposal(
    *,
    decision_effects: list[str],
    ask_type: str,
    state: Any,
) -> FollowupPolicyScore:
    """Score a catalog-free proposal using only validated semantic effect labels."""

    allowed = [
        item for item in dict.fromkeys(decision_effects)
        if item in DECISION_EFFECT_WEIGHTS
    ]
    synthetic = {
        "id": "dynamic",
        "kind": ask_type,
        "decision_dimension": "dynamic",
        "coverage": {"missing": ["动态决策信息"]},
        "priority": 100,
    }
    score = score_followup_candidate(synthetic, state)
    gain = _information_gain(allowed)
    net_score = max(0.0, min(1.0, gain - 0.45 * score.user_burden))
    minimum = _minimum_value_threshold(state)
    reasons = [] if allowed and net_score >= minimum else ["value_below_policy_threshold"]
    return score.model_copy(update={
        "decision_effects": allowed,
        "information_gain": round(gain, 4),
        "net_score": round(net_score, 4),
        "eligible": not reasons,
        "rejection_reasons": reasons,
    })
