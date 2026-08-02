"""Shared serialization for the case workspace returned to every client path."""
from __future__ import annotations

from typing import Any

from src.agents.legal_guide.state import GuideState


def guide_debug_payload(state: GuideState) -> dict[str, Any]:
    """Build one complete UI/debug projection for direct and Supervisor routes."""
    plan = state.followup_plan or {}
    followup_form = None
    if plan.get("plan_kind") in {"followup_form", "evidence_collection"}:
        followup_form = {
            "kind": plan.get("plan_kind"),
            "questions": plan.get("questions") or [],
            "planner_mode": plan.get("planner_mode") or "",
            "can_answer_in_chat": True,
            "can_conclude_now": True,
        }
    elif state.phase.value in {"CONCLUDE", "__end__"} and any(
        isinstance(item, dict) and item.get("active", True)
        for item in (state.evidence_requirements or [])
    ):
        # Keep the upload/assessment channel visible after a solution.  Users
        # can supplement missing evidence or replace a material and then ask
        # for a solution revision without starting a new case.
        followup_form = {
            "kind": "evidence_collection",
            "questions": [],
            "planner_mode": "post_solution_evidence_revision",
            "can_answer_in_chat": True,
            "can_conclude_now": True,
        }
    sufficiency = state.decision_sufficiency or {}
    unresolved = [
        item for item in (sufficiency.get("dimensions") or [])
        if isinstance(item, dict) and not item.get("satisfied")
    ]
    full_basis_refs = [
        item for item in (state.followup_basis_refs or state.retrieved_law_refs or [])
        if isinstance(item, dict)
    ]

    def enrich_basis_refs(requirement: dict) -> dict:
        """Backfill body text for checklist rows saved by older app versions."""

        enriched: list[dict] = []
        for basis in requirement.get("basis_refs") or []:
            if not isinstance(basis, dict):
                continue
            title = str(basis.get("title") or "").strip()
            article_no = str(basis.get("article_no") or "").strip()
            match = next(
                (
                    item for item in full_basis_refs
                    if str(item.get("title") or "").strip() == title
                    and str(item.get("article_no") or "").strip() == article_no
                ),
                None,
            )
            merged = dict(match or {})
            merged.update({
                key: value for key, value in basis.items()
                if value not in (None, "", [], {})
            })
            enriched.append(merged)
        return {**requirement, "basis_refs": enriched}

    evidence_checklist = [
        enrich_basis_refs(item)
        for item in (state.evidence_requirements or [])
        if isinstance(item, dict) and item.get("active", True)
    ]
    return {
        "case_id": state.case_id,
        "case_boundary_status": (
            "awaiting_confirmation" if state.awaiting_case_boundary else "resolved"
        ),
        "domain": state.legal_domain or "",
        "confidence_tier": state.confidence_tier or "GATHERING",
        "statute_hits": state.law_context_str or "",
        "case_hits": state.case_context_str or "",
        "graph_laws": state.candidate_laws or [],
        "graph_channels": state.relevant_channels or [],
        "followup_basis_refs": state.followup_basis_refs or [],
        "followup_basis_error": state.followup_basis_error or "",
        "fallback_guide": state.fallback_guide,
        "detail_store": [
            item for item in (state.case_facts or [])
            if isinstance(item, dict) and item.get("status") != "superseded"
        ],
        "followup_form": followup_form,
        "evidence_checklist": evidence_checklist,
        "evidence_requirement_version": state.evidence_requirement_version,
        "evidence_evaluation_version": state.evidence_evaluation_version,
        "solution_version": state.solution_version,
        "solution_evidence_version": state.solution_evidence_version,
        "convergence": {
            "facts_converged": not any(
                item.get("effect") != "evidence_gap" for item in unresolved
            ),
            "unresolved_dimensions": unresolved,
            "reason": sufficiency.get("reason") or "",
            "user_can_conclude_now": True,
        },
    }
