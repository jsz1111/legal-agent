"""Dynamic fact-blackboard update for one GuideGraph turn.

This module deliberately has no legal retrieval, question planning, evidence
evaluation, or solution-generation dependency.  It records what the user said
and what changed; the next node decides what information remains worth asking.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from loguru import logger

from src.agents.legal_guide.case_model import (
    active_case_facts,
    evidence_from_case_facts,
    fact_statements,
    format_case_context,
    legacy_fact_updates,
    reduce_case_facts,
)
from src.agents.legal_guide.evidence_analysis import split_uploaded_evidence_blocks
from src.agents.legal_guide.issue_normalizer import extract_case_facts
from src.agents.legal_guide.prepare_case import latest_user_input
from src.agents.legal_guide.state import GuidePhase, GuideState


_CHINA_TZ = timezone(timedelta(hours=8))
_TARGET_STATUS = {
    "asserted": "confirmed",
    "confirmed": "confirmed",
    "uncertain": "unclear",
    "unclear": "unclear",
    "denied": "denied",
    "conflicted": "conflicted",
    "unknown": "unknown",
    "not_provided": "not_provided",
    "superseded": "superseded",
}
_EVIDENCE_ALIAS_PATTERNS = (
    (re.compile(r"付(?:款|钱).{0,3}(?:记录|截图|凭证)|支付(?:记录|截图|凭证)|转账(?:记录|截图|凭证)"), "payment_record"),
    (re.compile(r"订单(?:详情|截图)?|商品(?:页面|截图)?"), "order_record"),
    (re.compile(r"聊天(?:记录|截图)?|对话(?:记录|截图)?"), "chat_record"),
    (re.compile(r"物流(?:记录|状态|截图)?|快递(?:记录|状态|截图)?"), "delivery_record"),
    (re.compile(r"投诉(?:工单|记录)?|客服(?:回复|记录)?"), "complaint_record"),
)
_TEMPORARILY_UNAVAILABLE_MARKERS = ("找不到", "暂时没有", "暂时找不到", "丢了", "遗失")


def _now() -> str:
    return datetime.now(_CHINA_TZ).isoformat()


def _compact(value: object, limit: int = 360) -> str:
    return " ".join(str(value or "").split())[:limit]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:18]}"


def _merge_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = _compact(value)
        if value and value not in result:
            result.append(value)
    return result


def _current_narrative(state: GuideState) -> tuple[str, list[dict[str, Any]]]:
    """Keep attached material out of user-confirmed facts."""

    raw = latest_user_input(state)
    narrative, inline_materials = split_uploaded_evidence_blocks(raw)
    narrative = narrative.strip()
    if not narrative and state.fact_payload.get("text"):
        narrative = str(state.fact_payload["text"]).strip()
    if not narrative and state.progress_payload.get("text"):
        narrative = str(state.progress_payload["text"]).strip()
    return narrative, [item for item in inline_materials if isinstance(item, dict)]


def _legacy_projection(blackboard: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a migrated canonical blackboard back to the old reducer shape."""

    records: list[dict[str, Any]] = []
    for item in blackboard:
        record = dict(item)
        target_status = str(record.get("status") or "unclear")
        record["status"] = {
            "confirmed": "asserted",
            "unclear": "uncertain",
        }.get(target_status, target_status)
        records.append(record)
    return records


def _form_fact_updates(form_updates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn application-owned form fields into grounded structured candidates."""

    updates: list[dict[str, Any]] = []
    for index, item in enumerate(form_updates or []):
        if not isinstance(item, dict):
            continue
        value = _compact(item.get("value") or item.get("text"))
        if not value:
            continue
        key = _compact(item.get("semantic_key") or item.get("key"))
        label = _compact(item.get("label") or item.get("field") or key)
        if not key:
            key = f"form.field_{index + 1}"
        updates.append(
            {
                "key": key,
                "category": _compact(item.get("category") or "event"),
                "statement": _compact(item.get("statement") or f"{label}：{value}"),
                "value": value,
                "certainty": _compact(item.get("certainty") or "asserted"),
                "operation": _compact(item.get("operation") or "add"),
                "source_text": value,
                "entity_scope": _compact(item.get("entity_scope")),
                "normalized_value": (
                    item.get("normalized_value")
                    if isinstance(item.get("normalized_value"), dict)
                    else None
                ),
            }
        )
    return updates


def _canonical_fact(record: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    """Project legacy case atoms into the normalized blackboard contract."""

    item = dict(record)
    semantic_key = _compact(item.get("semantic_key") or item.get("key"))
    source_text = _compact(item.get("source_text"))
    fact_id = _compact(item.get("fact_id"))
    if not fact_id:
        fact_id = _stable_id(
            "fact",
            case_id,
            semantic_key,
            source_text,
            item.get("turn"),
        )
    source_refs = [
        dict(source)
        for source in item.get("source_refs") or []
        if isinstance(source, dict)
    ]
    if not source_refs and source_text:
        source_refs = [
            {
                "message_id": f"turn-{item.get('turn', 0)}",
                "source_text": source_text,
                "source_type": item.get("source_type") or "user_message",
            }
        ]
    fact_status = _TARGET_STATUS.get(
        str(item.get("fact_status") or item.get("status") or "unclear"),
        "unclear",
    )
    verification = _compact(
        item.get("verification_status") or item.get("verification") or "user_stated"
    )
    return {
        **item,
        "fact_id": fact_id,
        "semantic_key": semantic_key,
        "key": semantic_key,
        "status": fact_status,
        "verification_status": verification,
        "source_refs": source_refs,
        "entity_scope": _compact(item.get("entity_scope")),
        "subject_id": _compact(item.get("subject_id")),
        "predicate": _compact(item.get("predicate") or item.get("relation")),
        "first_seen_round": int(item.get("first_seen_round") or item.get("turn") or 0),
        "last_updated_round": int(
            item.get("last_updated_round") or item.get("turn") or 0
        ),
        "supersedes_fact_id": item.get("supersedes_fact_id"),
        "superseded_by_fact_id": item.get("superseded_by_fact_id"),
        "conflict_group_id": item.get("conflict_group_id"),
    }


def _build_fact_changes(
    before: Iterable[dict[str, Any]],
    after: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    prior = {
        str(item.get("fact_id")): item
        for item in before
        if isinstance(item, dict) and item.get("fact_id")
    }
    changes: list[dict[str, Any]] = []
    for current in after:
        fact_id = str(current.get("fact_id") or "")
        if not fact_id:
            continue
        old = prior.get(fact_id)
        if old is None:
            change_type = (
                "superseded"
                if current.get("status") == "superseded"
                else "added"
            )
        elif old.get("status") != current.get("status"):
            change_type = (
                "superseded"
                if current.get("status") == "superseded"
                else "status_changed"
            )
        elif old.get("source_refs") != current.get("source_refs"):
            change_type = "source_added"
        else:
            continue
        changes.append(
            {
                "fact_id": fact_id,
                "semantic_key": current.get("semantic_key") or current.get("key"),
                "change_type": change_type,
                "status": current.get("status"),
                "supersedes_fact_id": current.get("supersedes_fact_id"),
                "conflict_group_id": current.get("conflict_group_id"),
            }
        )
    return changes


def _conflict_groups(facts: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for fact in facts:
        group_id = _compact(fact.get("conflict_group_id"))
        fact_id = _compact(fact.get("fact_id"))
        if group_id and fact_id:
            groups.setdefault(group_id, []).append(fact_id)
    return groups


def _evidence_inventory_key(name: str) -> str:
    compact = re.sub(r"\s+", "", name or "").lower()
    for pattern, alias in _EVIDENCE_ALIAS_PATTERNS:
        if pattern.search(compact):
            return alias
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)[:80] or "unnamed"


def _evidence_names_from_fact(fact: dict[str, Any]) -> list[str]:
    raw = _compact(fact.get("value") or fact.get("statement"))
    return [
        part.strip()
        for part in re.split(r"[、，,；;]+", raw)
        if part.strip()
    ]


def _update_evidence_name_inventory(
    state: GuideState,
    facts: Iterable[dict[str, Any]],
    material_refs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory = [
        dict(item) for item in state.evidence_name_inventory if isinstance(item, dict)
    ]
    changes: list[dict[str, Any]] = []

    def upsert(
        name: str,
        status: str,
        *,
        source_ref: dict[str, Any],
        fact_id: str = "",
    ) -> None:
        normalized_name = _evidence_inventory_key(name)
        index = next(
            (
                position
                for position, item in enumerate(inventory)
                if item.get("normalized_name") == normalized_name
            ),
            None,
        )
        if index is None:
            record = {
                "evidence_name_id": _stable_id(
                    "evidence-name", state.case_id, normalized_name
                ),
                "normalized_name": normalized_name,
                "display_name": name,
                "original_names": [name],
                "status": status,
                "source_refs": [source_ref],
                "fact_ids": [fact_id] if fact_id else [],
            }
            inventory.append(record)
            changes.append(
                {
                    "evidence_name_id": record["evidence_name_id"],
                    "change_type": "added",
                    "status": status,
                }
            )
            return

        record = inventory[index]
        changed = False
        if name not in record.get("original_names", []):
            record["original_names"] = [*record.get("original_names", []), name]
            changed = True
        if source_ref not in record.get("source_refs", []):
            record["source_refs"] = [*record.get("source_refs", []), source_ref]
            changed = True
        if fact_id and fact_id not in record.get("fact_ids", []):
            record["fact_ids"] = [*record.get("fact_ids", []), fact_id]
            changed = True
        if record.get("status") != status:
            record["status"] = status
            changed = True
        if changed:
            changes.append(
                {
                    "evidence_name_id": record["evidence_name_id"],
                    "change_type": "updated",
                    "status": record["status"],
                }
            )

    for fact in facts:
        if fact.get("category") != "evidence" or fact.get("status") == "superseded":
            continue
        fact_status = fact.get("status")
        status = {
            "confirmed": "user_claimed_present",
            "denied": "explicitly_absent",
            "unknown": "temporarily_unavailable",
            "unclear": "unknown",
        }.get(fact_status, "unknown")
        if any(marker in _compact(fact.get("source_text")) for marker in _TEMPORARILY_UNAVAILABLE_MARKERS):
            status = "temporarily_unavailable"
        source_ref = (
            (fact.get("source_refs") or [{}])[0]
            if isinstance(fact.get("source_refs"), list)
            else {}
        )
        for name in _evidence_names_from_fact(fact):
            upsert(name, status, source_ref=source_ref, fact_id=_compact(fact.get("fact_id")))

    for material in material_refs:
        name = _compact(material.get("name") or material.get("file_name"))
        if not name:
            continue
        upsert(
            name,
            "submitted",
            source_ref={
                "message_id": state.current_message_id or f"turn-{state.round}",
                "source_text": name,
                "source_type": "material_observation",
            },
        )
    return inventory, changes


def _material_observations(
    state: GuideState,
    inline_materials: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        dict(item)
        for item in state.material_fact_observations
        if isinstance(item, dict)
    ]
    candidates: list[dict[str, Any]] = [
        item for item in inline_materials if isinstance(item, dict)
    ]
    candidates.extend(
        item
        for item in state.current_attachments
        if isinstance(item, dict)
    )
    for item in candidates:
        file_name = _compact(item.get("name") or item.get("file_name"))
        if not file_name:
            continue
        observation_id = _stable_id(
            "material-observation",
            state.case_id,
            state.current_message_id,
            file_name,
            item.get("sha256") or item.get("digest"),
        )
        if any(existing.get("observation_id") == observation_id for existing in records):
            continue
        records.append(
            {
                "observation_id": observation_id,
                "material_id": _compact(
                    item.get("material_id") or item.get("document_id") or file_name
                ),
                "observation_type": "attachment_submitted",
                "normalized_value": {"file_name": file_name},
                "source_locator": _compact(
                    item.get("source_locator") or item.get("file_name") or file_name
                ),
                "parser_confidence": None,
                "requires_user_confirmation": True,
                "status": "pending_confirmation",
            }
        )
    return records[-120:]


def _downstream_invalidations(
    changes: Iterable[dict[str, Any]],
    state: GuideState,
) -> list[str]:
    meaningful = [
        item
        for item in changes
        if item.get("change_type") not in {"source_added"}
    ]
    if not meaningful:
        return []
    invalidations = ["decision_sufficiency", "followup_plan"]
    if any(
        _compact(item.get("semantic_key")).startswith(
            (
                "claim.",
                "relationship.",
                "actor.",
                "event.",
                "transaction.",
                "procedure.",
                "time.",
                "location.",
                "harm.",
            )
        )
        for item in meaningful
    ):
        invalidations.extend(["legal_model", "evidence_plan"])
    if state.evidence_review_version:
        invalidations.append("evidence_review")
    if state.plan_version:
        invalidations.append("plan")
    return _merge_unique(invalidations)


async def run_update_facts(state: GuideState, deps: Any) -> dict[str, Any]:
    """Apply the current event to the fact blackboard and return a checkpoint."""

    if state.awaiting_case_boundary or state.case_boundary_read_only:
        logger.info("节点③案件边界尚未确认，跳过事实写入 | case={}", state.case_id)
        return {}

    event_key = _stable_id(
        "fact-event",
        state.case_id,
        state.case_generation,
        state.event_sequence,
        state.current_message_id,
    )
    if event_key == state.last_processed_fact_event_key:
        return {
            "fact_changes": [],
            "evidence_name_changes": [],
            "downstream_invalidations": [],
            "fact_update_audit_id": _stable_id("fact-audit", event_key),
        }

    narrative, inline_materials = _current_narrative(state)
    resumed_safety_case = bool(
        state.safety_pause_case_message
        and not state.safety_pause_active
        and state.current_safety_status == "safe"
    )
    resume_observation = _compact(state.safety_pause_case_message)
    if resumed_safety_case and resume_observation:
        narrative = "\n".join(
            part for part in (resume_observation, narrative) if part
        )
    event_types = {item.get("type") for item in state.input_events}
    factual_event_types = {
        "fact_added",
        "fact_corrected",
        "fact_denied",
        "fact_batch_answered",
        "case_progress_updated",
        "mixed_update",
        "case_started",
    }
    if (
        state.turn_control_intent
        and not state.turn_contains_case_details
        and not event_types.intersection(factual_event_types)
    ):
        narrative = ""
    form_updates = _form_fact_updates(state.current_form_updates)
    source_for_forms = "\n".join(
        [narrative, *[str(item.get("source_text") or "") for item in form_updates]]
    ).strip()
    prior_blackboard = [
        _canonical_fact(item, case_id=state.case_id)
        for item in (state.fact_blackboard or state.case_facts)
        if isinstance(item, dict)
    ]
    base_case_records = state.case_facts or _legacy_projection(state.fact_blackboard)
    extractor_input = narrative
    if narrative and "【首次案件材料包】" not in narrative and state.case_facts:
        extractor_input = (
            f"[当前用户消息]\n{narrative}\n\n"
            "[已有事实，仅用于匹配同一事实或明确更正]\n"
            f"{format_case_context(state.case_facts)}\n\n"
            "只提取当前用户消息中能逐字回链的新增、更正、否认或明确不知道的事实。"
        )

    extracted = None
    llm = getattr(deps, "llm", None)
    if narrative and llm is not None:
        extracted = await extract_case_facts(
            extractor_input,
            llm,
            fallback_text=narrative,
        )
    raw_updates = (
        [item.model_dump() for item in extracted.case_updates]
        if extracted and extracted.case_updates
        else legacy_fact_updates([], user_text=narrative)
        if narrative
        else []
    )
    fact_source_type = "user_message"
    case_facts = reduce_case_facts(
        base_case_records,
        raw_updates,
        user_text=narrative,
        turn=state.round,
        case_id=state.case_id,
        message_id=state.current_message_id,
        source_type=fact_source_type,
    )
    if form_updates:
        case_facts = reduce_case_facts(
            case_facts,
            form_updates,
            user_text=source_for_forms,
            turn=state.round,
            case_id=state.case_id,
            message_id=state.current_message_id,
            source_type="structured_form",
        )

    blackboard = [
        _canonical_fact(item, case_id=state.case_id)
        for item in case_facts
        if isinstance(item, dict)
    ]
    fact_changes = _build_fact_changes(prior_blackboard, blackboard)
    inventory, evidence_changes = _update_evidence_name_inventory(
        state,
        blackboard,
        [*inline_materials, *state.current_attachments],
    )
    observations = _material_observations(state, inline_materials)
    invalidations = _downstream_invalidations(fact_changes, state)
    current_facts = active_case_facts(case_facts)
    statements = fact_statements(case_facts)
    draftable = fact_statements(case_facts, draftable_only=True)
    present, unavailable = evidence_from_case_facts(current_facts)
    submitted_names = [
        item["display_name"]
        for item in inventory
        if item.get("status") == "submitted"
    ]
    present = _merge_unique([*state.evidence_confirmed, *present, *submitted_names])
    unavailable = _merge_unique([*state.evidence_unavailable, *unavailable])
    issue_candidates = _merge_unique(
        [*state.issue_candidates, *(extracted.issues if extracted else [])]
    )
    domain_candidate = _compact(
        (extracted.domain if extracted else "") or state.domain_candidate
    )
    # ``confirmed_issues`` / ``legal_domain`` remain a compatibility projection
    # until node four is migrated. They are never treated as fact-blackboard
    # entries and the actual retrieval still validates their legal grounding.
    confirmed_issues = _merge_unique([*state.confirmed_issues, *issue_candidates])
    can_revise_early_domain = bool(
        state.legal_domain
        and state.legal_domain != "other"
        and domain_candidate
        and domain_candidate != state.legal_domain
        and not state.retrieval_completed
        and state.confidence_tier in {"", "LOW"}
        and fact_changes
    )
    legal_domain = (
        domain_candidate
        if not state.legal_domain
        or state.legal_domain == "other"
        or can_revise_early_domain
        else state.legal_domain
    )
    region = state.region or _compact(extracted.region if extracted else "")
    time_info = state.time_info or _compact(extracted.time_info if extracted else "")
    fact_audit_id = _stable_id("fact-audit", event_key)
    audit = {
        "fact_update_audit_id": fact_audit_id,
        "case_id": state.case_id,
        "case_generation": state.case_generation,
        "event_sequence": state.event_sequence,
        "message_id": state.current_message_id,
        "accepted_fact_ids": [
            item["fact_id"]
            for item in blackboard
            if any(change.get("fact_id") == item["fact_id"] for change in fact_changes)
        ],
        "fact_changes": fact_changes,
        "downstream_invalidations": invalidations,
        "fact_blackboard_version_before": state.fact_blackboard_version,
        "fact_blackboard_version_after": (
            state.fact_blackboard_version + 1 if fact_changes else state.fact_blackboard_version
        ),
        "created_at": _now(),
    }
    factual_event = bool(factual_event_types & event_types)
    phase = (
        GuidePhase.ISSUE_SEARCH
        if confirmed_issues or (legal_domain and legal_domain != "other")
        else GuidePhase.CLARIFY
    )
    fact_changed = bool(
        any(item.get("change_type") != "source_added" for item in fact_changes)
    )
    return {
        "case_facts": case_facts,
        "fact_blackboard": blackboard,
        "fact_blackboard_version": (
            state.fact_blackboard_version + 1
            if fact_changes
            else state.fact_blackboard_version
        ),
        "fact_changes": fact_changes,
        "fact_conflict_groups": _conflict_groups(blackboard),
        "collected_facts": statements,
        "draftable_facts": draftable,
        "evidence_name_inventory": inventory,
        "evidence_name_inventory_version": (
            state.evidence_name_inventory_version + 1
            if evidence_changes
            else state.evidence_name_inventory_version
        ),
        "evidence_name_changes": evidence_changes,
        "material_fact_observations": observations,
        "downstream_invalidations": invalidations,
        "fact_update_audit_id": fact_audit_id,
        "fact_update_audit_history": [*state.fact_update_audit_history, audit][-80:],
        "fact_update_degraded": bool(extracted and extracted.degraded),
        "last_processed_fact_event_key": event_key,
        "issue_candidates": issue_candidates,
        "domain_candidate": domain_candidate,
        "confirmed_issues": confirmed_issues,
        "legal_domain": legal_domain,
        "region": region,
        "time_info": time_info,
        "evidence_confirmed": present,
        "evidence_unavailable": unavailable,
        "safety_pause_case_message": "" if resumed_safety_case else state.safety_pause_case_message,
        "phase": phase,
        "workflow_stage": (
            "fact_clarification" if fact_changed else state.workflow_stage
        ),
        "fact_snapshot_confirmed": (
            False if fact_changed else state.fact_snapshot_confirmed
        ),
        "retrieval_completed": (
            False
            if "legal_model" in invalidations
            else state.retrieval_completed
        ),
        "issue_refresh_needed": bool(fact_changes),
        "pending_ask_details": [] if factual_event else state.pending_ask_details,
        "pending_ask_type": "" if factual_event else state.pending_ask_type,
        "pending_followup_ids": [] if factual_event else state.pending_followup_ids,
        "followup_plan": {} if factual_event else state.followup_plan,
    }
