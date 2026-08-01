"""Structured case facts and deterministic cross-turn reduction.

The LLM extracts atomic facts, but this module owns persistence semantics. It
does not know about legal scenarios or vocabulary such as "barber", "leak", or
"rider". Every accepted fact must point back to text from the current user
message, which keeps later planning and document generation grounded.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from pydantic import BaseModel, Field


CASE_CATEGORIES = {
    "actor", "relationship", "event", "claim", "amount", "time",
    "location", "evidence", "procedure", "harm", "uncertainty",
}
CASE_OPERATIONS = {"add", "replace", "deny"}
FACT_STATUSES = {
    "confirmed",
    "denied",
    "unclear",
    "conflicted",
    "unknown",
    "not_provided",
    "superseded",
}
_CHINA_TZ = timezone(timedelta(hours=8))
_EXPLICIT_DENIAL_MARKERS = (
    "没有", "没拍", "没留", "没保存", "未拍", "未留", "未保存", "无此",
    "不存在", "不在手里", "找不到", "丢了", "遗失", "拿不出", "无法提供",
    "不是", "并非", "否认", "不承认", "不准确", "不对", "错误", "说错",
)
_EXPLICIT_UNKNOWN_MARKERS = (
    "不知道", "不清楚", "无法确认", "不能确定", "记不清", "不确定",
)


class CaseFactUpdate(BaseModel):
    """One fact extracted from one user message."""

    key: str = Field(default="", description="Stable semantic key, for example transaction.total_paid")
    category: str = Field(default="event")
    statement: str = Field(default="", description="Concise normalized statement without legal inference")
    subject: str = ""
    relation: str = ""
    value: str = ""
    certainty: str = Field(default="asserted", description="asserted, uncertain, or denied")
    operation: str = Field(default="add", description="add, replace, or deny")
    source_text: str = Field(default="", description="Exact quote from the current user message")
    entity_scope: str = ""
    subject_id: str = ""
    predicate: str = ""
    normalized_value: dict[str, Any] | None = None


def _compact(value: object, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _source_is_grounded(source_text: str, user_text: str) -> bool:
    source = _compact(source_text)
    user = _compact(user_text, limit=4000)
    return bool(source and user and source in user)


def _fact_key(update: CaseFactUpdate) -> str:
    key = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", update.key.strip()).strip("_.:-").lower()
    if key:
        return key[:100]
    material = "|".join((update.category, update.subject, update.relation, update.source_text))
    return f"{update.category}.{hashlib.sha1(material.encode('utf-8')).hexdigest()[:12]}"


def _now() -> str:
    return datetime.now(_CHINA_TZ).isoformat()


def _stable_fact_id(
    key: str,
    source_text: str,
    *,
    case_id: str,
    turn: int,
) -> str:
    raw = "|".join((case_id, key, source_text, str(max(int(turn or 0), 0))))
    return f"fact-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:18]}"


def _target_fact_status(certainty: str, source_text: str) -> str:
    if certainty == "denied":
        return "denied"
    if certainty in {"unknown"} or any(
        marker in source_text for marker in _EXPLICIT_UNKNOWN_MARKERS
    ):
        return "unknown"
    if certainty in {"uncertain", "unclear"}:
        return "unclear"
    return "confirmed"


def _legacy_fact_status(fact_status: str) -> str:
    return {
        "confirmed": "asserted",
        "unclear": "uncertain",
    }.get(fact_status, fact_status)


def normalize_case_updates(
    raw_updates: Iterable[dict[str, Any] | CaseFactUpdate],
    *,
    user_text: str,
    turn: int,
    case_id: str = "",
    message_id: str = "",
    source_type: str = "user_message",
) -> list[dict[str, Any]]:
    """Validate model output and attach provenance owned by the application."""
    normalized: list[dict[str, Any]] = []
    for raw in raw_updates or []:
        try:
            update = raw if isinstance(raw, CaseFactUpdate) else CaseFactUpdate.model_validate(raw)
        except Exception:
            continue
        statement = _compact(update.statement, limit=300)
        source_text = _compact(update.source_text, limit=300)
        if not statement or not _source_is_grounded(source_text, user_text):
            continue
        category = update.category if update.category in CASE_CATEGORIES else "event"
        certainty = (
            update.certainty
            if update.certainty
            in {"asserted", "confirmed", "uncertain", "unclear", "denied", "unknown"}
            else "uncertain"
        )
        operation = update.operation if update.operation in CASE_OPERATIONS else "add"
        if (certainty == "denied" or operation == "deny") and not any(
            marker in source_text for marker in _EXPLICIT_DENIAL_MARKERS
        ):
            # A quoted source proves provenance, not semantic entailment. In
            # particular, a declarative detail such as "现场发现" must never be
            # transformed into "没有拍照" merely because both concern evidence.
            continue
        key = _fact_key(update)
        fact_status = _target_fact_status(certainty, source_text)
        now = _now()
        fact_id = _stable_fact_id(
            key,
            source_text,
            case_id=case_id,
            turn=turn,
        )
        source_ref = {
            "message_id": message_id or f"turn-{max(int(turn or 0), 0)}",
            "source_text": source_text,
            "source_type": source_type,
        }
        normalized.append({
            "fact_id": fact_id,
            "key": key,
            "semantic_key": key,
            "category": category,
            "statement": statement,
            "subject": _compact(update.subject, limit=100),
            "relation": _compact(update.relation, limit=100),
            "value": _compact(update.value, limit=160),
            # ``status`` remains the compatibility representation consumed by
            # the existing planner. ``fact_status`` is the seven-state contract
            # used by the new fact blackboard.
            "status": _legacy_fact_status(fact_status),
            "fact_status": fact_status,
            "operation": operation,
            "source_text": source_text,
            "source_type": source_type,
            "source_refs": [source_ref],
            "initial_source_ref": source_ref,
            "turn": max(int(turn or 0), 0),
            "verification": "user_stated",
            "verification_status": "user_stated",
            "entity_scope": _compact(update.entity_scope, limit=100),
            "subject_id": _compact(update.subject_id, limit=100),
            "predicate": _compact(update.predicate or update.relation, limit=100),
            "normalized_value": update.normalized_value,
            "first_seen_round": max(int(turn or 0), 0),
            "last_updated_round": max(int(turn or 0), 0),
            "created_at": now,
            "updated_at": now,
            "supersedes_fact_id": None,
            "superseded_by_fact_id": None,
            "conflict_group_id": None,
        })
    return normalized


def legacy_fact_updates(facts: Iterable[str], *, user_text: str) -> list[dict[str, Any]]:
    """Grounded fallback for model responses that omit ``case_updates``.

    Older prompts returned free-form summaries. Attaching those summaries to the
    whole user message only looked traceable: the summary itself could still add
    facts. The fallback therefore persists the user's own words as one atom and
    ignores the unstructured model summaries. A later turn can still replace it
    once the model emits stable semantic keys.
    """
    del facts
    source = _compact(user_text, limit=500)
    if not source:
        return []
    return [{
        "key": f"legacy.raw.{hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]}",
        "category": "event",
        "statement": source,
        "certainty": "asserted",
        "operation": "add",
        "source_text": source,
    }]


def reduce_case_facts(
    existing: Iterable[dict[str, Any]],
    raw_updates: Iterable[dict[str, Any] | CaseFactUpdate],
    *,
    user_text: str,
    turn: int,
    case_id: str = "",
    message_id: str = "",
    source_type: str = "user_message",
) -> list[dict[str, Any]]:
    """Merge new facts while preserving correction and conflict history."""
    records = [dict(item) for item in (existing or []) if isinstance(item, dict)]
    updates = normalize_case_updates(
        raw_updates,
        user_text=user_text,
        turn=turn,
        case_id=case_id,
        message_id=message_id,
        source_type=source_type,
    )
    for update in updates:
        active_indexes = [
            index for index, record in enumerate(records)
            if record.get("key") == update["key"] and record.get("status") != "superseded"
        ]
        exact = next(
            (
                index for index in active_indexes
                if records[index].get("statement") == update["statement"]
                and records[index].get("status") == update["status"]
            ),
            None,
        )
        if exact is not None:
            existing_record = records[exact]
            source_refs = list(existing_record.get("source_refs") or [])
            for source_ref in update.get("source_refs") or []:
                if source_ref not in source_refs:
                    source_refs.append(source_ref)
            records[exact] = {
                **existing_record,
                "source_refs": source_refs,
                "last_updated_round": update["last_updated_round"],
                "updated_at": update["updated_at"],
            }
            continue
        if active_indexes and update["operation"] in {"replace", "deny"}:
            superseded_ids: list[str] = []
            for index in active_indexes:
                records[index]["status"] = "superseded"
                records[index]["fact_status"] = "superseded"
                records[index]["superseded_by_turn"] = update["turn"]
                records[index]["superseded_by_fact_id"] = update["fact_id"]
                if records[index].get("fact_id"):
                    superseded_ids.append(str(records[index]["fact_id"]))
            if superseded_ids:
                update["supersedes_fact_id"] = superseded_ids[-1]
        elif active_indexes:
            # Same semantic key with a different value is never silently resolved.
            conflict_seed = "|".join(
                [
                    update["key"],
                    *[
                        str(records[index].get("fact_id") or index)
                        for index in active_indexes
                    ],
                    update["fact_id"],
                ]
            )
            conflict_group_id = (
                "conflict-"
                + hashlib.sha256(conflict_seed.encode("utf-8")).hexdigest()[:16]
            )
            for index in active_indexes:
                records[index]["status"] = "conflicted"
                records[index]["fact_status"] = "conflicted"
                records[index]["verification_status"] = "disputed"
                records[index]["conflict_group_id"] = conflict_group_id
            update["status"] = "conflicted"
            update["fact_status"] = "conflicted"
            update["verification_status"] = "disputed"
            update["conflict_group_id"] = conflict_group_id
        records.append(update)
    return records[-120:]


def active_case_facts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in (records or []) if item.get("status") != "superseded"]


def latest_case_facts(records: Iterable[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    return [item for item in active_case_facts(records) if int(item.get("turn") or 0) == int(turn or 0)]


def fact_statements(records: Iterable[dict[str, Any]], *, draftable_only: bool = False) -> list[str]:
    result: list[str] = []
    for item in active_case_facts(records):
        if draftable_only and (
            item.get("status") != "asserted"
            and item.get("fact_status") != "confirmed"
        ):
            continue
        statement = _compact(item.get("statement"), limit=300)
        if statement and statement not in result:
            result.append(statement)
    return result


def evidence_from_case_facts(records: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return user-claimed present and unavailable evidence from generic atoms."""
    present: list[str] = []
    unavailable: list[str] = []
    for item in active_case_facts(records):
        if item.get("category") != "evidence":
            continue
        target = unavailable if item.get("status") == "denied" else present
        raw_name = _compact(item.get("value") or item.get("statement"), limit=300)
        names = [
            part.strip()
            for part in re.split(r"[、，,；;]+", raw_name)
            if part.strip()
        ]
        for name in names:
            if name not in target:
                target.append(name)
    return present, unavailable


def format_case_context(records: Iterable[dict[str, Any]], *, limit: int = 40) -> str:
    """Compact, provenance-aware context for the planner and conclusion prompt."""
    lines: list[str] = []
    for item in active_case_facts(records)[-limit:]:
        lines.append(
            "- [{key}] {statement}（状态：{status}；用户原话：{source}；轮次：{turn}）".format(
                key=item.get("key", ""), statement=item.get("statement", ""),
                status=item.get("status", "uncertain"), source=item.get("source_text", ""),
                turn=item.get("turn", 0),
            )
        )
    return "\n".join(lines) or "- 暂无已落盘的原子事实"
