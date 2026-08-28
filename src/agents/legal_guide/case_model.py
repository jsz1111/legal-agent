"""Structured case facts and deterministic cross-turn reduction.

The LLM extracts atomic facts, but this module owns persistence semantics. It
does not know about legal scenarios or vocabulary such as "barber", "leak", or
"rider". Every accepted fact must point back to text from the current user
message, which keeps later planning and document generation grounded.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field


CASE_CATEGORIES = {
    "actor", "relationship", "event", "claim", "amount", "time",
    "location", "evidence", "procedure", "harm", "uncertainty",
}
CASE_OPERATIONS = {"add", "replace", "deny"}
EVIDENCE_STATUSES = {"obtained", "lead", "unavailable", "unknown"}
_CONTROL_ONLY_TEXTS = {
    "好的", "好", "嗯", "收到", "明白", "知道了", "继续补充",
    "现在生成方案", "生成方案", "按现有信息生成方案", "按目前情况生成",
    "直接生成方案", "不补充了", "就这些",
}
_EXPLICIT_DENIAL_MARKERS = (
    "没有", "没拍", "没留", "没保存", "未拍", "未留", "未保存", "无此",
    "不存在", "不在手里", "找不到", "丢了", "遗失", "拿不出", "无法提供",
    "不是", "并非", "否认", "不承认", "不准确", "不对", "错误", "说错",
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
    evidence_status: str = Field(
        default="unknown",
        description="For category=evidence only: obtained, lead, unavailable, or unknown",
    )


def _compact(value: object, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _source_is_grounded(source_text: str, user_text: str) -> bool:
    source = _compact(source_text)
    user = _compact(user_text, limit=4000)
    return bool(source and user and source in user)


def _is_control_only_text(value: object) -> bool:
    """Reject conversation controls and acknowledgements as case facts.

    This is deliberately about message function, not legal vocabulary: a user
    may ask to generate a plan in the same message as facts, but a standalone
    acknowledgement or workflow command never becomes an evidentiary fact.
    """
    compact = re.sub(r"[\s，。！？、,.!?:：‘’'\"“”【】\[\]()（）_-]+", "", _compact(value, 300))
    candidates = {compact}
    for prefix in ("用户回应", "用户回复", "用户答复", "用户表示", "用户确认", "回应", "回复", "答复"):
        if compact.startswith(prefix):
            candidates.add(compact[len(prefix):])
    return any(candidate in _CONTROL_ONLY_TEXTS for candidate in candidates)


def _evidence_status(update: CaseFactUpdate, source_text: str) -> str:
    if update.category != "evidence":
        return ""
    if update.certainty == "denied" or update.operation == "deny":
        return "unavailable"
    if update.evidence_status in EVIDENCE_STATUSES and update.evidence_status != "unknown":
        return update.evidence_status
    # A user stating that a clue exists is not the same as holding its original
    # material.  Default conservatively to a lead unless possession is explicit.
    material = f"{update.statement} {source_text}"
    held_markers = ("已取得", "已经取得", "已保存", "保存了", "手里有", "持有", "拍了", "录了", "上传了")
    explicit_markers = (
        "保留了", "保存了", "已经保存", "已有", "我有", "持有", "掌握", "留存", "提供了",
    )
    return "obtained" if (
        any(marker in material for marker in held_markers)
        or any(marker in material for marker in explicit_markers)
    ) else "lead"


def _fact_key(update: CaseFactUpdate) -> str:
    key = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", update.key.strip()).strip("_.:-").lower()
    if key:
        return key[:100]
    material = "|".join((update.category, update.subject, update.relation, update.source_text))
    return f"{update.category}.{hashlib.sha1(material.encode('utf-8')).hexdigest()[:12]}"


def normalize_case_updates(
    raw_updates: Iterable[dict[str, Any] | CaseFactUpdate],
    *,
    user_text: str,
    turn: int,
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
        if (
            not statement
            or _is_control_only_text(statement)
            or _is_control_only_text(source_text)
            or not _source_is_grounded(source_text, user_text)
        ):
            continue
        category = update.category if update.category in CASE_CATEGORIES else "event"
        certainty = update.certainty if update.certainty in {"asserted", "uncertain", "denied"} else "uncertain"
        operation = update.operation if update.operation in CASE_OPERATIONS else "add"
        if (certainty == "denied" or operation == "deny") and not any(
            marker in source_text for marker in _EXPLICIT_DENIAL_MARKERS
        ):
            # A quoted source proves provenance, not semantic entailment. In
            # particular, a declarative detail such as "现场发现" must never be
            # transformed into "没有拍照" merely because both concern evidence.
            continue
        normalized.append({
            "key": _fact_key(update),
            "category": category,
            "statement": statement,
            "subject": _compact(update.subject, limit=100),
            "relation": _compact(update.relation, limit=100),
            "value": _compact(update.value, limit=160),
            "status": certainty,
            "operation": operation,
            "source_text": source_text,
            "turn": max(int(turn or 0), 0),
            "verification": "user_stated",
            "evidence_status": _evidence_status(update, source_text),
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
    if not source or _is_control_only_text(source):
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
) -> list[dict[str, Any]]:
    """Merge new facts while preserving correction and conflict history."""
    records = [dict(item) for item in (existing or []) if isinstance(item, dict)]
    updates = normalize_case_updates(raw_updates, user_text=user_text, turn=turn)
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
            records[exact] = {**records[exact], **update}
            continue
        if active_indexes and update["operation"] in {"replace", "deny"}:
            for index in active_indexes:
                records[index]["status"] = "superseded"
                records[index]["superseded_by_turn"] = update["turn"]
        elif active_indexes:
            # Same semantic key with a different value is never silently resolved.
            for index in active_indexes:
                records[index]["status"] = "conflicted"
            update["status"] = "conflicted"
        records.append(update)
    return records[-120:]


def active_case_facts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Apply the same guard to historical records created before the control-text
    # filter existed, so a stored "generate the plan now" never reappears in a
    # later conclusion as if it were case evidence.
    return [
        dict(item) for item in (records or [])
        if item.get("status") != "superseded"
        and not _is_control_only_text(item.get("statement"))
        and not _is_control_only_text(item.get("source_text"))
    ]


def latest_case_facts(records: Iterable[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    return [item for item in active_case_facts(records) if int(item.get("turn") or 0) == int(turn or 0)]


def fact_statements(records: Iterable[dict[str, Any]], *, draftable_only: bool = False) -> list[str]:
    result: list[str] = []
    for item in active_case_facts(records):
        if draftable_only and item.get("status") != "asserted":
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
        evidence_status = str(item.get("evidence_status") or "")
        if item.get("status") == "denied" or evidence_status == "unavailable":
            target = unavailable
        elif evidence_status == "obtained":
            target = present
        else:
            # Historical records predate evidence_status. Preserve only an
            # explicit possession statement; generic old evidence atoms stay
            # conservative and are treated as leads by the final renderer.
            historical_material = " ".join(
                str(item.get(field) or "")
                for field in ("statement", "value", "source_text")
            )
            held_markers = ("持有", "手里有", "已有", "已保存", "保存了", "拍了", "录了", "上传了")
            if evidence_status in {"", "lead", "unknown"} and (
                any(marker in historical_material for marker in held_markers)
                or any(marker in historical_material for marker in (
                    "保留了", "保存了", "已经保存", "已有", "我有", "持有", "掌握", "留存", "提供了",
                ))
            ):
                target = present
            else:
                # A clue remains in the structured fact packet for AI analysis,
                # but is not promoted to the user's confirmed evidence inventory.
                continue
        raw_name = _compact(item.get("value") or item.get("statement"), limit=300)
        names = [
            part.strip()
        for part in re.split(r"[、，,；;和]+", raw_name)
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
            "- [{key}] {statement}（状态：{status}{evidence_status}；用户原话：{source}；轮次：{turn}）".format(
                key=item.get("key", ""), statement=item.get("statement", ""),
                status=item.get("status", "uncertain"), source=item.get("source_text", ""),
                evidence_status=(f"；证据状态：{item.get('evidence_status')}" if item.get("evidence_status") else ""),
                turn=item.get("turn", 0),
            )
        )
    return "\n".join(lines) or "- 暂无已落盘的原子事实"
