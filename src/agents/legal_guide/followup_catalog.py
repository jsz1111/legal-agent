"""结构化追问题库及事实、证据回答的保守评估。"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = PROJECT_ROOT / "data" / "legal_guide" / "followup_catalog.json"


class FollowupSource(BaseModel):
    authority_level: str
    issuer: str
    title: str
    url: str = ""
    usage_note: str


class FactFollowup(BaseModel):
    id: str
    slot: str
    question: str
    why: str
    answer_hint: str = ""
    resolve_keywords: list[str] = Field(default_factory=list)
    priority: int = 100


class EvidenceFollowup(BaseModel):
    id: str
    evidence_key: str
    item: str
    question: str
    purpose: str
    alternatives: list[str] = Field(default_factory=list)
    match_keywords: list[str] = Field(default_factory=list)
    priority: int = 100


class DomainFollowups(BaseModel):
    source: FollowupSource
    facts: list[FactFollowup]
    evidence: list[EvidenceFollowup]


class FollowupCatalog(BaseModel):
    schema_version: int
    domains: dict[str, DomainFollowups]


@lru_cache(maxsize=1)
def load_followup_catalog() -> FollowupCatalog:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return FollowupCatalog.model_validate(payload)


def get_domain_followups(domain: str) -> DomainFollowups:
    catalog = load_followup_catalog()
    return catalog.domains.get(domain) or catalog.domains["other"]


def fact_followups(domain: str) -> list[FactFollowup]:
    return sorted(get_domain_followups(domain).facts, key=lambda item: (item.priority, item.id))


def evidence_followups(domain: str) -> list[EvidenceFollowup]:
    return sorted(get_domain_followups(domain).evidence, key=lambda item: (item.priority, item.id))


def find_fact_followup(domain: str, rule_id: str) -> FactFollowup | None:
    return next((rule for rule in fact_followups(domain) if rule.id == rule_id), None)


def find_evidence_followup(domain: str, rule_id: str) -> EvidenceFollowup | None:
    return next((rule for rule in evidence_followups(domain) if rule.id == rule_id), None)


_UNKNOWN_MARKERS = ("不知道", "不清楚", "不记得", "记不清", "忘了", "没注意", "说不准")
_APPROXIMATE_MARKERS = ("大概", "差不多", "可能", "好像", "左右", "约", "应该")
_CORRECTION_MARKERS = ("更正", "说错了", "不是", "改一下", "准确说")


def _short(value: str, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def fact_rule_resolved(rule: FactFollowup, state: Any) -> bool:
    record = (getattr(state, "fact_records", {}) or {}).get(rule.id) or {}
    if record.get("status") in {"user_stated", "approximate", "corrected"}:
        return True
    if rule.slot == "event_time" and getattr(state, "time_info", ""):
        return True
    if rule.slot == "region" and getattr(state, "region", ""):
        return True
    context = "".join(getattr(state, "collected_facts", []) or [])
    context += "".join(
        str(getattr(message, "content", ""))
        for message in (getattr(state, "messages", []) or [])[-8:]
        if getattr(message, "type", "") == "human"
    )
    return bool(rule.resolve_keywords) and any(keyword in context for keyword in rule.resolve_keywords)


def evidence_rule_resolved(rule: EvidenceFollowup, known_items: list[str]) -> bool:
    for known in known_items:
        if not known:
            continue
        if known in rule.item or rule.item in known:
            return True
        if any(keyword in known for keyword in rule.match_keywords):
            return True
    return False


def assess_fact_answer(
    rule: FactFollowup,
    answer: str,
    previous: dict | None = None,
) -> dict:
    """只评估陈述清晰度，不把用户回答误标为已查证事实。"""
    value = _short(answer)
    if any(marker in value for marker in _UNKNOWN_MARKERS):
        status = "unknown"
    elif any(marker in value for marker in _CORRECTION_MARKERS):
        status = "corrected"
    elif any(marker in value for marker in _APPROXIMATE_MARKERS):
        status = "approximate"
    else:
        status = "user_stated"
    if (
        previous
        and previous.get("value")
        and previous.get("value") != value
        and status not in {"unknown", "corrected"}
    ):
        status = "conflicted"
    return {
        "rule_id": rule.id,
        "slot": rule.slot,
        "value": value,
        "status": status,
        "verification": "not_independently_verified",
        "why": rule.why,
        "source": "user_statement",
    }


def assess_initial_facts(facts: list[str], existing: dict[str, dict] | None = None) -> dict[str, dict]:
    records = dict(existing or {})
    for fact in facts:
        value = _short(fact)
        if not value:
            continue
        key = "statement_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        if any(marker in value for marker in _UNKNOWN_MARKERS):
            status = "unknown"
        elif any(marker in value for marker in _APPROXIMATE_MARKERS):
            status = "approximate"
        elif value.startswith("更正"):
            status = "corrected"
        else:
            status = "user_stated"
        records[key] = {
            "rule_id": key,
            "slot": "case_statement",
            "value": value,
            "status": status,
            "verification": "not_independently_verified",
            "why": "用于形成暂定案情时间线",
            "source": "user_statement",
        }
    return records


def assess_evidence_answer(
    rule: EvidenceFollowup,
    answer: str,
    *,
    unavailable: bool,
    uploaded: bool,
    mentioned_as_present: bool,
    previous: dict | None = None,
) -> dict:
    value = _short(answer)
    if unavailable:
        availability = "unavailable"
    elif uploaded:
        availability = "uploaded_copy"
    elif mentioned_as_present:
        availability = "user_claimed_present"
    else:
        availability = "unclear"
    if previous and previous.get("availability") == "unavailable" and availability in {
        "uploaded_copy",
        "user_claimed_present",
    }:
        availability = "conflicted"
    limitations = []
    if availability not in {"unavailable", "unclear"}:
        limitations.append("真实性、完整性、取得方式和证明力尚未由系统核验")
        limitations.append("截图或复制件应尽量保留原始载体、完整上下文和形成时间")
    return {
        "rule_id": rule.id,
        "evidence_key": rule.evidence_key,
        "canonical_item": rule.item,
        "availability": availability,
        "authenticity": "not_verified" if availability not in {"unavailable", "unclear"} else "not_assessed",
        "relevance": "potentially_relevant" if availability not in {"unavailable", "unclear"} else "not_assessed",
        "legal_admissibility": "not_determined",
        "purpose": rule.purpose,
        "alternatives": list(rule.alternatives),
        "limitations": limitations,
        "answer_excerpt": value,
    }


def assess_initial_evidence(
    evidence_items: list[str],
    existing: dict[str, dict] | None = None,
) -> dict[str, dict]:
    records = dict(existing or {})
    for item in evidence_items:
        canonical = _short(item, 120)
        if not canonical:
            continue
        key = "evidence_" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
        uploaded = canonical.startswith("已上传")
        records[key] = {
            "rule_id": key,
            "evidence_key": key,
            "canonical_item": canonical,
            "availability": "uploaded_copy" if uploaded else "user_claimed_present",
            "authenticity": "not_verified",
            "relevance": "potentially_relevant",
            "legal_admissibility": "not_determined",
            "purpose": "需结合具体争议判断证明目的",
            "alternatives": [],
            "limitations": ["真实性、完整性、取得方式和证明力尚未由系统核验"],
            "answer_excerpt": canonical,
        }
    return records


def evidence_effective_count(evidence_items: list[str], assessments: dict[str, dict]) -> float:
    """仅用于方案准备度评分，不代表法律上的证据效力。"""
    if not evidence_items:
        return 0.0
    remaining = list(assessments.values())
    total = 0.0
    for item in evidence_items:
        matched = next(
            (
                record for record in remaining
                if item in str(record.get("canonical_item") or "")
                or str(record.get("canonical_item") or "") in item
            ),
            None,
        )
        availability = (matched or {}).get("availability")
        if availability == "uploaded_copy":
            total += 0.70
        elif availability in {"user_claimed_present", "conflicted"}:
            total += 0.45 if availability == "user_claimed_present" else 0.20
    return total


_FACT_STATUS_LABELS = {
    "user_stated": "用户明确陈述，未独立核验",
    "approximate": "用户提供约数，未独立核验",
    "corrected": "用户已更正，需以新陈述为准",
    "conflicted": "与前述信息不一致，需谨慎使用",
    "unknown": "用户暂不清楚",
}

_EVIDENCE_STATUS_LABELS = {
    "uploaded_copy": "已上传可查看副本，但真实性和证明力未核验",
    "user_claimed_present": "用户称持有，系统尚未查看原件",
    "conflicted": "关于是否持有的陈述前后不一致",
    "unavailable": "用户明确表示暂时没有",
    "unclear": "是否持有尚不明确",
}


def format_fact_assessments(records: dict[str, dict]) -> str:
    if not records:
        return "（暂无结构化事实评估）"
    lines = []
    for record in list(records.values())[-8:]:
        label = _FACT_STATUS_LABELS.get(record.get("status"), "状态待确认")
        lines.append(f"- {record.get('value') or record.get('slot')}：{label}")
    return "\n".join(lines)


def format_evidence_assessments(records: dict[str, dict]) -> str:
    if not records:
        return "（暂无结构化证据评估）"
    lines = []
    for record in list(records.values())[-8:]:
        label = _EVIDENCE_STATUS_LABELS.get(record.get("availability"), "状态待确认")
        purpose = record.get("purpose") or "证明目的待结合案情判断"
        lines.append(f"- {record.get('canonical_item')}：{label}；可能用途：{purpose}；法律上的可采性尚未确定")
    return "\n".join(lines)
