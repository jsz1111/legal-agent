"""Formal node six: conservative assessment of submitted evidence materials.

The node is intentionally separate from both fact collection and evidence
planning.  It consumes the current formal evidence plan, stores a batch-level
checkpoint, and produces auditable material observations and proof-target
coverage.  It never upgrades a user statement or an uploaded copy into a
judicial finding of authenticity, legality, admissibility, or final proof.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src.agents.legal_guide.evidence_analysis import (
    EvidenceCoverage,
    EvidenceEvaluationReport,
    EvidenceItem,
    EvidenceLink,
    inspect_uploaded_evidence_blocks,
    split_uploaded_evidence_blocks,
)
from src.agents.legal_guide.evidence_rules import resolve_state_evidence_checklist
from src.agents.legal_guide.state import GuidePhase


ASSESSMENT_SCHEMA_VERSION = 1
ASSESSMENT_STATUSES = {
    "awaiting_batch",
    "completed",
    "partial",
    "needs_verification",
    "received_pending_remap",
    "degraded",
}

_ACTIVE_REQUIREMENT_STATUSES = {
    "",
    "active",
    "not_submitted",
    "temporarily_unavailable",
    "user_claimed_present",
    "available_for_third_party_request",
    "user_claimed_unavailable",
    "submitted",
}
_CLOSED_REQUIREMENT_STATUSES = {"stale", "not_applicable", "superseded"}
_UPLOADED_STATUSES = {"uploaded", "received", "stored"}
_UNKNOWN_WORDS = ("不清楚", "不知道", "不确定", "说不准", "无法确认")

_QUALITY_LABELS = {
    "source_form": "原件或原始电子载体情况",
    "completeness": "内容完整性和上下文",
    "identity_visibility": "相关主体或账号是否清晰",
    "time_visibility": "形成时间是否清晰",
    "acquisition_method": "材料取得或导出方式",
    "case_specificity": "是否包含本案具体记录",
}

_SOURCE_FORM_BY_SUFFIX = {
    ".pdf": "exported_file",
    ".docx": "native_electronic",
    ".txt": "native_electronic",
    ".csv": "exported_file",
}

_CLAIM_FACT_MARKERS = {
    "transaction.amount": ("金额", "付款", "支付", "转账", "元"),
    "transaction.payment_date": ("付款时间", "支付时间", "下单时间", "日期", "年"),
    "transaction.order_id": ("订单", "交易号", "流水号"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(value: Any, limit: int = 420) -> str:
    return " ".join(str(value or "").split())[:limit]


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _compact(value)
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:18]}"


def _hash_payload(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latest_message(state: Any) -> str:
    current = _compact(getattr(state, "current_message_text", ""), 20_000)
    if current:
        return current
    for message in reversed(getattr(state, "messages", []) or []):
        if isinstance(message, HumanMessage) and _compact(message.content):
            return _compact(message.content, 20_000)
    return ""


def _active_requirements(state: Any) -> list[dict[str, Any]]:
    rows = []
    for raw in getattr(state, "formal_evidence_requirements", []) or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "active")
        if status in _CLOSED_REQUIREMENT_STATUSES:
            continue
        if status not in _ACTIVE_REQUIREMENT_STATUSES:
            continue
        rows.append(dict(raw))
    return rows


def _requirement_map(state: Any) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("requirement_id")): item
        for item in _active_requirements(state)
        if item.get("requirement_id")
    }


def _target_id(requirement: dict[str, Any]) -> str:
    return _compact(
        requirement.get("proof_target_id")
        or requirement.get("target_id")
        or f"proof.{requirement.get('requirement_id')}",
        180,
    )


def _target_label(requirement: dict[str, Any]) -> str:
    return _compact(
        requirement.get("label")
        or requirement.get("requirement_id")
        or "未命名证明目标",
        180,
    )


def _infer_source_form(file_name: str, file_type: str = "") -> str:
    suffix = ""
    try:
        suffix = "." + str(file_name or "").rsplit(".", 1)[-1].lower()
    except Exception:
        suffix = ""
    if suffix in _SOURCE_FORM_BY_SUFFIX:
        return _SOURCE_FORM_BY_SUFFIX[suffix]
    if str(file_type or "").startswith("image/") or suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
    }:
        return "screenshot"
    return "unknown"


def _material_rows(state: Any) -> list[dict[str, Any]]:
    """Read only this turn's explicit attachment references.

    Attachment metadata is the authoritative identity of a submission.  Text
    extracted into the message is used as a bounded observation, never as a
    replacement for the metadata or the original file.
    """

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    payloads = [
        *(getattr(state, "current_attachments", []) or []),
        *(
            (getattr(state, "evidence_payload", {}) or {}).get("attachments")
            or []
        ),
    ]
    for raw in payloads:
        if not isinstance(raw, dict):
            continue
        name = _compact(raw.get("file_name") or raw.get("name"), 180)
        digest = _compact(raw.get("sha256") or raw.get("digest"), 120).lower()
        material_id = _compact(
            raw.get("material_id")
            or raw.get("document_id")
            or _stable_id("material", state.case_id, name, digest),
            180,
        )
        if not name and not digest:
            continue
        dedupe_key = digest or material_id or name
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "material_id": material_id,
                "file_name": name or "未命名材料",
                "file_type": _compact(raw.get("file_type"), 80),
                "sha256": digest,
                "upload_status": _compact(raw.get("upload_status") or "uploaded", 40),
                "requirement_id": _compact(raw.get("evidence_requirement_id"), 160),
                "evidence_batch_id": _compact(raw.get("evidence_batch_id"), 160),
                "proof_target_id": _compact(raw.get("proof_target_id"), 180),
                "source_form": _compact(raw.get("source_form"), 80),
                "original_available": raw.get("original_available"),
                "acquisition_method": _compact(raw.get("acquisition_method"), 100),
                "user_note": _compact(raw.get("user_note"), 400),
            }
        )
    return rows[:30]


def validate_evidence_plan_version(
    state: Any,
    materials: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the plan/fact versions carried by the current submission."""

    requirements = _requirement_map(state)
    if not requirements or int(getattr(state, "evidence_plan_version", 0) or 0) <= 0:
        return {
            "valid": False,
            "status": "received_pending_remap",
            "reason": "evidence_plan_missing",
            "message": "当前还没有可用于评估的正式证据清单，材料会先保留并等待归类。",
            "remap_material_ids": [
                str(item.get("material_id"))
                for item in materials or []
                if item.get("material_id")
            ],
        }

    mismatches: list[str] = []
    base_generation = getattr(state, "base_case_generation", None)
    if base_generation is not None and int(base_generation) != int(
        getattr(state, "case_generation", 1) or 1
    ):
        mismatches.append("case_generation")
    base_snapshot = getattr(state, "base_fact_snapshot_version", None)
    if base_snapshot is not None and int(base_snapshot) != int(
        getattr(state, "fact_snapshot_version", 0) or 0
    ):
        mismatches.append("fact_snapshot_version")
    base_plan = getattr(state, "base_evidence_plan_version", None)
    if base_plan is not None and int(base_plan) != int(
        getattr(state, "evidence_plan_version", 0) or 0
    ):
        mismatches.append("evidence_plan_version")

    remap_ids: list[str] = []
    current_batch_id = _compact(getattr(state, "evidence_batch_id", ""))
    for item in materials or []:
        batch_id = _compact(item.get("evidence_batch_id"))
        if batch_id and current_batch_id and batch_id != current_batch_id:
            remap_ids.append(str(item.get("material_id") or item.get("file_name")))
        requirement_id = _compact(item.get("requirement_id"))
        if requirement_id and requirement_id not in requirements:
            remap_ids.append(str(item.get("material_id") or item.get("file_name")))

    if mismatches:
        return {
            "valid": False,
            "status": "received_pending_remap",
            "reason": "upstream_version_stale",
            "message": "材料提交时使用的事实或证据清单版本已经变化，原文件会保留，待按最新清单重新归类。",
            "version_mismatches": mismatches,
            "remap_material_ids": _unique(remap_ids),
        }
    return {
        "valid": True,
        "status": "active",
        "reason": "",
        "message": "",
        "version_mismatches": [],
        "remap_material_ids": _unique(remap_ids),
    }


def _observation_for(
    material: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    name = str(material.get("file_name") or "")
    digest = str(material.get("sha256") or "").lower()
    matches = [
        item
        for item in observations
        if str(item.get("name") or "") == name
        or (
            digest
            and str(item.get("content_digest") or "").lower() == digest
        )
    ]
    return dict(matches[0]) if matches else None


def _source_locator(material: dict[str, Any], observation: dict[str, Any] | None) -> str:
    if observation and observation.get("source_locator"):
        return _compact(observation.get("source_locator"), 240)
    if observation and observation.get("uploaded_copy"):
        if str(material.get("file_type") or "").startswith("image/"):
            return f"{material.get('file_name')}（视觉识别摘要，需核对原图）"
        return f"{material.get('file_name')}（解析文本，需核对原文件）"
    return f"{material.get('file_name')}（材料元数据）"


def _material_observation(
    state: Any,
    material: dict[str, Any],
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize a material observation while preserving unknown attributes."""

    observed = dict(observation or {})
    excerpt = _compact(
        observed.get("content_excerpt")
        or observed.get("source_text")
        or material.get("user_note"),
        1200,
    )
    parser_status = "parsed" if excerpt and observed.get("uploaded_copy") else (
        "received_but_unparsed" if material.get("upload_status") in _UPLOADED_STATUSES else "unknown"
    )
    source_form = (
        _compact(observed.get("source_form"))
        or _compact(material.get("source_form"))
        or _infer_source_form(material.get("file_name"), material.get("file_type"))
    )
    record = {
        "observation_id": _stable_id(
            "material-observation",
            state.case_id,
            material.get("material_id"),
            material.get("sha256"),
        ),
        "material_id": material.get("material_id"),
        "file_name": material.get("file_name"),
        "requirement_id": material.get("requirement_id"),
        "proof_target_id": material.get("proof_target_id"),
        "source_locator": _source_locator(material, observed),
        "observed_text": excerpt,
        "source_excerpt": excerpt[:300],
        "source_form": source_form or "unknown",
        "completeness": _compact(observed.get("completeness") or "unknown"),
        "identity_visibility": _compact(observed.get("identity_visibility") or "unknown"),
        "time_visibility": _compact(observed.get("time_visibility") or "unknown"),
        "acquisition_method": _compact(
            observed.get("acquisition_method")
            or material.get("acquisition_method")
            or "unknown"
        ),
        "case_specificity": _compact(observed.get("case_specificity") or "unknown"),
        "proof_roles": [
            str(role)
            for role in (observed.get("proof_roles") or [])
            if role
        ][:6],
        "parser_status": parser_status,
        "content_digest": _compact(
            observed.get("content_digest") or material.get("sha256")
        ),
        "material_claims": [
            dict(item)
            for item in (observed.get("material_claims") or [])
            if isinstance(item, dict)
        ],
        "original_available": material.get("original_available"),
        "user_note": material.get("user_note") or "",
    }
    return record


async def parse_material_batch(
    state: Any,
    deps: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse only the current batch and return material rows plus observations."""

    materials = _material_rows(state)
    message = _latest_message(state)
    _narrative, inline_observations = split_uploaded_evidence_blocks(message)
    inspected: list[dict[str, Any]] = []
    llm = getattr(deps, "llm", None) if deps is not None else None
    if materials and llm is not None and message:
        try:
            inspected = await inspect_uploaded_evidence_blocks(message, llm)
        except Exception as exc:
            logger.warning("节点⑥材料观察降级为确定性元数据 | error={}", exc)
    all_observations = [*inline_observations, *inspected]
    observations: list[dict[str, Any]] = []
    for material in materials:
        observations.append(_material_observation(
            state,
            material,
            _observation_for(material, all_observations),
        ))

    # A programmatically parsed block may arrive without an attachment ref in
    # old Gradio sessions.  Keep it as an unclassified material instead of
    # silently dropping it.
    known_names = {
        str(item.get("file_name") or "")
        for item in materials
    }
    for raw in all_observations:
        name = _compact(raw.get("name"))
        if not name or name in known_names:
            continue
        material = {
            "material_id": _stable_id(
                "material",
                state.case_id,
                name,
                raw.get("content_digest"),
            ),
            "file_name": name,
            "file_type": "",
            "sha256": _compact(raw.get("content_digest")),
            "upload_status": "uploaded",
            "requirement_id": "",
            "evidence_batch_id": _compact(getattr(state, "evidence_batch_id", "")),
            "proof_target_id": "",
            "source_form": _compact(raw.get("source_form")),
            "original_available": None,
            "acquisition_method": "",
            "user_note": "",
        }
        materials.append(material)
        observations.append(_material_observation(state, material, raw))
    return materials[:30], observations[:30]


def _material_matches_requirement(
    material: dict[str, Any],
    requirement: dict[str, Any],
) -> bool:
    text = " ".join(
        [
            str(material.get("file_name") or ""),
            str(material.get("user_note") or ""),
            str(material.get("observed_text") or ""),
        ]
    ).lower()
    markers = [
        requirement.get("label"),
        requirement.get("purpose"),
        *(requirement.get("recommended_materials") or []),
        *(requirement.get("alternative_materials") or []),
    ]
    return any(
        marker
        and (
            str(marker).lower() in text
            or text in str(marker).lower()
        )
        for marker in markers
    )


def map_materials_to_requirements(
    state: Any,
    materials: Iterable[dict[str, Any]],
    observations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach each material to an explicit requirement or keep it unclassified."""

    requirements = _requirement_map(state)
    observation_map = {
        str(item.get("material_id")): item
        for item in observations
        if isinstance(item, dict) and item.get("material_id")
    }
    mapped: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    for raw in materials:
        material = dict(raw)
        material_id = str(material.get("material_id") or "")
        explicit_id = str(material.get("requirement_id") or "")
        requirement = requirements.get(explicit_id) if explicit_id else None
        if requirement is None and material.get("proof_target_id"):
            requirement = next(
                (
                    item
                    for item in requirements.values()
                    if _target_id(item) == material.get("proof_target_id")
                ),
                None,
            )
        if requirement is None:
            requirement = next(
                (
                    item
                    for item in requirements.values()
                    if _material_matches_requirement(
                        {
                            **material,
                            **(observation_map.get(material_id) or {}),
                        },
                        item,
                    )
                ),
                None,
            )
        if requirement is None:
            material["material_status"] = "unclassified"
            material["submission_status"] = (
                "submitted"
                if material.get("upload_status") in _UPLOADED_STATUSES
                else "received_but_unstored"
            )
            unclassified.append(material)
            mapped.append(material)
            continue
        material["requirement_id"] = str(requirement.get("requirement_id"))
        material["proof_target_id"] = _target_id(requirement)
        material["material_status"] = "mapped"
        material["submission_status"] = (
            "submitted"
            if material.get("upload_status") in _UPLOADED_STATUSES
            else "received_but_unstored"
        )
        mapped.append(material)
    return mapped, unclassified


def _quality_gaps(observation: dict[str, Any]) -> list[str]:
    if observation.get("parser_status") != "parsed":
        return ["材料内容尚未成功解析或缺少可定位文字"]
    if observation.get("case_specificity") == "blank_or_reference":
        return ["材料是空白模板或参考资料，不包含本案具体记录"]
    fields = (
        "source_form",
        "completeness",
        "identity_visibility",
        "time_visibility",
        "acquisition_method",
        "case_specificity",
    )
    return [
        _QUALITY_LABELS[field]
        for field in fields
        if str(observation.get(field) or "unknown")
        in {"", "unknown", "unclear", "partial"}
    ]


def _default_limitations(requirement: dict[str, Any] | None) -> list[str]:
    purpose = _compact((requirement or {}).get("purpose"))
    limitations = [
        "系统未核验真实性、取得方式的合法性、可采性或最终证明力",
        "材料只能在其可见内容和定位范围内支持事实，不能自动证明对方责任",
    ]
    if purpose:
        limitations.append(f"不能仅凭该材料完整证明“{purpose}”")
    return limitations


def mark_material_content_conflicts(
    observations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find contradictory, source-anchored fields without selecting a winner."""

    rows = [dict(item) for item in observations if isinstance(item, dict)]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for claim in row.get("material_claims") or []:
            if not isinstance(claim, dict):
                continue
            key = _compact(claim.get("key"))
            value = _compact(claim.get("value"))
            source_text = _compact(claim.get("source_text"))
            if key and value and source_text:
                groups.setdefault(key, []).append(
                    {
                        "material_id": row.get("material_id"),
                        "source_text": source_text,
                        "value": value,
                    }
                )
    conflicts: list[dict[str, Any]] = []
    for key, claims in groups.items():
        values = sorted({str(item.get("value")) for item in claims})
        if len(values) <= 1:
            continue
        conflicts.append(
            {
                "conflict_id": _stable_id("material-conflict", key, values),
                "claim_key": key,
                "values": values,
                "materials": claims,
                "status": "pending_fact_confirmation",
            }
        )
    conflict_by_material: dict[str, list[dict[str, Any]]] = {}
    for conflict in conflicts:
        for item in conflict.get("materials") or []:
            conflict_by_material.setdefault(
                str(item.get("material_id") or ""),
                [],
            ).append(conflict)
    for row in rows:
        row["content_conflicts"] = conflict_by_material.get(
            str(row.get("material_id") or ""),
            [],
        )
    return rows, conflicts


def assess_material_item(
    material: dict[str, Any],
    observation: dict[str, Any],
    requirement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create one material report from observations and plan metadata."""

    submission_status = str(material.get("submission_status") or "submitted")
    gaps = _quality_gaps(observation)
    if submission_status != "submitted":
        status = "received_but_unstored"
    elif observation.get("parser_status") != "parsed":
        status = "received_but_unparsed"
    elif observation.get("case_specificity") == "blank_or_reference":
        status = "not_relevant_to_current_target"
    elif observation.get("content_conflicts"):
        status = "conflicted"
    elif gaps:
        status = "partially_supports"
    else:
        status = "supports"

    purpose = _compact(
        (requirement or {}).get("purpose")
        or "需结合案件证明目标判断材料用途",
        320,
    )
    limitations = _unique([
        *_default_limitations(requirement),
        *(
            ["材料属性存在冲突，不能自行选择其中一种解释"]
            if observation.get("content_conflicts")
            else []
        ),
        *gaps,
    ])
    support_scope = (
        f"在当前可见内容和定位范围内，可初步支持：{purpose}"
        if status in {"supports", "partially_supports"}
        else "当前不能可靠确定可支持的具体范围"
    )
    return {
        "evidence_id": _stable_id(
            "evidence-item",
            material.get("material_id"),
            material.get("sha256"),
        ),
        "material_id": material.get("material_id"),
        "requirement_id": material.get("requirement_id") or "",
        "proof_target_id": material.get("proof_target_id") or "",
        "name": material.get("file_name") or "未命名材料",
        "file_name": material.get("file_name") or "未命名材料",
        "file_type": material.get("file_type") or "",
        "sha256": material.get("sha256") or "",
        "availability": (
            "submitted"
            if submission_status == "submitted"
            else "received_but_unstored"
        ),
        "submission_status": submission_status,
        "assessment_status": status,
        "parser_status": observation.get("parser_status") or "unknown",
        "source_form": observation.get("source_form") or "unknown",
        "original_available": observation.get("original_available"),
        "acquisition_method": observation.get("acquisition_method") or "unknown",
        "completeness": observation.get("completeness") or "unknown",
        "identity_visibility": observation.get("identity_visibility") or "unknown",
        "time_visibility": observation.get("time_visibility") or "unknown",
        "case_specificity": observation.get("case_specificity") or "unknown",
        "authenticity_status": "unknown",
        "legality_risk": "unknown",
        "admissibility_note": "not_determined",
        "probative_scope": support_scope,
        "source_excerpt": observation.get("source_excerpt") or "",
        "source_locator": observation.get("source_locator") or "",
        "observed_text": observation.get("observed_text") or "",
        "content_conflicts": list(observation.get("content_conflicts") or []),
        "quality_gaps": gaps,
        "possible_support": support_scope,
        "cannot_establish_alone": [
            "不能单独确认材料真实、未被修改或一定会被采信",
            "不能单独确认对方责任、损失因果关系或最终请求成立",
        ],
        "limitations": limitations,
        "basis_refs": [
            {
                "basis_type": "material_observation",
                "material_id": material.get("material_id"),
                "locator": observation.get("source_locator") or "",
                "excerpt": observation.get("source_excerpt") or "",
            }
        ],
        "user_note": material.get("user_note") or "",
    }


def _formal_target_rows(state: Any) -> list[dict[str, Any]]:
    requirements = _active_requirements(state)
    targets = {
        str(item.get("requirement_id")): item
        for item in requirements
        if item.get("requirement_id")
    }
    result: list[dict[str, Any]] = []
    for raw in getattr(state, "proof_targets", []) or []:
        if not isinstance(raw, dict):
            continue
        requirement_id = str(
            raw.get("requirement_id")
            or raw.get("rule_id")
            or raw.get("evidence_key")
            or ""
        )
        requirement = targets.get(requirement_id)
        if requirement is None:
            requirement = next(
                (
                    item
                    for item in requirements
                    if _target_id(item) == raw.get("proof_target_id")
                    or _target_id(item) == raw.get("id")
                ),
                None,
            )
        if requirement is None:
            continue
        result.append(
            {
                **dict(raw),
                **requirement,
                "requirement_id": requirement.get("requirement_id"),
                "proof_target_id": _target_id(requirement),
                "label": _target_label(requirement),
                "purpose": _compact(
                    requirement.get("purpose")
                    or raw.get("purpose")
                    or raw.get("proposition")
                    or "支持当前证明目标",
                    320,
                ),
            }
        )
    known = {str(item.get("requirement_id")) for item in result}
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id"))
        if requirement_id in known:
            continue
        result.append(
            {
                **requirement,
                "requirement_id": requirement_id,
                "proof_target_id": _target_id(requirement),
                "label": _target_label(requirement),
                "purpose": _compact(
                    requirement.get("purpose") or "支持当前证明目标",
                    320,
                ),
            }
        )
    return result


def _coverage_status(
    requirement: dict[str, Any],
    material_assessments: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], str]:
    statuses = {
        str(item.get("assessment_status") or "")
        for item in material_assessments
    }
    quality_gaps = _unique(
        gap
        for item in material_assessments
        for gap in item.get("quality_gaps") or []
    )
    supporting_ids = [
        str(item.get("evidence_id"))
        for item in material_assessments
        if item.get("assessment_status")
        in {"supports", "partially_supports", "conflicted"}
        and item.get("evidence_id")
    ]
    if "conflicted" in statuses:
        return (
            "conflicted",
            supporting_ids,
            quality_gaps,
            "先核对同一字段在不同材料中的金额、日期、账号或主体，不自行选择其中一份为真。",
        )
    if "supports" in statuses:
        return (
            "covered",
            supporting_ids,
            quality_gaps,
            "保留原始载体和可回查定位，提交前仍需由受理机关依法核验。",
        )
    if "partially_supports" in statuses or "received_but_unparsed" in statuses:
        return (
            "partially_covered",
            supporting_ids,
            quality_gaps,
            "优先补充" + "、".join(quality_gaps[:3] or ["可定位的原始内容"]) + "。",
        )
    state = str(requirement.get("user_material_state") or "not_submitted")
    if state == "user_claimed_unavailable":
        return (
            "explicitly_absent",
            [],
            quality_gaps,
            "记录为当前明确缺口，寻找替代材料或申请向第三方调取。",
        )
    if state == "available_for_third_party_request":
        return (
            "third_party_available",
            [],
            quality_gaps,
            "保留调取对象、渠道和申请时间，必要时请求平台、银行或机构出具记录。",
        )
    if state == "temporarily_unavailable":
        return (
            "not_submitted",
            [],
            quality_gaps,
            "先记录暂时找不到，不等同于材料不存在；可稍后补交。",
        )
    return (
        "not_submitted",
        [],
        quality_gaps,
        "当前没有收到该项材料；未提交不等于没有，可以稍后补交。",
    )


def calculate_target_coverage(
    state: Any,
    assessments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in assessments if isinstance(item, dict)]
    result: list[dict[str, Any]] = []
    for target in _formal_target_rows(state):
        requirement_id = str(target.get("requirement_id") or "")
        target_id = _target_id(target)
        linked = [
            item
            for item in rows
            if str(item.get("requirement_id") or "") == requirement_id
            or str(item.get("proof_target_id") or "") == target_id
        ]
        status, supporting_ids, quality_gaps, next_action = _coverage_status(
            target,
            linked,
        )
        result.append(
            {
                "target_id": target_id,
                "requirement_id": requirement_id,
                "label": _target_label(target),
                "purpose": _compact(target.get("purpose") or "支持当前证明目标", 320),
                "status": status,
                "supporting_evidence_ids": supporting_ids,
                "quality_gaps": quality_gaps,
                "limitations": _unique(
                    limitation
                    for item in linked
                    for limitation in item.get("limitations") or []
                ),
                "next_action": next_action,
                "authority_refs": list(
                    target.get("basis_refs")
                    or target.get("authority_refs")
                    or []
                ),
            }
        )
    return result


def build_assessment_basis_refs(
    state: Any,
    target_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reuse node-five sources and expose missing precision instead of guessing."""

    refs: list[dict[str, Any]] = []
    limitations: list[str] = []
    checklist = resolve_state_evidence_checklist(state)
    for item in getattr(state, "plan_basis_refs", []) or []:
        if isinstance(item, dict):
            refs.append({**item, "basis_type": item.get("basis_type") or "authority_rule"})
    for target in target_rows:
        for item in target.get("basis_refs") or target.get("authority_refs") or []:
            if isinstance(item, dict):
                refs.append({**item, "basis_type": item.get("basis_type") or "authority_rule"})
    refs = [
        item
        for item in refs
        if item.get("locator")
        or item.get("article_no")
        or item.get("source_url")
        or item.get("source_id")
    ]
    if not refs:
        source = checklist.source or {}
        refs.append(
            {
                "basis_type": "structured_checklist",
                "title": checklist.title,
                "authority_level": checklist.authority_level,
                "source_id": source.get("template_id") or "",
                "mapping_status": "system_guidance",
                "locator": "",
            }
        )
        limitations.append(checklist.usage_note or "当前没有可精确定位的证据规则依据")
    limitations.extend(
        str(item)
        for item in (getattr(state, "plan_basis_limitations", []) or [])
        if item
    )
    return refs, _unique(limitations)


def detect_new_fact_candidates(
    state: Any,
    observations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Surface source-anchored material claims that conflict with known facts."""

    fact_rows = (
        getattr(state, "fact_blackboard", None)
        or getattr(state, "case_facts", None)
        or []
    )
    fact_text = " ".join(
        _compact(
            item.get("statement")
            or item.get("value")
            or item.get("source_text")
        )
        for item in fact_rows
        if isinstance(item, dict)
    )
    candidates: list[dict[str, Any]] = []
    for observation in observations:
        for claim in observation.get("material_claims") or []:
            if not isinstance(claim, dict):
                continue
            key = _compact(claim.get("key"))
            value = _compact(claim.get("value"))
            source_text = _compact(claim.get("source_text"))
            if not key or not value or not source_text:
                continue
            markers = _CLAIM_FACT_MARKERS.get(key, ())
            if not any(marker in fact_text for marker in markers):
                continue
            if value in fact_text:
                continue
            candidates.append(
                {
                    "candidate_id": _stable_id(
                        "evidence-fact-candidate",
                        observation.get("material_id"),
                        key,
                        value,
                    ),
                    "material_id": observation.get("material_id"),
                    "fact_key": key,
                    "observed_value": value,
                    "source_text": source_text,
                    "source_locator": observation.get("source_locator") or "",
                    "status": "pending_fact_confirmation",
                    "reason": "材料观察与当前事实账本存在可定位差异",
                }
            )
    return candidates


def _apply_verification_answer(
    assessments: dict[str, dict[str, Any]],
    state: Any,
) -> dict[str, dict[str, Any]]:
    pending = list(getattr(state, "pending_evidence_verification", []) or [])
    if not pending:
        return assessments
    answer = _latest_message(state)
    if not answer or any(word in answer for word in _UNKNOWN_WORDS):
        return assessments
    result = {key: dict(value) for key, value in assessments.items()}
    target_material = str(pending[0].get("material_id") or "")
    for key, record in result.items():
        if str(record.get("material_id") or "") != target_material:
            continue
        text = answer
        if any(marker in text for marker in ("原件", "原始文件", "平台导出", "导出文件")):
            record["source_form"] = (
                "exported_file" if "导出" in text else "native_electronic"
            )
        elif "截图" in text:
            record["source_form"] = "screenshot"
        elif "复制" in text:
            record["source_form"] = "copy"
        if any(marker in text for marker in ("完整", "没有缺页", "上下文齐全")):
            record["completeness"] = "complete"
        elif any(marker in text for marker in ("裁剪", "缺页", "截断", "不完整")):
            record["completeness"] = "partial"
        if any(marker in text for marker in ("主体清楚", "账号清楚", "看得清")):
            record["identity_visibility"] = "clear"
        elif any(marker in text for marker in ("主体不清", "看不清", "账号不清")):
            record["identity_visibility"] = "unclear"
        if any(marker in text for marker in ("时间清楚", "日期清楚")):
            record["time_visibility"] = "clear"
        elif any(marker in text for marker in ("时间不清", "日期不清")):
            record["time_visibility"] = "unclear"
        record["verification_answer"] = _compact(answer, 400)
        result[key] = record
    return result


def plan_evidence_verification(
    state: Any,
    assessments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if int(getattr(state, "verification_round_count", 0) or 0) >= 1:
        return []
    for item in assessments:
        if item.get("assessment_status") not in {
            "partially_supports",
            "needs_verification",
            "received_but_unparsed",
        }:
            continue
        gaps = list(item.get("quality_gaps") or [])
        if not gaps:
            continue
        question = (
            f"关于“{item.get('name') or item.get('file_name')}”，请确认它是原始文件、"
            "平台导出还是截图/复制件？原始载体是否还保留，内容是否完整并能看清主体和时间？"
        )
        return [
            {
                "verification_id": _stable_id(
                    "evidence-verification",
                    item.get("material_id"),
                    item.get("sha256"),
                ),
                "material_id": item.get("material_id"),
                "evidence_id": item.get("evidence_id"),
                "question": question,
                "quality_gaps": gaps[:5],
                "reason": "材料属性可能改变当前证明范围或补强优先级",
                "status": "pending",
            }
        ]
    return []


def _legacy_evaluation_report(
    state: Any,
    assessments: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> EvidenceEvaluationReport:
    items = [
        EvidenceItem(
            id=str(item.get("evidence_id") or _stable_id("evidence", item.get("name"))),
            name=str(item.get("name") or "未命名材料"),
            availability=(
                "uploaded_copy"
                if item.get("availability") == "submitted"
                else str(item.get("availability") or "unclear")
            ),
            source_form=str(item.get("source_form") or "unknown"),
            completeness=str(item.get("completeness") or "unknown"),
            identity_visibility=str(item.get("identity_visibility") or "unknown"),
            time_visibility=str(item.get("time_visibility") or "unknown"),
            acquisition_method=str(item.get("acquisition_method") or "unknown"),
            case_specificity=str(item.get("case_specificity") or "unknown"),
            authenticity_status="not_verified",
            inspection_basis="uploaded_copy",
            source_excerpt=str(item.get("source_excerpt") or "")[:300],
            content_conflicts=list(item.get("content_conflicts") or []),
            limitations=list(item.get("limitations") or []),
        )
        for item in assessments
    ]
    targets = [
        {
            "id": str(row.get("target_id")),
            "rule_id": str(row.get("requirement_id") or row.get("target_id")),
            "evidence_key": str(row.get("requirement_id") or row.get("target_id")),
            "label": str(row.get("label") or "证明目标"),
            "purpose": str(row.get("purpose") or ""),
        }
        for row in coverage
    ]
    links = [
        EvidenceLink(
            evidence_id=str(item.get("evidence_id")),
            target_id=str(item.get("proof_target_id") or ""),
            proof_scope=str(item.get("probative_scope") or ""),
            basis="formal_evidence_plan",
            limitations=list(item.get("limitations") or []),
        )
        for item in assessments
        if item.get("evidence_id") and item.get("proof_target_id")
    ]
    legacy_status = {
        "covered": "preliminarily_covered",
        "partially_covered": "partially_covered",
        "conflicted": "conflicted",
        "explicitly_absent": "known_missing",
        "third_party_available": "known_missing",
        "not_submitted": "unresolved",
        "unresolved": "unresolved",
        "not_applicable": "unresolved",
    }
    coverage_models = [
        EvidenceCoverage(
            target_id=str(row.get("target_id")),
            label=str(row.get("label") or "证明目标"),
            purpose=str(row.get("purpose") or ""),
            status=legacy_status.get(str(row.get("status")), "unresolved"),
            supporting_evidence_ids=list(row.get("supporting_evidence_ids") or []),
            quality_gaps=list(row.get("quality_gaps") or []),
            limitations=list(row.get("limitations") or []),
            next_action=str(row.get("next_action") or ""),
        )
        for row in coverage
    ]
    counts = {
        "preliminarily_covered": sum(
            item.status == "preliminarily_covered" for item in coverage_models
        ),
        "partially_covered": sum(
            item.status == "partially_covered" for item in coverage_models
        ),
        "known_missing": sum(item.status == "known_missing" for item in coverage_models),
        "unresolved": sum(
            item.status in {"unresolved", "conflicted"} for item in coverage_models
        ),
    }
    return EvidenceEvaluationReport(
        schema_version=ASSESSMENT_SCHEMA_VERSION,
        items=items,
        targets=targets,
        links=links,
        coverage=coverage_models,
        target_count=len(coverage_models),
        preliminarily_covered_count=counts["preliminarily_covered"],
        partial_count=counts["partially_covered"],
        known_missing_count=counts["known_missing"],
        unresolved_count=counts["unresolved"],
    )


def _assessment_markdown(
    *,
    status: str,
    review_version: int,
    batch_version: int,
    assessments: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    basis_limitations: list[str],
    change_summary: dict[str, Any],
) -> str:
    status_label = {
        "completed": "本批次评估完成",
        "partial": "本批次部分完成",
        "needs_verification": "等待一次材料属性核验",
        "awaiting_batch": "材料已收到，等待完成本批次",
        "received_pending_remap": "材料已保存，等待按最新清单重新归类",
        "degraded": "评估服务降级，已保留材料和缺口",
    }.get(status, "评估状态待确认")
    lines = [
        "## 证据材料初步评估",
        "",
        f"> {status_label} · 评估版本 {review_version or 1} · 材料批次 {batch_version or 1}",
        "",
    ]
    if assessments:
        lines.extend(["### 本批材料", ""])
        for item in assessments:
            lines.extend(
                [
                    f"- **{item.get('name') or '未命名材料'}**："
                    f"{item.get('assessment_status') or '待评估'}；"
                    f"解析状态：{item.get('parser_status') or 'unknown'}",
                    f"  - 可见内容：{item.get('source_excerpt') or '当前没有可引用的文字摘录'}",
                    f"  - 可能支持：{item.get('probative_scope') or '待结合证明目标判断'}",
                    f"  - 不能单独证明：{'；'.join(item.get('cannot_establish_alone') or [])}",
                    f"  - 补强方向：{'、'.join(item.get('quality_gaps') or []) or '保留原始载体并准备提交前核对'}",
                ]
            )
        lines.append("")
    lines.extend(["### 证明目标覆盖", ""])
    if coverage:
        for row in coverage:
            lines.append(
                f"- **{row.get('label') or '证明目标'}**：{row.get('status') or 'unresolved'}；"
                f"{row.get('next_action') or '暂无下一步'}"
            )
    else:
        lines.append("- 当前没有可关联的正式证明目标。")
    if change_summary:
        lines.extend(
            [
                "",
                "### 本版变化",
                "",
                f"- 新增材料：{len(change_summary.get('added_material_ids') or [])} 份",
                f"- 更新材料：{len(change_summary.get('updated_material_ids') or [])} 份",
                f"- 受影响证明目标：{len(change_summary.get('changed_target_ids') or [])} 项",
            ]
        )
    if basis_limitations:
        lines.extend(
            [
                "",
                "### 依据限制",
                "",
                *[f"- {item}" for item in _unique(basis_limitations)[:5]],
            ]
        )
    lines.extend(
        [
            "",
            "> 本评估只用于梳理材料用途和补强方向，不认定真实性、合法性、可采性或最终证明力。",
        ]
    )
    return "\n".join(lines)


def _change_summary(
    previous: dict[str, Any],
    current: dict[str, Any],
    coverage: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_ids = set(previous)
    current_ids = set(current)
    added = sorted(current_ids - previous_ids)
    updated = sorted(
        key
        for key in current_ids & previous_ids
        if _hash_payload(current[key]) != _hash_payload(previous[key])
    )
    previous_targets = {
        str(item.get("proof_target_id") or ""): str(item.get("assessment_status") or "")
        for item in previous.values()
        if isinstance(item, dict)
    }
    changed_targets = sorted(
        {
            str(item.get("target_id") or "")
            for item in coverage
            if str(item.get("target_id") or "")
            and (
                str(item.get("status") or "") != previous_targets.get(
                    str(item.get("target_id") or ""),
                    "",
                )
            )
        }
    )
    return {
        "added_material_ids": added,
        "updated_material_ids": updated,
        "reused_material_ids": sorted(previous_ids & current_ids - set(updated)),
        "changed_target_ids": changed_targets,
    }


def _is_submission_completed(state: Any) -> bool:
    events = {
        str(item.get("type") or "")
        for item in (getattr(state, "input_events", []) or [])
        if isinstance(item, dict)
    }
    explicit = str(
        (getattr(state, "control_payload", {}) or {}).get("explicit_action") or ""
    )
    return bool(
        "evidence_batch_completed" in events
        or explicit == "complete_batch"
    )


async def run_assess_evidence(state: Any, deps: Any = None) -> dict[str, Any]:
    """Run node six and return a persistent evidence-review checkpoint."""

    materials, observations = await parse_material_batch(state, deps)
    validation = validate_evidence_plan_version(state, materials)
    target_rows = _formal_target_rows(state)
    basis_refs, basis_limitations = build_assessment_basis_refs(state, target_rows)
    submission_completed = _is_submission_completed(state)
    verification_answered = "evidence_verification_answered" in {
        str(item.get("type") or "")
        for item in (getattr(state, "input_events", []) or [])
        if isinstance(item, dict)
    }

    if not validation.get("valid"):
        material_rows = [
            {
                **dict(item),
                "material_status": "received_pending_remap",
                "submission_status": "submitted",
            }
            for item in materials
        ]
        report = {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "case_id": state.case_id,
            "evidence_plan_version": getattr(state, "evidence_plan_version", 0),
            "evidence_batch_id": getattr(state, "evidence_batch_id", ""),
            "reviewed_material_ids": [],
            "reused_material_assessment_ids": [],
            "items": [],
            "observations": observations,
            "proof_targets": target_rows,
            "links": [],
            "coverage": [],
            "basis_refs": basis_refs,
            "basis_limitations": _unique(
                [*basis_limitations, validation.get("message")]
            ),
            "new_fact_candidates": [],
            "content_conflicts": [],
            "quality_gaps": ["材料待按最新证据计划重新归类"],
            "unclassified_materials": material_rows,
            "pending_verification": [],
            "verification_round_count": getattr(state, "verification_round_count", 0),
            "assessment_change_summary": {},
            "assessment_status": str(validation.get("status") or "received_pending_remap"),
            "next_route": "plan_evidence",
            "disclaimer": (
                "材料已保存为待归类记录，不代表已经完成证据评估。"
            ),
        }
        return {
            "evidence_review_report": report,
            "evidence_review_status": "received_pending_remap",
            "evidence_reviewed_at": _now(),
            "evidence_observations": observations,
            "evidence_basis_refs": basis_refs,
            "evidence_basis_missing": basis_limitations,
            "unclassified_materials": material_rows,
            "evidence_review_version": int(
                getattr(state, "evidence_review_version", 0) or 0
            ),
            "evidence_verification_pending": False,
            "pending_evidence_verification": [],
            "pause_state": {
                "type": "awaiting_evidence_plan_remap",
                "pause_type": "awaiting_evidence_plan_remap",
            },
            "workflow_stage": "evidence_collection",
            "next_route": "plan_evidence",
            "decision_status": "evidence_received_pending_remap",
            "pending_ask_details": [],
            "pending_ask_type": "",
            "pending_followup_ids": [],
            "messages": [AIMessage(content=(
                "## 材料已收到，暂不覆盖当前证据结论\n\n"
                f"{validation.get('message') or '证据清单版本已变化，请先按最新清单重新归类。'}"
            ))],
        }

    mapped_materials, unclassified = map_materials_to_requirements(
        state,
        materials,
        observations,
    )
    observations_by_material = {
        str(item.get("material_id")): item
        for item in observations
        if item.get("material_id")
    }
    mapped_observations, conflicts = mark_material_content_conflicts(observations)
    observations_by_material.update(
        {
            str(item.get("material_id")): item
            for item in mapped_observations
            if item.get("material_id")
        }
    )
    previous_assessments = {
        str(key): dict(value)
        for key, value in (getattr(state, "evidence_assessments", {}) or {}).items()
        if isinstance(value, dict)
    }
    assessments_by_material: dict[str, dict[str, Any]] = {}
    for material in mapped_materials:
        material_id = str(material.get("material_id") or "")
        observation = observations_by_material.get(material_id) or {
            "material_id": material_id,
            "parser_status": "received_but_unparsed",
            "source_form": _infer_source_form(
                material.get("file_name"),
                material.get("file_type"),
            ),
            "source_locator": f"{material.get('file_name')}（材料元数据）",
            "source_excerpt": "",
            "observed_text": "",
            "quality_gaps": ["材料内容尚未成功解析或缺少可定位文字"],
        }
        requirement = _requirement_map(state).get(
            str(material.get("requirement_id") or "")
        )
        assessment = assess_material_item(material, observation, requirement)
        assessments_by_material[material_id] = assessment

    # Keep older assessments that are not part of this turn.  Their source
    # fingerprint and original observation remain auditable and reusable.
    for key, old in previous_assessments.items():
        material_id = str(old.get("material_id") or "")
        if material_id and material_id not in assessments_by_material:
            assessments_by_material[material_id] = old

    assessments = list(assessments_by_material.values())
    if verification_answered:
        assessments_by_material = _apply_verification_answer(
            assessments_by_material,
            state,
        )
        assessments = list(assessments_by_material.values())
        # Recompute quality-dependent status after a material-attribute answer.
        for key, item in list(assessments_by_material.items()):
            if item.get("assessment_status") in {
                "partially_supports",
                "needs_verification",
            }:
                gaps = [
                    _QUALITY_LABELS[field]
                    for field in (
                        "source_form",
                        "completeness",
                        "identity_visibility",
                        "time_visibility",
                        "acquisition_method",
                        "case_specificity",
                    )
                    if str(item.get(field) or "unknown")
                    in {"", "unknown", "unclear", "partial"}
                ]
                item["quality_gaps"] = _unique(gaps)
                if not gaps and item.get("parser_status") == "parsed":
                    item["assessment_status"] = "supports"
                assessments_by_material[key] = item
        assessments = list(assessments_by_material.values())

    coverage = calculate_target_coverage(state, assessments)
    new_fact_candidates = detect_new_fact_candidates(state, mapped_observations)
    pending_verification = (
        []
        if verification_answered
        else plan_evidence_verification(state, assessments)
        if submission_completed and not new_fact_candidates
        else []
    )
    previous_fingerprint = str(getattr(state, "evidence_review_fingerprint", "") or "")
    fingerprint_payload = {
        "plan": getattr(state, "evidence_plan_version", 0),
        "facts": getattr(state, "fact_snapshot_version", 0),
        "materials": [
            {
                "id": item.get("material_id"),
                "sha256": item.get("sha256"),
                "requirement_id": item.get("requirement_id"),
                "status": item.get("assessment_status"),
            }
            for item in assessments
        ],
        "coverage": coverage,
        "verification_round_count": getattr(state, "verification_round_count", 0)
        + int(bool(verification_answered)),
    }
    fingerprint = _hash_payload(fingerprint_payload)
    previous_version = int(getattr(state, "evidence_review_version", 0) or 0)
    review_version = previous_version if fingerprint == previous_fingerprint else previous_version + 1
    previous_batch_version = int(getattr(state, "evidence_batch_version", 0) or 0)
    current_material_ids = {
        str(item.get("material_id") or "")
        for item in materials
        if item.get("material_id")
    }
    previous_material_ids = {
        str(item.get("material_id") or "")
        for item in previous_assessments.values()
        if item.get("material_id")
    }
    batch_version = previous_batch_version
    if current_material_ids - previous_material_ids or submission_completed:
        batch_version = previous_batch_version + 1
    change_summary = _change_summary(
        previous_assessments,
        assessments_by_material,
        coverage,
    )

    if new_fact_candidates:
        review_status = "needs_verification"
        next_route = "update_facts"
        workflow_stage = "fact_clarification"
        pause_state = {
            "type": "awaiting_fact_batch",
            "pause_type": "awaiting_fact_batch",
            "reason": "材料观察发现需要用户确认的事实差异",
            "new_fact_candidates": new_fact_candidates,
        }
        reply = (
            "## 材料中发现需要确认的信息\n\n"
            "材料中的可定位内容与当前事实记录存在差异。系统不会自动选择其中一份为真，"
            "请确认后再更新案件事实：\n\n"
            + "\n".join(
                f"- **{item.get('fact_key')}**：材料显示“{item.get('observed_value')}”，"
                f"来源：{item.get('source_text')}"
                for item in new_fact_candidates[:5]
            )
        )
    elif pending_verification:
        review_status = "needs_verification"
        next_route = "assess_evidence"
        workflow_stage = "evidence_assessment"
        pause_state = {
            "type": "awaiting_evidence_verification",
            "pause_type": "awaiting_evidence_verification",
            "evidence_batch_id": getattr(state, "evidence_batch_id", ""),
            "verification_round_count": int(
                getattr(state, "verification_round_count", 0) or 0
            ),
        }
        reply = (
            "## 需要确认一项材料属性\n\n"
            f"{pending_verification[0].get('question')}\n\n"
            "这只用于判断材料的来源和完整性，不是在要求您证明材料一定有效。"
            "不清楚可以直接回复“不清楚”，本批次仍会按条件式结果继续。"
        )
    elif not submission_completed:
        review_status = "awaiting_batch"
        next_route = "await_evidence_batch"
        workflow_stage = "evidence_collection"
        pause_state = {
            "type": "awaiting_evidence_batch",
            "pause_type": "awaiting_evidence_batch",
            "evidence_batch_id": getattr(state, "evidence_batch_id", ""),
            "evidence_plan_version": getattr(state, "evidence_plan_version", 0),
        }
        names = _unique(item.get("file_name") for item in materials)
        reply = (
            "## 材料已暂存\n\n"
            f"本次已收到：{'、'.join(names) if names else '材料状态更新'}。"
            "可以继续补交，完成后点击“完成本批次并评估”。"
        )
    else:
        review_status = "completed"
        if any(
            row.get("status") in {"partially_covered", "conflicted", "not_submitted"}
            for row in coverage
        ):
            review_status = "partial"
        next_route = "conclude"
        workflow_stage = "evidence_assessment"
        pause_state = None
        legacy_report = _legacy_evaluation_report(state, assessments, coverage)
        reply = _assessment_markdown(
            status=review_status,
            review_version=review_version,
            batch_version=batch_version,
            assessments=assessments,
            coverage=coverage,
            basis_limitations=basis_limitations,
            change_summary=change_summary,
        )

    legacy_report = _legacy_evaluation_report(state, assessments, coverage)
    formal_report = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "case_id": state.case_id,
        "case_generation": getattr(state, "case_generation", 1),
        "fact_snapshot_version": getattr(state, "fact_snapshot_version", 0),
        "legal_model_version": getattr(state, "legal_model_version", 0),
        "evidence_plan_version": getattr(state, "evidence_plan_version", 0),
        "evidence_review_version": review_version,
        "evidence_review_id": _stable_id(
            "evidence-review",
            state.case_id,
            review_version,
            fingerprint,
        ),
        "evidence_batch_id": getattr(state, "evidence_batch_id", ""),
        "evidence_batch_version": batch_version,
        "reviewed_material_ids": [
            str(item.get("material_id"))
            for item in materials
            if item.get("material_id")
        ],
        "reused_material_assessment_ids": change_summary.get("reused_material_ids") or [],
        "items": assessments,
        "observations": mapped_observations,
        "proof_targets": target_rows,
        "links": [
            {
                "evidence_id": item.get("evidence_id"),
                "material_id": item.get("material_id"),
                "target_id": item.get("proof_target_id"),
                "requirement_id": item.get("requirement_id"),
                "direction": "supports",
                "relevance": (
                    "potentially_relevant"
                    if item.get("assessment_status")
                    in {"supports", "partially_supports"}
                    else "needs_review"
                ),
                "proof_scope": item.get("probative_scope"),
                "basis": "formal_evidence_plan",
                "basis_refs": item.get("basis_refs") or [],
                "limitations": item.get("limitations") or [],
            }
            for item in assessments
            if item.get("proof_target_id")
        ],
        "coverage": coverage,
        "basis_refs": basis_refs,
        "basis_limitations": basis_limitations,
        "new_fact_candidates": new_fact_candidates,
        "content_conflicts": conflicts,
        "quality_gaps": _unique(
            gap
            for item in assessments
            for gap in item.get("quality_gaps") or []
        ),
        "unclassified_materials": unclassified,
        "pending_verification": pending_verification,
        "verification_round_count": int(
            getattr(state, "verification_round_count", 0) or 0
        ) + int(bool(verification_answered)),
        "assessment_change_summary": change_summary,
        "assessment_status": review_status,
        "next_route": next_route,
        "disclaimer": (
            "本评估只用于梳理材料用途和补强方向，不认定真实性、合法性、"
            "可采性或最终证明力。"
        ),
    }
    return {
        "evidence_items": assessments,
        "evidence_assessments": {
            str(item.get("material_id") or item.get("evidence_id")): item
            for item in assessments
        },
        "evidence_observations": mapped_observations,
        "evidence_links": formal_report["links"],
        "evidence_coverage": legacy_report.model_dump(),
        "evidence_review_report": formal_report,
        "evidence_basis_refs": basis_refs,
        "evidence_basis_missing": basis_limitations,
        "evidence_review_version": review_version,
        "evidence_review_id": formal_report["evidence_review_id"],
        "evidence_review_fingerprint": fingerprint,
        "evidence_review_status": review_status,
        "evidence_reviewed_at": _now(),
        "evidence_batch_version": batch_version,
        "evidence_batch_completed": submission_completed,
        "evidence_verification_pending": bool(pending_verification),
        "pending_evidence_verification": pending_verification,
        "verification_round_count": int(
            getattr(state, "verification_round_count", 0) or 0
        ) + int(bool(verification_answered)),
        "new_fact_candidates_from_evidence": new_fact_candidates,
        "content_conflicts": conflicts,
        "quality_gaps": formal_report["quality_gaps"],
        "unclassified_materials": unclassified,
        "assessment_change_summary": change_summary,
        "pause_state": pause_state,
        "workflow_stage": workflow_stage,
        "next_route": next_route,
        "decision_status": (
            "evidence_review_needs_fact_confirmation"
            if new_fact_candidates
            else "evidence_review_needs_verification"
            if pending_verification
            else "evidence_review_awaiting_batch"
            if not submission_completed
            else "evidence_review_completed"
        ),
        "phase": (
            GuidePhase.DETAIL_GATHER
            if next_route in {"await_evidence_batch", "assess_evidence", "update_facts"}
            else GuidePhase.CONCLUDE
        ),
        "pending_ask_details": (
            [
                f"请确认材料中“{item.get('fact_key')}”显示的“{item.get('observed_value')}”"
                for item in new_fact_candidates[:5]
            ]
            if new_fact_candidates
            else []
        ),
        "pending_ask_type": "facts" if new_fact_candidates else "",
        "pending_followup_ids": [],
        "messages": [AIMessage(content=reply)],
    }


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "validate_evidence_plan_version",
    "parse_material_batch",
    "map_materials_to_requirements",
    "mark_material_content_conflicts",
    "assess_material_item",
    "calculate_target_coverage",
    "build_assessment_basis_refs",
    "detect_new_fact_candidates",
    "plan_evidence_verification",
    "run_assess_evidence",
]
