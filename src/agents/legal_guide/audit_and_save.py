"""Formal node eight: audit, publish, and persist a versioned legal plan."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage
from loguru import logger

from src.agents.legal_guide.db_queries import save_solution_version
from src.agents.legal_guide.generate_solution import (
    LIKELIHOOD_TIERS,
    build_evidence_effect_summary,
    build_likelihood_dimensions,
    derive_qualitative_likelihood,
    load_reusable_action_basis,
    render_solution_markdown,
)
from src.agents.legal_guide.state import GuidePhase


AUDIT_SCHEMA_VERSION = "audit-and-save.v1"
PUBLISHED_PLAN_SCHEMA_VERSION = "published-solution.v1"

_CONFIRMED_FACT_STATUSES = {
    "confirmed",
    "asserted",
    "user_stated",
    "corrected",
}
_REQUIRED_MAPPING_FIELDS = {
    "core_judgment",
    "likelihood_assessment",
    "evidence_effect_summary",
    "change_summary",
}
_REQUIRED_LIST_FIELDS = {
    "confirmed_facts",
    "recommended_routes",
    "alternative_routes",
    "immediate_actions",
    "case_tasks",
    "document_suggestions",
    "action_basis_refs",
    "action_basis_gaps",
}
_REQUIRED_HEADINGS = (
    "## 核心判断",
    "## 已确认事实",
    "## 法律依据与适用条件",
    "## 证据检验结果",
    "## 证据缺口与替代材料",
    "## 有利、不利和不确定因素",
    "## 当前维权可能性",
    "## 推荐行动方案",
    "## 替代与升级路径",
    "## 下一步任务清单",
    "## 参考文书",
    "## 版本变化与限制",
)
_GENERATED_TEXT_FIELDS = (
    "core_judgment",
    "likelihood_assessment",
    "evidence_effect_summary",
    "recommended_routes",
    "alternative_routes",
    "immediate_actions",
    "case_tasks",
    "document_suggestions",
    "action_basis_gaps",
)
_GUARANTEE_PATTERNS = (
    re.compile(
        r"(?:胜诉|成功|获赔|追回)(?:率|概率)?[^。；\n]{0,16}"
        r"\d+(?:\.\d+)?\s*%",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d+(?:\.\d+)?\s*%[^。；\n]{0,16}"
        r"(?:胜诉|成功|获赔|追回)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:保证|确保|必然|一定|肯定|百分之百)[^。；\n]{0,8}"
        r"(?:胜诉|成功|获赔|追回|得到支持)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:包赢|稳赢|必胜)", re.IGNORECASE),
)
_OVERREACH_REPLACEMENTS = (
    (
        re.compile(r"(?:已经|已|当然|必然)?构成诈骗"),
        "是否涉及诈骗等犯罪问题，应由有权机关依法核查判断",
    ),
    (
        re.compile(r"(?:证据|材料)(?:已经|已)?(?:完全)?(?:有效|充分|足够)"),
        "现有材料可能支持相应证明目标，仍需核对来源、完整性和争议情况",
    ),
    (
        re.compile(r"(?:法院|仲裁机构|有关机关)(?:一定|必然|肯定)采纳"),
        "是否采纳及证明范围由有权机关结合完整材料依法判断",
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_payload(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _compact(value: Any, limit: int = 800) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _canonical_draft_payload(draft: dict[str, Any]) -> dict[str, Any]:
    """Recreate the payload node seven fingerprinted before transient fields."""

    value = copy.deepcopy(draft)
    for key in (
        "plan_version_candidate",
        "change_summary",
        "generation_trace_id",
        "generated_at",
        "draft_markdown",
        "plan_version",
        "published_at",
        "reviewed_at",
        "audit_id",
        "audit_status",
        "published_markdown",
        "published_fingerprint",
    ):
        value.pop(key, None)
    for task in value.get("case_tasks") or []:
        if isinstance(task, dict):
            task.pop("plan_version_candidate", None)
            task.pop("introduced_in_plan_version", None)
            task.pop("last_updated_in_plan_version", None)
            task.pop("published_at", None)
    return value


def _fact_id(item: dict[str, Any]) -> str:
    return _compact(
        item.get("fact_id")
        or item.get("semantic_key")
        or item.get("key"),
        180,
    )


def _fact_text(item: dict[str, Any]) -> str:
    return _compact(
        item.get("statement")
        or item.get("value")
        or item.get("source_text"),
        420,
    )


def _confirmed_fact_map(state: Any) -> dict[str, str]:
    return {
        str(item["fact_id"]): str(item["statement"])
        for item in _confirmed_fact_records(state)
    }


def _confirmed_fact_records(state: Any) -> list[dict[str, str]]:
    rows = (
        getattr(state, "fact_blackboard", None)
        or getattr(state, "case_facts", None)
        or []
    )
    result: list[dict[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") not in _CONFIRMED_FACT_STATUSES:
            continue
        fact_id = _fact_id(item)
        statement = _fact_text(item)
        if fact_id and statement:
            result.append(
                {
                    "fact_id": fact_id,
                    "statement": statement,
                    "status": str(item.get("status") or "confirmed"),
                }
            )
    return result


def _basis_signature(item: dict[str, Any]) -> str:
    return "|".join(
        str(item.get(key) or "")
        for key in (
            "source_id",
            "source_version_id",
            "law_id",
            "article_no",
            "locator",
            "source_url",
            "official_url",
            "url",
            "title",
            "name",
        )
    )


def _record_issue(
    collection: list[dict[str, str]],
    *,
    code: str,
    message: str,
    route: str = "",
) -> None:
    item = {"code": code, "message": message}
    if route:
        item["route"] = route
    collection.append(item)


def _source_version_validation(
    state: Any,
    draft: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fatal: list[dict[str, str]] = []
    current = {
        "case_id": str(getattr(state, "case_id", "") or ""),
        "case_generation": int(getattr(state, "case_generation", 1) or 1),
        "fact_snapshot_version": int(
            getattr(state, "fact_snapshot_version", 0) or 0
        ),
        "fact_snapshot_hash": str(
            (getattr(state, "fact_snapshot_draft", None) or {}).get(
                "snapshot_hash"
            )
            or ""
        ),
        "legal_model_version": int(
            getattr(state, "legal_model_version", 0) or 0
        ),
        "evidence_plan_version": int(
            getattr(state, "evidence_plan_version", 0) or 0
        ),
        "evidence_review_version": int(
            getattr(state, "evidence_review_version", 0) or 0
        ),
    }
    checks = (
        ("case_id", "case_id", "prepare_case", "案件标识"),
        (
            "case_generation",
            "case_generation",
            "prepare_case",
            "案件代次",
        ),
        (
            "based_on_fact_snapshot_version",
            "fact_snapshot_version",
            "decide_facts",
            "事实快照版本",
        ),
        (
            "based_on_fact_snapshot_hash",
            "fact_snapshot_hash",
            "decide_facts",
            "事实快照指纹",
        ),
        (
            "based_on_legal_model_version",
            "legal_model_version",
            "plan_evidence",
            "法律模型版本",
        ),
        (
            "based_on_evidence_plan_version",
            "evidence_plan_version",
            "plan_evidence",
            "证据清单版本",
        ),
        (
            "based_on_evidence_review_version",
            "evidence_review_version",
            "assess_evidence",
            "证据评估版本",
        ),
    )
    for draft_key, current_key, route, label in checks:
        draft_value = draft.get(draft_key)
        current_value = current[current_key]
        if str(draft_value or "") != str(current_value or ""):
            _record_issue(
                fatal,
                code=f"stale_{current_key}",
                message=f"{label}已变化，草稿不能按旧版本发布。",
                route=route,
            )
    snapshot = getattr(state, "fact_snapshot_draft", None) or {}
    based_on_blackboard = int(
        snapshot.get("based_on_fact_blackboard_version") or 0
    )
    current_blackboard = int(
        getattr(state, "fact_blackboard_version", 0) or 0
    )
    if snapshot.get("stale") or (
        based_on_blackboard
        and current_blackboard
        and based_on_blackboard != current_blackboard
    ):
        _record_issue(
            fatal,
            code="fact_snapshot_stale",
            message="事实账本已变化，需要重新确认事实快照。",
            route="decide_facts",
        )
    if getattr(state, "stale_dependencies", []):
        _record_issue(
            fatal,
            code="upstream_dependencies_stale",
            message="法律模型或证据清单仍依赖已经变化的事实。",
            route="plan_evidence",
        )
    if getattr(state, "new_fact_candidates_from_evidence", []):
        _record_issue(
            fatal,
            code="evidence_fact_confirmation_pending",
            message="材料暴露的新事实仍待用户确认。",
            route="update_facts",
        )
    if getattr(state, "evidence_verification_pending", False):
        _record_issue(
            fatal,
            code="evidence_verification_pending",
            message="本批材料仍等待核验，不能发布方案。",
            route="assess_evidence",
        )
    return fatal, current


def validate_audit_inputs(state: Any) -> dict[str, Any]:
    """Validate immutable upstream inputs before any draft correction."""

    draft = getattr(state, "solution_draft", {}) or {}
    fatal: list[dict[str, str]] = []
    if not isinstance(draft, dict) or not draft:
        _record_issue(
            fatal,
            code="solution_draft_missing",
            message="节点七尚未生成可审校的方案草稿。",
            route="generate_solution",
        )
        return {
            "valid": False,
            "fatal_issues": fatal,
            "source_versions": {},
            "next_route": "generate_solution",
        }
    if str(getattr(state, "solution_draft_status", "") or "") not in {
        "awaiting_audit",
        "compatibility_presented",
        "published",
    }:
        _record_issue(
            fatal,
            code="solution_draft_not_ready",
            message="方案草稿尚未进入待审校状态。",
            route="generate_solution",
        )
    fatal_versions, source_versions = _source_version_validation(state, draft)
    fatal.extend(fatal_versions)
    expected_fingerprint = str(
        getattr(state, "solution_draft_fingerprint", "") or ""
    )
    actual_fingerprint = _hash_payload(_canonical_draft_payload(draft))
    if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
        _record_issue(
            fatal,
            code="solution_draft_fingerprint_mismatch",
            message="方案草稿内容与节点七登记的指纹不一致，需要重新生成。",
            route="generate_solution",
        )
    next_route = str(
        next(
            (
                item.get("route")
                for item in fatal
                if item.get("route") in {
                    "update_facts",
                    "decide_facts",
                    "plan_evidence",
                    "assess_evidence",
                    "generate_solution",
                }
            ),
            "generate_solution",
        )
    )
    return {
        "valid": not fatal,
        "fatal_issues": fatal,
        "source_versions": source_versions,
        "draft_fingerprint": expected_fingerprint,
        "next_route": next_route,
    }


def _sanitize_generated_text(value: Any) -> tuple[Any, int]:
    correction_count = 0

    def walk(item: Any) -> Any:
        nonlocal correction_count
        if isinstance(item, dict):
            return {key: walk(child) for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child) for child in item]
        if not isinstance(item, str):
            return item
        updated = item
        for pattern in _GUARANTEE_PATTERNS:
            updated, count = pattern.subn(
                "结果仍取决于事实、证据、程序和有权机关判断",
                updated,
            )
            correction_count += count
        for pattern, replacement in _OVERREACH_REPLACEMENTS:
            updated, count = pattern.subn(replacement, updated)
            correction_count += count
        return updated

    return walk(value), correction_count


def _unique_records(
    items: Iterable[Any],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = "|".join(str(item.get(field) or "") for field in key_fields)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def audit_solution_draft(
    state: Any,
    draft: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    """Correct presentation-level defects without changing upstream facts."""

    corrected = copy.deepcopy(draft)
    issues: list[dict[str, str]] = []
    for field in _REQUIRED_MAPPING_FIELDS:
        if not isinstance(corrected.get(field), dict):
            corrected[field] = {}
            _record_issue(
                issues,
                code=f"repair_{field}",
                message=f"已修复缺失或损坏的 {field} 结构。",
            )
    for field in _REQUIRED_LIST_FIELDS:
        if not isinstance(corrected.get(field), list):
            corrected[field] = []
            _record_issue(
                issues,
                code=f"repair_{field}",
                message=f"已修复缺失或损坏的 {field} 列表。",
            )

    confirmed = _confirmed_fact_map(state)
    safe_facts: list[dict[str, Any]] = _confirmed_fact_records(state)
    if corrected.get("confirmed_facts") != safe_facts:
        corrected["confirmed_facts"] = safe_facts
        _record_issue(
            issues,
            code="repair_fact_boundary",
            message="已按当前事实快照重建已确认事实展示，移除未知、冲突或失效事实。",
        )

    allowed_refs, basis_gaps = load_reusable_action_basis(state)
    allowed_signatures = {
        _basis_signature(item) for item in allowed_refs if _basis_signature(item)
    }
    draft_refs = [
        item
        for item in corrected.get("action_basis_refs") or []
        if isinstance(item, dict)
        and _basis_signature(item) in allowed_signatures
    ]
    if len(draft_refs) != len(corrected.get("action_basis_refs") or []):
        _record_issue(
            issues,
            code="repair_legal_boundary",
            message="已移除无法关联到本轮检索结果的法律或渠道依据。",
        )
    corrected["action_basis_refs"] = _unique_records(
        [*draft_refs, *allowed_refs],
        key_fields=(
            "source_id",
            "source_version_id",
            "locator",
            "source_url",
            "url",
            "title",
        ),
    )[:20]
    corrected["action_basis_gaps"] = list(dict.fromkeys(
        str(item).strip()
        for item in [
            *(corrected.get("action_basis_gaps") or []),
            *basis_gaps,
        ]
        if str(item).strip()
    ))

    expected_evidence = build_evidence_effect_summary(state)
    if corrected.get("evidence_effect_summary") != expected_evidence:
        corrected["evidence_effect_summary"] = expected_evidence
        _record_issue(
            issues,
            code="repair_evidence_boundary",
            message="已按当前证据评估重建已提交、未提交、明确缺失和可调取材料的边界。",
        )

    dimensions = build_likelihood_dimensions(
        state,
        action_basis_refs=corrected["action_basis_refs"],
        action_basis_gaps=corrected["action_basis_gaps"],
    )
    likelihood = derive_qualitative_likelihood(
        dimensions,
        basis_gaps=corrected["action_basis_gaps"],
    )
    if corrected.get("likelihood_assessment") != likelihood:
        corrected["likelihood_assessment"] = likelihood
        _record_issue(
            issues,
            code="repair_reasoning_dimensions",
            message="已按当前五个评估维度重算定性维权可能性。",
        )
    if likelihood.get("tier") not in LIKELIHOOD_TIERS:
        corrected["likelihood_assessment"]["tier"] = "不确定"
        _record_issue(
            issues,
            code="repair_likelihood_tier",
            message="已将无效的维权可能性等级降级为“不确定”。",
        )

    valid_fact_ids = set(confirmed)
    valid_requirement_ids = {
        str(item.get("requirement_id") or "")
        for item in (getattr(state, "formal_evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    for field in ("immediate_actions", "recommended_routes", "alternative_routes"):
        cleaned: list[dict[str, Any]] = []
        for item in corrected.get(field) or []:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("label")
            if not str(title or "").strip():
                _record_issue(
                    issues,
                    code=f"remove_untitled_{field}",
                    message=f"已移除 {field} 中缺少标题的无效项。",
                )
                continue
            value = dict(item)
            if "depends_on_fact_ids" in value:
                value["depends_on_fact_ids"] = [
                    fact_id
                    for fact_id in value.get("depends_on_fact_ids") or []
                    if str(fact_id) in valid_fact_ids
                ]
            if "required_fact_ids" in value:
                value["required_fact_ids"] = [
                    fact_id
                    for fact_id in value.get("required_fact_ids") or []
                    if str(fact_id) in valid_fact_ids
                ]
            if "depends_on_requirement_ids" in value:
                value["depends_on_requirement_ids"] = [
                    requirement_id
                    for requirement_id in value.get(
                        "depends_on_requirement_ids"
                    )
                    or []
                    if str(requirement_id) in valid_requirement_ids
                ]
            if "authority_refs" in value:
                value["authority_refs"] = [
                    ref
                    for ref in value.get("authority_refs") or []
                    if isinstance(ref, dict)
                    and _basis_signature(ref) in allowed_signatures
                ]
            cleaned.append(value)
        corrected[field] = cleaned

    cleaned_tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for item in corrected.get("case_tasks") or []:
        if not isinstance(item, dict) or not item.get("task_id"):
            _record_issue(
                issues,
                code="remove_invalid_task",
                message="已移除缺少稳定任务编号的无效任务。",
            )
            continue
        task_id = str(item["task_id"])
        if task_id in seen_task_ids:
            _record_issue(
                issues,
                code="remove_duplicate_task",
                message="已移除重复任务编号。",
            )
            continue
        seen_task_ids.add(task_id)
        value = dict(item)
        value["authority_refs"] = [
            ref
            for ref in value.get("authority_refs") or []
            if isinstance(ref, dict)
            and _basis_signature(ref) in allowed_signatures
        ]
        value["depends_on_fact_ids"] = [
            fact_id
            for fact_id in value.get("depends_on_fact_ids") or []
            if str(fact_id) in valid_fact_ids
        ]
        value["depends_on_requirement_ids"] = [
            requirement_id
            for requirement_id in value.get(
                "depends_on_requirement_ids"
            )
            or []
            if str(requirement_id) in valid_requirement_ids
        ]
        cleaned_tasks.append(value)
    corrected["case_tasks"] = cleaned_tasks

    for field in _GENERATED_TEXT_FIELDS:
        corrected[field], count = _sanitize_generated_text(
            corrected.get(field)
        )
        if count:
            _record_issue(
                issues,
                code=f"repair_forbidden_language_{field}",
                message=f"已修正 {field} 中的概率、保证或越权认定表述。",
            )

    old_markdown = str(
        corrected.get("draft_markdown")
        or getattr(state, "solution_draft_markdown", "")
        or ""
    )
    markdown_checks = {
        "required_headings_present": all(
            heading in old_markdown for heading in _REQUIRED_HEADINGS
        ),
        "no_duplicate_required_headings": all(
            old_markdown.count(heading) == 1 for heading in _REQUIRED_HEADINGS
        ),
        "no_debug_leakage": not bool(
            re.search(
                r"(?:guide_state:|guide_active:|Redis|traceback|"
                r"solution-generation)",
                old_markdown,
                re.IGNORECASE,
            )
        ),
    }
    if not all(markdown_checks.values()):
        _record_issue(
            issues,
            code="repair_markdown_structure",
            message="已用确定性模板重建 Markdown 层级并移除调试标识或重复栏目。",
        )
    return corrected, issues, markdown_checks


def _formalize_tasks(
    state: Any,
    tasks: Iterable[Any],
    *,
    plan_version: int,
    published_at: str,
) -> list[dict[str, Any]]:
    previous = {
        str(item.get("task_id") or ""): dict(item)
        for item in (getattr(state, "case_tasks", []) or [])
        if isinstance(item, dict) and item.get("task_id")
    }
    allowed_statuses = {
        "pending",
        "in_progress",
        "completed",
        "blocked",
        "abandoned",
        "superseded",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict) or not item.get("task_id"):
            continue
        task_id = str(item["task_id"])
        if task_id in seen:
            continue
        seen.add(task_id)
        old = previous.get(task_id, {})
        value = {**old, **dict(item)}
        status = str(old.get("status") or value.get("status") or "pending")
        value["status"] = status if status in allowed_statuses else "pending"
        value.pop("plan_version_candidate", None)
        value["introduced_in_plan_version"] = int(
            old.get("introduced_in_plan_version") or plan_version
        )
        value["last_updated_in_plan_version"] = plan_version
        value["published_at"] = published_at
        result.append(value)
    return result


def _existing_version(
    history: Iterable[Any],
    *,
    candidate: str,
    draft_fingerprint: str,
) -> dict[str, Any] | None:
    for item in reversed(list(history)):
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("plan_version_candidate") or "") == candidate
            and str(item.get("draft_fingerprint") or "") == draft_fingerprint
        ):
            return dict(item)
    return None


def build_published_version(
    state: Any,
    audited_draft: dict[str, Any],
    *,
    audit_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    """Assign or reuse a formal version and build the immutable version package."""

    history = [
        dict(item)
        for item in (getattr(state, "solution_versions", []) or [])
        if isinstance(item, dict)
    ]
    candidate = str(
        audited_draft.get("plan_version_candidate")
        or getattr(state, "plan_version_candidate", "")
        or ""
    )
    draft_fingerprint = str(
        getattr(state, "solution_draft_fingerprint", "") or ""
    )
    existing = _existing_version(
        history,
        candidate=candidate,
        draft_fingerprint=draft_fingerprint,
    )
    if existing:
        return (
            dict(existing.get("solution") or {}),
            existing,
            history,
            True,
        )

    previous_version = int(getattr(state, "plan_version", 0) or 0)
    formal_version = previous_version + 1
    published_at = _now()
    solution = copy.deepcopy(audited_draft)
    tasks = _formalize_tasks(
        state,
        solution.get("case_tasks") or [],
        plan_version=formal_version,
        published_at=published_at,
    )
    solution["schema_version"] = PUBLISHED_PLAN_SCHEMA_VERSION
    solution["previous_plan_version"] = previous_version
    solution["plan_version"] = formal_version
    solution["published_at"] = published_at
    solution["reviewed_at"] = audit_report["reviewed_at"]
    solution["audit_id"] = audit_report["audit_id"]
    solution["audit_status"] = audit_report["status"]
    solution["case_tasks"] = tasks
    solution.pop("draft_markdown", None)
    solution.pop("next_route", None)
    markdown = render_solution_markdown(solution)
    markdown = (
        f"{markdown}\n\n"
        "## 后续更新\n\n"
        "- 可以继续补充或更正事实，系统会保留旧事实和旧方案版本。\n"
        "- 证据清单开放后可以继续补交或替换材料，并只重评受影响部分。\n"
        "- 如需根据当前事实生成参考文书，可回复“生成文书”。"
    )
    solution["published_markdown"] = markdown
    solution["published_fingerprint"] = _hash_payload(
        {
            key: value
            for key, value in solution.items()
            if key != "published_fingerprint"
        }
    )
    version_record = {
        "schema_version": PUBLISHED_PLAN_SCHEMA_VERSION,
        "case_id": str(getattr(state, "case_id", "") or ""),
        "case_generation": int(
            getattr(state, "case_generation", 1) or 1
        ),
        "plan_version": formal_version,
        "previous_plan_version": previous_version,
        "plan_version_candidate": candidate,
        "draft_fingerprint": draft_fingerprint,
        "published_fingerprint": solution["published_fingerprint"],
        "published_at": published_at,
        "reviewed_at": audit_report["reviewed_at"],
        "source_versions": dict(audit_report.get("source_versions") or {}),
        "change_summary": copy.deepcopy(
            solution.get("change_summary") or {}
        ),
        "fact_snapshot": copy.deepcopy(
            getattr(state, "fact_snapshot_draft", None) or {}
        ),
        "legal_model": copy.deepcopy(
            getattr(state, "legal_model", {}) or {}
        ),
        "evidence_plan": {
            "formal_evidence_requirements": copy.deepcopy(
                getattr(state, "formal_evidence_requirements", []) or []
            ),
            "proof_targets": copy.deepcopy(
                getattr(state, "proof_targets", []) or []
            ),
            "delivery_entries": copy.deepcopy(
                getattr(state, "delivery_entries", []) or []
            ),
        },
        "evidence_review": copy.deepcopy(
            getattr(state, "evidence_review_report", {}) or {}
        ),
        "solution": solution,
        "case_tasks": tasks,
        "case_progress": copy.deepcopy(
            getattr(state, "case_progress", []) or []
        ),
        "audit_report": copy.deepcopy(audit_report),
    }
    return solution, version_record, [*history, version_record], False


def _blocked_result(
    state: Any,
    validation: dict[str, Any],
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    next_route = str(validation.get("next_route") or "generate_solution")
    messages = "\n".join(
        f"- {item.get('message')}"
        for item in validation.get("fatal_issues") or []
    )
    return {
        "solution_draft_status": "audit_blocked",
        "pending_solution_audit": False,
        "solution_audit_status": "blocked",
        "solution_audit_id": audit_report["audit_id"],
        "solution_reviewed_at": audit_report["reviewed_at"],
        "solution_audit_report": audit_report,
        "solution_audit_history": [
            *(getattr(state, "solution_audit_history", []) or []),
            audit_report,
        ],
        "next_route": next_route,
        "decision_status": "solution_audit_blocked",
        "workflow_stage": (
            "fact_clarification"
            if next_route in {"update_facts", "decide_facts"}
            else "evidence_collection"
            if next_route in {"plan_evidence", "assess_evidence"}
            else "solution_drafting"
        ),
        "phase": (
            GuidePhase.DETAIL_GATHER
            if next_route in {"plan_evidence", "assess_evidence"}
            else GuidePhase.ISSUE_SEARCH
            if next_route in {"update_facts", "decide_facts"}
            else GuidePhase.CONCLUDE
        ),
        "messages": [
            AIMessage(
                content=(
                    "## 方案发布已暂停\n\n"
                    "审校发现上游案件版本已经变化，旧草稿不会覆盖当前案件：\n\n"
                    f"{messages or '- 需要重新生成当前方案草稿。'}"
                )
            )
        ],
    }


async def run_audit_and_save(
    state: Any,
    deps: Any = None,
) -> dict[str, Any]:
    """Audit node-seven output, publish one formal version, and persist it."""

    validation = validate_audit_inputs(state)
    reviewed_at = _now()
    audit_id = _stable_id(
        "solution-audit",
        getattr(state, "case_id", ""),
        getattr(state, "plan_version_candidate", ""),
        validation.get("draft_fingerprint"),
    )
    audit_report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": audit_id,
        "reviewed_at": reviewed_at,
        "status": "blocked" if not validation.get("valid") else "reviewing",
        "source_versions": dict(validation.get("source_versions") or {}),
        "fatal_issues": list(validation.get("fatal_issues") or []),
        "corrected_issues": [],
        "checks": {
            "source_versions": bool(validation.get("valid")),
            "fact_boundary": False,
            "legal_boundary": False,
            "evidence_boundary": False,
            "reasoning_consistency": False,
            "expression_boundary": False,
            "markdown_structure": False,
            "version_persistence": False,
        },
    }
    if not validation.get("valid"):
        return _blocked_result(state, validation, audit_report)

    audited, corrected_issues, markdown_checks = audit_solution_draft(
        state,
        getattr(state, "solution_draft", {}) or {},
    )
    audit_report["corrected_issues"] = corrected_issues
    audit_report["checks"].update(
        {
            "fact_boundary": True,
            "legal_boundary": True,
            "evidence_boundary": True,
            "reasoning_consistency": True,
            "expression_boundary": True,
            "markdown_structure": all(markdown_checks.values())
            or any(
                item.get("code") == "repair_markdown_structure"
                for item in corrected_issues
            ),
        }
    )
    audit_report["status"] = (
        "passed_with_corrections" if corrected_issues else "passed"
    )
    solution, version_record, history, idempotent = build_published_version(
        state,
        audited,
        audit_report=audit_report,
    )
    if not solution:
        audit_report["status"] = "blocked"
        _record_issue(
            audit_report["fatal_issues"],
            code="published_solution_missing",
            message="已登记版本缺少正式方案内容，需要重新生成。",
            route="generate_solution",
        )
        validation = {
            **validation,
            "valid": False,
            "fatal_issues": audit_report["fatal_issues"],
            "next_route": "generate_solution",
        }
        return _blocked_result(state, validation, audit_report)

    persistence = {
        "status": "case_state_pending",
        "consultation_id": None,
        "plan_version": int(solution.get("plan_version") or 0),
    }
    try:
        persistence = await save_solution_version(
            user_id=(getattr(state, "user_context", {}) or {}).get("user_id"),
            session_id=str(getattr(state, "session_id", "") or ""),
            domain=str(getattr(state, "legal_domain", "") or ""),
            issues=list(getattr(state, "confirmed_issues", []) or []),
            version_record=version_record,
            version_history=history,
            db=getattr(deps, "db_session", None) if deps is not None else None,
        )
    except Exception as exc:
        logger.error(
            "正式方案数据库索引保存失败，保留案件状态版本 | case={} error={}",
            getattr(state, "case_id", ""),
            exc,
        )
        persistence = {
            "status": "case_state_only",
            "consultation_id": None,
            "plan_version": int(solution.get("plan_version") or 0),
            "error": "database_index_unavailable",
        }
    audit_report["checks"]["version_persistence"] = True
    audit_report["persistence"] = persistence
    audit_report["idempotent_replay"] = idempotent
    audit_report["status"] = (
        "passed_with_corrections" if corrected_issues else "passed"
    )
    version_record["audit_report"] = copy.deepcopy(audit_report)
    if history:
        history[-1] = version_record

    published_markdown = str(solution.get("published_markdown") or "")
    return {
        "phase": GuidePhase.END,
        "workflow_stage": "plan_issued",
        "decision_status": (
            "solution_version_reused"
            if idempotent
            else "solution_published"
        ),
        "next_route": "",
        "pause_state": None,
        "solution_draft": solution,
        "solution_draft_markdown": published_markdown,
        "solution_draft_status": "published",
        "pending_solution_audit": False,
        "solution_audit_status": audit_report["status"],
        "solution_audit_id": audit_id,
        "solution_reviewed_at": reviewed_at,
        "solution_audit_report": audit_report,
        "solution_audit_history": [
            *(getattr(state, "solution_audit_history", []) or []),
            audit_report,
        ],
        "published_solution": solution,
        "published_solution_markdown": published_markdown,
        "published_solution_fingerprint": str(
            solution.get("published_fingerprint") or ""
        ),
        "plan_version": int(solution.get("plan_version") or 0),
        "previous_plan_version": int(
            solution.get("previous_plan_version") or 0
        ),
        "plan_published_at": str(solution.get("published_at") or ""),
        "solution_versions": history,
        "solution_persistence_status": str(
            persistence.get("status") or "case_state_only"
        ),
        "case_tasks": list(solution.get("case_tasks") or []),
        "likelihood_assessment": dict(
            solution.get("likelihood_assessment") or {}
        ),
        "likelihood_tier": str(
            (solution.get("likelihood_assessment") or {}).get("tier")
            or "不确定"
        ),
        "action_basis_refs": list(
            solution.get("action_basis_refs") or []
        ),
        "action_basis_gaps": list(
            solution.get("action_basis_gaps") or []
        ),
        "messages": [AIMessage(content=published_markdown)],
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "PUBLISHED_PLAN_SCHEMA_VERSION",
    "validate_audit_inputs",
    "audit_solution_draft",
    "build_published_version",
    "run_audit_and_save",
]
