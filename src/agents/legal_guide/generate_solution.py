"""Formal node seven: build a versioned, source-bounded action-plan draft.

The node consumes the fact snapshot, legal/evidence plan, and evidence review.
It does not rewrite facts, reassess uploaded files, publish a final plan, or
invent a success percentage.  Its output is a structured draft for the
audit-and-save boundary; the legacy ``conclude`` node remains a compatibility
presenter until node eight is migrated.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage

from src.agents.legal_guide.state import GuidePhase


SOLUTION_SCHEMA_VERSION = 1
LIKELIHOOD_TIERS = ("较有利", "条件性有利", "不确定", "风险较高")
DIMENSION_STATUSES = {
    "favorable",
    "mixed",
    "unfavorable",
    "unknown",
    "not_applicable",
}

_CONFIRMED_FACT_STATUSES = {
    "confirmed",
    "asserted",
    "user_stated",
    "corrected",
}
_UNKNOWN_FACT_STATUSES = {
    "unknown",
    "unclear",
    "ambiguous",
    "uncertain",
}
_CONFLICT_FACT_STATUSES = {"conflicted"}
_CLOSED_REQUIREMENT_STATUSES = {"stale", "not_applicable", "superseded"}
_USABLE_REVIEW_STATUSES = {"completed", "partial"}
_PENDING_REVIEW_STATUSES = {
    "awaiting_batch",
    "needs_verification",
    "received_pending_remap",
}

_DIMENSION_LABELS = {
    "rights_basis": "权利基础",
    "fact_clarity": "事实清晰度",
    "evidence_coverage": "证据覆盖",
    "procedural_feasibility": "程序可行性",
    "performance_risk": "履行风险",
}

_DIMENSION_STATUS_LABELS = {
    "favorable": "有利",
    "mixed": "有条件",
    "unfavorable": "存在明显风险",
    "unknown": "暂不确定",
    "not_applicable": "暂不适用",
}

_COVERAGE_LABELS = {
    "covered": "初步覆盖",
    "partially_covered": "部分覆盖",
    "conflicted": "存在冲突",
    "not_submitted": "尚未提交",
    "explicitly_absent": "当前明确缺失",
    "third_party_available": "可向第三方调取",
    "unresolved": "待确认",
    "not_applicable": "暂不适用",
}

_TIER_RANK = {
    "风险较高": 0,
    "不确定": 1,
    "条件性有利": 2,
    "较有利": 3,
}

_DOCUMENT_BY_DOMAIN = {
    "consumer_market": "消费者投诉或平台申诉说明",
    "labor_employment": "劳动争议申请材料",
    "traffic_accident": "赔偿协商或调解申请材料",
    "lease_housing": "催告函或民事起诉状结构",
    "contract_dispute": "履行催告函或民事起诉状结构",
    "property_infringement": "停止侵害通知或民事起诉状结构",
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
            seen.add(clean)
            result.append(clean)
    return result


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]
    return f"{prefix}-{digest}"


def _hash_payload(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fact_rows(state: Any) -> list[dict[str, Any]]:
    rows = (
        getattr(state, "fact_blackboard", None)
        or getattr(state, "case_facts", None)
        or []
    )
    return [dict(item) for item in rows if isinstance(item, dict)]


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


def _confirmed_facts(state: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _fact_rows(state)
        if str(item.get("status") or "") in _CONFIRMED_FACT_STATUSES
        and _fact_text(item)
    ]


def _unknown_facts(state: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _fact_rows(state)
        if str(item.get("status") or "") in _UNKNOWN_FACT_STATUSES
    ]


def _conflicted_facts(state: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _fact_rows(state)
        if str(item.get("status") or "") in _CONFLICT_FACT_STATUSES
    ]


def _active_requirements(state: Any) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (getattr(state, "formal_evidence_requirements", []) or [])
        if isinstance(item, dict)
        and str(item.get("status") or "") not in _CLOSED_REQUIREMENT_STATUSES
    ]


def _coverage_rows(state: Any) -> list[dict[str, Any]]:
    report = getattr(state, "evidence_review_report", {}) or {}
    if isinstance(report, dict) and isinstance(report.get("coverage"), list):
        return [
            dict(item)
            for item in report["coverage"]
            if isinstance(item, dict)
        ]
    legacy = getattr(state, "evidence_coverage", {}) or {}
    if isinstance(legacy, dict) and isinstance(legacy.get("coverage"), list):
        return [
            dict(item)
            for item in legacy["coverage"]
            if isinstance(item, dict)
        ]
    return []


def _material_items(state: Any) -> list[dict[str, Any]]:
    report = getattr(state, "evidence_review_report", {}) or {}
    if isinstance(report, dict) and isinstance(report.get("items"), list):
        return [
            dict(item)
            for item in report["items"]
            if isinstance(item, dict)
        ]
    return [
        dict(item)
        for item in (getattr(state, "evidence_items", []) or [])
        if isinstance(item, dict)
    ]


def validate_solution_inputs(state: Any) -> dict[str, Any]:
    """Validate upstream checkpoints without changing any upstream object."""

    if getattr(state, "guard_pause_required", False) or getattr(
        state, "safety_pause_active", False
    ):
        return {
            "valid": False,
            "reason": "guard_pause_active",
            "next_route": "pause",
            "message": "当前存在尚未解除的安全或紧迫风险，普通行动方案继续暂停。",
        }

    snapshot_version = int(getattr(state, "fact_snapshot_version", 0) or 0)
    snapshot_confirmed = bool(
        getattr(state, "fact_snapshot_confirmed", False)
        or getattr(state, "proceed_under_uncertainty", False)
    )
    if not snapshot_version or not snapshot_confirmed:
        return {
            "valid": False,
            "reason": "fact_snapshot_missing",
            "next_route": "decide_facts",
            "message": "当前事实快照尚未确认，需要先完成事实收敛或明确按未知条件继续。",
        }

    snapshot = getattr(state, "fact_snapshot_draft", None) or {}
    based_on_blackboard = int(
        snapshot.get("based_on_fact_blackboard_version") or 0
    )
    current_blackboard = int(
        getattr(state, "fact_blackboard_version", 0) or 0
    )
    if (
        snapshot.get("stale")
        or (
            based_on_blackboard
            and current_blackboard
            and based_on_blackboard != current_blackboard
        )
    ):
        return {
            "valid": False,
            "reason": "fact_snapshot_stale",
            "next_route": "decide_facts",
            "message": "事实账本已经变化，旧事实快照不能继续生成新方案。",
        }

    if getattr(state, "new_fact_candidates_from_evidence", []):
        return {
            "valid": False,
            "reason": "evidence_fact_confirmation_pending",
            "next_route": "update_facts",
            "message": "材料中还有待用户确认的新事实，不能静默写入行动方案。",
        }

    if (
        int(getattr(state, "legal_model_version", 0) or 0) <= 0
        or int(getattr(state, "evidence_plan_version", 0) or 0) <= 0
        or str(getattr(state, "legal_model_status", "") or "")
        == "needs_fact_update"
    ):
        return {
            "valid": False,
            "reason": "legal_or_evidence_plan_missing",
            "next_route": "plan_evidence",
            "message": "当前法律模型或正式证据计划尚未形成，需要先完成节点五。",
        }

    if getattr(state, "stale_dependencies", []):
        return {
            "valid": False,
            "reason": "upstream_dependencies_stale",
            "next_route": "plan_evidence",
            "message": "上游法律或证据规划仍依赖已经变化的事实，需要先重新规划。",
        }

    report = getattr(state, "evidence_review_report", {}) or {}
    if isinstance(report, dict) and report:
        report_snapshot = int(report.get("fact_snapshot_version") or 0)
        report_plan = int(report.get("evidence_plan_version") or 0)
        if report_snapshot and report_snapshot != snapshot_version:
            return {
                "valid": False,
                "reason": "evidence_review_fact_version_stale",
                "next_route": "assess_evidence",
                "message": "证据评估仍绑定旧事实快照，需要按当前事实重新评估。",
            }
        current_plan = int(getattr(state, "evidence_plan_version", 0) or 0)
        if report_plan and report_plan != current_plan:
            return {
                "valid": False,
                "reason": "evidence_review_plan_version_stale",
                "next_route": "assess_evidence",
                "message": "证据评估仍绑定旧证据清单，需要按当前清单重新评估。",
            }

    review_status = str(
        getattr(state, "evidence_review_status", "") or "not_started"
    )
    if (
        getattr(state, "evidence_verification_pending", False)
        or review_status == "needs_verification"
    ):
        return {
            "valid": False,
            "reason": "evidence_verification_pending",
            "next_route": "assess_evidence",
            "message": "当前证据评估还在等待本批次唯一一次材料属性核验。",
        }
    if review_status == "received_pending_remap":
        return {
            "valid": False,
            "reason": "evidence_remap_pending",
            "next_route": "plan_evidence",
            "message": "已有材料正在等待按最新证据清单重新归类。",
        }
    if review_status == "awaiting_batch" and (
        _material_items(state) or getattr(state, "evidence_observations", [])
    ):
        return {
            "valid": False,
            "reason": "evidence_batch_incomplete",
            "next_route": "assess_evidence",
            "message": "当前材料批次尚未完成，需先完成或取消本批次后再生成方案。",
        }

    conditional = bool(
        getattr(state, "proceed_under_uncertainty", False)
        or _unknown_facts(state)
        or _conflicted_facts(state)
        or review_status not in _USABLE_REVIEW_STATUSES
    )
    return {
        "valid": True,
        "reason": "ready",
        "next_route": "audit_and_save",
        "message": "",
        "conditional": conditional,
        "fact_snapshot_version": snapshot_version,
        "fact_snapshot_hash": str(snapshot.get("snapshot_hash") or ""),
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


def _basis_key(item: dict[str, Any]) -> str:
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


def _is_usable_basis(item: dict[str, Any]) -> bool:
    status = str(item.get("status") or item.get("effective_status") or "")
    review = str(item.get("review_status") or "")
    if status in {"inactive", "expired", "repealed", "invalid"}:
        return False
    if review in {"needs_pinpoint", "pending", "unreviewed", "rejected"}:
        return False
    return bool(
        item.get("locator")
        or item.get("article_no")
        or item.get("source_url")
        or item.get("official_url")
        or item.get("url")
        or item.get("source_id")
    )


def load_reusable_action_basis(
    state: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reuse node-five/six references and expose precision gaps."""

    refs: list[dict[str, Any]] = []
    gaps: list[str] = []
    candidates: list[dict[str, Any]] = []
    for source in (
        getattr(state, "plan_basis_refs", []) or [],
        getattr(state, "evidence_basis_refs", []) or [],
        getattr(state, "retrieved_law_refs", []) or [],
    ):
        candidates.extend(
            dict(item) for item in source if isinstance(item, dict)
        )
    for coverage in _coverage_rows(state):
        candidates.extend(
            dict(item)
            for item in (
                coverage.get("authority_refs")
                or coverage.get("basis_refs")
                or []
            )
            if isinstance(item, dict)
        )
    for channel in getattr(state, "relevant_channels", []) or []:
        if not isinstance(channel, dict) or not channel.get("name"):
            continue
        candidates.append(
            {
                "basis_type": "official_channel",
                "title": channel.get("name"),
                "name": channel.get("name"),
                "phone": channel.get("phone") or "",
                "url": channel.get("url") or "",
                "source_url": channel.get("source_url") or "",
                "issuing_authority": channel.get("source_org") or "",
                "retrieved_at": channel.get("last_verified_on") or "",
                "applicable_region": channel.get("region") or "",
                "applicable_procedure": channel.get("route_stage") or "",
            }
        )

    seen: set[str] = set()
    for item in candidates:
        key = _basis_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        if _is_usable_basis(item):
            refs.append(item)
        else:
            title = _compact(item.get("title") or item.get("name"))
            if title:
                gaps.append(f"“{title}”缺少可核对的版本、定位或官方入口")

    gaps.extend(getattr(state, "plan_basis_limitations", []) or [])
    gaps.extend(getattr(state, "evidence_basis_missing", []) or [])
    if not refs:
        gaps.append("当前没有可精确定位的行动依据，只生成低风险条件式步骤")
    return refs[:20], _unique(gaps)


def _fact_basis(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "basis_type": "fact_snapshot",
            "fact_id": _fact_id(item),
            "statement": _fact_text(item),
        }
        for item in items
        if _fact_id(item) or _fact_text(item)
    ]


def _dimension(
    dimension_id: str,
    status: str,
    *,
    positive: Iterable[Any] = (),
    negative: Iterable[Any] = (),
    unknown: Iterable[Any] = (),
    basis_refs: Iterable[dict[str, Any]] = (),
    limitations: Iterable[Any] = (),
) -> dict[str, Any]:
    normalized_status = (
        status if status in DIMENSION_STATUSES else "unknown"
    )
    return {
        "dimension_id": dimension_id,
        "label": _DIMENSION_LABELS[dimension_id],
        "status": normalized_status,
        "status_label": _DIMENSION_STATUS_LABELS[normalized_status],
        "positive_factors": _unique(positive),
        "negative_factors": _unique(negative),
        "unknown_factors": _unique(unknown),
        "basis_refs": [dict(item) for item in basis_refs if isinstance(item, dict)],
        "limitations": _unique(limitations),
    }


def build_likelihood_dimensions(
    state: Any,
    *,
    action_basis_refs: list[dict[str, Any]] | None = None,
    action_basis_gaps: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build five conservative dimensions without exposing numeric scores."""

    refs = list(action_basis_refs or [])
    gaps = list(action_basis_gaps or [])
    confirmed = _confirmed_facts(state)
    unknown = _unknown_facts(state)
    conflicts = _conflicted_facts(state)
    legal_model = getattr(state, "legal_model", {}) or {}
    relations = (
        getattr(state, "relation_candidates", None)
        or legal_model.get("relation_candidates")
        or []
    )
    requests = (
        getattr(state, "request_models", None)
        or legal_model.get("request_models")
        or []
    )
    authority_refs = [
        item
        for item in refs
        if str(item.get("basis_type") or "") != "official_channel"
    ]

    if relations and requests and authority_refs:
        rights_status = "favorable"
    elif relations or requests:
        rights_status = "mixed"
    else:
        rights_status = "unknown"
    rights_positive = [
        _compact(item.get("label") or item.get("description"))
        for item in relations[:2]
    ] + [
        _compact(item.get("label") or item.get("requested_action"))
        for item in requests[:3]
    ]
    rights_unknown = list(legal_model.get("unknown_conditions") or [])
    if not authority_refs:
        rights_unknown.append("现有法律依据尚缺少可精确定位的有效来源")
    rights = _dimension(
        "rights_basis",
        rights_status,
        positive=rights_positive,
        unknown=rights_unknown,
        basis_refs=authority_refs[:8],
        limitations=gaps,
    )

    if conflicts:
        fact_status = "unfavorable" if not confirmed else "mixed"
    elif confirmed and not unknown:
        fact_status = "favorable"
    elif confirmed:
        fact_status = "mixed"
    else:
        fact_status = "unknown"
    facts = _dimension(
        "fact_clarity",
        fact_status,
        positive=[
            f"已有 {len(confirmed)} 项事实进入当前事实快照"
            if confirmed else ""
        ],
        negative=[
            f"仍有 {len(conflicts)} 组事实冲突未解决"
            if conflicts else ""
        ],
        unknown=[
            *[_fact_text(item) or _fact_id(item) for item in unknown[:6]],
            *[_fact_text(item) or _fact_id(item) for item in conflicts[:6]],
        ],
        basis_refs=_fact_basis([*confirmed, *unknown, *conflicts])[:16],
        limitations=[
            "事实快照记录的是当前可采用陈述，不代表已由第三方查证。",
        ],
    )

    coverage = _coverage_rows(state)
    status_by_target = {
        str(item.get("target_id") or item.get("proof_target_id") or ""):
        str(item.get("status") or "unresolved")
        for item in coverage
    }
    essential_target_ids = {
        str(
            item.get("proof_target_id")
            or item.get("target_id")
            or ""
        )
        for item in _active_requirements(state)
        if str(item.get("importance") or "") == "essential"
    }
    essential_statuses = [
        status
        for target_id, status in status_by_target.items()
        if not essential_target_ids or target_id in essential_target_ids
    ]
    covered = [
        item for item in coverage
        if str(item.get("status") or "") == "covered"
    ]
    partial = [
        item for item in coverage
        if str(item.get("status") or "") == "partially_covered"
    ]
    adverse_coverage = [
        item for item in coverage
        if str(item.get("status") or "")
        in {"conflicted", "explicitly_absent"}
    ]
    unresolved_coverage = [
        item for item in coverage
        if str(item.get("status") or "")
        in {"not_submitted", "third_party_available", "unresolved"}
    ]
    if coverage and essential_statuses and all(
        status == "covered" for status in essential_statuses
    ):
        evidence_status = "favorable"
    elif adverse_coverage and not covered:
        evidence_status = "unfavorable"
    elif coverage:
        evidence_status = "mixed"
    else:
        evidence_status = "unknown"
    evidence = _dimension(
        "evidence_coverage",
        evidence_status,
        positive=[
            f"{item.get('label') or '证明目标'}已初步覆盖"
            for item in covered[:6]
        ],
        negative=[
            f"{item.get('label') or '证明目标'}："
            f"{_COVERAGE_LABELS.get(str(item.get('status')), '存在缺口')}"
            for item in adverse_coverage[:6]
        ],
        unknown=[
            f"{item.get('label') or '证明目标'}："
            f"{_COVERAGE_LABELS.get(str(item.get('status')), '待确认')}"
            for item in [*partial, *unresolved_coverage][:8]
        ],
        basis_refs=[
            {
                "basis_type": "evidence_coverage",
                "target_id": item.get("target_id"),
                "status": item.get("status"),
                "supporting_evidence_ids":
                    item.get("supporting_evidence_ids") or [],
            }
            for item in coverage
        ],
        limitations=[
            "证据覆盖只表示材料可能支持的范围，不是司法机关的最终证据效力认定。",
        ],
    )

    channels = [
        item for item in (getattr(state, "relevant_channels", []) or [])
        if isinstance(item, dict) and item.get("name")
    ]
    risk = getattr(state, "deadline_risk", None) or {}
    risk_level = str(risk.get("level") or risk.get("status") or "").lower()
    if risk_level in {"critical", "urgent", "high", "expired"}:
        procedure_status = "unfavorable"
    elif channels and getattr(state, "region", ""):
        procedure_status = "favorable"
    elif channels:
        procedure_status = "mixed"
    else:
        procedure_status = "unknown"
    procedure = _dimension(
        "procedural_feasibility",
        procedure_status,
        positive=[
            f"已有可核对渠道：{item.get('name')}"
            for item in channels[:4]
        ],
        negative=[
            _compact(risk.get("trigger") or risk.get("reason"))
            if risk_level in {"critical", "urgent", "high", "expired"}
            else ""
        ],
        unknown=[
            "具体受理机构、管辖、期限或材料要求仍需通过官方渠道核对"
            if not channels or not getattr(state, "region", "") else ""
        ],
        basis_refs=[
            item for item in refs
            if str(item.get("basis_type") or "") == "official_channel"
        ],
        limitations=[
            "具体期限、费用、管辖和办理要求只在存在精确有效依据时才能确定。",
        ],
    )

    confirmed_text = " ".join(_fact_text(item) for item in confirmed)
    adverse_markers = ("拉黑", "失联", "拒绝", "未履行", "未发货", "不退款")
    positive_markers = ("已回复", "同意处理", "已经退款", "已经履行", "有送达地址")
    adverse_performance = [
        marker for marker in adverse_markers if marker in confirmed_text
    ]
    positive_performance = [
        marker for marker in positive_markers if marker in confirmed_text
    ]
    if positive_performance and not adverse_performance:
        performance_status = "favorable"
    elif adverse_performance:
        performance_status = "mixed"
    else:
        performance_status = "unknown"
    performance = _dimension(
        "performance_risk",
        performance_status,
        positive=[
            f"当前事实显示：{marker}" for marker in positive_performance
        ],
        negative=[
            f"当前事实显示：{marker}" for marker in adverse_performance
        ],
        unknown=[
            "对方身份、有效联系方式、送达或实际履行能力尚无可验证信息"
            if performance_status == "unknown" else ""
        ],
        basis_refs=_fact_basis(
            [
                item for item in confirmed
                if any(
                    marker in _fact_text(item)
                    for marker in (*adverse_markers, *positive_markers)
                )
            ]
        ),
        limitations=[
            "不根据模型常识猜测对方财产、经营状态或最终履行能力。",
        ],
    )
    return [rights, facts, evidence, procedure, performance]


def derive_qualitative_likelihood(
    dimensions: Iterable[dict[str, Any]],
    *,
    basis_gaps: Iterable[Any] = (),
) -> dict[str, Any]:
    rows = [dict(item) for item in dimensions if isinstance(item, dict)]
    by_id = {str(item.get("dimension_id")): item for item in rows}
    statuses = [str(item.get("status") or "unknown") for item in rows]
    favorable_count = statuses.count("favorable")
    unfavorable_count = statuses.count("unfavorable")
    rights_status = str(
        by_id.get("rights_basis", {}).get("status") or "unknown"
    )
    if rights_status == "unfavorable" or unfavorable_count >= 2:
        tier = "风险较高"
    elif (
        rights_status == "favorable"
        and all(
            str(by_id.get(key, {}).get("status") or "")
            == "favorable"
            for key in (
                "fact_clarity",
                "evidence_coverage",
                "procedural_feasibility",
            )
        )
        and unfavorable_count == 0
    ):
        tier = "较有利"
    elif (
        rights_status == "favorable"
        and favorable_count >= 1
        and unfavorable_count == 0
    ) or (favorable_count >= 3 and unfavorable_count == 0):
        tier = "条件性有利"
    else:
        tier = "不确定"
    if tier not in LIKELIHOOD_TIERS:
        tier = "不确定"
    return {
        "tier": tier,
        "dimensions": rows,
        "positive_factors": _unique(
            factor
            for item in rows
            for factor in item.get("positive_factors") or []
        ),
        "negative_factors": _unique(
            factor
            for item in rows
            for factor in item.get("negative_factors") or []
        ),
        "unknown_factors": _unique(
            factor
            for item in rows
            for factor in item.get("unknown_factors") or []
        ),
        "limitations": _unique(
            [
                *[
                    limitation
                    for item in rows
                    for limitation in item.get("limitations") or []
                ],
                *basis_gaps,
                "该等级是基于当前版本事实、法律依据和证据覆盖的定性判断，不代表具体概率或结果保证。",
            ]
        ),
    }


def build_evidence_effect_summary(state: Any) -> dict[str, Any]:
    coverage = _coverage_rows(state)
    counts: dict[str, int] = {}
    for item in coverage:
        status = str(item.get("status") or "unresolved")
        counts[status] = counts.get(status, 0) + 1
    critical_gaps = [
        {
            "target_id": item.get("target_id"),
            "requirement_id": item.get("requirement_id"),
            "label": item.get("label") or "证明目标",
            "status": item.get("status") or "unresolved",
            "next_action": item.get("next_action") or "",
        }
        for item in coverage
        if str(item.get("status") or "")
        in {
            "partially_covered",
            "conflicted",
            "not_submitted",
            "explicitly_absent",
            "third_party_available",
            "unresolved",
        }
    ]
    return {
        "review_status": str(
            getattr(state, "evidence_review_status", "") or "not_started"
        ),
        "coverage_counts": counts,
        "covered_target_ids": [
            item.get("target_id")
            for item in coverage
            if str(item.get("status") or "") == "covered"
        ],
        "critical_gaps": critical_gaps,
        "third_party_target_ids": [
            item.get("target_id")
            for item in coverage
            if str(item.get("status") or "") == "third_party_available"
        ],
        "conflicted_target_ids": [
            item.get("target_id")
            for item in coverage
            if str(item.get("status") or "") == "conflicted"
        ],
        "material_count": len(_material_items(state)),
        "disclaimer": (
            "未提交不等于没有；已经提交也不等于真实性、合法性、"
            "可采性或最终证明力已经确定。"
        ),
    }


def _action(
    state: Any,
    key: str,
    title: str,
    reason: str,
    *,
    priority: int,
    fact_ids: Iterable[Any] = (),
    requirement_ids: Iterable[Any] = (),
    authority_refs: Iterable[dict[str, Any]] = (),
    completion_criteria: str = "",
) -> dict[str, Any]:
    return {
        "action_id": _stable_id("action", state.case_id, key),
        "action_key": key,
        "title": _compact(title, 180),
        "priority": priority,
        "reason": _compact(reason, 360),
        "depends_on_fact_ids": _unique(fact_ids),
        "depends_on_requirement_ids": _unique(requirement_ids),
        "authority_refs": [
            dict(item) for item in authority_refs if isinstance(item, dict)
        ],
        "completion_criteria": _compact(
            completion_criteria or f"已完成“{title}”并保存可回查记录",
            320,
        ),
        "safe_now": True,
    }


def build_immediate_actions(
    state: Any,
    evidence_summary: dict[str, Any],
    *,
    action_basis_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    refs = list(action_basis_refs or [])
    guard = getattr(state, "guard_report", {}) or {}
    for item in guard.get("immediate_actions") or []:
        if not isinstance(item, dict) or not item.get("action"):
            continue
        actions.append(
            _action(
                state,
                f"guard.{item.get('action_id') or len(actions)}",
                str(item.get("action")),
                "节点二识别到需要优先处理的安全、期限或证据灭失风险。",
                priority=0,
                completion_criteria="已采取风险保护行动并保留回执或记录",
            )
        )

    if _material_items(state) or getattr(state, "evidence_name_inventory", []):
        actions.append(
            _action(
                state,
                "preserve_materials",
                "备份原始材料并保留原始载体",
                "后续核验需要能够回到完整原文件、平台记录或原始设备。",
                priority=1,
                completion_criteria="原始文件、导出记录和备份均可正常打开并按时间归档",
            )
        )
    else:
        actions.append(
            _action(
                state,
                "inventory_existing_records",
                "列出现有记录并先保存容易灭失的内容",
                "当前没有完成材料评估，先保存订单、沟通、支付或程序记录不会妨碍后续选择。",
                priority=1,
                completion_criteria="已列出现有记录名称，并保存当前能够取得的原始内容",
            )
        )

    for index, gap in enumerate(evidence_summary.get("critical_gaps") or []):
        next_action = _compact(gap.get("next_action"))
        label = _compact(gap.get("label") or "关键证明目标")
        if not next_action:
            next_action = f"核对并补充能够支持“{label}”的原始记录"
        actions.append(
            _action(
                state,
                f"evidence_gap.{gap.get('target_id') or index}",
                next_action,
                f"“{label}”当前状态为"
                f"{_COVERAGE_LABELS.get(str(gap.get('status')), '存在缺口')}。",
                priority=2,
                requirement_ids=[gap.get("requirement_id")],
                completion_criteria=(
                    f"已提交、标记无法提供，或记录“{label}”的合法调取渠道"
                ),
            )
        )
        if len(actions) >= 6:
            break

    channel_ref_map = {
        str(item.get("title") or item.get("name") or ""): item
        for item in refs
        if str(item.get("basis_type") or "") == "official_channel"
    }
    first_channel = next(
        (
            item
            for item in (getattr(state, "relevant_channels", []) or [])
            if isinstance(item, dict) and item.get("name")
        ),
        None,
    )
    if first_channel:
        name = _compact(first_channel.get("name"), 120)
        actions.append(
            _action(
                state,
                f"channel.{name}",
                f"通过{name}核对受理范围和下一步",
                "当前已有可联系渠道，但具体管辖、材料和时限仍以该渠道正式答复为准。",
                priority=3,
                authority_refs=[channel_ref_map[name]]
                if name in channel_ref_map else [],
                completion_criteria="已保存咨询或提交内容、工单编号和正式回复",
            )
        )
    return sorted(actions, key=lambda item: int(item.get("priority") or 99))[:7]


def _channel_route(
    state: Any,
    channel: dict[str, Any],
    *,
    index: int,
    authority_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = _compact(channel.get("name") or "可核对办理渠道", 160)
    phone = _compact(channel.get("phone"), 80)
    url = _compact(channel.get("url"), 260)
    contact = "；".join(
        item for item in (
            f"电话：{phone}" if phone else "",
            f"入口：{url}" if url else "",
        )
        if item
    )
    return {
        "route_id": _stable_id("route", state.case_id, name, index),
        "route_type": _compact(
            channel.get("route_stage") or channel.get("domain") or "official_channel",
            100,
        ),
        "label": name,
        "rationale": _compact(
            channel.get("recommendation_reason")
            or "通过现有渠道先核对受理范围、提交方式和所需材料。",
            360,
        ),
        "entry_conditions": _unique(
            channel.get("applicable_matters")
            or ["当前争议属于该渠道正式答复的受理范围"]
        ),
        "current_condition_status": (
            "available" if phone or url else "needs_official_confirmation"
        ),
        "required_fact_ids": [],
        "required_evidence_ids": [],
        "authority_refs": [authority_ref] if authority_ref else [],
        "contact": contact,
        "first_action": (
            f"先通过{name}{f'（{contact}）' if contact else ''}核对受理范围，"
            "再提交与当前证明目标对应的材料。"
        ),
        "expected_next_event": "取得咨询答复、受理回执、工单编号或不予受理说明",
        "escalation_condition": (
            "该路径未解决争议，且事实、主体、期限和关键材料已经核对后，"
            "再评估下一层级程序。"
        ),
        "stop_condition": (
            "确认不属于该渠道受理范围，或事实、证据版本发生实质变化时，"
            "先停止重复提交并重新核对路径。"
        ),
        "risks": [
            "受理范围、地域、期限和材料要求必须以该渠道当前正式说明为准。",
        ],
    }


def build_action_routes(
    state: Any,
    *,
    action_basis_refs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs = list(action_basis_refs or [])
    channels = [
        dict(item)
        for item in (getattr(state, "relevant_channels", []) or [])
        if isinstance(item, dict) and item.get("name")
    ]
    ref_by_title = {
        str(item.get("title") or item.get("name") or ""): item
        for item in refs
        if str(item.get("basis_type") or "") == "official_channel"
    }
    recommended: list[dict[str, Any]] = []
    alternative: list[dict[str, Any]] = []
    if channels:
        for index, channel in enumerate(channels[:3]):
            route = _channel_route(
                state,
                channel,
                index=index,
                authority_ref=ref_by_title.get(str(channel.get("name") or "")),
            )
            (recommended if index == 0 else alternative).append(route)
    else:
        confirmed = _confirmed_facts(state)
        confirmed_text = " ".join(_fact_text(item) for item in confirmed)
        platform_context = any(
            marker in confirmed_text
            for marker in ("平台", "投诉", "工单", "客服")
        )
        route_key = "platform_followup" if platform_context else "written_resolution"
        label = "继续现有平台或机构处理" if platform_context else "书面沟通与材料保全"
        recommended.append(
            {
                "route_id": _stable_id("route", state.case_id, route_key),
                "route_type": route_key,
                "label": label,
                "rationale": (
                    "当前缺少可精确定位的办理渠道，先完成低风险、可回查的记录和书面沟通。"
                ),
                "entry_conditions": ["不需要系统猜测具体受理机构即可先完成"],
                "current_condition_status": "conditional",
                "required_fact_ids": [],
                "required_evidence_ids": [],
                "authority_refs": [],
                "contact": "",
                "first_action": "保存现有记录，并书面提出明确请求、期限和联系方式。",
                "expected_next_event": "取得对方、平台或机构的可保存回复",
                "escalation_condition": "未解决且已核对主体、期限和正式办理渠道后再升级。",
                "stop_condition": "发现事实冲突、主体错误或期限风险时先停止重复沟通并核对。",
                "risks": ["当前具体受理渠道和程序条件仍需通过官方来源核对。"],
            }
        )
    alternative.append(
        {
            "route_id": _stable_id(
                "route", state.case_id, "professional_review"
            ),
            "route_type": "professional_review",
            "label": "专业法律服务核对与升级准备",
            "rationale": (
                "当主体、期限、管辖、金额或程序条件仍不确定时，先做定向核对，"
                "避免直接进入高成本程序。"
            ),
            "entry_conditions": [
                "重要决定仍受未知事实、证据缺口或程序条件影响"
            ],
            "current_condition_status": "available",
            "required_fact_ids": [],
            "required_evidence_ids": [],
            "authority_refs": [],
            "contact": "",
            "first_action": "携带事实时间线和证据清单进行定向咨询。",
            "expected_next_event": "确认可行请求、受理路径、期限和材料要求",
            "escalation_condition": "核对后确认存在明确、可执行的正式程序。",
            "stop_condition": "发现成本、期限或证明风险明显高于预期时重新比较路径。",
            "risks": ["咨询结论仍需以完整事实、原始材料和现行规则为基础。"],
        }
    )
    return recommended[:2], alternative[:3]


def build_case_tasks(
    state: Any,
    actions: Iterable[dict[str, Any]],
    routes: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous = {
        str(item.get("task_id") or ""): dict(item)
        for item in (getattr(state, "case_tasks", []) or [])
        if isinstance(item, dict) and item.get("task_id")
    }
    tasks: list[dict[str, Any]] = []
    priority_label = {0: "urgent", 1: "high", 2: "high", 3: "medium"}
    for action in actions:
        task_id = _stable_id("task", action.get("action_id"))
        old = previous.get(task_id, {})
        old_status = str(old.get("status") or "pending")
        status = (
            old_status
            if old_status
            in {
                "pending",
                "in_progress",
                "completed",
                "blocked",
                "abandoned",
                "superseded",
            }
            else "pending"
        )
        tasks.append(
            {
                "task_id": task_id,
                "title": action.get("title"),
                "status": status,
                "priority": priority_label.get(
                    int(action.get("priority") or 3), "medium"
                ),
                "route_id": "",
                "reason": action.get("reason") or "",
                "depends_on_fact_ids":
                    list(action.get("depends_on_fact_ids") or []),
                "depends_on_requirement_ids":
                    list(action.get("depends_on_requirement_ids") or []),
                "authority_refs": list(action.get("authority_refs") or []),
                "recommended_due_at": None,
                "due_basis_ref": None,
                "completion_criteria":
                    action.get("completion_criteria") or "",
                "blocking_reason": old.get("blocking_reason"),
            }
        )
    first_route = next(iter(routes), None)
    if first_route:
        task_id = _stable_id("task", first_route.get("route_id"), "start")
        old = previous.get(task_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "title": f"启动路径：{first_route.get('label')}",
                "status": str(old.get("status") or "pending"),
                "priority": "medium",
                "route_id": first_route.get("route_id"),
                "reason": first_route.get("rationale") or "",
                "depends_on_fact_ids":
                    list(first_route.get("required_fact_ids") or []),
                "depends_on_requirement_ids": [],
                "authority_refs":
                    list(first_route.get("authority_refs") or []),
                "recommended_due_at": None,
                "due_basis_ref": None,
                "completion_criteria":
                    first_route.get("expected_next_event") or "",
                "blocking_reason": old.get("blocking_reason"),
            }
        )
    return tasks[:10]


def build_document_suggestions(
    state: Any,
    *,
    action_basis_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    domain = str(getattr(state, "legal_domain", "") or "other")
    document_type = _DOCUMENT_BY_DOMAIN.get(
        domain,
        "书面协商、投诉或程序申请材料",
    )
    missing_fields = _unique(
        _fact_text(item) or _fact_id(item)
        for item in [*_unknown_facts(state), *_conflicted_facts(state)]
    )
    template_ref = next(
        (
            item
            for item in (action_basis_refs or [])
            if "template" in str(item.get("basis_type") or "").lower()
            or "示范" in str(item.get("title") or "")
        ),
        None,
    )
    return [
        {
            "suggestion_id": _stable_id(
                "document-suggestion", state.case_id, document_type
            ),
            "document_type": document_type,
            "reason": "用于把当前事实、请求、材料和办理记录整理为可提交文本。",
            "template_source_ref": template_ref,
            "can_generate_now": True,
            "missing_fields": missing_fields[:8],
            "placeholder_policy": (
                "缺失的主体、地址、日期、金额和受理机构必须保留"
                "[待补充]或[待核对]占位，不得补造。"
            ),
        }
    ]


def _core_judgment(
    state: Any,
    likelihood: dict[str, Any],
    conditional: bool,
) -> dict[str, Any]:
    legal_model = getattr(state, "legal_model", {}) or {}
    relations = (
        getattr(state, "relation_candidates", None)
        or legal_model.get("relation_candidates")
        or []
    )
    requests = (
        getattr(state, "request_models", None)
        or legal_model.get("request_models")
        or []
    )
    relation_labels = _unique(
        item.get("label") or item.get("description") or item.get("relation_id")
        for item in relations
        if isinstance(item, dict)
    )
    request_labels = _unique(
        item.get("label")
        or item.get("requested_action")
        or item.get("request_type")
        for item in requests
        if isinstance(item, dict)
    )
    return {
        "relation_labels": relation_labels,
        "request_labels": request_labels,
        "summary": (
            f"当前法律关系候选为"
            f"{'、'.join(relation_labels) or '一般争议关系'}；"
            f"当前请求为{'、'.join(request_labels) or '根据现有事实寻求适当救济'}。"
        ),
        "likelihood_tier": likelihood.get("tier") or "不确定",
        "conditional": conditional,
        "primary_condition": (
            (likelihood.get("unknown_factors") or ["仍需核对关键事实、证据或程序条件"])[0]
            if conditional else "当前主要条件已达到可行动程度"
        ),
        "disclaimer": (
            "这是基于当前案件版本的行动判断，不认定责任已经成立，"
            "也不保证平台、机构或法院作出特定结果。"
        ),
    }


def _likelihood_change(previous_tier: str, current_tier: str) -> str:
    if not previous_tier:
        return "newly_assessable"
    if previous_tier == current_tier:
        return "unchanged"
    if _TIER_RANK.get(current_tier, 1) > _TIER_RANK.get(previous_tier, 1):
        return "upgraded"
    return "downgraded"


def build_solution_change_summary(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    old = dict(previous or {})
    previous_likelihood = old.get("likelihood_assessment") or {}
    current_likelihood = current.get("likelihood_assessment") or {}
    previous_tier = str(previous_likelihood.get("tier") or "")
    current_tier = str(current_likelihood.get("tier") or "")
    changed_sections: list[str] = []
    section_fields = (
        ("core_judgment", "核心判断"),
        ("likelihood_assessment", "维权可能性"),
        ("evidence_effect_summary", "证据影响"),
        ("recommended_routes", "推荐路径"),
        ("alternative_routes", "替代路径"),
        ("immediate_actions", "立即行动"),
        ("case_tasks", "任务"),
        ("document_suggestions", "参考文书"),
    )
    for key, label in section_fields:
        if _hash_payload(old.get(key)) != _hash_payload(current.get(key)):
            changed_sections.append(label)
    old_tasks = {
        str(item.get("task_id"))
        for item in old.get("case_tasks") or []
        if isinstance(item, dict) and item.get("task_id")
    }
    new_tasks = {
        str(item.get("task_id"))
        for item in current.get("case_tasks") or []
        if isinstance(item, dict) and item.get("task_id")
    }
    likelihood_change = _likelihood_change(previous_tier, current_tier)
    reasons = []
    if likelihood_change == "upgraded":
        reasons.append("当前事实、证据覆盖或程序可行性相较上一版改善")
    elif likelihood_change == "downgraded":
        reasons.append("当前出现新的不利事实、证据缺口、冲突或程序风险")
    elif likelihood_change == "unchanged":
        reasons.append("影响定性等级的核心条件没有发生变化")
    else:
        reasons.append("首次形成可比较的定性判断")
    return {
        "likelihood_change": likelihood_change,
        "likelihood_change_reasons": reasons,
        "changed_sections": changed_sections,
        "added_task_ids": sorted(new_tasks - old_tasks),
        "retained_task_ids": sorted(new_tasks & old_tasks),
        "removed_task_ids": sorted(old_tasks - new_tasks),
    }


def _render_basis(ref: dict[str, Any]) -> str:
    title = _compact(
        ref.get("title") or ref.get("name") or ref.get("law_name"),
        180,
    ) or "已检索依据"
    locator = _compact(ref.get("locator") or ref.get("article_no"), 120)
    url = _compact(
        ref.get("source_url") or ref.get("official_url") or ref.get("url"),
        360,
    )
    label = f"[{title}]({url})" if url else title
    return f"{label}{f'，{locator}' if locator else ''}"


def render_solution_markdown(solution: dict[str, Any]) -> str:
    core = solution.get("core_judgment") or {}
    likelihood = solution.get("likelihood_assessment") or {}
    dimensions = list(likelihood.get("dimensions") or [])
    evidence = solution.get("evidence_effect_summary") or {}
    facts = list(solution.get("confirmed_facts") or [])
    refs = list(solution.get("action_basis_refs") or [])
    routes = list(solution.get("recommended_routes") or [])
    alternatives = list(solution.get("alternative_routes") or [])
    actions = list(solution.get("immediate_actions") or [])
    tasks = list(solution.get("case_tasks") or [])
    documents = list(solution.get("document_suggestions") or [])
    change = solution.get("change_summary") or {}
    formal_version = int(solution.get("plan_version") or 0)
    version_label = (
        "正式方案版本" if formal_version else "方案候选版本"
    )
    version_value = (
        f"第 {formal_version} 版"
        if formal_version
        else solution.get("plan_version_candidate")
    )
    lines = [
        "## 核心判断",
        "",
        f"- **当前方向：** {core.get('summary') or '按当前事实形成条件式行动方向。'}",
        f"- **方案性质：** {'条件式方案' if solution.get('conditional_plan') else '当前版本行动方案'}",
        f"- **关键条件：** {core.get('primary_condition') or '仍需结合事实、证据和程序核对'}",
        f"- **边界：** {core.get('disclaimer') or ''}",
        "",
        "## 已确认事实",
        "",
    ]
    lines.extend(
        f"- {item.get('statement') or item.get('value')}"
        for item in facts[:14]
        if item.get("statement") or item.get("value")
    )
    if not facts:
        lines.append("- 当前没有可作为稳定前提的事实；以下仅保留低风险行动。")
    lines.extend(["", "## 法律依据与适用条件", ""])
    lines.extend(f"- {_render_basis(item)}" for item in refs[:8])
    if not refs:
        lines.append("- 当前没有可精确定位的具体依据，不确定的机构、期限、费用和管辖需要通过官方渠道核对。")
    lines.extend(["", "## 证据检验结果", ""])
    counts = evidence.get("coverage_counts") or {}
    if counts:
        for status, count in counts.items():
            lines.append(
                f"- **{_COVERAGE_LABELS.get(str(status), str(status))}：** {count} 项证明目标"
            )
    else:
        lines.append("- 尚未完成材料评估；未提交不等于没有，当前方案按证据缺口理解。")
    lines.append(f"- {evidence.get('disclaimer') or ''}")
    lines.extend(["", "## 证据缺口与替代材料", ""])
    gaps = list(evidence.get("critical_gaps") or [])
    for item in gaps[:8]:
        lines.append(
            f"- **{item.get('label') or '证明目标'}**："
            f"{_COVERAGE_LABELS.get(str(item.get('status')), '存在缺口')}。"
            f"{item.get('next_action') or '可补交材料、说明无法提供，或记录合法调取渠道。'}"
        )
    if not gaps:
        lines.append("- 当前没有新增关键缺口；仍应保留原始载体并准备提交前核对。")
    lines.extend(["", "## 有利、不利和不确定因素", ""])
    lines.append("### 有利因素")
    lines.extend(
        f"- {item}" for item in likelihood.get("positive_factors") or []
    )
    if not likelihood.get("positive_factors"):
        lines.append("- 当前尚无足以单列的有利条件。")
    lines.extend(["", "### 不利因素"])
    lines.extend(
        f"- {item}" for item in likelihood.get("negative_factors") or []
    )
    if not likelihood.get("negative_factors"):
        lines.append("- 当前未识别到已有依据支持的明显不利因素。")
    lines.extend(["", "### 不确定因素"])
    lines.extend(
        f"- {item}" for item in likelihood.get("unknown_factors") or []
    )
    if not likelihood.get("unknown_factors"):
        lines.append("- 当前没有新增的关键未知项。")
    lines.extend(
        [
            "",
            "## 当前维权可能性",
            "",
            f"> **{likelihood.get('tier') or '不确定'}**",
            "",
        ]
    )
    for item in dimensions:
        lines.append(
            f"- **{item.get('label')}：** "
            f"{item.get('status_label') or _DIMENSION_STATUS_LABELS.get(str(item.get('status')), '暂不确定')}"
        )
    lines.extend(
        [
            "",
            "> 这是定性判断，不代表具体概率或结果保证。",
            "",
            "## 推荐行动方案",
            "",
        ]
    )
    for index, action in enumerate(actions, start=1):
        lines.append(
            f"{index}. **{action.get('title')}**：{action.get('reason')}"
        )
    if routes:
        lines.extend(["", "### 推荐路径", ""])
    for route in routes:
        lines.extend(
            [
                f"- **{route.get('label')}**",
                f"  - 为什么：{route.get('rationale')}",
                f"  - 第一步：{route.get('first_action')}",
                f"  - 升级条件：{route.get('escalation_condition')}",
                f"  - 停止条件：{route.get('stop_condition')}",
            ]
        )
    lines.extend(["", "## 替代与升级路径", ""])
    for route in alternatives:
        lines.extend(
            [
                f"- **{route.get('label')}**：{route.get('rationale')}",
                f"  - 第一步：{route.get('first_action')}",
                f"  - 升级条件：{route.get('escalation_condition')}",
            ]
        )
    if not alternatives:
        lines.append("- 当前没有依据支持的其他路径，不为完整性罗列无关程序。")
    lines.extend(["", "## 下一步任务清单", ""])
    for task in tasks[:10]:
        status = str(task.get("status") or "pending")
        marker = "x" if status == "completed" else " "
        lines.append(
            f"- [{marker}] **{task.get('title')}**"
            f"（{task.get('priority') or 'medium'}）"
        )
        if task.get("completion_criteria"):
            lines.append(
                f"  - 完成标准：{task.get('completion_criteria')}"
            )
    lines.extend(["", "## 参考文书", ""])
    for item in documents:
        lines.append(
            f"- **{item.get('document_type')}**：{item.get('reason')}"
        )
        if item.get("missing_fields"):
            lines.append(
                "  - 待补充或核对："
                + "、".join(str(value) for value in item["missing_fields"][:6])
            )
        lines.append(f"  - {item.get('placeholder_policy')}")
    lines.extend(["", "## 版本变化与限制", ""])
    lines.extend(
        [
            f"- **{version_label}：** {version_value}",
            f"- **可能性变化：** {change.get('likelihood_change') or 'newly_assessable'}",
            f"- **变化范围：** {'、'.join(change.get('changed_sections') or []) or '首次生成'}",
        ]
    )
    for limitation in likelihood.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines).strip()


def _invalid_result(state: Any, validation: dict[str, Any]) -> dict[str, Any]:
    next_route = str(validation.get("next_route") or "decide_facts")
    return {
        "solution_draft_status": "blocked_by_upstream",
        "pending_solution_audit": False,
        "solution_input_validation": validation,
        "next_route": next_route,
        "decision_status": "solution_blocked_by_upstream",
        "workflow_stage": (
            "fact_clarification"
            if next_route in {"update_facts", "decide_facts"}
            else "evidence_collection"
            if next_route in {"plan_evidence", "assess_evidence"}
            else "guard_pause"
        ),
        "pause_state": getattr(state, "pause_state", None),
        "phase": (
            GuidePhase.DETAIL_GATHER
            if next_route in {"plan_evidence", "assess_evidence"}
            else GuidePhase.ISSUE_SEARCH
        ),
        "messages": [
            AIMessage(
                content=(
                    "## 行动方案暂未生成\n\n"
                    f"{validation.get('message') or '上游案件版本尚未准备完成。'}"
                )
            )
        ],
    }


async def run_generate_solution(
    state: Any,
    deps: Any = None,
) -> dict[str, Any]:
    """Generate a structured draft for node eight or the legacy presenter."""

    del deps  # Node seven is deterministic; external retrieval is upstream.
    validation = validate_solution_inputs(state)
    if not validation.get("valid"):
        return _invalid_result(state, validation)

    basis_refs, basis_gaps = load_reusable_action_basis(state)
    dimensions = build_likelihood_dimensions(
        state,
        action_basis_refs=basis_refs,
        action_basis_gaps=basis_gaps,
    )
    likelihood = derive_qualitative_likelihood(
        dimensions,
        basis_gaps=basis_gaps,
    )
    evidence_summary = build_evidence_effect_summary(state)
    conditional = bool(
        validation.get("conditional")
        or evidence_summary.get("critical_gaps")
        or basis_gaps
        or likelihood.get("tier") != "较有利"
    )
    immediate_actions = build_immediate_actions(
        state,
        evidence_summary,
        action_basis_refs=basis_refs,
    )
    recommended_routes, alternative_routes = build_action_routes(
        state,
        action_basis_refs=basis_refs,
    )
    tasks = build_case_tasks(
        state,
        immediate_actions,
        [*recommended_routes, *alternative_routes],
    )
    documents = build_document_suggestions(
        state,
        action_basis_refs=basis_refs,
    )
    core = _core_judgment(state, likelihood, conditional)
    confirmed_facts = [
        {
            "fact_id": _fact_id(item),
            "statement": _fact_text(item),
            "status": item.get("status"),
        }
        for item in _confirmed_facts(state)
    ]

    draft_core = {
        "schema_version": SOLUTION_SCHEMA_VERSION,
        "case_id": str(getattr(state, "case_id", "")),
        "case_generation": int(
            getattr(state, "case_generation", 1) or 1
        ),
        "based_on_fact_snapshot_version":
            validation["fact_snapshot_version"],
        "based_on_fact_snapshot_hash":
            validation.get("fact_snapshot_hash") or "",
        "based_on_legal_model_version":
            validation["legal_model_version"],
        "based_on_evidence_plan_version":
            validation["evidence_plan_version"],
        "based_on_evidence_review_version":
            validation["evidence_review_version"],
        "previous_plan_version": int(
            getattr(state, "plan_version", 0) or 0
        ),
        "core_judgment": core,
        "confirmed_facts": confirmed_facts,
        "likelihood_assessment": likelihood,
        "evidence_effect_summary": evidence_summary,
        "recommended_routes": recommended_routes,
        "alternative_routes": alternative_routes,
        "immediate_actions": immediate_actions,
        "case_tasks": tasks,
        "document_suggestions": documents,
        "document_drafts": [],
        "action_basis_refs": basis_refs,
        "action_basis_gaps": basis_gaps,
        "action_retrieval_level": "reuse",
        "conditional_plan": conditional,
        "next_route": "audit_and_save",
    }
    fingerprint = _hash_payload(draft_core)
    previous_fingerprint = str(
        getattr(state, "solution_draft_fingerprint", "") or ""
    )
    previous_candidate = str(
        getattr(state, "plan_version_candidate", "") or ""
    )
    if fingerprint == previous_fingerprint and previous_candidate:
        candidate = previous_candidate
    else:
        candidate = (
            f"plan-draft:{state.case_id}:"
            f"{int(getattr(state, 'plan_version', 0) or 0) + 1}:"
            f"{fingerprint[:10]}"
        )
    for task in tasks:
        task["plan_version_candidate"] = candidate
    draft_core["plan_version_candidate"] = candidate
    draft_core["case_tasks"] = tasks
    change_summary = build_solution_change_summary(
        getattr(state, "solution_draft", {}) or {},
        draft_core,
    )
    draft_core["change_summary"] = change_summary
    generation_trace_id = _stable_id(
        "solution-generation",
        state.case_id,
        candidate,
        fingerprint,
    )
    draft_core["generation_trace_id"] = generation_trace_id
    draft_core["generated_at"] = _now()
    markdown = render_solution_markdown(draft_core)
    draft_core["draft_markdown"] = markdown

    return {
        "solution_draft": draft_core,
        "solution_draft_markdown": markdown,
        "solution_draft_status": "awaiting_audit",
        "solution_draft_fingerprint": fingerprint,
        "solution_generation_id": generation_trace_id,
        "solution_generated_at": draft_core["generated_at"],
        "solution_input_validation": validation,
        "plan_version_candidate": candidate,
        "solution_based_on_fact_snapshot_version":
            validation["fact_snapshot_version"],
        "solution_based_on_legal_model_version":
            validation["legal_model_version"],
        "solution_based_on_evidence_plan_version":
            validation["evidence_plan_version"],
        "solution_based_on_evidence_review_version":
            validation["evidence_review_version"],
        "likelihood_assessment": likelihood,
        "likelihood_tier": likelihood["tier"],
        "likelihood_change": change_summary["likelihood_change"],
        "solution_change_summary": change_summary,
        "recommended_routes": recommended_routes,
        "alternative_routes": alternative_routes,
        "immediate_actions": immediate_actions,
        "case_tasks": tasks,
        "document_suggestions": documents,
        "action_basis_refs": basis_refs,
        "action_basis_gaps": basis_gaps,
        "conditional_plan": conditional,
        "pending_solution_audit": True,
        "next_route": "audit_and_save",
        "decision_status": "solution_draft_ready",
        "workflow_stage": "solution_drafting",
        "pause_state": None,
        "phase": GuidePhase.CONCLUDE,
    }


__all__ = [
    "SOLUTION_SCHEMA_VERSION",
    "LIKELIHOOD_TIERS",
    "validate_solution_inputs",
    "load_reusable_action_basis",
    "build_likelihood_dimensions",
    "derive_qualitative_likelihood",
    "build_evidence_effect_summary",
    "build_immediate_actions",
    "build_action_routes",
    "build_case_tasks",
    "build_document_suggestions",
    "build_solution_change_summary",
    "render_solution_markdown",
    "run_generate_solution",
]
