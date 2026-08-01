"""Formal node five: legal modelling and evidence-plan construction.

The node is intentionally separate from both fact collection and material
assessment.  It consumes a confirmed (or explicitly conditional) fact
snapshot, builds an auditable legal/proof model, and opens stable evidence
delivery entries.  It never treats a user claim or an uploaded file as
authenticated evidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage
from loguru import logger

from src.agents.legal_guide.authority_registry import build_source_snapshots
from src.agents.legal_guide.case_model import active_case_facts
from src.agents.legal_guide.evidence_rules import resolve_state_evidence_checklist
from src.agents.legal_guide.followup_catalog import evidence_followups
from src.agents.legal_guide.retrieval_query import build_case_retrieval_inputs
from src.agents.legal_guide.state import GuidePhase


PLAN_SCHEMA_VERSION = 1
PLAN_STATUS_VALUES = {
    "draft",
    "active",
    "conditional",
    "needs_fact_update",
    "stale",
    "retrieval_degraded",
}

_CONFIRMED_STATUSES = {"confirmed", "asserted"}
_UNCERTAIN_STATUSES = {
    "unknown",
    "unclear",
    "conflicted",
    "pending_fact_confirmation",
    "not_provided",
}
_DENIED_STATUSES = {"denied", "superseded"}
_IMPORTANCE_LABELS = {
    "essential": "优先准备",
    "important": "重要补强",
    "reinforcing": "可选补充",
}
_MATERIAL_STATE_LABELS = {
    "submitted": "已上传待评估",
    "user_claimed_present": "用户称持有，尚未提交评估",
    "temporarily_unavailable": "暂时找不到",
    "user_claimed_unavailable": "用户明确表示没有",
    "available_for_third_party_request": "可向第三方调取",
    "not_submitted": "暂未提交",
    "unclassified": "待归类",
}
_RELATION_BY_DOMAIN = {
    "consumer_market": (
        "online_sale_transaction",
        "网络交易买卖关系候选",
        "用户与平台或卖方之间可能存在网络交易买卖关系",
    ),
    "labor_social_security": (
        "employment_relationship",
        "劳动关系候选",
        "用户与用人单位之间可能存在劳动关系或劳动争议关系",
    ),
    "contracts_property_housing": (
        "civil_contract",
        "民事合同关系候选",
        "双方之间可能存在合同、租赁或其他民事交易关系",
    ),
    "traffic_personal_injury": (
        "traffic_tort",
        "交通事故侵权关系候选",
        "事故参与方之间可能存在交通事故侵权或保险理赔关系",
    ),
    "family_vulnerable_groups": (
        "family_relationship",
        "婚姻家庭关系候选",
        "双方之间可能存在婚姻、亲属或家庭成员关系",
    ),
    "administrative_remedies": (
        "administrative_action",
        "行政行为关系候选",
        "用户与行政机关之间可能存在行政处理或行政救济关系",
    ),
    "intellectual_property": (
        "intellectual_property_dispute",
        "知识产权争议关系候选",
        "双方之间可能存在知识产权侵权或许可使用关系",
    ),
    "environment_pollution": (
        "environmental_tort",
        "环境侵权关系候选",
        "用户与污染行为人之间可能存在环境侵权关系",
    ),
    "cyber_data_fraud": (
        "cyber_data_dispute",
        "网络与数据争议关系候选",
        "双方之间可能存在网络交易、数据处理或网络侵害关系",
    ),
    "mediation_notary_arbitration": (
        "alternative_dispute_process",
        "调解、公证或仲裁程序关系候选",
        "当前案件可能进入调解、公证或仲裁等程序",
    ),
}

_REQUEST_ALIASES = {
    "refund": ("refund", "退款", "返还", "退钱"),
    "compensation": ("compensation", "赔偿", "补偿", "经济补偿"),
    "continue_performance": ("continue_performance", "继续履行", "发货", "交付"),
    "termination": ("termination", "解除", "终止合同", "退租"),
    "complaint": ("complaint", "投诉", "举报", "平台处理"),
    "arbitration": ("arbitration", "仲裁"),
    "litigation": ("litigation", "起诉", "诉讼"),
    "correction": ("correction", "更正", "删除", "停止侵害"),
    "preservation": ("preservation", "保全", "证据保全"),
}

_REQUIREMENT_ALIASES = {
    "payment_record": ("payment", "付款", "支付", "转账", "银行流水", "账单"),
    "order_record": ("order", "订单", "商品页面", "交易记录"),
    "chat_record": ("chat", "聊天", "沟通", "对话", "短信", "邮件"),
    "delivery_record": ("delivery", "发货", "物流", "交付", "履行"),
    "complaint_record": ("complaint", "投诉", "客服", "工单", "平台"),
    "contract_record": ("contract", "合同", "协议", "约定"),
    "identity_record": ("identity", "身份", "账号", "主体", "店铺", "单位"),
    "time_record": ("time", "时间", "日期", "期限", "时间线"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(value: Any, limit: int = 480) -> str:
    return " ".join(str(value or "").split())[:limit]


def _unique(values: Iterable[str]) -> list[str]:
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


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fact_rows(state: Any) -> list[dict[str, Any]]:
    rows = getattr(state, "fact_blackboard", None) or getattr(state, "case_facts", None) or []
    return [
        dict(item)
        for item in rows
        if isinstance(item, dict)
        and str(item.get("status") or "") not in _DENIED_STATUSES
    ]


def _fact_key(item: dict[str, Any]) -> str:
    return _compact(item.get("semantic_key") or item.get("key") or item.get("fact_id"))


def _fact_statement(item: dict[str, Any]) -> str:
    return _compact(item.get("statement") or item.get("value") or item.get("source_text"))


def _unknown_conditions(state: Any, rows: Iterable[dict[str, Any]] | None = None) -> list[str]:
    conditions: list[str] = []
    for item in rows or _fact_rows(state):
        if str(item.get("status") or "") in _UNCERTAIN_STATUSES:
            key = _fact_key(item)
            statement = _fact_statement(item)
            if key or statement:
                conditions.append(f"{key}: {statement}".strip(": "))
    conditions.extend(
        _compact(value)
        for value in (getattr(state, "risk_related_missing_facts", []) or [])
        if value
    )
    return _unique(conditions)


def validate_fact_snapshot(state: Any) -> dict[str, Any]:
    """Validate the snapshot boundary without mutating state."""

    draft = dict(getattr(state, "fact_snapshot_draft", None) or {})
    current_blackboard_version = int(getattr(state, "fact_blackboard_version", 0) or 0)
    draft_version = int(
        draft.get("based_on_fact_blackboard_version")
        or draft.get("fact_blackboard_version")
        or 0
    )
    explicit = bool(
        getattr(state, "fact_snapshot_confirmed", False)
        or getattr(state, "proceed_under_uncertainty", False)
        or getattr(state, "wants_conclude", False)
        or getattr(state, "input_event_type", "") == "fact_snapshot_confirmed"
        or (
            getattr(state, "requested_route", "") == "plan_evidence"
            and bool(draft)
        )
    )
    if not draft and not getattr(state, "fact_snapshot_confirmed", False):
        return {
            "valid": False,
            "status": "needs_fact_update",
            "reason": "fact_snapshot_missing",
            "message": "当前还没有可用于证据规划的事实快照。",
            "fact_snapshot_version": 0,
            "fact_snapshot_hash": "",
        }
    if draft and draft_version != current_blackboard_version:
        return {
            "valid": False,
            "status": "stale",
            "reason": "fact_snapshot_stale",
            "message": "事实快照已经过期，需要回到事实确认阶段重新确认。",
            "fact_snapshot_version": draft_version,
            "fact_snapshot_hash": str(draft.get("snapshot_hash") or ""),
        }
    if draft.get("stale"):
        return {
            "valid": False,
            "status": "stale",
            "reason": "fact_snapshot_marked_stale",
            "message": "事实快照被标记为过期，不能直接生成证据清单。",
            "fact_snapshot_version": draft_version,
            "fact_snapshot_hash": str(draft.get("snapshot_hash") or ""),
        }
    if not explicit:
        return {
            "valid": False,
            "status": "needs_fact_update",
            "reason": "fact_snapshot_not_confirmed",
            "message": "请先确认事实快照，或选择按当前信息继续。",
            "fact_snapshot_version": draft_version,
            "fact_snapshot_hash": str(draft.get("snapshot_hash") or ""),
        }
    return {
        "valid": True,
        "status": "conditional" if getattr(state, "proceed_under_uncertainty", False) else "active",
        "reason": "snapshot_confirmed",
        "message": "",
        "fact_snapshot_version": int(
            getattr(state, "fact_snapshot_version", 0) or draft_version or current_blackboard_version or 1
        ),
        "fact_snapshot_hash": str(draft.get("snapshot_hash") or ""),
        "fact_blackboard_version": current_blackboard_version,
    }


def build_legal_model_input(state: Any) -> dict[str, Any]:
    """Build retrieval input from confirmed facts and isolate unknowns."""

    rows = _fact_rows(state)
    confirmed = [_fact_statement(item) for item in rows if item.get("status") in _CONFIRMED_STATUSES]
    issues = _unique(
        [
            *(getattr(state, "confirmed_issues", []) or []),
            *(getattr(state, "issue_candidates", []) or []),
            *(getattr(state, "unmatched_issues", []) or []),
        ]
    )
    retrieval = build_case_retrieval_inputs(
        issues,
        [item for item in rows if item.get("status") in _CONFIRMED_STATUSES],
    )
    procedure_text = " ".join(
        _fact_statement(item)
        for item in rows
        if str(item.get("category") or "") in {"procedure", "claim"}
    )
    relation_candidates = build_relation_candidates(state, rows=rows)
    requests = build_request_models(state, rows=rows)
    return {
        "case_id": str(getattr(state, "case_id", "")),
        "legal_domain": str(getattr(state, "legal_domain", "") or getattr(state, "domain_candidate", "") or "other"),
        "relation_candidates": [item["relation_id"] for item in relation_candidates],
        "requests": [item["request_type"] for item in requests],
        "confirmed_fact_summary": confirmed[:24],
        "unknown_conditions": _unknown_conditions(state, rows),
        "region": str(getattr(state, "region", "") or "未知"),
        "procedure_context": procedure_text or "当前尚未确定程序阶段",
        "as_of_date": datetime.now(timezone.utc).date().isoformat(),
        "sparse_query": str(retrieval.get("sparse_query") or ""),
        "semantic_phrases": list(retrieval.get("semantic_phrases") or [])[:24],
        "lexical_phrases": list(retrieval.get("lexical_phrases") or [])[:24],
    }


def build_relation_candidates(
    state: Any,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_list = list(rows or _fact_rows(state))
    domain = str(getattr(state, "legal_domain", "") or getattr(state, "domain_candidate", "") or "other")
    relation_id, label, description = _RELATION_BY_DOMAIN.get(
        domain,
        ("general_civil_dispute", "一般民事争议关系候选", "当前案件可能属于一般民事争议关系"),
    )
    refs = [_fact_key(item) for item in rows_list if _fact_key(item)]
    statements = " ".join(_fact_statement(item) for item in rows_list)
    parties: list[dict[str, Any]] = []
    party_terms = (
        ("buyer", "买方", ("买", "付款人", "用户")),
        ("seller", "卖方或相对方", ("卖家", "商家", "收款方", "对方")),
        ("employer", "用人单位", ("公司", "单位", "雇主")),
        ("employee", "劳动者", ("员工", "本人", "劳动者")),
        ("platform", "平台", ("平台", "闲鱼", "客服")),
    )
    for role, role_label, markers in party_terms:
        if any(marker in statements for marker in markers):
            party_refs = [
                _fact_key(item)
                for item in rows_list
                if any(marker in _fact_statement(item) for marker in markers)
            ]
            parties.append({"role": role, "label": role_label, "fact_refs": _unique(party_refs)})
    missing_conditions: list[str] = []
    if not refs:
        missing_conditions.append("case_facts")
    if not getattr(state, "legal_domain", "") and not getattr(state, "domain_candidate", ""):
        missing_conditions.append("legal_domain")
    has_claim_fact = any(
        str(item.get("category") or "") == "claim"
        or str(item.get("semantic_key") or "").startswith("claim.")
        for item in rows_list
    )
    if (
        not getattr(state, "confirmed_issues", [])
        and not getattr(state, "issue_candidates", [])
        and not has_claim_fact
    ):
        missing_conditions.append("issue_or_request")
    return [
        {
            "relation_id": relation_id,
            "label": label,
            "description": description,
            "parties": parties,
            "supporting_fact_keys": _unique(refs),
            "missing_conditions": missing_conditions,
            "confidence_tier": "candidate",
            "basis_refs": [],
        }
    ]


def _request_type_from_text(text: str) -> str:
    compact = str(text or "").lower()
    for request_type, markers in _REQUEST_ALIASES.items():
        if any(marker.lower() in compact for marker in markers):
            return request_type
    return ""


def _amount_from_text(text: str) -> str:
    match = re.search(r"(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*(?:元|人民币|块)?", str(text or ""))
    return match.group(1).replace(",", "") if match else ""


def build_request_models(
    state: Any,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_list = list(rows or _fact_rows(state))
    candidates: list[tuple[str, list[str], list[str]]] = []
    for item in rows_list:
        statement = _fact_statement(item)
        request_type = _request_type_from_text(
            f"{item.get('semantic_key', '')} {statement}"
        )
        if request_type:
            candidates.append((request_type, [_fact_key(item)], [statement]))
    for issue in (
        *(getattr(state, "confirmed_issues", []) or []),
        *(getattr(state, "unmatched_issues", []) or []),
    ):
        request_type = _request_type_from_text(issue)
        if request_type:
            candidates.append((request_type, [], [str(issue)]))
    if not candidates:
        # Keep an explicit generic request so the legal model remains
        # inspectable when the user has only described the dispute.
        candidates = [("general_remedy", [], [])]

    merged: dict[str, dict[str, Any]] = {}
    all_text = " ".join(
        [_fact_statement(item) for item in rows_list]
        + list(getattr(state, "confirmed_issues", []) or [])
        + list(getattr(state, "unmatched_issues", []) or [])
    )
    labels = {
        "refund": ("退款或返还款项", "解除交易并返还已付款项"),
        "compensation": ("赔偿或补偿", "请求赔偿实际损失或依法计算的补偿"),
        "continue_performance": ("继续履行", "请求对方继续交付或履行约定"),
        "termination": ("解除或终止关系", "请求解除交易、合同或其他持续性关系"),
        "complaint": ("投诉或平台处理", "请求平台或主管机关处理争议"),
        "arbitration": ("仲裁", "准备进入仲裁程序"),
        "litigation": ("诉讼", "准备通过诉讼解决争议"),
        "correction": ("停止侵害或更正", "请求停止侵害、删除、更正或恢复相关权益"),
        "preservation": ("证据或财产保全", "请求采取必要的保全措施"),
        "general_remedy": ("一般救济请求", "根据事实和适用程序确定具体请求"),
    }
    for request_type, fact_refs, source_texts in candidates:
        record = merged.setdefault(
            request_type,
            {
                "request_id": f"request.{request_type}",
                "request_type": request_type,
                "label": labels.get(request_type, (request_type, request_type))[0],
                "requested_action": labels.get(request_type, (request_type, request_type))[1],
                "request_scope": "",
                "requested_amount": "",
                "supporting_fact_keys": [],
                "unknown_conditions": [],
            },
        )
        record["supporting_fact_keys"] = _unique(
            [*record["supporting_fact_keys"], *fact_refs]
        )
        if source_texts:
            record["request_scope"] = _compact(
                "；".join([record["request_scope"], *source_texts]).strip("；"),
                260,
            )
    amount = _amount_from_text(all_text)
    for record in merged.values():
        if amount:
            record["requested_amount"] = f"{amount} CNY"
        record["unknown_conditions"] = _unknown_conditions(state, rows_list)
    return list(merged.values())


def build_legal_model(
    state: Any,
    *,
    relation_candidates: list[dict[str, Any]] | None = None,
    request_models: list[dict[str, Any]] | None = None,
    basis_refs: list[dict[str, Any]] | None = None,
    retrieval_trace: dict[str, Any] | None = None,
    basis_limitations: list[str] | None = None,
) -> dict[str, Any]:
    relations = relation_candidates or build_relation_candidates(state)
    requests = request_models or build_request_models(state)
    unknown = _unknown_conditions(state)
    missing = _unique(
        [
            condition
            for relation in relations
            for condition in relation.get("missing_conditions", [])
        ]
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "legal_domain": str(getattr(state, "legal_domain", "") or "other"),
        "relation_candidates": relations,
        "request_models": requests,
        "procedure_candidates": _unique(
            [
                str(item.get("statement") or "")
                for item in _fact_rows(state)
                if str(item.get("category") or "") == "procedure"
            ]
        ),
        "limitation_conditions": unknown,
        "jurisdiction_conditions": [str(getattr(state, "region", "") or "未知")],
        "proof_target_ids": [],
        "evidence_basis_refs": basis_refs or [],
        "unknown_conditions": unknown,
        "blocking_fact_keys": missing,
        "retrieval_trace_id": str((retrieval_trace or {}).get("retrieval_trace_id") or ""),
        "basis_limitations": _unique(basis_limitations or []),
        "created_at": _now(),
    }


def _request_for_requirement(requirement_id: str, requests: list[dict[str, Any]]) -> str:
    text = requirement_id.lower()
    preferred = (
        "refund" if any(token in text for token in ("payment", "transaction", "delivery", "platform")) else ""
    )
    if preferred and any(item.get("request_type") == preferred for item in requests):
        return f"request.{preferred}"
    return str((requests[0] if requests else {}).get("request_id") or "request.general_remedy")


def _importance_for_requirement(requirement_id: str) -> str:
    if any(token in requirement_id for token in ("payment", "relationship", "agreement", "non_performance", "event")):
        return "essential"
    if any(token in requirement_id for token in ("complaint", "timeline", "request", "harm", "loss")):
        return "important"
    return "reinforcing"


def _burden_note() -> str:
    return (
        "通常需要准备能够支持该事实的材料；建议保留由用户控制的原始记录。"
        "具体证明责任可能受程序和对方抗辩影响。"
    )


def build_proof_targets(
    state: Any,
    *,
    legal_model: dict[str, Any] | None = None,
    basis_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    model = legal_model or build_legal_model(state)
    requests = list(model.get("request_models") or build_request_models(state))
    basis = list(basis_refs if basis_refs is not None else model.get("evidence_basis_refs") or [])
    internal = [
        dict(item)
        for item in (getattr(state, "internal_evidence_requirements", []) or [])
        if isinstance(item, dict)
        and str(item.get("status") or "") not in {"superseded", "not_applicable"}
    ]
    targets: list[dict[str, Any]] = []
    for item in internal:
        requirement_id = _compact(item.get("requirement_id"), 120)
        if not requirement_id:
            continue
        target_id = _compact(
            item.get("proof_target_id") or f"proof.{requirement_id}",
            160,
        )
        status = "active"
        if str(item.get("status") or "") == "pending_fact_confirmation":
            status = "pending_fact_confirmation"
        targets.append(
            {
                "id": target_id,
                "proof_target_id": target_id,
                "rule_id": requirement_id,
                "evidence_key": requirement_id,
                "request_id": _request_for_requirement(requirement_id, requests),
                "relation_id": str(
                    (model.get("relation_candidates") or [{}])[0].get("relation_id")
                    or "general_civil_dispute"
                ),
                "label": _compact(item.get("label") or requirement_id, 180),
                "proposition": _compact(
                    item.get("purpose") or item.get("label") or requirement_id,
                    260,
                ),
                "purpose": _compact(
                    item.get("purpose") or item.get("label") or requirement_id,
                    260,
                ),
                "dependent_fact_keys": _unique(item.get("dependent_fact_keys") or []),
                "proof_roles": [],
                "importance": _importance_for_requirement(requirement_id),
                "status": status,
                "legal_condition": _compact(
                    item.get("purpose") or "支持当前请求所需的事实条件",
                    240,
                ),
                "procedure_relevance": "需结合具体受理程序核对",
                "burden_note": _burden_note(),
                "basis_refs": [dict(ref) for ref in basis],
                "limitations": [
                    "证明目标是系统规划对象，不代表材料已经证明该事实。",
                ],
                "unknown_conditions": _unknown_conditions(state),
            }
        )
    if not targets:
        # A domain catalogue is a conservative fallback when node four did not
        # produce a candidate yet.  It still creates planning objects, not
        # verified evidence conclusions.
        for rule in evidence_followups(str(getattr(state, "legal_domain", "") or "other")):
            requirement_id = _compact(rule.evidence_key or rule.id, 120)
            target_id = f"proof.{requirement_id}"
            targets.append(
                {
                    "id": target_id,
                    "proof_target_id": target_id,
                    "rule_id": rule.id,
                    "evidence_key": requirement_id,
                    "request_id": _request_for_requirement(requirement_id, requests),
                    "relation_id": str(
                        (model.get("relation_candidates") or [{}])[0].get("relation_id")
                        or "general_civil_dispute"
                    ),
                    "label": _compact(rule.item, 180),
                    "proposition": _compact(rule.purpose, 260),
                    "purpose": _compact(rule.purpose, 260),
                    "dependent_fact_keys": [],
                    "proof_roles": [],
                    "importance": _importance_for_requirement(requirement_id),
                    "status": "active",
                    "legal_condition": _compact(rule.purpose, 240),
                    "procedure_relevance": "需结合具体受理程序核对",
                    "burden_note": _burden_note(),
                    "basis_refs": [dict(ref) for ref in basis],
                    "limitations": [
                        "当前使用领域材料规则作为保守规划参考，不是机关固定清单。",
                    ],
                    "unknown_conditions": _unknown_conditions(state),
                    "recommended_materials": list(rule.alternatives or []),
                    "alternative_materials": list(rule.alternatives or []),
                }
            )
    return targets


def _material_classes(item: dict[str, Any]) -> list[str]:
    return _unique(
        [
            *(item.get("recommended_material_classes") or []),
            *(item.get("alternative_material_classes") or []),
            *(item.get("recommended_materials") or []),
            *(item.get("alternative_materials") or []),
        ]
    )


def _inventory_text(item: dict[str, Any]) -> str:
    return " ".join(
        _compact(item.get(field))
        for field in ("display_name", "normalized_name", "original_names")
        if item.get(field)
    ).lower()


def _matches_requirement(item: dict[str, Any], requirement: dict[str, Any]) -> bool:
    haystack = _inventory_text(item)
    if not haystack:
        return False
    classes = [value.lower() for value in _material_classes(requirement)]
    if any(value and (value in haystack or haystack in value) for value in classes):
        return True
    for alias, markers in _REQUIREMENT_ALIASES.items():
        if alias in str(requirement.get("requirement_id") or "").lower():
            if any(marker.lower() in haystack for marker in markers):
                return True
    return False


def _material_ids_for_inventory(item: dict[str, Any], state: Any) -> list[str]:
    ids: list[str] = []
    for ref in item.get("source_refs") or []:
        if isinstance(ref, dict):
            ids.extend(
                str(ref.get(key) or "")
                for key in ("material_id", "document_id", "attachment_id")
                if ref.get(key)
            )
    haystack = _inventory_text(item)
    for observation in getattr(state, "material_fact_observations", []) or []:
        if not isinstance(observation, dict):
            continue
        name = _compact(
            (observation.get("normalized_value") or {}).get("file_name")
            if isinstance(observation.get("normalized_value"), dict)
            else ""
        ).lower()
        if name and (name in haystack or haystack in name):
            if observation.get("material_id"):
                ids.append(str(observation["material_id"]))
    return _unique(ids)


def link_evidence_name_inventory(
    requirements: Iterable[dict[str, Any]],
    inventory: Iterable[dict[str, Any]],
    state: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Link names/materials without upgrading their evidentiary status."""

    requirements_list = [dict(item) for item in requirements if isinstance(item, dict)]
    inventory_list = [dict(item) for item in inventory if isinstance(item, dict)]
    links: list[dict[str, Any]] = []
    for name in inventory_list:
        matched = [
            str(req.get("requirement_id") or "")
            for req in requirements_list
            if _matches_requirement(name, req)
        ]
        links.append(
            {
                "evidence_name_id": str(name.get("evidence_name_id") or ""),
                "display_name": _compact(name.get("display_name") or name.get("normalized_name"), 180),
                "requirement_ids": _unique(matched),
                "status": str(name.get("status") or "unknown"),
                "material_ids": _material_ids_for_inventory(name, state) if state is not None else [],
                "link_status": "linked" if matched else "unclassified",
            }
        )
    for requirement in requirements_list:
        matched_names = [
            name
            for name in inventory_list
            if _matches_requirement(name, requirement)
        ]
        matched_ids = _unique(
            str(name.get("evidence_name_id") or "")
            for name in matched_names
            if name.get("evidence_name_id")
        )
        submitted_ids = _unique(
            material_id
            for name in matched_names
            if str(name.get("status") or "") == "submitted"
            for material_id in _material_ids_for_inventory(name, state) if state is not None
        )
        if state is not None and not submitted_ids:
            submitted_ids = _unique(
                str(item.get("material_id") or item.get("document_id") or "")
                for item in getattr(state, "material_fact_observations", []) or []
                if isinstance(item, dict)
                and any(
                    _matches_requirement(
                        {
                            "display_name": (
                                item.get("normalized_value") or {}
                            ).get("file_name")
                            if isinstance(item.get("normalized_value"), dict)
                            else item.get("source_locator"),
                        },
                        requirement,
                    )
                    for _ in [0]
                )
            )
        requirement["matched_evidence_name_ids"] = matched_ids
        requirement["submitted_material_ids"] = submitted_ids
        states = [str(item.get("status") or "") for item in matched_names]
        if "submitted" in states:
            requirement["user_material_state"] = "submitted"
        elif "user_claimed_present" in states:
            requirement["user_material_state"] = "user_claimed_present"
        elif "temporarily_unavailable" in states:
            requirement["user_material_state"] = "temporarily_unavailable"
        elif "explicitly_absent" in states:
            requirement["user_material_state"] = "user_claimed_unavailable"
        else:
            requirement["user_material_state"] = "not_submitted"
    return requirements_list, links


def formalize_evidence_requirements(
    state: Any,
    proof_targets: Iterable[dict[str, Any]],
    *,
    basis_refs: list[dict[str, Any]] | None = None,
    previous: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert temporary node-four needs into stable formal requirements."""

    basis = list(basis_refs or [])
    previous_map = {
        str(item.get("requirement_id")): dict(item)
        for item in (previous or getattr(state, "formal_evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    requirements: dict[str, dict[str, Any]] = {}
    internal_map = {
        str(item.get("requirement_id")): item
        for item in (getattr(state, "internal_evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    requests = list(getattr(state, "request_models", []) or build_request_models(state))
    for target in proof_targets:
        target = dict(target)
        requirement_id = _compact(
            target.get("requirement_id")
            or target.get("evidence_key")
            or target.get("rule_id")
            or target.get("proof_target_id"),
            120,
        )
        if not requirement_id:
            continue
        internal = dict(internal_map.get(requirement_id) or {})
        old = dict(previous_map.get(requirement_id) or {})
        requirement = {
            **old,
            "requirement_id": requirement_id,
            "proof_target_id": _compact(
                target.get("proof_target_id") or target.get("id") or f"proof.{requirement_id}",
                160,
            ),
            "request_id": _compact(
                target.get("request_id")
                or old.get("request_id")
                or _request_for_requirement(requirement_id, requests),
                120,
            ),
            "label": _compact(
                target.get("label")
                or internal.get("label")
                or old.get("label")
                or requirement_id,
                180,
            ),
            "purpose": _compact(
                target.get("purpose")
                or target.get("proposition")
                or internal.get("purpose")
                or old.get("purpose")
                or "支持当前证明目标",
                280,
            ),
            "importance": str(
                target.get("importance")
                or old.get("importance")
                or _importance_for_requirement(requirement_id)
            ),
            "status": (
                "pending_fact_confirmation"
                if target.get("status") == "pending_fact_confirmation"
                else "active"
            ),
            "dependent_fact_keys": _unique(
                [
                    *(old.get("dependent_fact_keys") or []),
                    *(internal.get("dependent_fact_keys") or []),
                    *(target.get("dependent_fact_keys") or []),
                ]
            ),
            "recommended_materials": _unique(
                [
                    *(internal.get("recommended_material_classes") or []),
                    *(target.get("recommended_materials") or []),
                    *(old.get("recommended_materials") or []),
                ]
            ),
            "alternative_materials": _unique(
                [
                    *(internal.get("alternative_material_classes") or []),
                    *(target.get("alternative_materials") or []),
                    *(old.get("alternative_materials") or []),
                ]
            ),
            "submission_modes": [
                "text",
                "image",
                "pdf",
                "docx",
                "native_electronic",
            ],
            "user_material_state": old.get("user_material_state") or "not_submitted",
            "matched_evidence_name_ids": list(old.get("matched_evidence_name_ids") or []),
            "submitted_material_ids": list(old.get("submitted_material_ids") or []),
            "basis_refs": [dict(ref) for ref in (target.get("basis_refs") or basis)],
            "basis_limitations": _unique(
                [
                    *(old.get("basis_limitations") or []),
                    *(target.get("limitations") or []),
                ]
            ),
            "generation_round": int(old.get("generation_round") or getattr(state, "round", 0) or 0),
            "last_updated_round": int(getattr(state, "round", 0) or 0),
            "change_reason": "fact_snapshot_confirmed",
        }
        requirements[requirement_id] = requirement

    # Preserve old requirements for audit.  A later assessment or fact change
    # can mark them stale/superseded; they must not disappear from the case.
    for requirement_id, old in previous_map.items():
        if requirement_id in requirements:
            continue
        retained = dict(old)
        if retained.get("status") not in {"not_applicable", "superseded"}:
            retained["status"] = "stale"
            retained["change_reason"] = "not_in_current_fact_snapshot"
        requirements[requirement_id] = retained
    return list(requirements.values())


def build_delivery_entries(
    requirements: Iterable[dict[str, Any]],
    *,
    case_id: str,
    evidence_plan_version: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        if not requirement_id:
            continue
        entries.append(
            {
                "delivery_entry_id": _stable_id(
                    "delivery-entry", case_id, evidence_plan_version, requirement_id
                ),
                "case_id": case_id,
                "evidence_plan_version": evidence_plan_version,
                "requirement_id": requirement_id,
                "proof_target_id": str(requirement.get("proof_target_id") or ""),
                "accepted_input_modes": list(
                    requirement.get("submission_modes")
                    or ["text", "image", "pdf", "docx", "native_electronic"]
                ),
                "upload_limits": {
                    "max_files": 10,
                    "max_file_size_mb": 50,
                    "allow_later_submission": True,
                },
                "text_schema": {
                    "source_form": "optional",
                    "acquisition_method": "optional",
                    "original_carrier_available": "optional",
                    "formation_time_known": "optional",
                    "identity_visibility": "optional",
                    "completeness_note": "optional",
                    "user_note": "optional",
                },
                "status": "open" if requirement.get("status") not in {"stale", "not_applicable"} else "closed",
            }
        )
    return entries


def validate_plan_citations(
    candidates: Iterable[dict[str, Any]],
    *,
    return_limitations: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[str]]:
    """Keep only citations with explicit source/version/review/locator data."""

    valid: list[dict[str, Any]] = []
    limitations: list[str] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        status = str(row.get("status") or "active").lower()
        version = str(row.get("version") or row.get("version_key") or "").strip()
        review = str(row.get("review_status") or row.get("mapping_status") or "").lower()
        locator = str(row.get("locator") or row.get("article_no") or "").strip()
        source = str(
            row.get("source_url")
            or row.get("official_url")
            or row.get("source_id")
            or row.get("law_id")
            or ""
        ).strip()
        if (
            status == "active"
            and version
            and version.lower() not in {"unknown", "invalid", "outdated"}
            and review in {"approved", "pinpointed", "source_located"}
            and locator
            and source
        ):
            row["citation_status"] = "validated"
            row["locator"] = locator
            valid.append(row)
            continue
        reason = str(row.get("title") or row.get("rule_id") or row.get("law_id") or "一条检索依据")
        if review in {"needs_pinpoint", "pending_legal_review", ""}:
            limitations.append(f"{reason}尚未完成精确条文或法律审校定位")
        elif status != "active":
            limitations.append(f"{reason}不是当前有效来源")
        else:
            limitations.append(f"{reason}缺少可审计的版本、来源或定位信息")
    result = (valid, _unique(limitations))
    return result if return_limitations else result[0]


async def retrieve_plan_authorities(
    state: Any,
    deps: Any = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve law/authority candidates, with explicit degraded fallback."""

    query = dict(query or build_legal_model_input(state))
    query_hash = _hash_payload(query)
    law_refs: list[dict[str, Any]] = [
        dict(item)
        for item in (getattr(state, "retrieved_law_refs", []) or [])
        if isinstance(item, dict)
    ]
    basis_candidates: list[dict[str, Any]] = [
        dict(item)
        for item in (getattr(state, "retrieval_basis_candidates", []) or [])
        if isinstance(item, dict)
    ]
    similar_cases: list[dict[str, Any]] = [
        dict(item)
        for item in (getattr(state, "similar_cases", []) or [])
        if isinstance(item, dict)
    ]
    case_context = str(getattr(state, "case_context_str", "") or "")
    fallback_guide = getattr(state, "fallback_guide", None)
    gaps: list[str] = []
    domain = str(query.get("legal_domain") or "")
    question = "；".join(
        [
            domain,
            "；".join(query.get("relation_candidates") or []),
            "；".join(query.get("requests") or []),
            "；".join(query.get("confirmed_fact_summary") or []),
        ]
    )
    embedding_model = getattr(deps, "embedding_model", None) if deps is not None else None
    milvus_client = getattr(deps, "milvus_client", None) if deps is not None else None
    llm = getattr(deps, "llm", None) if deps is not None else None
    settings = None
    try:
        from src.core.config import get_settings

        settings = get_settings()
    except Exception:
        settings = None

    if embedding_model is not None and milvus_client is not None:
        try:
            from src.agents.legal_knowledge.statute_rag import (
                _fetch_law_titles,
                search_statutes_raw,
            )

            raw_hits = await asyncio.wait_for(
                search_statutes_raw(
                    question=question or "法律证据规划",
                    embedding_model=embedding_model,
                    milvus_client=milvus_client,
                    domain=domain if domain != "other" else "",
                    llm=llm,
                    use_hyde=False,
                    use_rrf=bool(query.get("sparse_query")),
                    sparse_query=str(query.get("sparse_query") or ""),
                    top_k=20,
                    skip_rerank=False,
                ),
                timeout=float(getattr(settings, "GUIDE_RETRIEVE_TIMEOUT_STATUTE", 8.0) if settings else 8.0),
            )
            titles: dict[str, str] = {}
            db_session = getattr(deps, "db_session", None)
            if raw_hits and db_session is not None:
                try:
                    titles = await _fetch_law_titles(raw_hits, db_session)
                except Exception:
                    titles = {}
            law_refs = [
                {
                    "law_id": str(hit.get("law_id") or ""),
                    "title": titles.get(str(hit.get("law_id") or ""), ""),
                    "article_no": str(hit.get("article_no") or ""),
                    "locator": str(hit.get("article_no") or ""),
                    "text": _compact(hit.get("text"), 1800),
                    "domain": str(hit.get("domain") or domain),
                    "score": float(hit.get("score") or 0.0),
                    "status": "active",
                    "version_key": "retrieved-current",
                    "review_status": "needs_pinpoint",
                    "source_id": str(hit.get("law_id") or ""),
                }
                for hit in (raw_hits or [])[:12]
            ]
        except asyncio.TimeoutError:
            gaps.append("statute_retrieval_timeout")
        except Exception as exc:
            logger.warning("节点五法条检索降级 | error={}", exc)
            gaps.append("statute_retrieval_failed")
    elif not law_refs:
        gaps.append("statute_retrieval_unavailable")

    if embedding_model is not None and milvus_client is not None:
        try:
            from src.agents.legal_guide.authority_rag import search_authority_basis_raw

            basis_hits = await search_authority_basis_raw(
                question="；".join(query.get("confirmed_fact_summary") or [])
                or "法律证据规划依据",
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                domain=domain if domain != "other" else "",
                top_k=8,
            )
            basis_candidates = [dict(item) for item in basis_hits or []]
        except Exception:
            gaps.append("authority_basis_retrieval_failed")
    if not basis_candidates:
        gaps.append("authority_basis_unavailable")

    if embedding_model is not None and milvus_client is not None:
        try:
            from src.agents.legal_knowledge.case_rag import search_cases_context

            case_result = await asyncio.wait_for(
                search_cases_context(
                    question=question or "案件事实与请求",
                    embedding_model=embedding_model,
                    milvus_client=milvus_client,
                    db_session=getattr(deps, "db_session", None),
                    domain=domain if domain != "other" else "",
                    sparse_query=str(query.get("sparse_query") or ""),
                    llm=llm,
                    use_hyde=False,
                ),
                timeout=float(
                    getattr(
                        settings,
                        "GUIDE_RETRIEVE_TIMEOUT_CASE",
                        5.0,
                    )
                    if settings
                    else 5.0
                ),
            )
            similar_cases = [
                dict(item)
                for item in (case_result.get("cases") or [])
                if isinstance(item, dict)
            ]
            case_context = str(case_result.get("context") or "")
            fallback_guide = case_result.get("fallback_guide")
            if not similar_cases:
                gaps.append("similar_case_retrieval_empty")
        except asyncio.TimeoutError:
            gaps.append("similar_case_retrieval_timeout")
        except Exception as exc:
            logger.warning("节点五类案检索降级 | error={}", exc)
            gaps.append("similar_case_retrieval_failed")
    elif not similar_cases:
        gaps.append("similar_case_retrieval_unavailable")

    trace = {
        "retrieval_trace_id": _stable_id("plan-retrieval", state.case_id, query_hash),
        "query_hash": f"sha256:{query_hash}",
        "retrieved_ids": _unique(
            [
                str(item.get("law_id") or item.get("id") or "")
                for item in [*law_refs, *basis_candidates, *similar_cases]
            ]
        ),
        "scores": [
            float(item.get("score") or 0.0)
            for item in [*law_refs, *basis_candidates]
            if item.get("score") is not None
        ][:20],
        "rerank_scores": [],
        "source_versions": _unique(
            str(item.get("version") or item.get("version_key") or "")
            for item in [*law_refs, *basis_candidates]
            if item.get("version") or item.get("version_key")
        ),
        "applied_filters": {
            "domain": domain,
            "as_of_date": query.get("as_of_date"),
        },
        "created_at": _now(),
        "retrieval_types": [
            "statute",
            "authority_basis",
            "similar_case",
        ],
    }
    return {
        "query": query,
        "law_refs": law_refs,
        "basis_candidates": basis_candidates,
        "retrieval_trace": trace,
        "retrieval_gaps": _unique(gaps),
        "similar_cases": similar_cases,
        "case_context": case_context,
        "fallback_guide": fallback_guide,
    }


def detect_blocking_fact_gaps(
    state: Any,
    *,
    legal_model: dict[str, Any],
) -> list[str]:
    """Return only gaps that can change the legal/procedure model."""

    gaps = list(legal_model.get("blocking_fact_keys") or [])
    domain = str(getattr(state, "legal_domain", "") or getattr(state, "domain_candidate", "") or "")
    requests = list(legal_model.get("request_models") or [])
    if not domain or domain == "other":
        gaps.append("legal_domain")
    if not requests:
        gaps.append("claim.request")
    return _unique(gaps)


def _change_summary(changes: dict[str, list[str]], *, first: bool) -> str:
    if first:
        return "首次根据确认事实快照建立证据规划"
    parts: list[str] = []
    for label, values in (
        ("新增", changes.get("added_requirement_ids") or []),
        ("更新", changes.get("updated_requirement_ids") or []),
        ("停用", changes.get("deactivated_requirement_ids") or []),
        ("恢复", changes.get("reactivated_requirement_ids") or []),
    ):
        if values:
            parts.append(f"{label}{len(values)}项")
    return "；".join(parts) if parts else "事实快照未改变，复用当前证据清单"


def version_evidence_plan(
    state: Any,
    *,
    legal_model: dict[str, Any],
    requirements: list[dict[str, Any]],
    proof_targets: list[dict[str, Any]],
    delivery_entries: list[dict[str, Any]] | None = None,
    fact_snapshot: dict[str, Any] | None = None,
    request_id: str = "",
) -> dict[str, Any]:
    snapshot = fact_snapshot or validate_fact_snapshot(state)
    payload = {
        "fact_snapshot_version": snapshot.get("fact_snapshot_version"),
        "fact_snapshot_hash": snapshot.get("fact_snapshot_hash"),
        "legal_domain": legal_model.get("legal_domain"),
        "relations": legal_model.get("relation_candidates"),
        "requests": legal_model.get("request_models"),
        "proof_targets": [
            {
                "id": item.get("proof_target_id") or item.get("id"),
                "status": item.get("status"),
                "dependencies": item.get("dependent_fact_keys"),
            }
            for item in proof_targets
        ],
        "requirements": [
            {
                "id": item.get("requirement_id"),
                "status": item.get("status"),
                "importance": item.get("importance"),
                "materials": item.get("recommended_materials"),
                "alternatives": item.get("alternative_materials"),
            }
            for item in requirements
        ],
    }
    fingerprint = _hash_payload(payload)
    previous_fingerprint = str(getattr(state, "evidence_plan_fingerprint", "") or "")
    previous_version = int(getattr(state, "evidence_plan_version", 0) or 0)
    reused = bool(previous_version and previous_fingerprint == fingerprint)
    version = previous_version if reused else previous_version + 1
    old_map = {
        str(item.get("requirement_id")): item
        for item in (getattr(state, "formal_evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    new_map = {
        str(item.get("requirement_id")): item
        for item in requirements
        if item.get("requirement_id")
    }
    added = sorted(set(new_map) - set(old_map))
    deactivated = sorted(
        key
        for key, item in new_map.items()
        if str(item.get("status") or "") in {"stale", "not_applicable", "superseded"}
        and str(old_map.get(key, {}).get("status") or "") not in {"stale", "not_applicable", "superseded"}
    )
    reactivated = sorted(
        key
        for key, item in new_map.items()
        if str(item.get("status") or "") == "active"
        and str(old_map.get(key, {}).get("status") or "") in {"stale", "not_applicable", "superseded"}
    )
    updated = sorted(
        key
        for key in set(new_map) & set(old_map)
        if _hash_payload(new_map[key]) != _hash_payload(old_map[key])
        and key not in deactivated
        and key not in reactivated
    )
    changes = {
        "added_requirement_ids": added,
        "updated_requirement_ids": updated,
        "deactivated_requirement_ids": deactivated,
        "reactivated_requirement_ids": reactivated,
        "changed_proof_target_ids": _unique(
            item.get("proof_target_id") or item.get("id") or ""
            for item in proof_targets
            if item.get("requirement_id") in set(updated)
        ),
    }
    return {
        "evidence_plan_version": version,
        "previous_version": previous_version or None,
        "legal_model_version": (
            int(getattr(state, "legal_model_version", 0) or 0)
            if reused
            else int(getattr(state, "legal_model_version", 0) or 0) + 1
        ),
        "fingerprint": fingerprint,
        "request_id": request_id or _stable_id("plan-request", state.case_id, fingerprint),
        "reused": reused,
        "changes": changes,
        "change_summary": _change_summary(changes, first=not previous_version),
        "delivery_entries": delivery_entries
        if delivery_entries is not None
        else build_delivery_entries(
            requirements,
            case_id=str(state.case_id),
            evidence_plan_version=version,
        ),
        "created_at": _now(),
    }


def build_plan_markdown(
    state: Any,
    *,
    requirements: list[dict[str, Any]],
    legal_model: dict[str, Any],
    status: str,
    basis_limitations: list[str],
    change_summary: str,
) -> str:
    groups = {
        "essential": [],
        "important": [],
        "reinforcing": [],
    }
    for item in requirements:
        if str(item.get("status") or "") in {"stale", "not_applicable", "superseded"}:
            continue
        groups.setdefault(str(item.get("importance") or "reinforcing"), []).append(item)
    lines = [
        "## 证据准备清单",
        "",
        f"> 当前规划状态：{'按当前信息形成条件式清单' if status == 'conditional' else '已形成案件专属清单'}",
        f"> 规划变化：{change_summary}",
        "",
    ]
    number = 1
    for importance in ("essential", "important", "reinforcing"):
        items = groups.get(importance) or []
        if not items:
            continue
        lines.extend([f"### {_IMPORTANCE_LABELS[importance]}", ""])
        for item in items:
            state_label = _MATERIAL_STATE_LABELS.get(
                str(item.get("user_material_state") or "not_submitted"),
                "暂未提交",
            )
            lines.extend(
                [
                    f"#### {number}. {item.get('label') or item.get('requirement_id')}",
                    "",
                    f"- **用于证明：** {item.get('purpose') or '支持当前证明目标'}",
                    f"- **建议准备：** {'、'.join(item.get('recommended_materials') or []) or '能够直接支持该事实的原始记录'}",
                    f"- **替代材料：** {'、'.join(item.get('alternative_materials') or []) or '如无法提供，可先说明原因和可调取来源'}",
                    f"- **当前状态：** {state_label}",
                    "",
                ]
            )
            number += 1
    if not any(groups.values()):
        lines.extend(["### 当前没有可固化的材料需求", "", "- 请先补充或确认案件事实。", ""])
    lines.extend(
        [
            "> 以上是根据当前事实、诉求和可检索依据整理的建议清单，不代表受理机关固定材料目录。",
            "> 用户称持有不等于已经提交，已经提交也需要在后续节点单独评估。",
        ]
    )
    if legal_model.get("relation_candidates"):
        relation = legal_model["relation_candidates"][0]
        lines.extend(
            [
                "",
                "### 当前法律关系候选",
                "",
                f"- **关系：** {relation.get('label') or relation.get('relation_id')}",
                "- **说明：** 这是基于当前事实的候选关系，不是最终责任结论。",
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
            "接下来可以分批提交材料，也可以先标记“稍后提交”“暂时找不到”或“可向第三方调取”。",
        ]
    )
    return "\n".join(lines).strip()


def checkpoint_evidence_plan(
    state: Any,
    *,
    snapshot: dict[str, Any],
    legal_model: dict[str, Any],
    proof_targets: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    evidence_name_links: list[dict[str, Any]],
    delivery_entries: list[dict[str, Any]],
    plan_version: dict[str, Any],
    status: str,
    retrieval: dict[str, Any],
    basis_refs: list[dict[str, Any]],
    basis_limitations: list[str],
    blocking_gaps: list[str],
) -> dict[str, Any]:
    plan_version_number = int(plan_version["evidence_plan_version"])
    next_route = "decide_facts" if blocking_gaps else "await_evidence_batch"
    if blocking_gaps:
        plan_status = "needs_fact_update"
        workflow_stage = "fact_clarification"
        pause_state = {
            "type": "awaiting_fact_batch",
            "pause_type": "awaiting_fact_batch",
            "missing_fact_keys": blocking_gaps,
            "reason": "法律建模发现新的阻断事实条件",
        }
        reply = (
            "## 还需要确认少量关键事实\n\n"
            "法律和证据规划发现以下信息可能改变适用关系或处理路径：\n\n"
            + "\n".join(f"- {item}" for item in blocking_gaps)
            + "\n\n请补充或明确说明“不清楚”，系统不会自行补全。"
        )
    else:
        plan_status = (
            "retrieval_degraded"
            if retrieval.get("retrieval_gaps") and not basis_refs
            else status
        )
        workflow_stage = "evidence_collection"
        evidence_batch_id = _stable_id(
            "evidence-batch",
            state.case_id,
            plan_version_number,
        )
        pause_state = {
            "type": "awaiting_evidence_batch",
            "pause_type": "awaiting_evidence_batch",
            "evidence_batch_id": evidence_batch_id,
            "evidence_plan_version": plan_version_number,
        }
        reply = build_plan_markdown(
            state,
            requirements=requirements,
            legal_model=legal_model,
            status=status,
            basis_limitations=basis_limitations,
            change_summary=str(plan_version.get("change_summary") or ""),
        )
    evidence_batch_id = (
        str(pause_state.get("evidence_batch_id") or getattr(state, "evidence_batch_id", ""))
        if pause_state
        else str(getattr(state, "evidence_batch_id", "") or "")
    )
    audit_id = _stable_id(
        "plan-audit",
        state.case_id,
        snapshot.get("fact_snapshot_version"),
        plan_version_number,
        plan_version.get("fingerprint"),
    )
    audit = {
        "plan_audit_id": audit_id,
        "case_id": state.case_id,
        "case_generation": getattr(state, "case_generation", 1),
        "fact_snapshot_version": snapshot.get("fact_snapshot_version"),
        "fact_blackboard_version": snapshot.get("fact_blackboard_version"),
        "legal_model_version": plan_version.get("legal_model_version"),
        "evidence_plan_version": plan_version_number,
        "retrieval_trace_id": (retrieval.get("retrieval_trace") or {}).get("retrieval_trace_id", ""),
        "proof_target_ids": [
            item.get("proof_target_id") or item.get("id") for item in proof_targets
        ],
        "requirement_changes": plan_version.get("changes") or {},
        "evidence_name_links": evidence_name_links,
        "delivery_entry_ids": [
            item.get("delivery_entry_id") for item in delivery_entries
        ],
        "basis_refs": basis_refs,
        "basis_limitations": basis_limitations,
        "unknown_conditions": legal_model.get("unknown_conditions") or [],
        "stale_dependencies": blocking_gaps,
        "next_route": next_route,
        "created_at": _now(),
    }
    return {
        "fact_snapshot_confirmed": True if snapshot.get("valid") else state.fact_snapshot_confirmed,
        "fact_snapshot_version": int(
            snapshot.get("fact_snapshot_version")
            or getattr(state, "fact_snapshot_version", 0)
            or 0
        ),
        "legal_model": legal_model,
        "legal_model_version": int(plan_version.get("legal_model_version") or 0),
        "legal_model_status": "candidate" if not blocking_gaps else "needs_fact_update",
        "relation_candidates": list(legal_model.get("relation_candidates") or []),
        "request_models": list(legal_model.get("request_models") or []),
        "plan_retrieval_trace": retrieval.get("retrieval_trace") or {},
        "plan_retrieval_gaps": list(retrieval.get("retrieval_gaps") or []),
        "retrieval_basis_candidates": list(retrieval.get("basis_candidates") or []),
        "retrieved_law_refs": list(retrieval.get("law_refs") or []),
        "similar_cases": list(retrieval.get("similar_cases") or []),
        "case_context_str": str(retrieval.get("case_context") or ""),
        "fallback_guide": retrieval.get("fallback_guide"),
        "proof_targets": proof_targets,
        "formal_evidence_requirements": requirements,
        "evidence_name_links": evidence_name_links,
        "delivery_entries": delivery_entries,
        "plan_basis_refs": basis_refs,
        "plan_basis_limitations": basis_limitations,
        "plan_change_summary": str(plan_version.get("change_summary") or ""),
        "plan_audit_id": audit_id,
        "evidence_plan_request_id": str(plan_version.get("request_id") or ""),
        "evidence_plan_fingerprint": str(plan_version.get("fingerprint") or ""),
        "previous_evidence_plan_version": int(plan_version.get("previous_version") or 0),
        "evidence_plan_version": plan_version_number,
        "evidence_plan_status": plan_status,
        "evidence_collection_status": "open" if not blocking_gaps else "not_open",
        "evidence_batch_id": evidence_batch_id,
        "evidence_batch_completed": False if not blocking_gaps else state.evidence_batch_completed,
        "evidence_verification_pending": False,
        "pause_state": pause_state,
        "stale_dependencies": blocking_gaps,
        "decision_status": "evidence_plan_active" if not blocking_gaps else "needs_fact_update",
        "next_route": next_route,
        "workflow_stage": workflow_stage,
        "phase": GuidePhase.DETAIL_GATHER if not blocking_gaps else GuidePhase.ISSUE_SEARCH,
        "messages": [AIMessage(content=reply)],
        "plan_audit": audit,
    }


async def run_plan_evidence(state: Any, deps: Any = None) -> dict[str, Any]:
    """Execute node five and return a persistent evidence-plan checkpoint."""

    snapshot = validate_fact_snapshot(state)
    if not snapshot.get("valid"):
        message = f"## 暂时不能建立证据清单\n\n{snapshot.get('message') or '请先完成事实确认。'}"
        return {
            "evidence_plan_status": str(snapshot.get("status") or "needs_fact_update"),
            "decision_status": "needs_fact_update",
            "next_route": "decide_facts",
            "pause_state": {
                "type": "awaiting_fact_snapshot_confirmation",
                "pause_type": "awaiting_fact_snapshot_confirmation",
            },
            "workflow_stage": "fact_snapshot",
            "messages": [AIMessage(content=message)],
            "stale_dependencies": [str(snapshot.get("reason") or "fact_snapshot")],
        }

    query = build_legal_model_input(state)
    retrieval = await retrieve_plan_authorities(state, deps, query)
    valid_basis, citation_limits = validate_plan_citations(
        [
            *(retrieval.get("basis_candidates") or []),
            *(retrieval.get("law_refs") or []),
        ],
        return_limitations=True,
    )
    relations = build_relation_candidates(state)
    requests = build_request_models(state)
    legal_model = build_legal_model(
        state,
        relation_candidates=relations,
        request_models=requests,
        basis_refs=valid_basis,
        retrieval_trace=retrieval.get("retrieval_trace"),
        basis_limitations=citation_limits,
    )
    blocking_gaps = detect_blocking_fact_gaps(state, legal_model=legal_model)
    proof_targets = build_proof_targets(
        state,
        legal_model=legal_model,
        basis_refs=valid_basis,
    )
    requirements = formalize_evidence_requirements(
        state,
        proof_targets,
        basis_refs=valid_basis,
        previous=getattr(state, "formal_evidence_requirements", []) or [],
    )
    requirements, evidence_name_links = link_evidence_name_inventory(
        requirements,
        getattr(state, "evidence_name_inventory", []) or [],
        state,
    )
    legal_model["proof_target_ids"] = [
        item.get("proof_target_id") or item.get("id") for item in proof_targets
    ]
    legal_model["evidence_basis_refs"] = valid_basis
    legal_model["basis_limitations"] = citation_limits
    plan_version = version_evidence_plan(
        state,
        legal_model=legal_model,
        requirements=requirements,
        proof_targets=proof_targets,
        fact_snapshot=snapshot,
        request_id=str(getattr(state, "current_request_id", "") or ""),
    )
    delivery_entries = build_delivery_entries(
        requirements,
        case_id=str(state.case_id),
        evidence_plan_version=int(plan_version["evidence_plan_version"]),
    )
    updates = checkpoint_evidence_plan(
        state,
        snapshot=snapshot,
        legal_model=legal_model,
        proof_targets=proof_targets,
        requirements=requirements,
        evidence_name_links=evidence_name_links,
        delivery_entries=delivery_entries,
        plan_version=plan_version,
        status=str(snapshot.get("status") or "active"),
        retrieval=retrieval,
        basis_refs=valid_basis,
        basis_limitations=_unique(
            [
                *citation_limits,
                *(retrieval.get("retrieval_gaps") or []),
                *(
                    resolve_state_evidence_checklist(state).usage_note,
                ),
            ]
        ),
        blocking_gaps=blocking_gaps,
    )
    logger.info(
        "节点⑤证据规划 | case={} snapshot={} plan={} status={} requirements={} gaps={}",
        state.case_id,
        snapshot.get("fact_snapshot_version"),
        updates.get("evidence_plan_version"),
        updates.get("evidence_plan_status"),
        len(requirements),
        len(blocking_gaps),
    )
    return updates


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "build_legal_model_input",
    "build_relation_candidates",
    "build_request_models",
    "build_legal_model",
    "build_proof_targets",
    "formalize_evidence_requirements",
    "link_evidence_name_inventory",
    "build_delivery_entries",
    "validate_plan_citations",
    "retrieve_plan_authorities",
    "detect_blocking_fact_gaps",
    "version_evidence_plan",
    "build_plan_markdown",
    "checkpoint_evidence_plan",
    "validate_fact_snapshot",
    "run_plan_evidence",
]
