"""Deterministic input preparation for the first legal-guide workflow node.

This module deliberately performs *event-level* classification only.  It does
not atomize legal facts, decide liability, retrieve law, or evaluate evidence.
Those responsibilities belong to later workflow nodes.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from langchain_core.messages import HumanMessage

from src.agents.legal_guide.state import GuidePhase, GuideState


_CONCLUDE_MARKERS = (
    "不要再问",
    "别再问",
    "不用再问",
    "现在生成方案",
    "直接生成方案",
    "生成方案",
    "给我方案",
    "给出方案",
    "按现有信息",
    "按目前情况",
    "按现在这些",
    "最终建议",
    "最终方案",
    "请收敛",
    "没有更多信息",
    "没有更多证据",
    "没更多信息",
)
_CONTINUE_MARKERS = (
    "继续补充",
    "继续问",
    "可以继续",
    "再问",
    "再补充",
    "继续梳理",
    "我继续说",
)
_REGENERATE_MARKERS = (
    "重新生成",
    "重新评估",
    "再评估",
    "更新方案",
)
_SNAPSHOT_CONFIRM_MARKERS = (
    "确认并继续",
    "事实无误",
    "案情无误",
    "确认事实",
)
_BATCH_COMPLETE_MARKERS = (
    "完成本批次并评估",
    "完成本批证据",
    "本批次提交完成",
    "开始评估证据",
)
_CORRECTION_MARKERS = (
    "之前说错",
    "前面说错",
    "更正一下",
    "纠正一下",
    "准确时间是",
    "实际是",
    "应当是",
    "改为",
)
_DENIAL_MARKERS = (
    "并没有",
    "不是这样",
    "没有这回事",
    "我没说过",
    "不属实",
)
_PROGRESS_MARKERS = (
    "已经投诉",
    "已投诉",
    "投诉了",
    "平台处理",
    "平台回复",
    "客服回复",
    "拒绝退款",
    "协商结果",
    "已经报警",
    "已报警",
    "已经申请仲裁",
    "已申请仲裁",
    "仲裁受理",
    "已经起诉",
    "已起诉",
    "法院受理",
    "收到传票",
    "对方回应",
    "执行进展",
)
_DOCUMENT_MARKERS = (
    "生成投诉信",
    "生成起诉状",
    "生成仲裁申请书",
    "生成律师函",
    "生成调解协议",
    "生成文书",
    "写一份投诉信",
    "写一份起诉状",
)
_EVIDENCE_NAME_PATTERN = re.compile(
    r"(订单(?:截图|详情|记录)|付款记录|支付记录|转账记录|银行流水|"
    r"聊天记录|通话记录|录音|录像|照片|视频|合同|协议|收据|发票|"
    r"物流记录|物流状态|投诉工单|客服记录|病历|诊断证明|工资流水|"
    r"考勤记录|劳动合同|商品页面|平台账单)"
)
_EVIDENCE_BLOCK_START = re.compile(r"(?=【(?:图片|文档)证据补充)")
_PURE_ACKNOWLEDGEMENTS = {
    "",
    "好",
    "好的",
    "行",
    "可以",
    "嗯",
    "哦",
    "知道了",
    "明白了",
}


def latest_user_input(state: GuideState) -> str:
    """Return the request text even when boundary review must stay read-only."""

    if str(state.current_message_text or "").strip():
        return str(state.current_message_text).strip()
    return next(
        (
            str(message.content).strip()
            for message in reversed(state.messages)
            if isinstance(message, HumanMessage) and str(message.content).strip()
        ),
        "",
    )


def _compact(value: str) -> str:
    return "".join(str(value or "").split())


def _contains_any(value: str, markers: Iterable[str]) -> bool:
    compact = _compact(value)
    return any(marker in compact for marker in markers)


def resolve_control_intent(
    message: str,
    semantic_intent: str = "",
    explicit_action: str = "",
) -> str:
    """Normalize legacy semantic output and deterministic UI controls."""

    explicit = str(explicit_action or "").strip().lower()
    semantic = str(semantic_intent or "").strip().lower()
    aliases = {
        "conclude": "conclude_now",
        "control_conclude_now": "conclude_now",
        "continue": "continue_gathering",
        "control_continue_gathering": "continue_gathering",
        "control_regenerate": "regenerate",
        "complete_batch": "complete_batch",
        "confirm": "confirm",
        "document_request": "document_request",
    }
    for candidate in (explicit, semantic):
        normalized = aliases.get(candidate, candidate)
        if normalized in {
            "conclude_now",
            "continue_gathering",
            "regenerate",
            "complete_batch",
            "confirm",
            "document_request",
            "case_detail",
        }:
            return normalized
    if _contains_any(message, _BATCH_COMPLETE_MARKERS):
        return "complete_batch"
    if _contains_any(message, _SNAPSHOT_CONFIRM_MARKERS):
        return "confirm"
    if _contains_any(message, _REGENERATE_MARKERS):
        return "regenerate"
    if _contains_any(message, _CONCLUDE_MARKERS):
        return "conclude_now"
    if _contains_any(message, _CONTINUE_MARKERS):
        return "continue_gathering"
    if _contains_any(message, _DOCUMENT_MARKERS):
        return "document_request"
    return "other"


def _uploaded_blocks(message: str) -> tuple[str, list[dict[str, Any]]]:
    """Split legacy inline upload blocks without interpreting their contents."""

    parts = _EVIDENCE_BLOCK_START.split(str(message or ""))
    narrative: list[str] = []
    blocks: list[dict[str, Any]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped.startswith(("【图片证据补充", "【文档证据补充")):
            file_match = re.search(r"^文件[：:]\s*([^\n]+)", stripped, re.MULTILINE)
            sha_match = re.search(r"SHA-256[：:]\s*([0-9a-fA-F]{32,64})", stripped)
            blocks.append(
                {
                    "source": "legacy_inline_block",
                    "kind": "image" if stripped.startswith("【图片") else "document",
                    "file_name": file_match.group(1).strip() if file_match else "",
                    "sha256": sha_match.group(1).lower() if sha_match else "",
                    # Only keep a reference-sized excerpt in node state.  Full parsed text
                    # remains in the message and is handled by the evidence node.
                    "content_excerpt": stripped[:500],
                }
            )
        else:
            narrative.append(stripped)
    return "\n\n".join(narrative).strip(), blocks


def _strip_control_text(message: str) -> str:
    value = str(message or "")
    for marker in sorted(
        {
            *_CONCLUDE_MARKERS,
            *_CONTINUE_MARKERS,
            *_REGENERATE_MARKERS,
            *_SNAPSHOT_CONFIRM_MARKERS,
            *_BATCH_COMPLETE_MARKERS,
            *_DOCUMENT_MARKERS,
        },
        key=len,
        reverse=True,
    ):
        value = value.replace(marker, "")
    return value.strip(" \t\r\n，。；：、！？!?")


def split_mixed_payload(
    message: str,
    *,
    attachments: list[dict] | None = None,
    form_updates: list[dict] | None = None,
    control_intent: str = "other",
    message_id: str = "",
) -> dict[str, dict]:
    """Separate facts, materials, progress and flow control by reference."""

    narrative, legacy_blocks = _uploaded_blocks(message)
    case_text = _strip_control_text(narrative)
    if control_intent not in {"", "other", "case_detail"}:
        control_residue = re.sub(
            r"[，。；：、！？!?\s]|就|请|现在|直接|先|给我|给|生成|方案|评估|继续|了|吧",
            "",
            case_text,
        )
        if not control_residue:
            case_text = ""
    evidence_names = list(dict.fromkeys(_EVIDENCE_NAME_PATTERN.findall(narrative)))
    progress_text = case_text if _contains_any(case_text, _PROGRESS_MARKERS) else ""
    fact_text = "" if progress_text else case_text

    return {
        "fact_payload": {
            "text": fact_text,
            "source_message_id": message_id,
        },
        "evidence_payload": {
            "attachments": list(attachments or []),
            "legacy_blocks": legacy_blocks,
            "named_evidence": evidence_names,
            "source_message_id": message_id,
        },
        "progress_payload": {
            "text": progress_text,
            "form_updates": list(form_updates or []),
            "source_message_id": message_id,
        },
        "control_payload": {
            "intent": control_intent,
            "source_message_id": message_id,
        },
    }


def restore_pause_state(state: GuideState) -> dict | None:
    """Map legacy and target fields onto one stable pause-state contract."""

    if state.safety_pause_active:
        return {"type": "awaiting_safety_confirmation"}
    if state.awaiting_case_boundary:
        return {"type": "awaiting_case_boundary"}
    if state.evidence_verification_pending:
        return {
            "type": "awaiting_evidence_verification",
            "evidence_batch_id": state.evidence_batch_id,
        }
    if state.pending_ask_type == "facts" and state.pending_ask_details:
        return {
            "type": "awaiting_fact_batch",
            "pending_followup_ids": list(state.pending_followup_ids),
        }
    if (
        state.workflow_stage == "fact_snapshot"
        and not state.fact_snapshot_confirmed
    ):
        return {
            "type": "awaiting_fact_snapshot_confirmation",
            "fact_snapshot_version": state.fact_snapshot_version,
        }
    if (
        state.evidence_collection_status == "open"
        and not state.evidence_batch_completed
    ):
        return {
            "type": "awaiting_evidence_batch",
            "evidence_batch_id": state.evidence_batch_id,
        }
    return None


def derive_workflow_stage(state: GuideState, pause_state: dict | None) -> str:
    """Derive a target-stage label while old GuidePhase is still in service."""

    pause_type = str((pause_state or {}).get("type") or "")
    if pause_type == "awaiting_safety_confirmation":
        return "risk_guard"
    if pause_type == "awaiting_case_boundary":
        return "case_boundary"
    if pause_type == "awaiting_fact_snapshot_confirmation":
        return "fact_snapshot"
    if pause_type == "awaiting_evidence_verification":
        return "evidence_assessment"
    if pause_type == "awaiting_evidence_batch":
        return "evidence_collection"
    if state.evidence_collection_status == "open":
        return "evidence_collection"
    if state.phase == GuidePhase.DETAIL_GATHER or state.pending_ask_details:
        return "fact_gathering"
    if state.phase == GuidePhase.ISSUE_SEARCH:
        return "fact_modeling"
    if state.phase == GuidePhase.CONCLUDE:
        return "solution_generation"
    if state.phase == GuidePhase.END and state.plan_version:
        return "case_management"
    return "case_intake"


def _looks_like_question(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and (
            any(marker in text for marker in ("？", "?", "为什么", "怎么", "如何", "什么是"))
            or text.endswith(("吗", "呢", "么"))
        )
    )


def _event(type_: str, payload_ref: str = "") -> dict[str, str]:
    result = {"type": type_}
    if payload_ref:
        result["payload_ref"] = payload_ref
    return result


def classify_input_events(
    state: GuideState,
    message: str,
    payloads: dict[str, dict],
    *,
    control_intent: str,
    is_first_turn: bool,
) -> tuple[str, list[dict[str, str]]]:
    """Classify all events in one message without making legal judgments."""

    if state.case_boundary_read_only:
        return "unknown", []

    pause_type = str((restore_pause_state(state) or {}).get("type") or "")
    fact_text = str(payloads["fact_payload"].get("text") or "").strip()
    progress_text = str(payloads["progress_payload"].get("text") or "").strip()
    evidence_payload = payloads["evidence_payload"]
    has_uploaded_evidence = bool(
        evidence_payload.get("attachments") or evidence_payload.get("legacy_blocks")
    )
    named_evidence = list(evidence_payload.get("named_evidence") or [])
    events: list[dict[str, str]] = []

    if pause_type == "awaiting_case_boundary":
        events.append(_event("case_boundary_answered"))
    elif pause_type == "awaiting_fact_batch" and fact_text:
        events.append(_event("fact_batch_answered", "fact_payload"))
    elif pause_type == "awaiting_fact_snapshot_confirmation" and control_intent == "confirm":
        events.append(_event("fact_snapshot_confirmed", "control_payload"))
    elif pause_type == "awaiting_evidence_verification" and fact_text:
        events.append(_event("evidence_verification_answered", "fact_payload"))

    if progress_text:
        events.append(_event("case_progress_updated", "progress_payload"))
    elif fact_text and _compact(fact_text) not in _PURE_ACKNOWLEDGEMENTS:
        if _contains_any(fact_text, _CORRECTION_MARKERS):
            events.append(_event("fact_corrected", "fact_payload"))
        elif _contains_any(fact_text, _DENIAL_MARKERS):
            events.append(_event("fact_denied", "fact_payload"))
        elif not any(item["type"].startswith("fact_") for item in events):
            if _looks_like_question(fact_text):
                events.append(_event("case_related_question", "fact_payload"))
            else:
                events.append(_event("fact_added", "fact_payload"))

    if named_evidence:
        events.append(_event("evidence_named", "evidence_payload"))
    if has_uploaded_evidence:
        events.append(_event("evidence_added", "evidence_payload"))
    if control_intent == "complete_batch":
        events.append(_event("evidence_batch_completed", "control_payload"))
    elif control_intent == "conclude_now":
        events.append(_event("control_conclude_now", "control_payload"))
    elif control_intent == "continue_gathering":
        events.append(_event("control_continue_gathering", "control_payload"))
    elif control_intent == "regenerate":
        events.append(_event("control_regenerate", "control_payload"))
    elif control_intent == "document_request":
        events.append(_event("document_requested", "control_payload"))

    # Preserve deterministic order while removing accidental duplicates.
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in events:
        if item["type"] in seen:
            continue
        seen.add(item["type"])
        unique.append(item)

    if is_first_turn:
        return "case_started", unique or [_event("fact_added", "fact_payload")]
    if not unique:
        return "case_continued", []
    if len(unique) > 1:
        return "mixed_update", unique
    return unique[0]["type"], unique


def determine_requested_route(
    state: GuideState,
    input_event_type: str,
    input_events: list[dict[str, str]],
) -> tuple[str, list[str]]:
    """Return the immediate post-guard target and a non-binding candidate chain."""

    event_types = {item.get("type", "") for item in input_events}
    if state.awaiting_case_boundary or state.case_boundary_read_only:
        return "guard_case_read_only", []
    if event_types & {
        "fact_added",
        "fact_corrected",
        "fact_denied",
        "fact_batch_answered",
        "case_progress_updated",
    }:
        chain = ["update_facts", "decide_facts"]
        if "control_conclude_now" in event_types:
            chain.extend(["plan_evidence", "assess_evidence", "generate_solution"])
        return "update_facts", chain
    if "fact_snapshot_confirmed" in event_types:
        return "plan_evidence", ["plan_evidence"]
    if "evidence_batch_completed" in event_types:
        return "assess_evidence", ["assess_evidence"]
    if "evidence_verification_answered" in event_types:
        return "assess_evidence", ["assess_evidence"]
    if "evidence_added" in event_types:
        if state.evidence_collection_status == "open":
            return "assess_evidence", ["assess_evidence"]
        return "update_facts", ["update_facts", "decide_facts"]
    if "control_regenerate" in event_types:
        return "assess_evidence", ["assess_evidence", "generate_solution"]
    if "control_conclude_now" in event_types:
        if state.retrieval_completed:
            return "generate_solution", ["generate_solution"]
        return "decide_facts", ["decide_facts", "plan_evidence", "generate_solution"]
    if "document_requested" in event_types:
        return "document_service", ["document_service"]
    if "case_related_question" in event_types:
        return "case_question", ["case_question"]
    if input_event_type == "case_started":
        return "update_facts", ["update_facts"]
    return "update_facts", ["update_facts"]
