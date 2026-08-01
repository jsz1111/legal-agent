"""Structured risk gate for every legal-guide case turn.

The guard is intentionally conservative: it can pause for current personal
danger, surface time-sensitive preservation actions, and block unsafe methods,
but it never decides liability or announces a legal deadline as established.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from langchain_core.messages import AIMessage, SystemMessage
from loguru import logger

from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage
from src.agents.legal_guide.prepare_case import latest_user_input
from src.agents.legal_guide.state import GuideState
from src.core.config import get_settings


settings = get_settings()
_CHINA_TZ = timezone(timedelta(hours=8))

RISK_TYPES = {
    "personal_safety",
    "custody_or_coercion",
    "deadline",
    "evidence_loss",
    "asset_emergency",
    "unlawful_collection",
    "dangerous_confrontation",
}
RISK_LEVELS = {"clear", "warning", "urgent", "critical", "unknown"}
_LEVEL_WEIGHT = {
    "clear": 0,
    "warning": 1,
    "urgent": 2,
    "unknown": 3,
    "critical": 4,
}

_SAFE_MARKERS = (
    "今天暂时安全",
    "现在暂时安全",
    "目前暂时安全",
    "现在安全",
    "目前安全",
    "已经安全",
    "我安全了",
    "没有安全危险",
    "现在没有危险",
    "目前没有危险",
    "已经离开现场",
    "已离开现场",
    "已经到朋友家",
    "已经到安全地点",
    "对方不在附近",
)
_CURRENT_DANGER_PATTERNS = (
    r"(?:现在|目前|此刻|正在).{0,12}(?:打我|打人|施暴|追我|跟踪我|威胁我|伤害我)",
    r"(?:拿刀|持刀|持械).{0,12}(?:门外|附近|冲过来|威胁|要伤害|对着我)",
    r"(?:就在门外|赶过来|马上过来).{0,12}(?:打|杀|伤害|报复|威胁)",
    r"(?:我|他人).{0,8}(?:被困|被关|被扣住|无法离开|不让我走)",
    r"(?:现在|目前).{0,8}(?:有危险|无法脱身|不能离开)",
    r"(?:正在|马上|即将).{0,10}(?:杀|伤害|绑架|强迫签字)",
)
_SAFETY_RELEVANT_PATTERN = re.compile(
    r"殴打|打伤|被.{0,4}打|打过|家暴|暴力|威胁|恐吓|跟踪|持刀|拿刀|被困|拘禁|"
    r"不让我走|伤害|杀我|强迫签字|限制人身自由"
)
_HISTORICAL_OR_QUOTED_PATTERN = re.compile(
    r"去年|前年|以前|曾经|当时|过去|之前发生|聊天记录(?:里|中)|"
    r"截图(?:里|中)|录音(?:里|中)|只是举例|不是我现在|并非当前"
)
_EVIDENCE_LOSS_PATTERNS = (
    re.compile(r"(?:监控|录像).{0,18}(?:今晚|今天|明天|即将|马上).{0,10}(?:覆盖|删除|清空)"),
    re.compile(r"(?:网页|商品页面|直播|帖子|链接|平台记录).{0,18}(?:即将|马上|可能).{0,10}(?:下架|删除|失效)"),
    re.compile(r"(?:商品|设备|现场|原件).{0,18}(?:即将|马上|准备).{0,10}(?:销毁|维修|拆除|返还|格式化)"),
)
_DEADLINE_URGENT_PATTERN = re.compile(
    r"(?:(?:今天|明天|后天|只剩\d+天|不到\d+天|马上|即将).{0,10}(?:截止|到期|届满))"
    r"|(?:(?:截止|到期|届满).{0,10}(?:今天|明天|后天|只剩\d+天|不到\d+天))"
)
_DEADLINE_WARNING_PATTERN = re.compile(
    r"诉讼时效|仲裁时效|申请期限|上诉期限|复议期限|起诉期限|"
    r"是否来得及|还来得及吗|会不会过期|已经过期|收到.{0,8}(?:裁决书|决定书|判决书|通知书|传票)"
)
_ASSET_URGENT_PATTERN = re.compile(
    r"(?:平台款项|保证金|货款).{0,16}(?:今天|明天|马上|即将).{0,8}(?:放行|打给|结算)"
    r"|(?:正在|马上|即将).{0,8}(?:转移|隐匿|处分).{0,6}(?:财产|资产)"
)
_ASSET_WARNING_PATTERN = re.compile(
    r"银行卡被冻结|账户被冻结|资金被冻结|账号冻结|财产可能转移|经营异常|即将注销"
)
_UNLAWFUL_COLLECTION_PATTERN = re.compile(
    r"破解.{0,8}(?:账号|密码|手机|电脑)|盗号|黑进.{0,8}(?:账号|系统|设备)|"
    r"木马|窃取密码|冒充.{0,8}(?:本人|客服|警察|律师)|"
    r"伪造.{0,8}(?:证据|合同|聊天记录|签名)|篡改.{0,8}(?:记录|证据|截图)|"
    r"偷拍.{0,8}(?:卧室|浴室|更衣室)|公开.{0,8}(?:身份证|住址|手机号)"
)
_DANGEROUS_CONFRONTATION_PATTERN = re.compile(
    r"(?:我要|我准备|我打算|想去|准备去).{0,12}(?:堵人|堵他|上门闹|上门打|打他|"
    r"威胁他|扣押|砸店|抢回|强行拿走)"
)

_GUARD_PROMPT = """你负责法律维权工作流的风险语义分类，不判断案件责任或胜负。

只分析[当前用户陈述]，结构化附件元数据不能证明用户当前处于危险。必须区分：
- 正在发生或明确即将发生；
- 过去发生；
- 用户已经明确安全；
- 当前状态未知；
- 只是材料中的引用。

风险类型只能是：
personal_safety, custody_or_coercion, deadline, evidence_loss,
asset_emergency, unlawful_collection, dangerous_confrontation。

等级只能是 warning, urgent, critical, unknown：
- critical 只用于当前现实人身危险或当前人身自由受限；
- unknown 只用于涉及人身安全但当前状态没有说清；
- urgent 用于建议今天或明确短期内先行动的风险；
- deadline 条件不足时只能 warning，不能计算或宣布截止日；
- 账户冻结和普通金钱损失不是人身危险；
- 用户提到明确日期不等于临近法定期限；
- 违法取证和危险对抗需标记，但不要提供做法。

[当前用户陈述]
{current_input}

[最近安全状态摘要]
{recent_safety}

[当前结构化状态]
- 既有安全暂停：{safety_pause}
- 法律领域：{legal_domain}
- 地区：{region}
- 本轮事件：{input_events}

只输出 JSON：
{{
  "risks": [
    {{
      "risk_type": "personal_safety|custody_or_coercion|deadline|evidence_loss|asset_emergency|unlawful_collection|dangerous_confrontation",
      "level_candidate": "warning|urgent|critical|unknown",
      "trigger": "不超过80字的当前陈述摘要",
      "current_or_historical": "current|historical|quoted|unknown",
      "confidence": 0.0,
      "missing_conditions": ["尚缺条件"],
      "source_refs": ["message_id"]
    }}
  ],
  "safety_relevant": true,
  "current_safety_status": "danger|safe|unknown|not_applicable",
  "time_clues": []
}}"""


def _now() -> str:
    return datetime.now(_CHINA_TZ).isoformat()


def _redact(value: str, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())[:limit]
    text = re.sub(r"(?<!\d)1\d{10}(?!\d)", "1**********", text)
    text = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "账号已脱敏", text)
    return text


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def collect_guard_context(state: GuideState) -> dict[str, Any]:
    """Use structured current-turn payloads and a bounded safety-only history."""

    raw_input = latest_user_input(state)
    fact_text = str(state.fact_payload.get("text") or "").strip()
    progress_text = str(state.progress_payload.get("text") or "").strip()
    structured_text = "\n".join(
        part for part in (fact_text, progress_text) if part
    ).strip()
    has_material_only = bool(
        state.evidence_payload.get("attachments")
        or state.evidence_payload.get("legacy_blocks")
    )
    if state.case_boundary_read_only or state.safety_pause_active:
        current_input = raw_input
    elif structured_text:
        current_input = structured_text
    elif has_material_only:
        current_input = "用户本轮仅提交附件；附件内容不代表用户当前现实状态。"
    else:
        current_input = raw_input

    recent_safety: list[str] = []
    for message in state.messages[-8:]:
        content = _redact(getattr(message, "content", ""), 220)
        if content and (
            _SAFETY_RELEVANT_PATTERN.search(content)
            or any(marker in content for marker in _SAFE_MARKERS)
        ):
            recent_safety.append(content)
    if current_input and (
        _SAFETY_RELEVANT_PATTERN.search(current_input)
        or any(marker in current_input for marker in _SAFE_MARKERS)
    ):
        if not recent_safety or recent_safety[-1] != current_input:
            recent_safety.append(_redact(current_input, 220))

    return {
        "current_input": current_input,
        "raw_input": raw_input,
        "recent_safety": recent_safety[-4:],
        "source_ref": state.current_message_id or f"event-{state.event_sequence}",
    }


def _risk(
    state: GuideState,
    risk_type: str,
    level: str,
    trigger: str,
    source_ref: str,
    *,
    missing_conditions: list[str] | None = None,
    decision_source: str = "deterministic_rule",
    temporal_status: str = "current",
    confidence: float = 1.0,
) -> dict[str, Any]:
    cleaned_trigger = _redact(trigger)
    return {
        "risk_id": _stable_id(
            "risk",
            state.case_id,
            state.case_generation,
            state.event_sequence,
            risk_type,
            cleaned_trigger,
        ),
        "risk_type": risk_type,
        "level": level,
        "status": "active",
        "trigger": cleaned_trigger,
        "source_refs": [source_ref] if source_ref else [],
        "basis_refs": [],
        "missing_conditions": list(missing_conditions or []),
        "decision_source": decision_source,
        "temporal_status": temporal_status,
        "confidence": round(max(0.0, min(float(confidence), 1.0)), 4),
    }


def detect_deterministic_safety_risk(
    state: GuideState,
    context: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Return current safety status without treating history/quotes as current."""

    text = str(context["current_input"] or "")
    recent = "\n".join(context["recent_safety"])
    source_ref = str(context["source_ref"])
    current_danger = any(re.search(pattern, text) for pattern in _CURRENT_DANGER_PATTERNS)
    explicitly_safe = any(marker in text for marker in _SAFE_MARKERS)
    inherited_safe = any(marker in recent for marker in _SAFE_MARKERS)
    historical_or_quoted = bool(_HISTORICAL_OR_QUOTED_PATTERN.search(text))
    safety_relevant = bool(_SAFETY_RELEVANT_PATTERN.search(text))
    matches: list[str] = []

    if current_danger:
        matches.append("current_personal_danger")
        risk_type = (
            "custody_or_coercion"
            if re.search(r"被困|被关|无法离开|不让我走|强迫签字", text)
            else "personal_safety"
        )
        return (
            "danger",
            [_risk(state, risk_type, "critical", text, source_ref)],
            matches,
        )
    if explicitly_safe:
        matches.append("explicit_current_safety")
        return "safe", [], matches
    if inherited_safe and not safety_relevant:
        matches.append("recent_explicit_safety")
        return "safe", [], matches
    if state.safety_pause_active:
        matches.append("existing_safety_pause_unresolved")
        return (
            "unknown",
            [
                _risk(
                    state,
                    "personal_safety",
                    "unknown",
                    text or state.safety_pause_case_message,
                    source_ref,
                    missing_conditions=["用户当前是否已经脱离现场并处于安全位置"],
                )
            ],
            matches,
        )
    if safety_relevant:
        matches.append(
            "historical_or_quoted_safety_reference"
            if historical_or_quoted
            else "safety_status_not_stated"
        )
        return (
            "safe" if inherited_safe else "unknown",
            []
            if inherited_safe
            else [
                _risk(
                    state,
                    "personal_safety",
                    "unknown",
                    text,
                    source_ref,
                    missing_conditions=["当前是否安全"],
                    temporal_status="historical" if historical_or_quoted else "unknown",
                )
            ],
            matches,
        )
    return "not_applicable", [], matches


def detect_deterministic_non_safety_risks(
    state: GuideState,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    text = str(context["current_input"] or "")
    source_ref = str(context["source_ref"])
    risks: list[dict[str, Any]] = []
    matches: list[str] = []

    if any(pattern.search(text) for pattern in _EVIDENCE_LOSS_PATTERNS):
        matches.append("imminent_evidence_loss")
        risks.append(_risk(state, "evidence_loss", "urgent", text, source_ref))
    if _DEADLINE_URGENT_PATTERN.search(text):
        matches.append("explicit_short_deadline")
        risks.append(
            _risk(
                state,
                "deadline",
                "urgent",
                text,
                source_ref,
                missing_conditions=["适用程序、起算事件和官方送达时间仍需核对"],
            )
        )
    elif _DEADLINE_WARNING_PATTERN.search(text):
        matches.append("possible_deadline")
        risks.append(
            _risk(
                state,
                "deadline",
                "warning",
                text,
                source_ref,
                missing_conditions=[
                    "请求类型",
                    "起算事件",
                    "送达或知悉时间",
                    "是否存在中止、中断或特殊期间",
                ],
            )
        )
    if _ASSET_URGENT_PATTERN.search(text):
        matches.append("imminent_asset_change")
        risks.append(_risk(state, "asset_emergency", "urgent", text, source_ref))
    elif _ASSET_WARNING_PATTERN.search(text):
        matches.append("asset_status_risk")
        risks.append(
            _risk(state, "asset_emergency", "warning", text, source_ref)
        )
    if _UNLAWFUL_COLLECTION_PATTERN.search(text):
        matches.append("restricted_evidence_method")
        risks.append(
            _risk(state, "unlawful_collection", "warning", text, source_ref)
        )
    if _DANGEROUS_CONFRONTATION_PATTERN.search(text):
        matches.append("dangerous_confrontation_plan")
        risks.append(
            _risk(state, "dangerous_confrontation", "urgent", text, source_ref)
        )
    return risks, matches


def _json_payload(content: Any) -> dict[str, Any]:
    value = str(content or "").strip()
    if "```" in value:
        pieces = value.split("```")
        value = pieces[1] if len(pieces) > 1 else value
        value = re.sub(r"^\s*json\s*", "", value, flags=re.IGNORECASE).strip()
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


async def classify_guard_risks(
    state: GuideState,
    context: dict[str, Any],
    llm: Any,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], str]:
    """Return validated model candidates; invalid output safely degrades."""

    prompt = _GUARD_PROMPT.format(
        current_input=context["current_input"] or "无可分类文本",
        recent_safety="\n".join(context["recent_safety"]) or "无",
        safety_pause=state.safety_pause_active,
        legal_domain=state.legal_domain or "尚未确定",
        region=state.region or "尚未说明",
        input_events=json.dumps(state.input_events, ensure_ascii=False),
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=900),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_URGENCY,
            stage="guard_case",
        )
        payload = _json_payload(response.content)
    except Exception as exc:
        logger.warning("guard_case语义分类失败，使用确定性规则降级 | error={}", exc)
        return [], "", {}, str(exc)[:200]

    # Backward compatibility with the old CRITICAL/TIME/NORMAL contract.
    if "risks" not in payload and "urgency" in payload:
        urgency = str(payload.get("urgency") or "NORMAL").upper()
        safety_relevant = bool(payload.get("safety_relevant"))
        safety_status = str(payload.get("safety_status") or "").lower()
        if urgency == "CRITICAL" and not safety_status:
            safety_relevant = True
            safety_status = "danger"
        risks_payload: list[dict[str, Any]] = []
        if urgency == "CRITICAL":
            risks_payload.append(
                {
                    "risk_type": "personal_safety",
                    "level_candidate": (
                        "critical" if safety_status == "danger" else "unknown"
                    ),
                    "trigger": payload.get("reason") or context["current_input"],
                    "current_or_historical": (
                        "current" if safety_status == "danger" else "unknown"
                    ),
                    "confidence": 0.7,
                    "missing_conditions": [],
                    "source_refs": [context["source_ref"]],
                }
            )
        elif urgency == "TIME":
            risks_payload.append(
                {
                    "risk_type": "deadline",
                    "level_candidate": "warning",
                    "trigger": payload.get("time_clue") or context["current_input"],
                    "current_or_historical": "unknown",
                    "confidence": 0.6,
                    "missing_conditions": ["期限适用条件和起算时间"],
                    "source_refs": [context["source_ref"]],
                }
            )
        payload = {
            **payload,
            "risks": risks_payload,
            "safety_relevant": safety_relevant,
            "current_safety_status": safety_status or (
                "unknown" if safety_relevant else "not_applicable"
            ),
        }

    model_safety_status = str(
        payload.get("current_safety_status")
        or payload.get("safety_status")
        or ""
    ).lower()
    if model_safety_status not in {"danger", "safe", "unknown", "not_applicable"}:
        model_safety_status = ""

    candidates: list[dict[str, Any]] = []
    for item in payload.get("risks") or []:
        if not isinstance(item, dict):
            continue
        risk_type = str(item.get("risk_type") or "")
        level = str(item.get("level_candidate") or item.get("level") or "").lower()
        temporal = str(item.get("current_or_historical") or "unknown").lower()
        if risk_type not in RISK_TYPES or level not in RISK_LEVELS - {"clear"}:
            continue
        if risk_type in {"personal_safety", "custody_or_coercion"}:
            if temporal in {"historical", "quoted"}:
                level = "unknown" if model_safety_status == "unknown" else "warning"
            if level == "critical" and (
                temporal != "current" or model_safety_status != "danger"
            ):
                level = "unknown"
        if risk_type == "deadline" and level == "critical":
            level = "urgent"
        candidates.append(
            _risk(
                state,
                risk_type,
                level,
                str(item.get("trigger") or context["current_input"]),
                str(context["source_ref"]),
                missing_conditions=[
                    str(value)[:120]
                    for value in (item.get("missing_conditions") or [])
                    if str(value).strip()
                ],
                decision_source="semantic_classifier",
                temporal_status=temporal,
                confidence=float(item.get("confidence") or 0.0),
            )
        )
    return candidates, model_safety_status, payload, ""


def _merge_risks(
    deterministic: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer deterministic matches for the same risk type."""

    result: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, Any]] = {}
    for item in [*deterministic, *semantic]:
        risk_type = str(item["risk_type"])
        current = by_type.get(risk_type)
        if current is None:
            by_type[risk_type] = item
            result.append(item)
            continue
        deterministic_wins = (
            current.get("decision_source") == "deterministic_rule"
            and item.get("decision_source") != "deterministic_rule"
        )
        validated_current_upgrade = (
            current.get("level") == "unknown"
            and item.get("level") == "critical"
            and item.get("temporal_status") == "current"
        )
        if deterministic_wins and not validated_current_upgrade:
            continue
        if _LEVEL_WEIGHT[item["level"]] > _LEVEL_WEIGHT[current["level"]]:
            index = result.index(current)
            result[index] = item
            by_type[risk_type] = item
    return result


async def retrieve_guard_authorities(
    state: GuideState,
    risks: list[dict[str, Any]],
    deps: Any,
) -> dict[str, Any]:
    """Retrieve internal candidates without exposing unverified pinpoint citations."""

    retrieval_types = {
        item["risk_type"]
        for item in risks
        if item["risk_type"] in {"deadline", "evidence_loss", "asset_emergency"}
    }
    if not retrieval_types:
        return {"status": "not_required", "trace_id": "", "basis_refs": []}

    embedding_model = getattr(deps, "embedding_model", None)
    milvus_client = getattr(deps, "milvus_client", None)
    llm = getattr(deps, "llm", None)
    db_session = getattr(deps, "db_session", None)
    trace_id = _stable_id(
        "guard-retrieval",
        state.case_id,
        state.state_version,
        state.event_sequence,
        ",".join(sorted(retrieval_types)),
    )
    if embedding_model is None or milvus_client is None:
        return {
            "status": "required_but_deferred",
            "trace_id": trace_id,
            "basis_refs": [],
            "reason": "当前运行环境未提供法条检索依赖",
            "candidates": [],
        }

    query_parts = []
    if "deadline" in retrieval_types:
        query_parts.append(
            "法定期限 起算 送达 中止 中断 上诉 复议 仲裁 诉讼时效"
        )
    if "evidence_loss" in retrieval_types:
        query_parts.append("证据保全 电子数据 调查取证 诉前保全")
    if "asset_emergency" in retrieval_types:
        query_parts.append("财产保全 冻结异议 平台资金 申请条件")
    question = "；".join(query_parts)
    try:
        from src.agents.legal_knowledge.statute_rag import (
            _fetch_law_titles,
            search_statutes_raw,
        )

        hits = await asyncio.wait_for(
            search_statutes_raw(
                question=question,
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                top_k=12,
                rerank_top_k=5,
                domain=state.legal_domain or "",
                llm=llm,
                use_hyde=False,
                use_rrf=True,
            ),
            timeout=8,
        )
        titles = (
            await _fetch_law_titles(hits, db_session)
            if hits and db_session is not None
            else {}
        )
    except Exception as exc:
        logger.warning("guard_case权威依据检索失败，保持保守提示 | error={}", exc)
        return {
            "status": "unavailable",
            "trace_id": trace_id,
            "basis_refs": [],
            "reason": str(exc)[:200],
            "candidates": [],
        }

    candidates = []
    for hit in hits[:5]:
        candidates.append(
            {
                "law_id": str(hit.get("law_id") or ""),
                "title": titles.get(
                    str(hit.get("law_id") or ""),
                    f"法律ID:{hit.get('law_id') or '未知'}",
                ),
                "article_or_locator": str(hit.get("article_no") or ""),
                "domain": str(hit.get("domain") or ""),
                "score": float(hit.get("score") or 0.0),
                # The current statute index does not carry all publication/effect
                # metadata required by the guard citation gate.
                "needs_pinpoint": True,
                "visibility": "internal_candidate_only",
            }
        )
    return {
        "status": "candidates_only" if candidates else "no_hits",
        "trace_id": trace_id,
        "basis_refs": [],
        "reason": (
            "候选依据缺少完整效力、地域或官方回链元数据，不向用户展示精确结论"
            if candidates
            else "当前知识库未返回可用候选"
        ),
        "candidates": candidates,
    }


def resolve_guard_level(
    risks: list[dict[str, Any]],
    safety_status: str,
) -> tuple[str, bool]:
    if any(item["level"] == "critical" for item in risks):
        return "critical", True
    if safety_status == "unknown" and any(
        item["risk_type"] in {"personal_safety", "custody_or_coercion"}
        for item in risks
    ):
        return "unknown", True
    if any(item["level"] == "urgent" for item in risks):
        return "urgent", False
    if risks:
        return "warning", False
    return "clear", False


def build_immediate_actions(
    state: GuideState,
    risks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    templates: dict[str, list[tuple[str, str]]] = {
        "personal_safety": [
            ("立即", "在不增加风险的前提下离开现场，前往有人可求助的安全位置"),
            ("立即", "联系当地紧急服务或可信任的人；如在中国大陆可拨打110"),
            ("立即", "不要为了取证返回危险现场或与对方对峙"),
        ],
        "custody_or_coercion": [
            ("立即", "优先设法让可信任的人知道您的位置和当前处境"),
            ("立即", "在不增加风险的前提下联系当地紧急服务或法律援助"),
            ("立即", "不要在被威胁或无法理解内容时勉强签署文件"),
        ],
        "deadline": [
            ("今天", "保存决定书、通知书或其他文件的完整页面和送达信息"),
            ("尽快", "核对适用程序、起算事件和是否存在中止、中断或特殊期间"),
            ("尽快", "通过对应官方受理渠道确认期限和可采取的保护措施"),
        ],
        "evidence_loss": [
            ("今天", "向材料保管方提出书面保留请求，并保存发送或签收记录"),
            ("今天", "记录准确地点、日期、时间段和需要保留的内容范围"),
            ("尽快", "保存本人有权访问的原始记录、完整上下文和来源信息"),
        ],
        "asset_emergency": [
            ("尽快", "先通过银行、平台或作出措施的机构核对冻结或放行状态"),
            ("尽快", "保存通知、工单、账户状态和提交异议或申诉的记录"),
            ("核对后", "需要采取保全或异议程序时，先确认适用条件和受理机构"),
        ],
        "unlawful_collection": [
            ("立即停止", "不要入侵账号、破解设备、冒充他人或伪造、篡改材料"),
            ("改用合法方式", "保存本人有权访问的原始记录和完整上下文"),
            ("改用合法方式", "通过平台导出、书面申请或依法申请调查取证"),
        ],
        "dangerous_confrontation": [
            ("立即停止", "不要单独上门堵人、威胁对方或强行扣押财物"),
            ("今天", "改用书面沟通、平台投诉、调解或依法申请处理"),
            ("需要见面时", "优先选择公开场所并让可信任的人知情"),
        ],
    }
    actions: list[dict[str, Any]] = []
    priority = 1
    for risk in risks:
        for time_window, action in templates.get(risk["risk_type"], []):
            actions.append(
                {
                    "action_id": _stable_id(
                        "action", risk["risk_id"], priority, action
                    ),
                    "risk_id": risk["risk_id"],
                    "priority": priority,
                    "action": action,
                    "recommended_by": "guard_case",
                    "time_window": time_window,
                    "reason": risk["trigger"],
                    "basis_refs": list(risk.get("basis_refs") or []),
                    "requires_user_confirmation": risk["risk_type"]
                    in {"personal_safety", "custody_or_coercion"},
                }
            )
            priority += 1
    return actions


def build_guard_notice(
    status: str,
    risks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> str:
    if status == "clear":
        return ""
    if status == "critical":
        return (
            "## 请先确保现实安全\n\n"
            "您描述的情况可能正在危及人身安全，普通维权步骤先暂停。\n\n"
            "### 现在先做\n\n"
            "1. 在不增加风险的前提下离开现场，前往有人可求助的安全位置。\n"
            "2. 联系当地紧急服务或可信任的人；如您在中国大陆，可拨打 **110**。\n"
            "3. 不要为了取证返回危险现场或与对方对峙。\n\n"
            "安全后回复“我现在安全了”，本案件会从当前进度继续。"
        )
    if status == "unknown":
        return (
            "## 先确认一件事\n\n"
            "请告诉我：您现在是否已经脱离现场并处于安全位置？\n\n"
            "如果危险仍在，请先联系当地紧急服务或身边可信任的人。"
        )

    title = "## 先处理一项紧迫事项" if status == "urgent" else "## 先注意一项风险"
    risk_labels = {
        "deadline": "可能存在期限风险，但适用条件和起算时间仍需核对",
        "evidence_loss": "相关材料或数据可能很快灭失",
        "asset_emergency": "账户、平台款项或财产状态可能需要先核对和止损",
        "unlawful_collection": "拟采用的取证方式可能产生新的法律和证据风险",
        "dangerous_confrontation": "拟采取的当面对抗方式可能增加人身和法律风险",
        "personal_safety": "陈述涉及人身安全风险",
        "custody_or_coercion": "陈述涉及人身自由或受强迫风险",
    }
    lines = [title, "", "### 当前风险", ""]
    for risk in risks:
        lines.append(f"- {risk_labels.get(risk['risk_type'], risk['trigger'])}")
    visible_actions = actions[:6]
    if visible_actions:
        lines.extend(["", "### 建议先做", ""])
        for index, item in enumerate(visible_actions, start=1):
            lines.append(f"{index}. **{item['time_window']}**：{item['action']}。")
    if any(risk["risk_type"] == "deadline" for risk in risks):
        lines.extend(
            [
                "",
                "> 目前不会仅凭事情发生时间直接认定已经超过期限；适用条件不明时也不会给出确定的期限结论。",
            ]
        )
    lines.extend(["", "我会继续按您本轮提供的信息梳理案件。"])
    return "\n".join(lines)


def _active_risk_merge(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    resolve_safety: bool,
    resolved_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*previous, *current]:
        risk_type = str(item.get("risk_type") or "")
        if resolve_safety and risk_type in {"personal_safety", "custody_or_coercion"}:
            resolved.append(
                {
                    **item,
                    "status": "resolved",
                    "resolved_at": resolved_at,
                    "resolution_source": "user_confirmed_safe",
                }
            )
            continue
        key = str(item.get("risk_id") or f"{risk_type}:{item.get('trigger')}")
        if key in seen:
            continue
        seen.add(key)
        active.append(item)
    return active[-30:], resolved[-30:]


async def run_guard_case(state: GuideState, deps: Any) -> dict[str, Any]:
    """Execute deterministic rules, bounded semantics, grading and presentation."""

    checked_at = _now()
    llm = getattr(deps, "llm", None)
    context = collect_guard_context(state)
    deterministic_safety, safety_risks, safety_matches = (
        detect_deterministic_safety_risk(state, context)
    )
    other_risks, other_matches = detect_deterministic_non_safety_risks(
        state, context
    )
    deterministic_risks = [*safety_risks, *other_risks]

    model_risks: list[dict[str, Any]] = []
    model_safety = ""
    model_payload: dict[str, Any] = {}
    model_error = ""
    # Current danger must not wait for a model. Explicit safety remains authoritative,
    # while the model may still identify a separate deadline or preservation risk.
    if deterministic_safety != "danger":
        model_risks, model_safety, model_payload, model_error = (
            await classify_guard_risks(state, context, llm)
        )

    safety_status = deterministic_safety
    if safety_status == "not_applicable" and model_safety:
        safety_status = model_safety
    elif safety_status == "unknown" and model_safety == "danger":
        # A model cannot upgrade a historical/quoted deterministic observation unless
        # its validated candidate is explicitly current and critical.
        current_model_danger = any(
            item["risk_type"] in {"personal_safety", "custody_or_coercion"}
            and item["level"] == "critical"
            and item["temporal_status"] == "current"
            for item in model_risks
        )
        if current_model_danger:
            safety_status = "danger"
    if safety_status == "safe":
        model_risks = [
            item
            for item in model_risks
            if item["risk_type"] not in {"personal_safety", "custody_or_coercion"}
        ]

    risks = _merge_risks(deterministic_risks, model_risks)
    if safety_status == "danger" and not any(
        item["risk_type"] in {"personal_safety", "custody_or_coercion"}
        for item in risks
    ):
        risks.insert(
            0,
            _risk(
                state,
                "personal_safety",
                "critical",
                context["current_input"],
                context["source_ref"],
            ),
        )
    if safety_status == "unknown" and (
        state.safety_pause_active
        or _SAFETY_RELEVANT_PATTERN.search(context["current_input"])
    ) and not any(
        item["risk_type"] in {"personal_safety", "custody_or_coercion"}
        for item in risks
    ):
        risks.insert(
            0,
            _risk(
                state,
                "personal_safety",
                "unknown",
                context["current_input"],
                context["source_ref"],
                missing_conditions=["当前是否已经脱离现场并处于安全位置"],
            ),
        )

    status, pause_required = resolve_guard_level(risks, safety_status)
    retrieval_trace = await retrieve_guard_authorities(state, risks, deps)
    actions = build_immediate_actions(state, risks)
    notice = build_guard_notice(status, risks, actions)
    audit_id = _stable_id(
        "guard-audit",
        state.case_id,
        state.case_generation,
        state.state_version,
        state.event_sequence,
        checked_at,
    )
    notice_hash = (
        hashlib.sha256(notice.encode("utf-8")).hexdigest() if notice else ""
    )
    report = {
        "guard_status": status,
        "guard_checked_at": checked_at,
        "risks": risks,
        "current_safety_status": safety_status,
        "immediate_actions": actions,
        "user_notice_markdown": notice,
        "pause_required": pause_required,
        "next_route": state.requested_route or "update_facts",
        "route_after_guard": list(state.route_after_guard),
        "retrieval_trace_id": retrieval_trace.get("trace_id", ""),
        "guard_audit_id": audit_id,
        "degraded": bool(model_error),
    }
    audit = {
        "guard_audit_id": audit_id,
        "case_id": state.case_id,
        "case_generation": state.case_generation,
        "state_version": state.state_version,
        "event_id": state.current_message_id or f"event-{state.event_sequence}",
        "input_source_refs": [context["source_ref"]],
        "deterministic_matches": [*safety_matches, *other_matches],
        "model_candidate": model_risks,
        "model_error": model_error,
        "retrieval_trace_id": retrieval_trace.get("trace_id", ""),
        "basis_refs": [],
        "final_risks": risks,
        "guard_status": status,
        "immediate_actions": actions,
        "pause_decision": pause_required,
        "resume_route": list(state.route_after_guard),
        "user_notice_hash": notice_hash,
        "created_at": checked_at,
    }

    resolved_safety = state.safety_pause_active and safety_status == "safe"
    active, newly_resolved = _active_risk_merge(
        state.active_risk_flags,
        risks,
        resolve_safety=resolved_safety,
        resolved_at=checked_at,
    )
    resolved_flags = [*state.resolved_risk_flags, *newly_resolved][-50:]
    deadline_risk = next(
        (item for item in risks if item["risk_type"] == "deadline"),
        state.deadline_risk,
    )
    evidence_loss_risk = next(
        (item for item in risks if item["risk_type"] == "evidence_loss"),
        state.evidence_loss_risk,
    )
    asset_risk = next(
        (item for item in risks if item["risk_type"] == "asset_emergency"),
        state.asset_emergency_risk,
    )
    restricted = [
        item
        for item in risks
        if item["risk_type"] in {"unlawful_collection", "dangerous_confrontation"}
    ]
    risk_observations = [
        {
            "risk_id": item["risk_id"],
            "risk_type": item["risk_type"],
            "statement": item["trigger"],
            "source_refs": item["source_refs"],
            "certainty": "user_stated_or_model_candidate",
        }
        for item in risks
    ]
    missing_facts = [
        {
            "risk_id": item["risk_id"],
            "risk_type": item["risk_type"],
            "missing_conditions": item["missing_conditions"],
            "priority": item["level"],
        }
        for item in risks
        if item["missing_conditions"]
    ]
    updates: dict[str, Any] = {
        "guard_status": status,
        "guard_checked_at": checked_at,
        "guard_report": report,
        "guard_pause_required": pause_required,
        "guard_notice_markdown": notice,
        "guard_notice_pending": bool(notice and not pause_required),
        "guard_next_route": state.requested_route or "update_facts",
        "active_risk_flags": active,
        "resolved_risk_flags": resolved_flags,
        "guard_audit_history": [*state.guard_audit_history, audit][-50:],
        "risk_observations": risk_observations,
        "risk_related_missing_facts": missing_facts,
        "current_safety_status": safety_status,
        "safety_relevant": safety_status != "not_applicable",
        "safety_confirmation_required": status == "unknown",
        "deadline_risk": deadline_risk,
        "evidence_loss_risk": evidence_loss_risk,
        "asset_emergency_risk": asset_risk,
        "restricted_action_flags": restricted,
        "guard_retrieval_trace": {
            **retrieval_trace,
            "model_degraded": bool(model_error),
        },
        # Compatibility fields for the old graph, API debug and stored cases.
        "urgency_level": (
            "critical"
            if status == "critical"
            else "time"
            if any(item["risk_type"] == "deadline" for item in risks)
            else "normal"
        ),
        "time_warning": notice
        if any(item["risk_type"] == "deadline" for item in risks)
        else "",
    }

    if pause_required:
        pause_state = {
            "type": "safety",
            "pause_type": "safety",
            "pause_reason": (
                "current_personal_danger"
                if status == "critical"
                else "current_safety_unknown"
            ),
            "paused_at": state.safety_pause_started_at or checked_at,
            "paused_event_id": state.current_message_id
            or f"event-{state.event_sequence}",
            "pending_input_events": list(state.input_events),
            "resume_route": list(state.route_after_guard)
            or ([state.requested_route] if state.requested_route else []),
            "confirmation_required": "current_safety",
            "case_id": state.case_id,
            "case_generation": state.case_generation,
        }
        updates.update(
            {
                "safety_pause_active": True,
                "safety_pause_started_at": state.safety_pause_started_at
                or checked_at,
                "safety_pause_case_message": state.safety_pause_case_message
                or context["raw_input"],
                "safety_resume_route": list(state.route_after_guard)
                or ([state.requested_route] if state.requested_route else []),
                "safety_resume_stage": (
                    state.safety_resume_stage
                    or (
                        state.workflow_stage
                        if state.workflow_stage
                        not in {"risk_guard", "paused_for_safety"}
                        else "case_intake"
                    )
                ),
                "safety_pause_pending_events": list(state.input_events),
                "workflow_stage": "paused_for_safety",
                "pause_state": pause_state,
                "messages": [AIMessage(content=notice)],
            }
        )
    elif resolved_safety:
        updates.update(
            {
                "safety_pause_active": False,
                "safety_confirmation_required": False,
                "guard_pause_required": False,
                "workflow_stage": state.safety_resume_stage or "case_intake",
                "pause_state": None,
            }
        )
    else:
        updates["safety_pause_active"] = False

    return updates
