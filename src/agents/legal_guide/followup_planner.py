"""Dynamic, context-grounded planning for the existing follow-up node."""
from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.agents.legal_guide.case_model import (
    active_case_facts,
    format_case_context,
    latest_case_facts,
)
from src.agents.legal_guide.followup_catalog import (
    evidence_rule_resolved,
    fact_rule_resolved,
    get_domain_followups,
)
from src.agents.legal_guide.followup_policy import (
    rank_followup_candidates,
    score_dynamic_proposal,
)
from src.agents.legal_guide.evidence_analysis import coverage_for_rule
from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage
from src.core.config import get_settings


FOLLOWUP_PLANNER_PROMPT = """你是法律咨询工作流中的动态追问规划器，不是固定问卷生成器。

你的目标不是把字段问完，而是判断再问一个问题是否会实质改变以下任一判断：
责任主体、请求类型或金额、时效、管辖、程序路径、关键证据缺口、当前安全措施。

当前领域：{domain}
法律问题：{issues}
当前轮次：{turn}
已追问轮数：{ask_rounds}
体验软上限：{soft_ask_rounds}

结构化案情（每项都带用户原文；不得补写未出现的事实）：
{case_context}

用户称持有的材料：{evidence_present}
用户明确没有的材料：{evidence_unavailable}
已经问过的问题：
{asked_questions}

本轮检索到的真实法条候选（只能按编号引用，不得自行编造法条）：
{law_sources}

应用层已经选出的最高价值决策维度（不得改选其他维度；你的职责是结合本案自然改写问题）：
{decision_hints}

请判断该决策维度能否被自然、准确地询问，并生成一个问题。规则：
1. 已在结构化案情或材料中出现的信息不得重复询问；用户刚说的话必须在 acknowledgement 中自然承接。
2. question 只服务于一个法律决策目标，而不是机械限制为一个字段。可以在一个自然句中组合
   2-3 个紧密相关、用户通常能一起回答的要素（例如交易对象、商品和金额），但只能有一个问号，
   不能把责任、时效、证据、诉求等不同决策目标拼成表单。
3. 行为事实、用户主张和法律结论必须分开；不得把任何尚未核验的事实直接写成违法、违约、侵权、犯罪或责任成立。
4. 若继续追问不会明显改变方案，should_ask=false。不要为了填满字段而追问。
5. basis_kind="law" 时 law_index 必须对应上方真实法条编号；否则用 basis_kind="official_elements"，表示问题来自官方办事/示范文本要素。
6. acknowledgement 只复述已提供内容，不表达胜诉判断，不超过80字。
7. decision_key 应描述法律决策点而非具体行业，例如 counterparty_identity、claim_scope、limitation_start、proof_of_payment。
8. decision_effects 只能从 responsibility、claim_scope、limitation、jurisdiction、procedure、evidence_gap、safety 中选择；不要在 reason、answer_hint 或 question 中写法律名称、条号或自行解释法条。
9. 每个候选维度都带 coverage，其中 known 是本案已经说清的内容，missing 才是仍可追问的内容。question 只能询问 missing，不能把 known 换一种说法再问一次。
10. question 必须带入至少一个本案具体锚点，例如当事人、商品/服务、金额、地点、时间或具体行为；不要直接照抄 seed_question 中的“商品或服务”“发生了什么”等通用占位表达。
11. contextual_reason 用一句自然语言解释为什么此刻问这个问题，必须承接本案具体事实，不得写法律结论或法条。acknowledgement、question、contextual_reason 连起来应像同一段真实对话，而不是三段表单文案。
12. ask_type=evidence 时，question 必须询问某项材料是否存在、是否保留或能否找到；不能把它改成询问付款方式、行为方式等新的事实问题。
13. contextual_reason 不得擅自认定“属于违法/不符合标准/构成侵权或犯罪”，也不要假设对方一定会否认、拒绝或抗辩。
14. 候选排序、信息增益、用户负担和是否超过追问阈值由应用程序决定；不得自行给这些项目打分。

只输出 JSON：
{{
  "should_ask": true,
  "ask_type": "facts或evidence",
  "decision_key": "稳定的英文语义键",
  "candidate_id": "采用的候选维度ID；完全动态生成时可为空",
  "question": "一个自然、贴合上下文的问题",
  "reason": "该答案会怎样影响法律判断",
  "contextual_reason": "结合本案已知事实说明为什么此刻问它",
  "answer_hint": "用户不知道时怎样轻松回答",
  "decision_effects": ["procedure"],
  "acknowledgement": "自然承接本轮新增信息",
  "acknowledged_fact_keys": ["实际承接的案情事实key"],
  "basis_kind": "law或official_elements",
  "law_index": 0
}}"""


class FollowupPlanProposal(BaseModel):
    """Untrusted planner output before application-level policy checks."""

    should_ask: bool = False
    ask_type: str = "facts"
    decision_key: str = ""
    # Models commonly encode an optional identifier as JSON null. An omitted
    # catalog id still represents a valid fully dynamic proposal.
    candidate_id: str | None = ""
    question: str = ""
    reason: str = ""
    contextual_reason: str = ""
    answer_hint: str = ""
    decision_effects: list[str] = Field(default_factory=list)
    acknowledgement: str = ""
    acknowledged_fact_keys: list[str] = Field(default_factory=list)
    basis_kind: str = "official_elements"
    law_index: int = -1


def _json_content(value: str) -> dict[str, Any]:
    content = str(value or "").strip()
    if "```" in content:
        content = content.split("```", 2)[1].lstrip("json").strip()
    return json.loads(content)


_CATEGORY_DIMENSIONS = {
    "actor": {"actor", "counterparty"},
    "relationship": {"relationship", "counterparty"},
    "event": {"event"},
    "claim": {"claim"},
    "amount": {"amount"},
    "time": {"time"},
    "location": {"location"},
    "evidence": {"evidence"},
    "procedure": {"procedure"},
    "harm": {"harm"},
}
_KEY_DIMENSIONS = {
    "transaction": "transaction", "payment": "amount", "paid": "amount",
    "price": "amount", "amount": "amount", "total": "amount",
    "balance": "amount", "loss": "amount", "counterparty": "counterparty",
    "merchant": "counterparty", "operator": "counterparty",
    "employer": "counterparty", "landlord": "counterparty",
    "institution": "counterparty", "item": "subject_matter",
    "product": "subject_matter", "service": "subject_matter",
    "food": "subject_matter", "problem": "event", "defect": "event",
    "issue": "event", "incident": "event", "contamination": "event",
    "closure": "event", "response": "procedure", "date": "time",
    "time": "time", "discovery": "time", "claim": "claim",
    "request": "claim", "remedy": "claim", "negotiation": "procedure",
    "complaint": "procedure", "report": "procedure", "injury": "harm",
    "damage": "harm", "harm": "harm", "relationship": "relationship",
    "safety": "safety", "danger": "safety", "threat": "safety",
    "contract": "relationship", "employment": "relationship",
    "location": "location", "address": "location", "region": "location",
}
_DIMENSION_LABELS = {
    "actor": "相关主体", "counterparty": "对方或经营者身份",
    "relationship": "双方关系", "transaction": "具体消费或交易",
    "subject_matter": "涉及的商品或服务", "amount": "涉及金额",
    "event": "具体问题或行为", "time": "发现或发生时间",
    "location": "发生地点", "claim": "希望解决的结果",
    "procedure": "此前沟通或处理经过", "evidence": "相关材料",
    "harm": "造成的损失或影响", "safety": "当前是否安全",
}
_SLOT_ALIASES = {
    "administrative_action": (("event",),),
    "agreement": (("relationship",),),
    "children": (("relationship",),),
    "employment_status": (("relationship",),),
    "event": (("event",),),
    "event_and_liability": (("event",), ("procedure",)),
    "event_and_urgency": (("event",),),
    "current_safety": (("safety",),),
    "event_time": (("event", "harm"), ("time",)),
    "harm": (("harm",),),
    "infringement": (("event",),),
    "insurance_and_claim": (("relationship",), ("claim",)),
    "legal_relationship": (("event",), ("relationship", "counterparty", "actor")),
    "procedure": (("procedure",),),
    "property_and_safety": (("harm", "event"),),
    "right_type": (("relationship", "subject_matter"),),
    "source_and_harm": (("counterparty", "actor"), ("harm",)),
    "transaction": (("amount",), ("counterparty", "relationship", "transaction", "location")),
    "claim": (("claim",),),
}

_DIMENSION_FOLLOWUP_QUESTIONS = {
    "actor": "这件事主要涉及哪些人或单位？",
    "counterparty": "对方是什么身份，例如个人、公司还是政府机构？",
    "relationship": "您和对方是什么关系，例如买卖、借贷、劳动或租赁关系？",
    "transaction": "这笔交易具体买了什么或约定了什么服务？",
    "subject_matter": "争议涉及的商品、服务或其他标的具体是什么？",
    "amount": "这件事涉及多少钱，已经实际支付或损失了多少？",
    "event": "对方具体做了什么，当前争议结果是什么？",
    "time": "事情大约是什么时候发生的？",
    "location": "事情发生在哪里，或者通过哪个平台办理？",
    "claim": "您现在最希望对方怎么处理？",
    "procedure": "您此前是否已经联系、投诉、报警或申请处理，结果怎样？",
    "evidence": "目前有哪些能够反映事情经过的材料？",
    "harm": "这件事目前给您造成了哪些实际损失或影响？",
    "safety": "您现在是否安全？",
}
_DIMENSION_ANSWER_HINTS = {
    "actor": "简单说明涉及的人或单位即可，不清楚可以直接说“不清楚”。",
    "counterparty": "简单说对方是个人、公司或其他机构即可。",
    "relationship": "简单说明您和对方是什么关系即可，不确定可以说“不清楚”。",
    "transaction": "简单说购买或约定的内容即可。",
    "subject_matter": "说出商品、服务或其他争议对象即可。",
    "amount": "可以说准确金额，也可以说大概数。",
    "event": "按先后顺序简单说最关键的行为和结果即可。",
    "time": "记不清具体日期时，说大概月份或时间范围即可。",
    "location": "说出线下地点或线上平台即可。",
    "claim": "直接说您最希望实现的结果即可。",
    "procedure": "没有处理过也可以直接说“没有”。",
    "evidence": "有、没有或暂时找不到都可以直接说。",
    "harm": "可以只说目前已经发生的实际损失或影响。",
    "safety": "如果危险仍在，先说“现在有危险”即可。",
}


def _focused_candidate_copy(candidate: dict[str, Any] | None) -> tuple[str, str]:
    """Render only the still-missing part of a partially covered decision."""

    coverage = (candidate or {}).get("coverage") or {}
    known = list(coverage.get("known_dimension_keys") or [])
    missing = list(coverage.get("missing_dimension_keys") or [])
    if not known or not missing:
        return "", ""
    dimension = missing[0]
    return (
        _DIMENSION_FOLLOWUP_QUESTIONS.get(dimension, ""),
        _DIMENSION_ANSWER_HINTS.get(dimension, ""),
    )


def _case_dimension_context(state: Any) -> tuple[set[str], dict[str, list[str]]]:
    dimensions: set[str] = set()
    statements: dict[str, list[str]] = {}
    for item in active_case_facts(getattr(state, "case_facts", []) or []):
        if item.get("status") not in {"asserted", "uncertain"}:
            continue
        item_dimensions = set(_CATEGORY_DIMENSIONS.get(str(item.get("category") or ""), set()))
        key_tokens = re.split(r"[._:-]+", str(item.get("key") or "").lower())
        item_dimensions.update(
            _KEY_DIMENSIONS[token] for token in key_tokens if token in _KEY_DIMENSIONS
        )
        statement = " ".join(str(item.get("statement") or "").split())
        for dimension in item_dimensions:
            dimensions.add(dimension)
            if statement and statement not in statements.setdefault(dimension, []):
                statements[dimension].append(statement)
    if getattr(state, "time_info", ""):
        dimensions.add("time")
        statements.setdefault("time", []).append(str(state.time_info))
    if getattr(state, "region", ""):
        dimensions.add("location")
        statements.setdefault("location", []).append(str(state.region))
    if getattr(state, "current_safety_status", "not_applicable") in {"safe", "danger"}:
        dimensions.add("safety")
        statements.setdefault("safety", []).append(
            "当前安全"
            if getattr(state, "current_safety_status", "") == "safe"
            else "当前有危险"
        )
    return dimensions, statements


def candidate_coverage(slot: str, state: Any) -> dict[str, list[str]]:
    """Return known and missing semantic dimensions for a catalog slot."""
    dimensions, statements = _case_dimension_context(state)
    requirements = _SLOT_ALIASES.get(slot, ())
    known_dimensions: list[str] = []
    missing_dimensions: list[str] = []
    for alternatives in requirements:
        matched = next((item for item in alternatives if item in dimensions), "")
        if matched:
            known_dimensions.append(matched)
        else:
            missing_dimensions.append(alternatives[0])
    relevant = []
    for dimension in known_dimensions:
        for statement in statements.get(dimension, [])[-2:]:
            if statement not in relevant:
                relevant.append(statement)
    if not relevant:
        relevant = [
            str(item.get("statement") or "")
            for item in active_case_facts(getattr(state, "case_facts", []) or [])[-2:]
            if item.get("statement")
        ]
    return {
        "known": relevant[-3:],
        "missing": [_DIMENSION_LABELS.get(item, item) for item in missing_dimensions],
        "known_dimension_keys": known_dimensions,
        "missing_dimension_keys": missing_dimensions,
    }


def build_followup_candidates(state: Any) -> tuple[list[dict[str, Any]], Any]:
    """Build unresolved catalog candidates from structured case state."""
    domain_rules = get_domain_followups(getattr(state, "legal_domain", ""))
    asked = set(getattr(state, "asked_followup_ids", []) or [])
    asked.update(getattr(state, "asked_decision_keys", []) or [])
    rows: list[dict[str, Any]] = []
    for rule in domain_rules.facts:
        if rule.slot == "current_safety" and not getattr(state, "safety_relevant", False):
            continue
        coverage = candidate_coverage(rule.slot, state)
        resolved_by_dimensions = bool(_SLOT_ALIASES.get(rule.slot)) and not coverage["missing"]
        if rule.id not in asked and not fact_rule_resolved(rule, state) and not resolved_by_dimensions:
            rows.append({
                "id": rule.id, "kind": "facts", "decision_dimension": rule.slot,
                "seed_question": rule.question, "legal_effect": rule.why,
                "low_burden_hint": rule.answer_hint, "coverage": coverage,
                "priority": rule.priority,
            })
    known_evidence = list(getattr(state, "evidence_confirmed", []) or []) + list(
        getattr(state, "evidence_unavailable", []) or []
    )
    for rule in domain_rules.evidence:
        resolved = evidence_rule_resolved(rule, known_evidence)
        if rule.id not in asked and not resolved:
            rows.append({
                "id": rule.id, "kind": "evidence", "decision_dimension": rule.evidence_key,
                "seed_question": rule.question, "legal_effect": rule.purpose,
                "alternatives": rule.alternatives,
                "priority": rule.priority,
                "coverage": {
                    "known": known_evidence[-3:],
                    "missing": [rule.item],
                },
            })
            continue
        coverage = coverage_for_rule(
            getattr(state, "evidence_coverage", {}) or {},
            rule.id,
        )
        if (
            rule.id not in asked
            and resolved
            and coverage
            and coverage.status == "partially_covered"
            and coverage.quality_gaps
        ):
            rows.append({
                "id": rule.id,
                "kind": "evidence",
                "decision_dimension": rule.evidence_key,
                "seed_question": (
                    f"您提到已有{rule.item}，请确认原始载体是否还在、内容是否完整，"
                    "以及能否看清相关主体和形成时间？"
                ),
                "legal_effect": "判断该材料的具体证明范围和优先补强方向",
                "alternatives": rule.alternatives,
                "priority": rule.priority + 1,
                "low_burden_hint": "不确定的项目可以直接说“不清楚”。",
                "coverage": {
                    "known": [
                        str(item.get("name") or "")
                        for item in (
                            getattr(state, "evidence_items", []) or []
                        )
                        if isinstance(item, dict) and item.get("name")
                    ][-3:],
                    "missing": coverage.quality_gaps[:3],
                },
                "evaluation_mode": "quality",
            })
    return rows, domain_rules.source


_DECISION_EFFECT_LABELS = {
    "responsibility": "判断责任主体和责任范围",
    "claim_scope": "确定请求类型和范围",
    "limitation": "判断时效和关键时间节点",
    "jurisdiction": "判断受理机构和管辖",
    "procedure": "判断下一步处理程序",
    "evidence_gap": "识别当前最关键的证明缺口",
    "safety": "判断是否需要优先采取安全措施",
}
_FREE_TEXT_LEGAL_CLAIM = re.compile(
    r"《[^》]+》|第[零〇一二三四五六七八九十百千万两\d]+条|"
    r"(?:法律|法规|条例|司法解释).{0,8}(?:规定|明确|要求)"
)


def _trusted_explanation(
    plan: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> tuple[str, str]:
    """Keep legal explanation outside the model's free-text trust boundary."""
    if candidate:
        hint = " ".join(str(plan.get("answer_hint") or "").split())[:180]
        if (
            not hint
            or _FREE_TEXT_LEGAL_CLAIM.search(hint)
            or any(marker in hint for marker in (
                "如果", "或者只想", "市场监管", "公安", "报警", "起诉", "仲裁",
                "法院", "行政处罚", "肯定", "一定",
            ))
        ):
            hint = str(candidate.get("low_burden_hint") or "")
        if "；" in hint:
            hint = hint.split("；", 1)[0].rstrip("。") + "。"
        reason = str(candidate.get("legal_effect") or "判断下一步处理方式")
        for prefix in ("为了用于", "用于", "为了"):
            if reason.startswith(prefix):
                reason = reason[len(prefix):].strip()
                break
        return (
            reason,
            hint,
        )
    effects = [
        _DECISION_EFFECT_LABELS[item]
        for item in (plan.get("decision_effects") or [])
        if item in _DECISION_EFFECT_LABELS
    ]
    reason = "、".join(dict.fromkeys(effects)) or "判断下一步处理方式"
    hint = " ".join(str(plan.get("answer_hint") or "").split())[:180]
    if _FREE_TEXT_LEGAL_CLAIM.search(hint):
        hint = ""
    return reason, hint


def _safe_contextual_reason(plan: dict[str, Any], state: Any) -> str:
    """Allow conversational context, while keeping legal claims outside model control."""
    value = " ".join(str(plan.get("contextual_reason") or "").split())[:180]
    if not value or _FREE_TEXT_LEGAL_CLAIM.search(value):
        return ""
    if any(marker in value for marker in (
        "一定胜诉", "必然", "已经构成", "构成违法", "构成违约", "构成侵权",
        "构成犯罪", "属于违法", "属于违约", "属于侵权", "属于犯罪",
        "不符合食品安全标准", "依法必须", "依法应当", "有权要求", "肯定可以",
        "通常可以主张", "可以主张", "能够主张", "有权主张", "可以获赔",
        "能够获赔", "十倍赔偿", "三倍赔偿", "赔偿范围",
        "对方一定会", "对方可能会", "对方可能否认", "对方会否认", "对方会拒绝",
        "对方可能拒绝", "对方会抗辩", "对方可能抗辩",
        "最直接的依据", "最关键的依据", "就能证明", "足以证明", "证据已经完整",
        "补充责任", "连带责任", "共同责任",
        "最直接", "如果没有", "如果没", "万一", "非常困难", "很难让",
        "不承认", "否认", "拒绝", "没有任何凭证", "没有任何材料",
        "治安报案", "刑事立案", "难以立案", "不能立案", "不予立案",
        "警察可以", "警方可以", "公安可以",
    )):
        return ""
    unsupported_denial = any(
        marker in value
        for marker in ("没有拍", "没拍", "没有保留", "没保留", "没有留", "没留", "未拍", "未保留")
    )
    denied_evidence_exists = bool(getattr(state, "evidence_unavailable", []) or []) or any(
        item.get("category") == "evidence" and item.get("status") == "denied"
        for item in active_case_facts(getattr(state, "case_facts", []) or [])
    )
    if unsupported_denial and not denied_evidence_exists:
        return ""
    return value.rstrip("。；")


def _law_rows(state: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in (getattr(state, "retrieved_law_refs", []) or [])[:8]:
        if isinstance(item, dict):
            result.append({
                "title": str(item.get("title") or "").strip(),
                "article_no": str(item.get("article_no") or "").strip(),
                "text": str(item.get("text") or "").strip()[:500],
            })
    return result


def _question_is_duplicate(question: str, asked: list[str]) -> bool:
    normalized = re.sub(r"\W+", "", question)
    if not normalized:
        return True
    for previous in asked:
        old = re.sub(r"\W+", "", str(previous or ""))
        if old and (old in normalized or normalized in old or SequenceMatcher(None, old, normalized).ratio() >= 0.76):
            return True
    return False


def _single_question(question: str) -> str:
    """Keep one decision objective while allowing related fields in one sentence."""
    compact = " ".join(str(question or "").split())
    marks = [index for index, char in enumerate(compact) if char in "？?"]
    if len(marks) <= 1:
        return compact
    return compact[: marks[0] + 1].strip()


def _stop_plan(
    mode: str,
    *,
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed instead of falling back to the first catalog question."""
    result: dict[str, Any] = {"should_ask": False, "planner_mode": mode}
    if decision_trace:
        result["decision_trace"] = decision_trace
    return result


def _policy_trace(
    *,
    mode: str,
    scores: list[Any],
    selected_id: str = "",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "selected_candidate_id": selected_id,
        "candidates": [item.model_dump() for item in scores[:8]],
    }


def _grounded_acknowledgement(state: Any, requested_keys: list[str]) -> tuple[str, list[str]]:
    """Render continuity from current-turn provenance, never from free text."""
    latest = latest_case_facts(
        getattr(state, "case_facts", []), getattr(state, "round", 0)
    )
    requested = set(requested_keys)
    selected = [item for item in latest if item.get("key") in requested]
    if not selected:
        selected = latest[:2]
    quotes: list[str] = []
    keys: list[str] = []
    for item in selected:
        quote = " ".join(str(item.get("source_text") or "").split())
        key = str(item.get("key") or "")
        if quote and quote not in quotes:
            quotes.append(quote[:80])
        if key and key not in keys:
            keys.append(key)
    if not quotes:
        return "", []
    return f"您刚补充的“{'；'.join(quotes)}”我已经记下", keys


def _selected_candidate_fallback(
    *,
    state: Any,
    candidate: dict[str, Any] | None,
    score: Any,
    scores: list[Any],
    source: Any,
    mode: str,
) -> dict[str, Any]:
    """Use the application-owned catalog question when model phrasing fails."""

    if not candidate or not score:
        return _stop_plan(
            mode,
            decision_trace=_policy_trace(
                mode=mode,
                scores=scores,
                selected_id="",
            ),
        )
    focused_question, focused_hint = _focused_candidate_copy(candidate)
    question = _single_question(
        focused_question or str(candidate.get("seed_question") or "").strip()
    )
    if not question:
        return _stop_plan(
            mode,
            decision_trace=_policy_trace(
                mode=mode,
                scores=scores,
                selected_id=str(candidate.get("id") or ""),
            ),
        )
    if "？" not in question and "?" not in question:
        question += "？"
    acknowledgement, acknowledged = _grounded_acknowledgement(state, [])
    reason = str(candidate.get("legal_effect") or "判断下一步处理方式")
    for prefix in ("为了用于", "用于", "为了"):
        if reason.startswith(prefix):
            reason = reason[len(prefix):].strip()
            break
    return {
        "should_ask": True,
        "ask_type": str(candidate.get("kind") or "facts"),
        "decision_key": str(candidate.get("id") or ""),
        "candidate_id": str(candidate.get("id") or ""),
        "question": question,
        "reason": reason,
        "contextual_reason": "",
        "answer_hint": str(
            focused_hint
            or candidate.get("low_burden_hint")
            or "不确定时可以直接说“不清楚”。"
        ),
        "acknowledgement": acknowledgement,
        "acknowledged_fact_keys": acknowledged,
        "basis_kind": "official_elements",
        "law_index": -1,
        "law_source": {},
        "official_source": source.model_dump(),
        "information_gain": score.information_gain,
        "user_burden": score.user_burden,
        "policy_score": score.net_score,
        "decision_effects": score.decision_effects,
        "decision_trace": _policy_trace(
            mode=mode,
            scores=scores,
            selected_id=str(candidate.get("id") or ""),
        ),
        "planner_mode": mode,
    }


async def plan_next_followup(state: Any, llm: Any) -> dict[str, Any]:
    """Choose and phrase the single highest-value next question."""
    candidates, source = build_followup_candidates(state)
    if not candidates and not getattr(state, "allow_extra_followups", False):
        return _stop_plan("no_candidates")
    policy_scores = rank_followup_candidates(candidates, state)
    selected_score = next((item for item in policy_scores if item.eligible), None)
    selected_candidate = next(
        (
            item for item in candidates
            if selected_score and item["id"] == selected_score.candidate_id
        ),
        None,
    )
    if candidates and selected_candidate is None:
        return _stop_plan(
            "policy_no_eligible_candidate",
            decision_trace=_policy_trace(
                mode="policy_no_eligible_candidate",
                scores=policy_scores,
            ),
        )
    law_rows = _law_rows(state)
    asked = list(getattr(state, "asked_details", []) or [])
    settings = get_settings()
    safety_gate = next(
        (item for item in candidates if item.get("decision_dimension") == "current_safety"),
        None,
    )
    if (
        safety_gate
        and getattr(state, "safety_relevant", False)
        and getattr(state, "current_safety_status", "not_applicable") == "unknown"
    ):
        acknowledgement, acknowledged = _grounded_acknowledgement(state, [])
        return {
            "should_ask": True,
            "ask_type": "facts",
            "decision_key": "current_safety",
            "candidate_id": safety_gate["id"],
            "question": str(safety_gate.get("seed_question") or "您现在是否安全？"),
            "reason": str(safety_gate.get("legal_effect") or "判断是否需要优先采取安全措施"),
            "contextual_reason": "",
            "answer_hint": str(safety_gate.get("low_burden_hint") or "如果危险仍在，先说“现在有危险”即可。"),
            "acknowledgement": acknowledgement,
            "acknowledged_fact_keys": acknowledged,
            "basis_kind": "official_elements",
            "law_index": -1,
            "law_source": {},
            "official_source": source.model_dump(),
            "information_gain": 1.0,
            "user_burden": 0.05,
            "policy_score": 1.0,
            "decision_effects": ["safety"],
            "decision_trace": _policy_trace(
                mode="mandatory_safety_gate",
                scores=policy_scores,
                selected_id=safety_gate["id"],
            ),
            "planner_mode": "mandatory_safety_gate",
        }
    prompt = FOLLOWUP_PLANNER_PROMPT.format(
        domain=getattr(state, "legal_domain", "") or "other",
        issues="；".join(getattr(state, "confirmed_issues", []) or []) or "尚未稳定归类",
        turn=getattr(state, "round", 0),
        ask_rounds=getattr(state, "ask_rounds", 0),
        soft_ask_rounds=settings.GUIDE_SOFT_ASK_ROUNDS,
        case_context=format_case_context(getattr(state, "case_facts", []) or []),
        evidence_present="；".join(getattr(state, "evidence_confirmed", []) or []) or "暂无",
        evidence_unavailable="；".join(getattr(state, "evidence_unavailable", []) or []) or "暂无",
        asked_questions="\n".join(f"- {item}" for item in asked) or "- 暂无",
        law_sources=json.dumps(law_rows, ensure_ascii=False, indent=2),
        decision_hints=json.dumps(
            [selected_candidate] if selected_candidate else [],
            ensure_ascii=False,
            indent=2,
        ),
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=700),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_FOLLOWUP,
            stage="followup_planner",
        )
        proposal = FollowupPlanProposal.model_validate(_json_content(response.content))
        plan = proposal.model_dump()
    except Exception as exc:
        logger.warning("动态追问表达失败，回退到确定性题库问题: {}", exc)
        return _selected_candidate_fallback(
            state=state,
            candidate=selected_candidate,
            score=selected_score,
            scores=policy_scores,
            source=source,
            mode="deterministic_fallback_planner_error",
        )

    if not bool(plan.get("should_ask")):
        return _selected_candidate_fallback(
            state=state,
            candidate=selected_candidate,
            score=selected_score,
            scores=policy_scores,
            source=source,
            mode="deterministic_fallback_model_declined",
        )

    proposed_candidate_id = str(plan.get("candidate_id") or "").strip()
    if selected_candidate:
        if proposed_candidate_id and proposed_candidate_id != selected_candidate["id"]:
            return _selected_candidate_fallback(
                state=state,
                candidate=selected_candidate,
                score=selected_score,
                scores=policy_scores,
                source=source,
                mode="deterministic_fallback_model_changed_candidate",
            )
        candidate_id = selected_candidate["id"]
        policy_score = selected_score
    else:
        candidate_id = ""
        policy_score = score_dynamic_proposal(
            decision_effects=list(plan.get("decision_effects") or []),
            ask_type=str(plan.get("ask_type") or "facts"),
            state=state,
        )
        if not policy_score.eligible:
            return _stop_plan(
                "policy_rejected_dynamic",
                decision_trace={
                    "mode": "policy_rejected_dynamic",
                    "selected_candidate_id": "",
                    "candidates": [policy_score.model_dump()],
                },
            )

    raw_question = " ".join(str(plan.get("question") or "").split())
    question = _single_question(raw_question)
    if question != raw_question:
        logger.info("动态追问包含多个问题，保留首个中心问题 | raw={}", raw_question)
    focused_question, focused_hint = _focused_candidate_copy(selected_candidate)
    if focused_question:
        question = focused_question
        plan["answer_hint"] = focused_hint
    ask_type = str(plan.get("ask_type") or "facts")
    if ask_type == "evidence" and selected_candidate and not any(
        marker in question
        for marker in (
            "有", "保留", "保存", "还在", "找到", "提供", "记录", "材料",
            "凭证", "截图", "照片", "视频", "原件", "收据", "发票", "订单", "回执",
        )
    ):
        logger.info("证据追问偏离材料存在性，回退到权威题库问题 | raw={}", question)
        question = str(selected_candidate.get("seed_question") or question).strip()
    decision_key = re.sub(
        r"[^a-zA-Z0-9_.:-]+", "_", str(plan.get("decision_key") or "")
    ).strip("_")
    asked_decision_keys = set(getattr(state, "asked_decision_keys", []) or [])
    if (
        ask_type not in {"facts", "evidence"} or not question
        or _question_is_duplicate(question, asked)
        or (decision_key and decision_key in asked_decision_keys)
        or _FREE_TEXT_LEGAL_CLAIM.search(question)
    ):
        logger.warning("动态追问计划未通过结构校验，按现有信息收敛 | plan={}", plan)
        if selected_candidate:
            return _selected_candidate_fallback(
                state=state,
                candidate=selected_candidate,
                score=selected_score,
                scores=policy_scores,
                source=source,
                mode="deterministic_fallback_invalid_expression",
            )
        return _stop_plan(
            "policy_rejected",
            decision_trace=_policy_trace(
                mode="policy_rejected",
                scores=policy_scores,
                selected_id=candidate_id,
            ),
        )
    if "？" not in question and "?" not in question:
        question += "？"

    acknowledgement, acknowledged = _grounded_acknowledgement(
        state, plan.get("acknowledged_fact_keys") or []
    )

    law_index = int(plan.get("law_index") or 0)
    basis_kind = str(plan.get("basis_kind") or "official_elements")
    if basis_kind != "law" or law_index < 0 or law_index >= len(law_rows):
        basis_kind, law_index = "official_elements", -1
    if not decision_key:
        decision_key = candidate_id or "dynamic." + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    reason, answer_hint = _trusted_explanation(plan, selected_candidate)
    contextual_reason = _safe_contextual_reason(plan, state)
    return {
        "should_ask": True, "ask_type": ask_type, "decision_key": decision_key,
        "candidate_id": candidate_id, "question": question,
        "reason": reason,
        "contextual_reason": contextual_reason,
        "answer_hint": answer_hint,
        "acknowledgement": acknowledgement, "acknowledged_fact_keys": acknowledged,
        "basis_kind": basis_kind, "law_index": law_index,
        "law_source": law_rows[law_index] if law_index >= 0 else {},
        "official_source": source.model_dump(),
        "information_gain": policy_score.information_gain,
        "user_burden": policy_score.user_burden,
        "policy_score": policy_score.net_score,
        "decision_effects": policy_score.decision_effects,
        "decision_trace": _policy_trace(
            mode="deterministic_policy",
            scores=policy_scores if policy_scores else [policy_score],
            selected_id=candidate_id,
        ),
        "planner_mode": "deterministic_policy",
    }


def format_followup_authority(plan: dict[str, Any]) -> str:
    reason = str(plan.get("reason") or "判断下一步处理方式").strip().rstrip("。；")
    if plan.get("basis_kind") == "law" and plan.get("law_source"):
        law = plan["law_source"]
        title = law.get("title") or "本轮检索法条"
        article = law.get("article_no") or "相关规定"
        return f"追问依据：结合本轮检索到的《{title}》{article}，这个信息用于{reason}；它不是要求您必须提交的固定材料。"
    source = plan.get("official_source") or {}
    title, issuer, url = source.get("title") or "通用案情整理规则", source.get("issuer") or "", source.get("url") or ""
    label = f"[{title}]({url})" if url else title
    prefix = f"{issuer}发布的" if issuer else ""
    return f"追问依据：参考{prefix}{label}中的办理或示范文本要素，用于{reason}；它不是官方固定问卷。"
