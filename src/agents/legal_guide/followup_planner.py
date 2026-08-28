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
    fact_followups,
    fact_rule_resolved,
    get_domain_followups,
)
from src.agents.legal_guide.followup_policy import (
    candidate_decision_effects,
    rank_followup_candidates,
    score_dynamic_proposal,
)
from src.agents.legal_guide.evidence_analysis import (
    case_context_text,
    coverage_for_rule,
    evidence_rule_relevant,
)
from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage
from src.core.config import get_settings


FOLLOWUP_PLANNER_PROMPT = """你是法律咨询工作流中的动态追问规划器，不是固定问卷生成器。

你的目标不是把字段问完，而是判断再问一个问题是否会实质改变以下任一判断：
责任主体、请求类型或金额、时效、管辖、程序路径、关键证据缺口、当前安全措施、实际场景归属。
不要重复询问结构化案情已经明确回答的内容；只问真正缺失或影响结论的细节。

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
8. decision_effects 只能从 responsibility、claim_scope、limitation、jurisdiction、procedure、evidence_gap、safety、scenario 中选择；不要在 reason、answer_hint 或 question 中写法律名称、条号或自行解释法条。
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


def _fixed_rule_answered(rule: Any, state: Any) -> bool:
    """答案终结性：固定阶段一次展示即算已问；有/没有/不清楚都终结该题。

    与 `fact_rule_resolved` 的区别在于 `unknown`（用户不清楚）也算已答——
    终结性保证"不清楚"不会被固定阶段或后续动态阶段反复盘问。
    """
    rule_id = str(getattr(rule, "id", "") or "")
    if not rule_id:
        return True
    if rule_id in set(getattr(state, "asked_followup_ids", []) or []):
        return True
    record = (getattr(state, "fact_records", {}) or {}).get(rule_id) or {}
    # ambiguous/conflicted 仍需澄清，不因终结性被当作已答。
    if record.get("status") not in {None, "ambiguous", "conflicted"}:
        return True
    if bool(_SLOT_ALIASES.get(getattr(rule, "slot", ""))) and not candidate_coverage(
        getattr(rule, "slot", ""), state
    )["missing_dimension_keys"]:
        return True
    if fact_rule_resolved(rule, state):
        return True
    return False


def remaining_fixed_rules(state: Any) -> list[Any]:
    """固定阶段剩余必问事实规则，按目录优先级排序，不含证据规则。"""
    domain = str(getattr(state, "legal_domain", "") or "other")
    safety_relevant = getattr(state, "safety_relevant", False)
    return [
        rule
        for rule in fact_followups(domain)
        if not (rule.slot == "current_safety" and not safety_relevant)
        and not _fixed_rule_answered(rule, state)
    ]


def _slot_input_type(slot: str, question: str) -> tuple[str, list[str]]:
    """按问题性质给出输入控件类型（时间/金额→short_text，经过/损失→long_text，是/否→single_choice）。

    模型改写失败时该启发式是权威兜底；可枚举的选择题优先由模型提出 options，
    这里对非 yes/no 槽位一律回退文本框，避免"单选题配错选项"的体验。
    """
    question = str(question or "")
    if slot == "current_safety":
        return "single_choice", ["我现在安全", "仍有现实危险", "无法确认"]
    if slot == "procedure":
        return "single_choice", ["是/已经处理", "否/尚未处理", "不清楚/无法确认"]
    if slot in {"employment_status", "children", "property_and_safety", "agreement", "insurance_and_claim"}:
        return "single_choice", ["是", "否", "不清楚/无法确认"]
    if slot in {"event_time", "region", "transaction", "administrative_action", "right_type"}:
        return "short_text", []
    if slot in {"event", "harm", "infringement", "source_and_harm", "event_and_liability", "event_and_urgency"}:
        return "long_text", []
    # 通用 yes/no 判定：题干含是否/有没有/吗，且不是叙述性描述
    choice_markers = ("是否", "有没有", "是不是", "吗", "签了", "签过", "写过", "办过")
    narrative_markers = ("经过", "过程", "怎么", "如何", "内容", "损失", "情况", "发生了什么")
    if any(marker in question for marker in choice_markers) and not any(
        marker in question for marker in narrative_markers
    ):
        return "single_choice", ["是", "否", "不清楚/无法确认"]
    return "short_text", []


def _dimension_unknown_only(dimension: dict[str, Any], state: Any) -> bool:
    """该决策维度剩余缺口是否都已按"不清楚/不知道"答过（终结性收敛）。"""
    unresolved_rule_ids = list(dimension.get("unresolved_rule_ids") or [])
    if not unresolved_rule_ids:
        return False
    records = getattr(state, "fact_records", {}) or {}
    return all(
        (records.get(str(rule_id)) or {}).get("status") == "unknown"
        for rule_id in unresolved_rule_ids
    )


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
        if rule.id not in asked and not _fixed_rule_answered(rule, state):
            rows.append({
                "id": rule.id, "kind": "facts", "decision_dimension": rule.slot,
                "seed_question": rule.question, "legal_effect": rule.why,
                "low_burden_hint": rule.answer_hint, "coverage": coverage,
                "priority": rule.priority,
            })
    known_evidence = list(getattr(state, "evidence_confirmed", []) or []) + list(
        getattr(state, "evidence_unavailable", []) or []
    )
    context_text = case_context_text(state)
    for rule in domain_rules.evidence:
        # 与本案细节库不相关的证据点不追问（与评估/收敛/渲染同一判定）。
        if not evidence_rule_relevant(rule, context_text):
            continue
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
                "seed_question": coverage.seed_question,
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
    "scenario": "确认最接近的实际场景",
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


def _question_punctuation(question: str) -> str:
    """Return one clean Chinese question mark without combinations such as `。？`."""

    compact = " ".join(str(question or "").split()).strip()
    if not compact:
        return ""
    if compact.endswith(("？", "?")):
        stem = re.sub(r"[。；，、：:]+$", "", compact[:-1].rstrip())
        return stem + "？"
    return re.sub(r"[。；，、：:]+$", "", compact) + "？"


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


BATCH_FOLLOWUP_PROMPT = """你是法律咨询工作流中的动态批量追问规划器。

你必须根据本案已经入库的事实、尚未解决的法律决策维度和本轮真实检索依据，实时生成本轮表单；
不是从固定问卷挑题，也不得询问已经出现、已经否认、明确不知道或已经问过的信息。

结构化事实库：
{case_context}

尚未解决的决策维度（固定问卷全部覆盖后此列表可能为空）：
{unresolved_dimensions}

已经问过的决策键：{asked_keys}
已经问过的问题：
{asked_questions}

本轮追问阶段检索依据（不含类案；只能按数组下标引用）：
{basis_rows}

先判断本轮是否还需要追问。请像执业律师一样，把上方**每条检索依据**拆成构成要件
（主体、行为、结果、情节、期限、前置程序等），与结构化事实库**逐条联合盘查**，再下结论
（不要因为某个方向在固定问卷里没有就跳过）：
0. **要件 × 事实联合盘查（核心，先做这一步）**：对每条检索依据的每个要件，判定它目前
   对用户是**有利 / 不利 / 未知**，追问的目的是”撑住有利要件、争取不利要件、补足未知要件”：
   - **有利要件**：是否已有事实支撑？若用户还能补强（锁定对方身份、保留记录、补证明、
     留下协商记录），必须问。
   - **不利要件**：用户能否通过陈述或行动争取有利解释（谁先动手、是否知情、是否留存记录、
     是否有可引用的例外或从轻情节）？能 → 必须问。
   - **未知要件**：用户能陈述 → 必须问。
   例：故意伤害/寻衅滋事——“谁先动手”通常决定责任，必须先问；”是否持械、是否多人”是不利
   情节，若用户能澄清必须问；”现场是否有监控或证人”是补强有利要件的线索，必须问。
   消费类要确认购买时间、金额、沟通记录；欠薪类要确认劳动合同与在职/离职状态。
1. **要件补全**：把上方每条检索依据拆成构成要件，逐一对照结构化事实库；
   只要存在”用户能陈述、能补足某要件”的事实缺口，就必须问。
2. **证据补强**：找出当前最可能落空的环节（如无法锁定对方身份、关键金额无凭据、关键承诺只有口头），
   检查是否有可补强的线索——目击者、现场监控、第三方在场、通话/转账/聊天记录、现场照片——
   以及用户是否掌握或能尽快调取。必须问。
3. **时间敏感**：识别会随时间消失或失效的证据（监控覆盖、聊天记录可删、伤情自愈、记忆模糊），
   在对应问题里提示用户尽快调取/保存。
4. **前提风险**：识别用户乐观预期或本方案依赖的未证实前提（如”有车牌/账号就能找到人””对方会承认”），
   若存在能验证或推翻该前提的事实，必须问。
5. **自身风险**：从办案机关/对方视角审视用户自己的行为是否可能使其成为被追责方
   （是否也动了手、谁先动手、对方伤情、是否参与、行为是否可能违约/侵权/构成犯罪要件）。
   若存在能改变自身责任认定的事实缺口，必须问。

只有当”有利要件已撑住、不利要件没有可争取的补救空间、未知要件用户也无法陈述”、
且上述其他缺口都明确没有高价值信息时，才返回 should_ask=false 与空 fields；
不要硬凑问题，也不得遗漏可能改变定性的细节。

需要追问时，请一次生成 2 至 {max_fields} 个当前信息增益最高、彼此不重复的事实问题；确实只剩一个高价值缺口时可只生成一个。
问题之间可以混合以下展示类型：
- short_text：金额、日期、地点、主体名称等简短确定值；
- long_text：经过、沟通内容、损失等需要叙述的信息；
- single_choice：只能有一个状态成立；
- multi_choice：多个状态可以同时成立。

规则：
1. 只问用户能够陈述的行为事实，不让用户判断违法、违约、侵权、犯罪、责任或证据效力。
2. 每题只影响一个主要决策目标；不同目标拆成不同字段。
3. choice 类型必须给 2 至 6 个互不重叠的案情化选项；不得把法律结论当选项。
4. 每题都允许用户回答“不清楚/无法确认”，required 必须为 false。
5. question 必须带入本案具体主体、交易、行为、时间或地点锚点，不能照抄通用问卷。
6. basis_indices 只能引用上方真实依据；没有直接对应依据时留空，不能编造。
7. decision_effects 只能使用 responsibility、claim_scope、limitation、jurisdiction、procedure、safety、scenario。
8. field_id 使用稳定英文语义键；不得与已经问过的决策键重复。
9. 不询问证据是否持有。证据需求由事实变化在后台增量生成，事实收敛后集中展示。
10. “用户希望实现的结果”和“此前联系、投诉、报警或协商的经过”必须拆成两个字段，禁止一题同时承担 claim_scope 和 procedure。

只输出 JSON：
{{
  "should_ask": true,
  "fields": [
    {{
      "field_id": "transaction_total",
      "question": "本案具体问题",
      "input_type": "short_text",
      "options": [],
      "placeholder": "填写提示",
      "answer_hint": "不知道时如何回答",
      "decision_effects": ["claim_scope"],
      "basis_indices": [0],
      "acknowledged_fact_keys": []
    }}
  ]
}}"""


class DynamicFollowupFieldProposal(BaseModel):
    field_id: str = ""
    question: str = ""
    input_type: str = "long_text"
    options: list[str] = Field(default_factory=list)
    placeholder: str = ""
    answer_hint: str = ""
    decision_effects: list[str] = Field(default_factory=list)
    basis_indices: list[int] = Field(default_factory=list)
    acknowledged_fact_keys: list[str] = Field(default_factory=list)
    required: bool = False


ADVERSARIAL_GAP_PROMPT = """你是法律咨询工作流中的二次审视审阅员。

主审员判定当前没有更多需要追问的问题，准备收敛出方案。请你换一个独立视角——
站在**对方当事人**和**办案/受理机关**的立场——回顾以下信息，找出主审员可能
漏掉、但会实质影响结果的细节缺口。你的职责不是重复主审员的结论，而是补位挑刺。

结构化事实库：
{case_context}

本轮检索依据：
{basis_rows}

已经问过的问题：
{asked_questions}

请对上方**每条检索依据的构成要件**过一遍，双向挑刺：
1. **对用户不利、但主审没问的**：对方/办案机关最可能用哪个要件反驳用户？用户能否通过
   陈述补救（谁先动手、是否知情、是否有例外）？能 → 必须问。
2. **对用户有利、但主审没用上的**：哪条法条要件或事实还没被用户说出来、说出来就能增强
   维权地位（可另行主张的请求项、可申请的减免、可补强的证明）？用户能陈述 → 必须问。

只提出真正高价值、且与已问问题不重复的缺口问题；若确实没有，返回 should_ask=false 与空 fields。

只输出 JSON：
{{
  "should_ask": true,
  "fields": [
    {{
      "field_id": "stable_english_key",
      "question": "从对方/办案机关视角最可能被问住的具体问题",
      "input_type": "short_text",
      "options": [],
      "placeholder": "填写提示",
      "answer_hint": "不知道时如何回答",
      "decision_effects": ["responsibility"],
      "basis_indices": []
    }}
  ]
}}"""


class DynamicFollowupBatchProposal(BaseModel):
    should_ask: bool = False
    fields: list[DynamicFollowupFieldProposal] = Field(default_factory=list)


_ALLOWED_INPUT_TYPES = {"short_text", "long_text", "single_choice", "multi_choice"}
_ALLOWED_BATCH_EFFECTS = {
    "responsibility", "claim_scope", "limitation", "jurisdiction",
    "procedure", "safety", "scenario",
}


def _batch_reason(effects: list[str]) -> str:
    labels = [
        _DECISION_EFFECT_LABELS[item]
        for item in effects
        if item in _DECISION_EFFECT_LABELS
    ]
    return "、".join(dict.fromkeys(labels)) or "判断下一步处理方式"


async def _batch_converge_or_adversarial(
    state: Any, llm: Any, max_fields: int, catalog_dims_pending: bool
) -> dict[str, Any]:
    """主审无题可出（或 LLM 失败）时：目录维度仍待补先回退目录，否则二次对抗审视。

    目录维度待补 → 先回退目录（确定性、省一次 LLM）；固定阶段已全部覆盖 →
    收敛前跑一次对方/办案机关视角的二次审视，抓到主审漏掉的本案特有缺口；
    两者都无结果才以 fact_dimensions_converged 收敛。
    """
    if catalog_dims_pending:
        fallback = _batch_fallback_fields(state, max_fields)
        if fallback:
            return {
                "should_ask": True,
                "plan_kind": "followup_form",
                "ask_type": "facts",
                "questions": fallback,
                "planner_mode": "catalog_fallback_batch",
            }
    adversarial = await _adversarial_gap_scan(state, llm, max_fields)
    if adversarial:
        return adversarial
    return _stop_plan("fact_dimensions_converged")


async def _adversarial_gap_scan(
    state: Any, llm: Any, max_fields: int
) -> dict[str, Any] | None:
    """收敛前二次审视：从对方/办案机关视角挑刺，抓本案特有缺口。

    返回可直接展示的动态批次计划；无新缺口或 LLM 失败时返回 None（由调用方收敛）。
    与主审共用同一套字段过滤与去重（_build_batch_fields），不会重复已问问题。
    """
    basis_rows = list(getattr(state, "followup_basis_refs", []) or [])[:8]
    asked_questions = list(getattr(state, "asked_details", []) or [])
    prompt = ADVERSARIAL_GAP_PROMPT.format(
        case_context=format_case_context(getattr(state, "case_facts", []) or []),
        basis_rows=json.dumps(basis_rows, ensure_ascii=False, indent=2),
        asked_questions="\n".join(f"- {item}" for item in asked_questions) or "- 暂无",
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=1200),
            [SystemMessage(content=prompt)],
            timeout=get_settings().GUIDE_LLM_TIMEOUT_FOLLOWUP,
            stage="followup_adversarial_scan",
        )
        proposal = DynamicFollowupBatchProposal.model_validate(_json_content(response.content))
        if not proposal.should_ask or not proposal.fields:
            return None
        fields = _build_batch_fields(proposal, state, basis_rows, max_fields)
    except Exception as exc:
        logger.warning("二次对抗审视失败，按主审结论收敛: {}", exc)
        return None
    if not fields:
        return None
    return {
        "should_ask": True,
        "plan_kind": "followup_form",
        "ask_type": "facts",
        "questions": fields,
        "planner_mode": "adversarial_retrieval_batch",
    }


def _build_batch_fields(
    proposal: DynamicFollowupBatchProposal,
    state: Any,
    basis_rows: list[dict[str, Any]],
    max_fields: int,
) -> list[dict[str, Any]]:
    """把模型提议的批量字段规整为可展示的动态表单（主审与二次审视共用）。

    规整：去重已问键、去重重复问题、拦截自由文本法律断言、选择类补齐
    "不清楚/无法确认"兜底、校验 basis_indices 引用、按 effect 广度去重。
    """
    settings = get_settings()
    asked_questions = list(getattr(state, "asked_details", []) or [])
    asked_keys = set(getattr(state, "asked_decision_keys", []) or [])
    fields: list[dict[str, Any]] = []
    used_effects: set[str] = set()
    for raw in proposal.fields[: max_fields * 2]:
        field_id = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", raw.field_id).strip("_.:-")[:100]
        question = " ".join(raw.question.split())[:300]
        input_type = raw.input_type if raw.input_type in _ALLOWED_INPUT_TYPES else "long_text"
        effects = [item for item in dict.fromkeys(raw.decision_effects) if item in _ALLOWED_BATCH_EFFECTS]
        if (
            not field_id or field_id in asked_keys or not question or not effects
            or {"claim_scope", "procedure"}.issubset(set(effects))
            or _question_is_duplicate(question, asked_questions + [item["question"] for item in fields])
            or _FREE_TEXT_LEGAL_CLAIM.search(question)
        ):
            continue
        # Prefer breadth across independent legal decisions within one compact batch.
        if used_effects and set(effects).issubset(used_effects) and len(fields) >= int(settings.GUIDE_FOLLOWUP_BATCH_MIN):
            continue
        options = [" ".join(str(item).split())[:80] for item in raw.options if str(item).strip()]
        options = list(dict.fromkeys(options))[:6]
        if input_type in {"single_choice", "multi_choice"}:
            if len(options) < 2:
                input_type, options = "long_text", []
            elif not any("不清楚" in item or "无法确认" in item for item in options):
                options = (options + ["不清楚/无法确认"])[:6]
        else:
            options = []
        basis_indices = [
            index for index in dict.fromkeys(raw.basis_indices)
            if isinstance(index, int) and 0 <= index < len(basis_rows)
        ]
        basis = [basis_rows[index] for index in basis_indices[:3]]
        fields.append({
            "field_id": field_id,
            "candidate_id": "",
            "question": _question_punctuation(question),
            "input_type": input_type,
            "options": options,
            "placeholder": " ".join(raw.placeholder.split())[:120] or "请按您知道的情况填写",
            "answer_hint": " ".join(raw.answer_hint.split())[:160] or "不清楚时可填写“不清楚”。",
            "required": False,
            "decision_effects": effects,
            "reason": _batch_reason(effects),
            "basis_refs": basis,
            "official_source": {},
            "acknowledged_fact_keys": raw.acknowledged_fact_keys[:6],
        })
        used_effects.update(effects)
        asked_keys.add(field_id)
        if len(fields) >= max_fields:
            break
    return fields


def _batch_fallback_fields(state: Any, limit: int) -> list[dict[str, Any]]:
    """Last-resort compatibility fallback; normal batches are model-generated."""
    candidates, source = build_followup_candidates(state)
    fallback_basis = [
        item for item in (getattr(state, "followup_basis_refs", []) or [])
        if isinstance(item, dict) and item.get("text")
    ][:2]
    scores = rank_followup_candidates(
        [item for item in candidates if item.get("kind") == "facts"],
        state,
    )
    by_id = {str(item.get("id") or ""): item for item in candidates}
    fields: list[dict[str, Any]] = []
    for score in scores:
        if not score.eligible or len(fields) >= limit:
            continue
        if {"claim_scope", "procedure"}.issubset(set(score.decision_effects)):
            continue
        candidate = by_id.get(score.candidate_id)
        if not candidate:
            continue
        question, hint = _focused_candidate_copy(candidate)
        question = _single_question(question or candidate.get("seed_question") or "")
        if not question or _question_is_duplicate(question, list(getattr(state, "asked_details", []) or [])):
            continue
        question = _question_punctuation(question)
        field_id = str(candidate.get("id") or "").strip()
        input_type = "long_text"
        options: list[str] = []
        if "是否" in question:
            input_type = "single_choice"
            options = ["是/已经处理", "否/尚未处理", "不清楚/无法确认"]
        fields.append({
            "field_id": field_id,
            "candidate_id": field_id,
            "question": question,
            "input_type": input_type,
            "options": options,
            "placeholder": hint or "请按您知道的情况填写",
            "answer_hint": hint or "不清楚时可填写“不清楚”。",
            "required": False,
            "decision_effects": score.decision_effects,
            "reason": _batch_reason(score.decision_effects),
            "basis_refs": fallback_basis,
            "official_source": source.model_dump(),
            "information_gain": score.information_gain,
            "policy_score": score.net_score,
        })
    return fields


async def plan_followup_batch(state: Any, llm: Any) -> dict[str, Any]:
    """动态批量追问：优先补目录维度缺口；目录维度全部覆盖后仍运行检索驱动缺口扫描。

    收敛由 LLM 基于当前案情与检索法条判断——不再因"目录维度已满足"提前收敛，
    避免固定阶段答完就跳过"根据法条再追问"的动态补充。
    """
    settings = get_settings()
    max_fields = max(1, int(settings.GUIDE_FOLLOWUP_BATCH_MAX))
    sufficiency = getattr(state, "decision_sufficiency", {}) or {}
    unresolved = [
        item for item in (sufficiency.get("dimensions") or [])
        if isinstance(item, dict)
        and not item.get("satisfied")
        and item.get("effect") != "evidence_gap"
        and not _dimension_unknown_only(item, state)
    ]
    # 不再因"目录维度已满足"提前收敛：固定阶段全部覆盖后仍会进入检索驱动缺口扫描，
    # 由 LLM 基于当前案情与检索法条判断是否还有场景特有的关键缺口。
    catalog_dims_pending = bool(unresolved)

    safety_gap = next((item for item in unresolved if item.get("effect") == "safety"), None)
    if safety_gap and getattr(state, "safety_relevant", False):
        return {
            "should_ask": True,
            "plan_kind": "followup_form",
            "ask_type": "facts",
            "questions": [{
                "field_id": "current_safety",
                "candidate_id": "current_safety",
                "question": "您现在是否已经脱离现场并处于安全位置？",
                "input_type": "single_choice",
                "options": ["我现在安全", "仍有现实危险", "无法确认"],
                "placeholder": "",
                "answer_hint": "仍有危险时请先联系 110 或身边可信任的人。",
                "required": False,
                "decision_effects": ["safety"],
                "reason": _batch_reason(["safety"]),
                "basis_refs": [],
                "official_source": {},
            }],
            "planner_mode": "mandatory_safety_batch",
        }

    basis_rows = list(getattr(state, "followup_basis_refs", []) or [])[:8]
    prompt = BATCH_FOLLOWUP_PROMPT.format(
        case_context=format_case_context(getattr(state, "case_facts", []) or []),
        unresolved_dimensions=json.dumps(unresolved, ensure_ascii=False, indent=2),
        asked_keys=json.dumps(list(getattr(state, "asked_decision_keys", []) or []), ensure_ascii=False),
        asked_questions="\n".join(
            f"- {item}" for item in (getattr(state, "asked_details", []) or [])
        ) or "- 暂无",
        basis_rows=json.dumps(basis_rows, ensure_ascii=False, indent=2),
        max_fields=max_fields,
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=1800),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_FOLLOWUP,
            stage="followup_batch_planner",
        )
        proposal = DynamicFollowupBatchProposal.model_validate(_json_content(response.content))
        if not proposal.should_ask or not proposal.fields:
            # 缺口扫描判定无更多高价值缺口：收敛前先跑二次对抗审视（目录维度仍待补时先回退目录）。
            return await _batch_converge_or_adversarial(state, llm, max_fields, catalog_dims_pending)
    except Exception as exc:
        logger.warning("动态批量追问生成失败，使用兼容兜底: {}", exc)
        return await _batch_converge_or_adversarial(state, llm, max_fields, catalog_dims_pending)

    fields = _build_batch_fields(proposal, state, basis_rows, max_fields)
    if not fields:
        return await _batch_converge_or_adversarial(state, llm, max_fields, catalog_dims_pending)
    return {
        "should_ask": True,
        "plan_kind": "followup_form",
        "ask_type": "facts",
        "questions": fields,
        "planner_mode": "dynamic_retrieval_batch",
    }


FIXED_FOLLOWUP_PROMPT = """你是法律咨询工作流中的固定阶段追问改写器，不是新题生成器。

不要重复询问结构化案情已经明确回答的内容；只问真正缺失或影响结论的细节。

当前领域：{domain}
法律问题：{issues}
当前轮次：{turn}

结构化案情（每项都带用户原文；不得补写未出现的事实）：
{case_context}

本轮需要核对的必问事项（每项都必须在 output 中恰好出现一次，不得增删、合并、改名或换顺序）：
{fixed_rules}

你的唯一职责：把每个 rule_id 的 question 结合本案具体事实改写得更自然、更贴合上下文，
但必须保留它的语义和覆盖范围。禁止引入法条、条号或法律结论，禁止把选择题语义改成文本框。

规则：
1. 必须覆盖下方列出的全部 N 个 rule_id；缺一不可，不得新增规则、不得合并规则、不得改动 id。
2. 每个字段对应一个 rule_id；question 只能改写措辞，不得变成另一个问题。
3. 每个字段都可以用单选/多选/文本框；选择类必须包含"不清楚/无法确认"选项，required 一律 false。
4. 只问用户能陈述的行为事实，不让用户判断违法、违约、侵权、犯罪、责任或证据效力。
5. field_id 使用对应的 rule_id；不得改写。
6. 不询问证据是否持有；证据需求在事实收敛后单独集中展示。
7. question 必须带入至少一个本案具体锚点（当事人、商品/服务、金额、地点、时间或具体行为）。

只输出 JSON：
{{
  "should_ask": true,
  "fields": [
    {{
      "field_id": "catalog 中的 id（即规则 id）",
      "question": "结合本案的改写后问题",
      "input_type": "single_choice",
      "options": ["...", "不清楚/无法确认"],
      "placeholder": "填写提示",
      "answer_hint": "不知道时如何回答"
    }}
  ]
}}"""


def _fixed_catalog_field(rule: Any, state: Any, source: Any) -> dict[str, Any]:
    """固定阶段字段：以目录原文为权威内容，模型改写仅在覆盖之后微调措辞。"""
    rule_id = str(getattr(rule, "id", "") or "")
    slot = str(getattr(rule, "slot", "") or "")
    question = _question_punctuation(str(getattr(rule, "question", "") or ""))
    input_type, options = _slot_input_type(slot, question)
    effects = candidate_decision_effects({"kind": "facts", "decision_dimension": slot})
    reason = " ".join(str(getattr(rule, "why", "") or "").split())
    for prefix in ("为了用于", "用于", "为了"):
        if reason.startswith(prefix):
            reason = reason[len(prefix):].strip()
            break
    hint = str(getattr(rule, "answer_hint", "") or "").strip()
    basis_refs: list[dict[str, Any]] = []
    if source is not None:
        basis_refs = [{
            "source_type": "official_process",
            "title": str(getattr(source, "title", "") or "通用案情整理规则"),
            "article_no": "",
            "text": str(getattr(source, "usage_note", "") or "")[:500],
            "issuer": str(getattr(source, "issuer", "") or ""),
            "url": str(getattr(source, "url", "") or ""),
        }]
    return {
        "field_id": rule_id,
        "candidate_id": rule_id,
        "question": question,
        "input_type": input_type,
        "options": options,
        "placeholder": hint or "请按您知道的情况填写",
        "answer_hint": hint or "不清楚时可填写“不清楚”。",
        "required": False,
        "decision_effects": effects,
        "reason": reason or _batch_reason(effects),
        "basis_refs": basis_refs,
        "official_source": source.model_dump() if source is not None else {},
        "is_fixed_rule": True,
    }


def _apply_fixed_rewrite(
    field: dict[str, Any],
    rewrite: dict[str, Any],
) -> dict[str, Any]:
    """把模型改写结果应用到固定字段，带白名单校验；失败时保留字段原样。"""
    rewritten_question = " ".join(str(rewrite.get("question") or "").split())
    if (
        not rewritten_question
        or _FREE_TEXT_LEGAL_CLAIM.search(rewritten_question)
        or _question_is_duplicate(rewritten_question, [str(field["question"])])
    ):
        return field
    field["question"] = _question_punctuation(rewritten_question)
    proposed_type = str(rewrite.get("input_type") or "").strip()
    if proposed_type in _ALLOWED_INPUT_TYPES:
        options = [
            str(item).strip()[:80]
            for item in (rewrite.get("options") or []) if str(item).strip()
        ]
        options = list(dict.fromkeys(options))[:6]
        if proposed_type in {"single_choice", "multi_choice"}:
            if len(options) < 2:
                options = []
            elif not any("不清楚" in item or "无法确认" in item for item in options):
                options = (options + ["不清楚/无法确认"])[:6]
            field["input_type"] = proposed_type
            field["options"] = options
        else:
            field["input_type"] = proposed_type
            field["options"] = []
    placeholder = " ".join(str(rewrite.get("placeholder") or "").split())
    if placeholder:
        field["placeholder"] = placeholder[:120]
    answer_hint = " ".join(str(rewrite.get("answer_hint") or "").split())
    if answer_hint:
        field["answer_hint"] = answer_hint[:160]
    return field


async def plan_fixed_batch(state: Any, llm: Any) -> dict[str, Any]:
    """固定阶段：一次性规划领域必问事实表单，模型仅改写措辞，覆盖优先。

    返回结构与动态批量一致（plan_kind=followup_form），由同一前端表单渲染。
    planner_mode：fixed_catalog_batch（模型改写）/ fixed_catalog_fallback（回退原文）。
    安全类问题（current_safety）永远使用目录原文，不经过模型改写。
    """
    settings = get_settings()
    if not settings.GUIDE_FIXED_STAGE_ENABLED:
        return _stop_plan("fixed_stage_disabled")
    rules = remaining_fixed_rules(state)
    if not rules:
        return _stop_plan("fixed_facts_done")
    domain = str(getattr(state, "legal_domain", "") or "other")
    source = get_domain_followups(domain).source
    # 安全类问题禁止模型改写，防止措辞被软化；其余规则允许结合案情改写措辞。
    safety_ids = {
        str(rule.id) for rule in rules if rule.slot == "current_safety"
    }
    model_rules = [rule for rule in rules if str(rule.id) not in safety_ids]
    rule_specs = [
        {
            "rule_id": rule.id,
            "question": rule.question,
            "why": rule.why,
            "answer_hint": rule.answer_hint,
        }
        for rule in model_rules
    ]
    rewritten: dict[str, dict[str, Any]] = {}
    if model_rules:
        prompt = FIXED_FOLLOWUP_PROMPT.format(
            domain=domain,
            issues="；".join(getattr(state, "confirmed_issues", []) or []) or "尚未稳定归类",
            turn=getattr(state, "round", 0),
            case_context=format_case_context(getattr(state, "case_facts", []) or []),
            fixed_rules=json.dumps(rule_specs, ensure_ascii=False, indent=2),
        )
        try:
            response = await ainvoke_bounded(
                llm_for_stage(llm, max_tokens=1800),
                [SystemMessage(content=prompt)],
                timeout=settings.GUIDE_LLM_TIMEOUT_FOLLOWUP,
                stage="followup_fixed_planner",
            )
            proposal = DynamicFollowupBatchProposal.model_validate(
                _json_content(response.content)
            )
            if proposal.should_ask:
                model_rule_ids = {rule.id for rule in model_rules}
                for raw in proposal.fields:
                    rule_id = str(raw.field_id or "").strip()
                    if rule_id not in model_rule_ids:
                        continue
                    rewritten[rule_id] = {
                        "question": " ".join(str(raw.question).split())[:300],
                        "input_type": raw.input_type,
                        "options": [
                            str(item).strip()[:80]
                            for item in raw.options if str(item).strip()
                        ],
                        "placeholder": " ".join(str(raw.placeholder).split()),
                        "answer_hint": " ".join(str(raw.answer_hint).split()),
                    }
        except Exception as exc:
            logger.warning("固定表单改写失败，回退目录原文: {}", exc)

    fields: list[dict[str, Any]] = []
    rewritten_count = 0
    for rule in rules:
        field = _fixed_catalog_field(rule, state, source)
        if str(rule.id) in safety_ids:
            fields.append(field)
            continue
        model_field = rewritten.get(str(rule.id))
        if model_field:
            fields.append(_apply_fixed_rewrite(field, model_field))
            rewritten_count += 1
        else:
            fields.append(field)
    mode = (
        "fixed_catalog_batch"
        if rewritten_count == len(model_rules)
        else "fixed_catalog_fallback"
    )
    return {
        "should_ask": True,
        "plan_kind": "followup_form",
        "ask_type": "facts",
        "questions": fields,
        "planner_mode": mode,
    }
