"""公民法律指引 LangGraph 状态机。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from difflib import SequenceMatcher
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pymilvus import MilvusClient

from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings
from src.agents.legal_guide.state import GuideState, GuidePhase
from src.agents.legal_guide.issue_normalizer import (
    _deterministic_case_frame,
    is_high_precision_fraud_report,
    normalize_legal_issues,
)
from src.agents.legal_guide.neo4j_queries import query_laws_and_channels
from src.agents.legal_guide.convergence import should_conclude
from src.agents.legal_guide.decision_sufficiency import (
    DecisionSufficiencyReport,
    assess_decision_sufficiency,
    unresolved_decision_summary,
)
from src.agents.legal_guide.confidence import score_confidence, tier_guidance
from src.agents.legal_guide.db_queries import (
    load_user_context,
    query_recommended_channels,
    save_guide_record,
)
from src.agents.legal_guide.channel_catalog import (
    extract_supported_region,
    normalize_region_name,
)
from src.agents.legal_guide.formatters import fmt_channels, fmt_evidence_checklist
from src.agents.legal_guide.case_model import (
    _is_control_only_text,
    active_case_facts,
    evidence_from_case_facts,
    format_case_context,
    legacy_fact_updates,
    latest_case_facts,
    reduce_case_facts,
)
from src.agents.legal_guide.followup_planner import (
    candidate_coverage,
    format_followup_authority,
    plan_fixed_batch,
    plan_followup_batch,
    plan_next_followup,
    _json_content,
)
from src.agents.legal_guide.retrieval_query import build_case_retrieval_inputs
from src.agents.legal_guide.authority_registry import format_domain_authority_summary
from src.agents.legal_guide.evidence_rules import (
    format_evidence_source,
    resolve_state_evidence_checklist,
)
from src.agents.legal_guide.evidence_analysis import (
    EvidenceEvaluationReport,
    evaluate_state_evidence,
    evidence_decay_banner,
    format_evidence_coverage,
    inspect_uploaded_evidence_blocks,
    merge_evidence_observations,
    merge_evidence_requirements,
    normalize_evidence_observations,
    split_uploaded_evidence_blocks,
)
from src.agents.legal_guide.llm_runtime import (
    ainvoke_bounded,
    build_chat_llm,
    llm_for_stage,
)
from src.agents.legal_guide.situation_review import (
    assess_user_situation,
    situation_guidance,
)
from src.agents.legal_guide.scenario_assessment import assess_scenario
from src.agents.legal_guide.progress import emit_guide_progress
from src.agents.legal_guide.followup_catalog import (
    assess_evidence_answer,
    assess_fact_answer,
    assess_initial_evidence,
    assess_initial_facts,
    evidence_effective_count,
    fact_followups,
    find_evidence_followup,
    find_fact_followup,
    format_evidence_assessments,
    format_fact_assessments,
    get_domain_followups,
)
from src.agents.legal_guide.prompts import (
    URGENCY_CHECK_PROMPT, CLARIFY_PROMPT,
    PARSE_DETAILS_PROMPT, CONCLUDE_PROMPT, ISSUE_MAP_PROMPT,
    ISSUE_APPLICATION_PROMPT, STRATEGY_SYNTHESIS_PROMPT, SELF_REVIEW_PROMPT,
    PLAN_CRITIQUE_PROMPT, PLAN_REVISION_PROMPT,
    COUNTER_QUESTION_RESPONSE_PROMPT,
    DOMAIN_DETAIL_TEMPLATES, DOMAIN_LABELS,
)

settings = get_settings()

_COMMON_CASE_REGIONS = (
    "北京", "上海", "天津", "重庆", "杭州", "广州", "深圳", "南京", "成都",
    "武汉", "西安", "郑州", "长沙", "苏州", "宁波", "青岛", "厦门", "福州",
    "济南", "合肥", "南昌", "昆明", "贵阳", "南宁", "海口", "沈阳", "大连",
    "长春", "哈尔滨", "石家庄", "太原", "呼和浩特", "兰州", "西宁", "银川",
    "乌鲁木齐", "拉萨",
)

URGENCY_CRITICAL_RESPONSE = """听到您的情况，我非常担心您的安全。

【立即行动】
- 人身安全威胁：立即拨打 **110**（警察）
- 家庭暴力求助：**12338**（全国妇女权益保护）或 **110**
- 免费法律援助：**12348**（全国法律援助热线）

请先确保安全。安全后直接回复“我现在安全了”，我会保留当前案件并继续帮您梳理。"""

URGENCY_SAFETY_CHECK_RESPONSE = """普通维权步骤先暂停一下，我需要先确认您的现实安全。

请只告诉我：您现在是否已经脱离现场、处于安全位置？
如果危险仍在，请优先联系身边可信任的人或当地紧急服务；确认安全后，我会从当前案件继续。"""

FRAUD_STOP_LOSS_RESPONSE = """### 先做紧急止损

您描述的情况具有较明确的诈骗风险信号，但这不等于系统已经认定构成犯罪。请先停止继续转账，并尽快：

- 联系付款银行或支付平台，申请止付、冻结或拦截；
- 在交易平台发起投诉和账号处置申请，保存受理编号；
- 完整保留订单、聊天记录、转账凭证、对方账号和拉黑页面；
- 拨打 **110** 或 **96110** 咨询、报案，并保存报警或受理记录。"""


class GuideDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session=None, fast_llm=None):
        self.llm = llm
        self.fast_llm = fast_llm
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.db_session = db_session


def _fast_llm_for(deps: GuideDeps):
    return getattr(deps, "fast_llm", None) or getattr(deps, "llm", None)


def _is_low_information_message(message: str) -> bool:
    """Detect bare acknowledgements and fragments that must not trigger memory or routing."""
    text = " ".join(str(message or "").split()).strip()
    if not text:
        return True
    if _is_control_only_text(text):
        return True
    return len(text) <= 2


def _long_term_memories(state: GuideState, limit: int = 5) -> list[str]:
    """返回已由 Supervisor/Worker 检索出的相关长期记忆，限制长度避免污染提示词。"""
    memories = state.user_context.get("long_term_memories") or []
    return [str(item).strip()[:300] for item in memories[:limit] if str(item).strip()]


async def _recall_relevant_memories(user_id: str, query: str, limit: int = 5) -> list[str]:
    """Relevance-search user memories before each guide turn when available."""
    if not user_id or _is_low_information_message(query):
        return []
    try:
        from src.infra.milvus_store import get_milvus_store
        store = get_milvus_store()
        results = await store.asearch(
            ("users", str(user_id), "memories"),
            query=str(query).strip(),
            limit=limit,
        )
    except Exception:
        return []
    memories: list[str] = []
    for item in results:
        value = getattr(item, "value", {}) or {}
        content = value.get("content") or value.get("text")
        if content and str(content).strip():
            memories.append(str(content).strip())
    return memories


_MEMORY_RECALL_MARKERS = (
    "之前说", "以前说", "上次说", "前面说", "还记得", "记得我", "我说过",
    "之前的", "上次的", "以前的",
)


def _active_long_term_memories(state: GuideState) -> list[str]:
    """Relevance-recalled memories are available without an explicit recall phrase."""
    return _long_term_memories(state)


def _with_memory_recall_preface(state: GuideState, user_message: str, reply: str) -> str:
    """用户明确追问历史时，先复述一条最相关记忆，再继续当前流程。

    长期记忆只作为可纠正的上下文，不把历史信息伪装成已经核验的事实。
    """
    if not any(marker in str(user_message or "") for marker in _MEMORY_RECALL_MARKERS):
        return reply
    memories = _active_long_term_memories(state)
    if not memories:
        return reply

    def _rank(value: str) -> tuple[int, int]:
        legal_summary = int("法律咨询摘要" in value or "案情事实" in value)
        substantive = int(any(term in value for term in ("争议", "拖欠", "纠纷", "证据", "合同", "事故")))
        return legal_summary + substantive, len(value)

    memory = max(memories, key=_rank)
    clean_memory = re.sub(r"^法律咨询摘要[:：]\s*", "", memory).strip().rstrip("。；")
    if not clean_memory or clean_memory in reply:
        return reply
    return (
        f"我记得您之前提到：{clean_memory}。\n"
        "如果情况已有变化，以您这次说明为准。\n\n"
        f"{reply}"
    )


def _merge_unique(old: list[str], new: list[str]) -> list[str]:
    seen: set[str] = set()
    return [item for item in old + new if item and not (item in seen or seen.add(item))]


def _split_evidence_names(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for part in re.split(r"[、，,；;和]+", str(value or "")):
            part = part.strip()
            if part and part not in result:
                result.append(part)
    return result


def _state_region_name(raw: str | None) -> str:
    """Preserve a user-stated region even when local channel data is not piloted there."""
    supported = normalize_region_name(raw)
    if supported:
        return supported
    value = str(raw or "").strip()
    if value in {"", "全国", "中国", "未说明", "未知", "不清楚", "所在地区"}:
        return ""
    if value in _COMMON_CASE_REGIONS:
        return value
    if re.fullmatch(r"[\u4e00-\u9fff]{2,12}(?:省|市|自治区|特别行政区|自治州|地区|盟|县|区)", value):
        return value
    return ""


def _extract_case_region(text: str) -> str:
    supported = extract_supported_region(text)
    if supported:
        return supported
    value = str(text or "")
    return next((region for region in _COMMON_CASE_REGIONS if region in value), "")


_SUPPLEMENT_CONCLUDE_MARKERS = (
    "现在生成", "直接生成", "生成方案", "给方案", "出方案", "先生成",
    "不补充", "不用补充", "不继续", "不问了", "就这些", "按现有信息",
)
_SUPPLEMENT_CONTINUE_MARKERS = (
    "继续补充", "继续问", "可以继续", "再问", "再补充", "完善一下",
    "还要补充", "我继续说", "继续",
)


def _supplement_choice_from_text(message: str) -> str:
    """仅在等待选择时解析用户意图，避免普通案情中的“继续”被误判。"""
    compact = "".join(str(message or "").strip().split())
    if any(marker in compact for marker in _SUPPLEMENT_CONCLUDE_MARKERS):
        return "conclude"
    if any(marker in compact for marker in _SUPPLEMENT_CONTINUE_MARKERS):
        return "continue"
    return ""


def _supplement_contains_case_details(message: str) -> bool:
    """Treat free-form facts as an implicit request to keep supplementing."""
    compact = "".join(str(message or "").strip().split())
    if compact in {"好", "好的", "行", "可以", "嗯", "哦", "知道了", "明白了"}:
        return False
    markers = sorted(
        {*_SUPPLEMENT_CONCLUDE_MARKERS, *_SUPPLEMENT_CONTINUE_MARKERS},
        key=len,
        reverse=True,
    )
    for marker in markers:
        compact = compact.replace(marker, "")
    compact = re.sub(r"[，。；：、！？?（）()\[\]【】‘’“”\-]", "", compact)
    if compact in {"", "现在", "直接", "先", "方案", "生成", "给", "出"}:
        return False
    compact = compact.strip("好的行可以嗯哦我请就吧")
    return len(compact) >= 2


def _current_turn_contains_case_details(state: GuideState, message: str) -> bool:
    """Prefer the semantic control result, retaining heuristics for direct graph calls."""

    if state.turn_control_intent:
        return state.turn_contains_case_details
    return _supplement_contains_case_details(message)


def _normalized_question_text(value: str) -> str:
    return re.sub(r"[\s，。；：、！？?（）()\[\]【】‘’“”\-]", "", str(value or ""))


def _looks_like_question_repetition(answer: str, questions: list[str]) -> bool:
    """识别用户把系统问题复制回来、但没有作出肯定或否定回答的情况。"""
    answer_norm = _normalized_question_text(answer)
    if not answer_norm:
        return False
    explicit_prefixes = ("有", "没有", "没", "是", "不是", "签了", "没签", "写了", "没写", "确认")
    if answer_norm.startswith(explicit_prefixes):
        return False
    looks_interrogative = any(marker in answer for marker in ("？", "?", "是否", "有没有", "是不是", "吗"))
    if not looks_interrogative:
        return False
    for question in questions:
        question_norm = _normalized_question_text(question)
        if not question_norm:
            continue
        if question_norm in answer_norm or answer_norm in question_norm:
            return True
        if SequenceMatcher(None, answer_norm, question_norm).ratio() >= 0.62:
            return True
    return False


def _looks_like_user_question(value: str) -> bool:
    """Conservative fallback when the detail parser mistakes a statement for a question."""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return False
    if any(marker in text for marker in ("？", "?", "为什么", "为何", "怎么", "如何", "什么是", "有什么用")):
        return True
    return text.endswith(("吗", "呢", "么"))


def _is_usable_case_fact(value: str) -> bool:
    """过滤疑问句、系统问题复述和纯推测，避免污染案情黑板。"""
    text = " ".join(str(value or "").split())
    if not text:
        return False
    if any(marker in text for marker in ("？", "?", "是否", "有没有", "是不是")):
        return False
    if text.endswith("吗"):
        return False
    if re.search(r"使用[‘'\"“]?[^’'\"”]{1,12}[’'\"”]?一词", text):
        return False
    return True


def _is_transport_wrapper_fact(value: str) -> bool:
    """Protocol envelopes identify form/evidence transport, not case facts."""
    text = str(value or "")
    return any(marker in text for marker in (
        "【动态追问表单回答】",
        "【文档证据补充（程序提取",
        "【图片证据补充（视觉模型识别",
    ))


_STRUCTURED_FORM_ANSWER_RE = re.compile(
    r"(?ms)^\s*\d+\.\s*\[([^\]]+)\][^\n]*\n\s*回答[：:]\s*"
    r"(.*?)(?=^\s*\d+\.\s*\[[^\]]+\]|\Z)"
)


def _structured_followup_answers(message: str) -> dict[str, str]:
    """Extract only user-entered values from the frontend form envelope."""

    answers: dict[str, str] = {}
    if "【动态追问表单回答】" not in str(message or ""):
        return answers
    for field_id, raw_answer in _STRUCTURED_FORM_ANSWER_RE.findall(message):
        key = str(field_id or "").strip()
        value = str(raw_answer or "").strip()
        if key and value:
            answers[key] = value
    return answers


# 只含碎片词的答案：无法独立读出完整语义，必须锚定原问题后再入库。
_FRAGMENT_ANSWERS = frozenset([
    "有", "没有", "无", "是", "否", "没", "未", "有的", "不是",
    "有呢", "没有呢", "没有了", "没有了呢", "没有过", "没有啊",
    "不知道", "不清楚", "不确定", "记不清", "不记得", "没法确认",
    "是的", "对", "对的", "可以", "不可以",
])


def _fragment_anchored_statement(question: str, value: str) -> str:
    """碎片词答案 → 问题锚定的自包含陈述（例："是否持有转账凭证：没有"）。

    用户回答本身已完整时原样保留；只有回答是"有/没有/不知道"这类无法独立
    读懂的碎片词时，才把原问题拼进来，避免"理解您的情况"只显示裸词。
    """
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    if len(text) <= 6 and text in _FRAGMENT_ANSWERS:
        question_text = " ".join(str(question or "").strip().split()).rstrip("？?。！!：:")
        return f"{question_text}：{text}" if question_text else text
    return text


def _humanized_followup_answers(message: str, questions: list[dict] | None = None) -> str:
    """Turn the frontend form envelope into user-facing dialogue text.

    The final prompt should never expose internal field ids or the
    ``【动态追问表单回答】`` control marker; only the user's answers and, for
    fragment answers, the anchored question are useful to the final model.
    """
    answers = _structured_followup_answers(message)
    if not answers:
        return str(message or "")
    question_by_id = {
        str(item.get("field_id") or "").strip(): str(item.get("question") or "")
        for item in (questions or [])
        if isinstance(item, dict)
    }
    parts: list[str] = []
    for field_id, raw_answer in answers.items():
        value = str(raw_answer or "").strip()
        if not value:
            continue
        parts.append(_fragment_anchored_statement(question_by_id.get(field_id, ""), value))
    return "用户补充：" + "；".join(parts)


def _clean_dialogue_message(message: str, questions: list[dict] | None = None) -> str:
    """Remove protocol wrappers before a user message enters the final prompt."""
    content, _ = split_uploaded_evidence_blocks(str(message or ""))
    content = content.strip()
    if not content:
        return "用户：提交了附件材料（内容按证据处理）"
    if _is_control_only_text(content):
        return "用户：流程控制语（非案件事实）"
    return _humanized_followup_answers(content, questions)


def _structured_answer_case_updates(
    questions: list[dict],
    answers: dict[str, str],
    *,
    domain: str,
) -> list[dict]:
    """Persist every submitted form value in the primary detail store."""

    slot_categories = {
        "legal_relationship": "relationship",
        "transaction": "event",
        "claim": "claim",
        "procedure": "procedure",
        "event_time": "time",
        "harm": "harm",
        "event": "event",
    }
    effect_categories = {
        "claim_scope": "claim",
        "procedure": "procedure",
        "limitation": "time",
        "responsibility": "relationship",
        "jurisdiction": "location",
        "safety": "event",
    }
    updates: list[dict] = []
    for item in questions:
        field_id = str(item.get("field_id") or "").strip()
        value = answers.get(field_id, "").strip()
        if not field_id or not value:
            continue
        rule = find_fact_followup(domain, str(item.get("candidate_id") or ""))
        category = slot_categories.get(rule.slot if rule else "", "")
        if not category:
            effects = [str(effect) for effect in (item.get("decision_effects") or [])]
            category = next(
                (effect_categories[effect] for effect in effects if effect in effect_categories),
                "event",
            )
        updates.append({
            "key": f"followup.{field_id}",
            "category": category,
            "statement": _fragment_anchored_statement(
                item.get("question") or "", value
            ),
            "value": value,
            "certainty": "asserted",
            "operation": "add",
            "source_text": value,
        })
    return updates


def _is_draftable_fact(value: str) -> bool:
    """只有清晰、非推测的用户陈述才能进入正式文书事实池。"""
    text = " ".join(str(value or "").split())
    if not _is_usable_case_fact(text):
        return False
    if text.startswith("待核验线索"):
        return False
    if any(marker in text for marker in ("可能", "好像", "应该", "猜测", "听说", "据说")):
        return False
    return True


# ════════════════════════════════════════════════════════════════════════
# 节点函数
# ════════════════════════════════════════════════════════════════════════

async def node_load_context(state: GuideState, deps: GuideDeps) -> dict:
    """准备阶段辅助函数：仅首轮加载用户历史咨询上下文。"""
    if state.round > 0:
        return {}
    user_id = state.user_context.get("user_id")
    logger.info("准备阶段加载上下文 | session={}", state.session_id)
    ctx = await load_user_context(user_id, deps.db_session)
    region = normalize_region_name(ctx.get("region", ""))
    # PG 历史上下文只能补充，不能覆盖 user_id 和 Supervisor 已检索的长期记忆。
    merged_context = {**state.user_context, **ctx}
    last_msg = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        "",
    )
    recalled = await _recall_relevant_memories(str(user_id or ""), str(last_msg or ""))
    existing_memories = list(state.user_context.get("long_term_memories") or [])
    merged_context["long_term_memories"] = list(
        dict.fromkeys([*existing_memories, *recalled])
    )[:8]
    return {"user_context": merged_context, "region": region or state.region}


async def node_prepare_turn(state: GuideState, deps: GuideDeps) -> dict:
    """节点①：首轮加载历史上下文，并且只在这里推进用户轮次。"""
    context_updates = await node_load_context(state, deps)
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    conclude_phrases = (
        "不要再问", "别再问", "不用再问", "给方案", "给我方案", "给出方案",
        "生成方案", "现在生成方案", "按现有信息", "按现在这些", "最终建议", "最终方案", "请收敛",
        "只能说这些", "只说这些", "没有更多信息", "没有更多证据", "没更多信息",
    )
    supplement_choice = ""
    supplement_has_details = _current_turn_contains_case_details(state, last_msg)
    awaiting_supplement_choice = state.awaiting_supplement_choice
    allow_extra_followups = state.allow_extra_followups
    semantic_control = state.turn_control_intent
    wants_conclude = (
        state.wants_conclude
        or semantic_control == "conclude_now"
        or any(p in last_msg for p in conclude_phrases)
    )
    # “继续补充”是对会话流程的控制，不是案件事实。该意图可能发生在方案已经
    # 生成之后，此时不存在 awaiting_supplement_choice 菜单状态，也必须直接
    # 回到追问规划，不能把控制语句送进案情提取节点。
    if semantic_control == "continue_gathering":
        supplement_choice = "continue"
        allow_extra_followups = True
    if state.awaiting_supplement_choice:
        if semantic_control == "conclude_now":
            supplement_choice = "conclude"
        elif semantic_control == "continue_gathering":
            supplement_choice = "continue"
        else:
            supplement_choice = _supplement_choice_from_text(last_msg)
        if not supplement_choice and supplement_has_details:
            supplement_choice = "continue"
        if supplement_choice == "conclude":
            wants_conclude = True
            awaiting_supplement_choice = False
        elif supplement_choice == "continue":
            allow_extra_followups = True
        # 旧版本持久化过这个菜单状态。新流程收到下一条消息后立即退出该状态，
        # 再由动态规划器选择一个明确缺口或直接生成方案。
        awaiting_supplement_choice = False
    total_rounds = state.total_rounds + 1
    return {
        **context_updates,
        "round": state.round + 1,
        "total_rounds": total_rounds,
        "wants_conclude": wants_conclude,
        "force_conclude": state.force_conclude or total_rounds >= settings.GUIDE_MAX_TOTAL_ROUNDS,
        "awaiting_supplement_choice": awaiting_supplement_choice,
        "supplement_choice": supplement_choice,
        "supplement_has_details": supplement_has_details,
        "allow_extra_followups": allow_extra_followups,
    }


async def node_check_urgency(state: GuideState, deps: GuideDeps) -> dict:
    """节点②：三级紧急分类（每一轮都执行）。CRITICAL → 立即给援助信息+END。

    关键安全设计：不能只在首轮检测。用户可能在多轮对话中途才追加高危案情
    （例：先聊租房纠纷，几轮后才说"对方上门殴打我"），因此每轮都必须重跑。
    """
    emit_guide_progress(
        "risk_check",
        "正在检查紧急风险",
        "识别是否需要先处理人身安全、止付、冻结或报警。",
    )
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if not last_msg:
        return {}
    logger.info("节点②紧急检测 | session={} round={}", state.session_id, state.round)
    recent_human_messages = [
        str(message.content).strip()
        for message in state.messages
        if isinstance(message, HumanMessage) and str(message.content).strip()
    ][-4:]
    recent_dialogue = "\n".join(
        f"- 第{index + 1}条：{message}"
        for index, message in enumerate(recent_human_messages)
    )
    explicitly_safe = any(
        marker in "\n".join(recent_human_messages)
        for marker in (
            "今天暂时安全", "现在暂时安全", "目前暂时安全", "现在安全",
            "目前安全", "已经安全", "没有安全危险", "现在没有危险",
            "目前没有危险", "已经离开现场", "现在不在现场",
        )
    )
    deterministic_current_danger = any(
        marker in last_msg
        for marker in (
            "现在有危险", "目前有危险", "正在打", "正在施暴", "正在追",
            "拿刀", "持刀", "要杀", "赶过来", "马上过来", "被困",
            "不让我走", "无法离开", "就在门外",
        )
    )
    deterministic_safety_relevant = bool(_deterministic_case_frame(last_msg))
    fraud_signal = is_high_precision_fraud_report(last_msg)
    fraud_updates = (
        {
            "fraud_stop_loss_relevant": True,
            "fraud_stop_loss_warning": FRAUD_STOP_LOSS_RESPONSE,
            "fraud_stop_loss_offered": False,
        }
        if fraud_signal and not state.fraud_stop_loss_offered
        else {}
    )
    # 高精度现实危险信号先于模型处理，避免模型超时让安全熔断失效。
    if deterministic_current_danger and not explicitly_safe:
        logger.warning("节点②确定性检测到当前危险，立即触发安全中断")
        return {
            **fraud_updates,
            "urgency_level": "critical",
            "safety_relevant": True,
            "current_safety_status": "danger",
            "safety_pause_active": True,
            "safety_pause_case_message": state.safety_pause_case_message or last_msg,
            "phase": GuidePhase.END,
            "messages": [AIMessage(content=URGENCY_CRITICAL_RESPONSE)],
        }
    prompt = URGENCY_CHECK_PROMPT.format(
        user_input=last_msg,
        recent_dialogue=recent_dialogue or f"- {last_msg}",
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(deps.llm, max_tokens=500),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_URGENCY,
            stage="urgency_check",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        urgency = result.get("urgency", "NORMAL")
        time_clue = result.get("time_clue", "")
        allowed_safety_statuses = {"danger", "safe", "unknown", "not_applicable"}
        model_safety_status = str(result.get("safety_status") or "").lower()
        safety_relevant = bool(result.get("safety_relevant"))
        safety_contract_present = (
            "safety_relevant" in result or "safety_status" in result
        )
        if deterministic_safety_relevant and not explicitly_safe:
            # 只要当前消息出现被打、受伤等身体侵害信号，就不能由模型降级成
            # not_applicable；是否正在危险中由用户确认，而不是默认无关。
            safety_relevant = True
            if model_safety_status not in {"danger", "unknown"}:
                model_safety_status = "unknown"
        if deterministic_current_danger:
            urgency = "CRITICAL"
            safety_relevant = True
            safety_status = "danger"
        elif explicitly_safe:
            safety_relevant = True
            safety_status = "safe"
        elif urgency == "CRITICAL" and not safety_contract_present:
            # Backward-compatible fail-safe for an older/malformed model reply.
            safety_relevant = True
            safety_status = "danger"
        elif safety_relevant:
            safety_status = (
                model_safety_status
                if model_safety_status in allowed_safety_statuses - {"not_applicable"}
                and (model_safety_status != "safe" or state.safety_pause_active)
                else "unknown"
            )
        else:
            safety_status = "not_applicable"
        current_danger = safety_status == "danger"
        if state.safety_pause_active and safety_status == "safe":
            logger.info("现实危险已经解除，恢复同一案件的普通法律梳理")
            return {
                **fraud_updates,
                "urgency_level": "normal",
                "safety_relevant": True,
                "current_safety_status": "safe",
                "safety_pause_active": False,
            }
        if urgency == "CRITICAL" and not current_danger and explicitly_safe:
            logger.info("近期已明确当前安全且没有新增危险，继续法律梳理而不触发紧急终止")
            return {
                **fraud_updates,
                "urgency_level": "normal",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
            }
        if urgency == "CRITICAL" and not current_danger:
            logger.info("涉及人身安全但当前危险未确认，转入单问题安全确认")
            return {
                **fraud_updates,
                "urgency_level": "normal",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
            }
        if urgency == "CRITICAL" and current_danger:
            logger.warning("节点②检测到CRITICAL紧急情形")
            return {
                **fraud_updates,
                "urgency_level": "critical",
                "safety_relevant": True,
                "current_safety_status": "danger",
                "safety_pause_active": True,
                "safety_pause_case_message": state.safety_pause_case_message or last_msg,
                "phase": GuidePhase.END,
                "messages": [AIMessage(content=URGENCY_CRITICAL_RESPONSE)],
            }
        if state.safety_pause_active:
            logger.info("现实危险是否解除仍不明确，保持安全中断")
            return {
                **fraud_updates,
                "urgency_level": "critical",
                "safety_relevant": True,
                "current_safety_status": "unknown",
                "safety_pause_active": True,
                "phase": GuidePhase.END,
                "messages": [AIMessage(content=URGENCY_SAFETY_CHECK_RESPONSE)],
            }
        if urgency == "TIME" and time_clue:
            warning = f'\n⚠️ **时效提醒**：您提到"{time_clue}"，请注意维权时效（劳动仲裁1年、一般民事3年），建议尽快行动。'
            logger.info("节点②检测到时效紧迫: {}", time_clue)
            return {
                **fraud_updates,
                "urgency_level": "time",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
                "time_warning": warning,
            }
        return {
            **fraud_updates,
            "urgency_level": "normal",
            "safety_relevant": safety_relevant,
            "current_safety_status": safety_status,
        }
    except Exception as e:
        logger.warning(f"紧急检测解析失败: {e}")
    if _deterministic_case_frame(last_msg) and not explicitly_safe:
        return {
            **fraud_updates,
            "urgency_level": "normal",
            "safety_relevant": True,
            "current_safety_status": "unknown",
        }
    return {
        **fraud_updates,
        "urgency_level": "normal",
        "safety_relevant": state.safety_relevant,
        "current_safety_status": state.current_safety_status,
    }


async def node_extract_issues(state: GuideState, deps: GuideDeps) -> dict:
    """节点③：标准化法律问题，并把当前消息归入通用原子案情。"""
    human_msgs = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {}
    combined_input = "\n".join(
        split_uploaded_evidence_blocks(message)[0]
        for message in human_msgs[-3:]
    )
    memories = _active_long_term_memories(state)
    prior_messages = [
        split_uploaded_evidence_blocks(message)[0]
        for message in human_msgs[-3:-1]
    ]
    current_user_input = human_msgs[-1]
    resumed_safety_case = bool(
        state.safety_pause_case_message
        and not state.safety_pause_active
        and state.current_safety_status == "safe"
    )
    if resumed_safety_case:
        current_user_input = (
            state.safety_pause_case_message
            + "\n当前安全状态补充："
            + human_msgs[-1]
        )
    narrative_input, uploaded_observations = split_uploaded_evidence_blocks(
        current_user_input
    )
    if uploaded_observations:
        emit_guide_progress(
            "evidence_intake",
            "正在读取并登记新证据",
            f"已收到 {len(uploaded_observations)} 份材料，正在核对可见内容、完整性和证明目标。",
        )
    else:
        emit_guide_progress(
            "fact_analysis",
            "正在整理案情与事实细节",
            "把本轮陈述更新到事实细节库，并识别需要判断的法律问题。",
        )
    if _is_low_information_message(narrative_input) and not uploaded_observations:
        logger.info(
            "低信息消息不进入领域提取 | message={}",
            narrative_input[:40],
        )
        return {
            "phase": GuidePhase.CLARIFY,
            "issue_refresh_needed": False,
        }
    attachment_inventory = "\n".join(
        f"- {item['name']}（系统已收到副本，内容不得自动当作用户确认事实）"
        for item in uploaded_observations
    )
    normalizer_input = (
        "[近期对话，仅用于理解语境]\n"
        + ("\n".join(prior_messages) or "无")
        + "\n\n[当前用户消息]\n"
        + (narrative_input or "用户本轮仅提交了附件")
        + (
            "\n\n[本轮附件清单]\n"
            + attachment_inventory
            + "\n附件由程序单独进入证据库存；不要把附件全文或其中陈述写成用户已经确认的案情事实。"
            if attachment_inventory
            else ""
        )
        + "\n\ncase_updates 只提取[当前用户消息]中的新增、更正或否定内容；"
          "source_text 必须来自该消息原文。"
    )
    if state.case_facts:
        normalizer_input += (
            "\n\n[已有结构化事实及语义键]\n"
            + format_case_context(state.case_facts)
            + "\n当前消息若只是重复已有事实，不要换一个 key 再次写入；"
              "若补充同一事实，沿用已有 key 或其下级 key。"
        )
    if memories and (
        _current_turn_contains_case_details(state, current_user_input)
        or any(marker in current_user_input for marker in _MEMORY_RECALL_MARKERS)
    ):
        normalizer_input += (
            "\n\n[相关长期记忆，仅作补充；与本轮冲突时以本轮为准]\n"
            + "\n".join(f"- {item}" for item in memories)
        )
    logger.info("节点③提取法律问题 | round={}", state.round)
    result, inspected_evidence_observations = await asyncio.gather(
        normalize_legal_issues(
            user_input=normalizer_input,
            llm=_fast_llm_for(deps),
            neo4j_driver=deps.neo4j_driver,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
            fallback_text=combined_input,
        ),
        inspect_uploaded_evidence_blocks(current_user_input, _fast_llm_for(deps)),
    )
    # 两个池分别累积，跨轮保序去重（不用 set，避免检索 query 每轮字符串顺序漂移）
    def _merge(old: list[str], new: list[str]) -> list[str]:
        seen: set[str] = set()
        return [x for x in old + new if not (x in seen or seen.add(x))]

    latest_user_text = current_user_input
    result_standard = result["standard"]
    new_confirmed = _merge(state.confirmed_issues, result_standard)
    # 已升级为标准术语的口语词，从口语池剔除，避免同一件事在两个池里各出现一次
    result_term_map = dict(result["term_map"])
    promoted = set(result_term_map)
    new_unmatched = [
        x for x in _merge(state.unmatched_issues, result["colloquial"])
        if x not in promoted
    ]
    proposed_domain = result["domain"] or state.legal_domain
    # The domain is stable after retrieval/conclusion, but an early low-information
    # label may be revised when a later message supplies grounded facts and a
    # concrete issue. This handles ordinary user corrections without treating
    # the first short answer as an irreversible routing decision.
    can_revise_early_domain = bool(
        state.legal_domain
        and state.legal_domain != "other"
        and proposed_domain
        and proposed_domain != "other"
        and proposed_domain != state.legal_domain
        and not state.retrieval_completed
        and state.confidence_tier in {"", "LOW"}
        and result_standard
        and (result.get("case_updates") or result.get("collected_facts"))
    )
    domain = (
        proposed_domain
        if not state.legal_domain or state.legal_domain == "other" or can_revise_early_domain
        else state.legal_domain
    )
    case_frame = str(result.get("case_frame") or "").strip()
    frame_confidence = float(result.get("frame_confidence") or 0.0)
    if case_frame == "personal_safety":
        # Personal-safety frames are a hard routing boundary: even if the LLM
        # or an earlier turn mislabeled the case as cyber/consumer, the current
        # physical-harm signal must switch the domain to the public-security
        # workflow instead of asking about platform or transfer details.
        safety_domain = (
            proposed_domain
            if proposed_domain in {
                "criminal_public_security",
                "traffic_personal_injury",
                "family_vulnerable_groups",
            }
            else "criminal_public_security"
        )
        if safety_domain != domain:
            logger.info(
                "人身安全事件框架覆盖早期领域 | old={} new={}",
                domain,
                safety_domain,
            )
            domain = safety_domain
    if can_revise_early_domain:
        logger.info(
            "早期领域依据被更具体的新事实修正 | old={} new={}",
            state.legal_domain,
            proposed_domain,
        )
    new_term_map = {**state.term_map, **result_term_map}
    if uploaded_observations and not narrative_input.strip():
        raw_case_updates = []
    else:
        raw_case_updates = result.get("case_updates") or legacy_fact_updates(
            result.get("collected_facts") or [],
            user_text=latest_user_text,
        )
        raw_case_updates = [
            item for item in raw_case_updates
            if not _is_transport_wrapper_fact(
                item.get("statement", "") if isinstance(item, dict) else ""
            )
        ]
    case_facts = reduce_case_facts(
        state.case_facts,
        raw_case_updates,
        user_text=latest_user_text,
        turn=state.round,
    )
    active_atoms = active_case_facts(case_facts)
    atomic_statements = [
        item["statement"] for item in active_atoms
        if item.get("category") != "evidence" and item.get("statement")
    ]
    new_facts = atomic_statements if case_facts else _merge(state.collected_facts, atomic_statements)
    fact_records = assess_initial_facts(atomic_statements, state.fact_records)
    active_draftable_facts = [
            item["statement"] for item in active_atoms
            if item.get("category") != "evidence"
            and item.get("status") == "asserted"
            and item.get("statement")
        ]
    new_draftable_facts = (
        active_draftable_facts
        if case_facts
        else _merge(state.draftable_facts, active_draftable_facts)
    )
    initial_evidence_observations = normalize_evidence_observations(
        result.get("evidence_details"),
        user_text=latest_user_text,
    )
    initial_evidence_observations.extend(inspected_evidence_observations)
    # Always merge the deterministic upload inventory after model output.  It
    # supplies the transport facts (received copy, source form and digest) that
    # must survive even when the model omits or misreads an attachment.
    initial_evidence_observations.extend(uploaded_observations)
    current_turn_atoms = latest_case_facts(case_facts, state.round)
    atom_evidence, atom_unavailable = evidence_from_case_facts(current_turn_atoms)
    if initial_evidence_observations:
        # A source-anchored material description is not necessarily material in
        # the user's possession. Only an uploaded copy is promoted here; a
        # claimed photo, witness or accessible surveillance remains a lead
        # unless its case atom explicitly says it was obtained.
        atom_evidence = _merge(
            atom_evidence,
            [item["name"] for item in initial_evidence_observations if item.get("uploaded_copy")],
        )
    observation_leads = [
        item["name"] for item in initial_evidence_observations
        if not item.get("uploaded_copy") and item.get("name")
    ]
    new_evidence = _merge(state.evidence_confirmed, atom_evidence)
    new_unavailable = _merge(state.evidence_unavailable, atom_unavailable)
    new_unverified = _merge(state.evidence_unverified, observation_leads)
    region_extracted = (
        _state_region_name(state.region)
        or _state_region_name(result.get("region", ""))
    )
    time_info = state.time_info or result.get("time_info", "")

    logger.info(
        "节点③结果 | standard={} colloquial={} domain={} evidence={} region={} time_info={}",
        new_confirmed, new_unmatched, domain, new_evidence, region_extracted, time_info,
    )

    has_case_substance = bool(
        new_confirmed
        or new_unmatched
        or new_facts
        or state.case_facts
        or state.collected_facts
    )
    effective_domain = domain if has_case_substance else state.legal_domain
    effective_frame = case_frame if has_case_substance else state.case_frame
    effective_confidence = (
        frame_confidence if has_case_substance else state.frame_confidence
    )
    updates = {
        "unmatched_issues": new_unmatched,
        "term_map": new_term_map,
        "issue_refresh_needed": False,
        "collected_facts": new_facts,
        "draftable_facts": new_draftable_facts,
        "case_facts": case_facts,
        "fact_records": fact_records,
        "evidence_unavailable": new_unavailable,
        "evidence_unverified": new_unverified,
        "legal_domain": effective_domain,
        "case_frame": effective_frame,
        "frame_confidence": effective_confidence,
        "safety_relevant": (
            state.safety_relevant
            or case_frame == "personal_safety"
            or state.current_safety_status in {"safe", "danger", "unknown"}
        ),
        "safety_pause_case_message": (
            "" if resumed_safety_case else state.safety_pause_case_message
        ),
    }

    evidence_assessments = state.evidence_assessments
    if new_evidence != state.evidence_confirmed:
        updates["evidence_confirmed"] = new_evidence
        newly_found = [item for item in new_evidence if item not in state.evidence_confirmed]
        evidence_assessments = assess_initial_evidence(
            newly_found,
            state.evidence_assessments,
        )
    if initial_evidence_observations:
        evidence_assessments = merge_evidence_observations(
            evidence_assessments,
            initial_evidence_observations,
            domain=domain,
        )
    if uploaded_observations:
        # Every explicit submission creates a new assessment revision.  The
        # material store itself remains de-duplicated by requirement/name and
        # content digest, so re-uploading updates the row instead of multiplying it.
        updates["evidence_evaluation_version"] = (
            state.evidence_evaluation_version + 1
        )
    if evidence_assessments != state.evidence_assessments:
        updates["evidence_assessments"] = evidence_assessments

    if region_extracted and not state.region:
        updates["region"] = region_extracted

    if time_info and time_info != state.time_info:
        updates["time_info"] = time_info

    if new_confirmed:
        updates.update({
            "confirmed_issues": new_confirmed,
            "phase": GuidePhase.ISSUE_SEARCH,
        })
    else:
        # 无标准术语：进澄清引导，让用户补充细节后重新提取（口语池保留，供兜底检索用）
        updates["phase"] = GuidePhase.CLARIFY

    return updates


async def node_clarify(state: GuideState, deps: GuideDeps) -> dict:
    """节点④：引导用户描述清楚法律情况。上限 2 轮，仍模糊则降级。"""
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    logger.info("节点④澄清引导 | round={} clarify_rounds={} total_rounds={}",
                state.round, state.clarify_rounds, state.total_rounds)
    recent_messages = state.messages[-8:]
    recent_dialogue = "\n".join(
        f"{'用户' if isinstance(message, HumanMessage) else '助手'}：{str(message.content)[:500]}"
        for message in recent_messages
    )
    prompt = CLARIFY_PROMPT.format(
        user_input=last_msg,
        recent_dialogue=recent_dialogue or f"用户：{last_msg}",
        case_context=format_case_context(state.case_facts),
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(_fast_llm_for(deps), max_tokens=350),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_EXTRACT,
            stage="clarify",
        )
        reply = str(response.content or "").strip()
    except Exception as exc:
        logger.warning("澄清生成失败，使用低负担固定澄清 | err={}", exc)
        reply = (
            "请用一两句话补充：事情发生在谁和谁之间、发生了什么，"
            "以及您现在最希望解决什么问题。暂时不清楚的部分可以直接说“不清楚”。"
        )
    # 澄清也是一个真实追问：下一轮必须先解析这道题的回答，不能把用户
    # 的短回答当成脱离上下文的新问题再次分类。
    if "？" in reply:
        reply = reply.split("？", 1)[0].strip() + "？"
    elif "?" in reply:
        reply = reply.split("?", 1)[0].strip() + "？"
    elif reply:
        reply = reply.rstrip("。；") + "？"
    return {
        "clarify_rounds": state.clarify_rounds + 1,
        "phase": GuidePhase.CLARIFY,
        "asked_details": _merge_unique(state.asked_details, [reply]),
        "pending_ask_details": [reply],
        "pending_ask_type": "facts",
        "pending_followup_ids": [],
        "messages": [AIMessage(content=reply)],
    }


async def node_score(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：纯规则打分（打分前置、零 I/O），决定是否值得深度检索。"""
    domain = state.legal_domain
    evidence_total = len(resolve_state_evidence_checklist(state).items)
    raw_evidence_report = state.evidence_coverage or {}
    evidence_report = (
        EvidenceEvaluationReport.model_validate(raw_evidence_report)
        if raw_evidence_report
        else evaluate_state_evidence(state)
    )
    if evidence_report.target_count:
        evidence_total = evidence_report.target_count
        # This is plan-preparation coverage, not a judicial proof score.
        # Partially covered targets receive limited credit so an uninspected
        # screenshot cannot inflate confidence as much as a source-anchored,
        # complete material.
        effective_evidence = (
            float(evidence_report.preliminarily_covered_count)
            + 0.35 * float(evidence_report.partial_count)
        )
    else:
        effective_evidence = evidence_effective_count(
            state.evidence_confirmed,
            state.evidence_assessments,
        )

    time_known = (
        bool(state.time_warning) or
        bool(state.time_info) or
        any(item.get("category") == "time" for item in active_case_facts(state.case_facts))
    )

    conf = score_confidence(
        confirmed_issues=state.confirmed_issues,
        evidence_confirmed=state.evidence_confirmed,
        evidence_total=evidence_total,
        domain_locked=bool(domain),
        region_known=bool(state.region),
        time_known=time_known,
        effective_evidence_count=effective_evidence,
    )
    logger.info("节点⑤打分 | score={:.2f} tier={} breakdown={}",
                conf["score"], conf["tier"], conf["breakdown"])
    return {
        "confidence_score": conf["score"],
        "confidence_tier": conf["tier"],
    }


def _rrf_fuse(hits_a: list[dict], hits_b: list[dict], k: int = 60, top_n: int = 10) -> list[dict]:
    """Reciprocal Rank Fusion：融合两个已排序 hit 列表，返回 top_n 条。

    同时出现在两个列表的条文得分叠加（说明跨检索策略都召回，相关性更高）。
    score 字段替换为 RRF 分（越大越靠前）。
    """
    rrf_scores: dict[tuple, float] = {}
    all_hits: dict[tuple, dict] = {}

    for rank, hit in enumerate(hits_a):
        key = (hit.get("law_id", ""), hit.get("article_no", ""))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        all_hits[key] = hit

    for rank, hit in enumerate(hits_b):
        key = (hit.get("law_id", ""), hit.get("article_no", ""))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        all_hits.setdefault(key, hit)  # domain 列表优先，全库做补充

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    result = []
    for key in sorted_keys[:top_n]:
        hit = dict(all_hits[key])
        hit["score"] = round(rrf_scores[key], 6)
        result.append(hit)
    return result


def _retrieval_fingerprint(state: GuideState) -> str:
    """Hash only facts that materially shape legal retrieval."""

    inputs = build_case_retrieval_inputs(
        state.confirmed_issues,
        active_case_facts(state.case_facts),
    )
    payload = {
        "domain": state.legal_domain,
        "issues": list(state.confirmed_issues),
        "unmatched": list(state.unmatched_issues[:5]),
        "lexical": list(inputs.get("lexical_phrases") or []),
        "semantic": list(inputs.get("semantic_phrases") or []),
        "evidence": list(state.evidence_confirmed[:5]),
        "time": state.time_info,
        "region": state.region,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _retrieval_snapshot_reusable(state: GuideState) -> bool:
    if not state.retrieval_completed:
        return False
    if not state.retrieval_fingerprint:
        # Compatibility with snapshots persisted before fingerprints existed.
        return True
    return state.retrieval_fingerprint == _retrieval_fingerprint(state)


async def node_retrieve_followup_basis(state: GuideState, deps: GuideDeps) -> dict:
    """Lightweight live grounding for follow-up planning; deliberately no cases."""

    emit_guide_progress(
        "followup_retrieval",
        "正在检索追问依据",
        "检索相关法条和官方知识；追问阶段不检索类案。",
    )

    fingerprint = _retrieval_fingerprint(state)
    if not all(
        getattr(deps, name, None) is not None
        for name in ("embedding_model", "milvus_client", "neo4j_driver")
    ):
        return {
            "followup_basis_fingerprint": fingerprint,
            "followup_basis_error": "追问依据检索依赖未注入，使用安全兜底",
        }
    if state.followup_basis_fingerprint == fingerprint and state.followup_basis_refs:
        return {}
    inputs = build_case_retrieval_inputs(
        state.confirmed_issues,
        active_case_facts(state.case_facts),
    )
    query_parts = [DOMAIN_LABELS.get(state.legal_domain, "")]
    query_parts.extend(state.confirmed_issues[:5])
    query_parts.extend(list(inputs.get("semantic_phrases") or [])[-10:])
    if state.unmatched_issues:
        query_parts.extend(state.unmatched_issues[:3])
    question = "；".join(item for item in query_parts if item).strip()
    if not question:
        return {
            "followup_basis_fingerprint": fingerprint,
            "followup_basis_error": "尚无可用于检索的稳定案情",
        }
    effective_domain = (
        state.legal_domain
        if state.legal_domain and state.legal_domain != "other"
        else ""
    )
    from src.agents.legal_knowledge.statute_rag import (
        _fetch_law_titles,
        search_statutes_raw,
    )

    statute_task = search_statutes_raw(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        domain=effective_domain,
        llm=deps.llm,
        use_hyde=False,
        use_rrf=bool(inputs.get("sparse_query")),
        sparse_query=str(inputs.get("sparse_query") or ""),
        skip_rerank=True,
    )
    graph_task = query_laws_and_channels(effective_domain, deps.neo4j_driver)
    raw_hits, graph_result = await asyncio.gather(
        asyncio.wait_for(statute_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_STATUTE),
        asyncio.wait_for(graph_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_GRAPH),
        return_exceptions=True,
    )
    errors: list[str] = []
    if isinstance(raw_hits, Exception):
        logger.warning("追问阶段法条检索失败: {}", raw_hits)
        law_hits: list[dict] = []
        errors.append("法条检索暂不可用")
    else:
        law_hits = list(raw_hits or [])[:8]
    if isinstance(graph_result, Exception):
        logger.warning("追问阶段知识图谱检索失败: {}", graph_result)
        graph_result = {"laws": [], "channels": []}
        errors.append("知识图谱暂不可用")

    law_titles: dict[str, str] = {}
    if law_hits and deps.db_session:
        try:
            law_titles = await asyncio.wait_for(
                _fetch_law_titles(law_hits, deps.db_session),
                timeout=settings.GUIDE_RETRIEVE_TIMEOUT_AUX,
            )
        except Exception as exc:
            logger.warning("追问阶段法律标题补充失败: {}", exc)
    refs = [
        {
            "source_type": "statute",
            "law_id": str(hit.get("law_id") or ""),
            "title": law_titles.get(str(hit.get("law_id") or ""), ""),
            "article_no": str(hit.get("article_no") or ""),
            "text": str(hit.get("text") or "")[:900],
        }
        for hit in law_hits
    ]
    graph_laws = [
        item for item in (graph_result.get("laws") or [])
        if isinstance(item, dict)
    ][:8]
    for item in graph_laws:
        title = str(item.get("title") or "").strip()
        if title and not any(ref.get("title") == title for ref in refs):
            refs.append({
                "source_type": "knowledge_graph",
                "title": title,
                "article_no": "",
                "text": str(item.get("category") or "适用法律关系")[:300],
            })
    authority_source = get_domain_followups(
        state.legal_domain or "other"
    ).source
    refs.append({
        "source_type": "official_process",
        "title": authority_source.title,
        "article_no": "",
        "text": authority_source.usage_note[:500],
        "issuer": authority_source.issuer,
        "url": authority_source.url,
    })
    logger.info(
        "追问阶段轻量检索完成（不含类案） | statutes={} graph_laws={}",
        len(refs), len(graph_laws),
    )
    emit_guide_progress(
        "followup_retrieval",
        "追问依据检索完成",
        f"已整理 {len(refs)} 条法条依据和 {len(graph_laws)} 条知识图谱依据。",
        status="completed",
    )
    return {
        "followup_basis_refs": refs,
        "followup_basis_graph": graph_laws,
        "followup_basis_fingerprint": fingerprint,
        "followup_basis_error": "；".join(errors),
    }


async def node_retrieve(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤内部检索：所有档位检索 statute+case+graph，HIGH 档额外自省。

    法条检索策略：
    - effective_domain 不为空 → domain-filtered + 全库 双路并发，RRF 融合
    - effective_domain 为空（domain=other）→ 仅全库向量检索
    避免 domain 识别错误时返回 0 条法律。
    """
    emit_guide_progress(
        "solution_retrieval",
        "正在检索方案依据",
        "收敛阶段检索法条、类案和可办理渠道，并核对来源。",
    )
    # ── 双查询构建：Dense 与 Sparse 走同一案情模型的两种投影 ────────────────
    # Dense 保留完整语义；Sparse 只保留已确认的关系、行为、请求、时间和
    # 程序词。两者都由原子事实生成，不再为具体行业维护关键词分支。
    domain = state.legal_domain
    retrieval_inputs = build_case_retrieval_inputs(
        state.confirmed_issues,
        active_case_facts(state.case_facts),
    )
    sparse_query = str(retrieval_inputs["sparse_query"])

    dense_parts: list[str] = []
    if domain_label := DOMAIN_LABELS.get(domain, ""):
        dense_parts.append(domain_label)
    if state.confirmed_issues:
        dense_parts.append("；".join(state.confirmed_issues))
        dense_parts.append("法律依据 权利义务")
    # 未标准化的口语描述：只进 Dense，保留用户原始意图不丢
    if state.unmatched_issues:
        dense_parts.append("；".join(state.unmatched_issues[:5]))
    # 对话积累的事实（取前3条）：例"在职3年、口头辞退"
    # → 召回经济补偿金计算/书面通知要求等具体条款
    if state.evidence_confirmed:
        dense_parts.append("、".join(state.evidence_confirmed[:3]))
    semantic_phrases = list(retrieval_inputs["semantic_phrases"])
    if semantic_phrases:
        dense_parts.append("；".join(semantic_phrases[-10:]))
    # Compatibility for states created before atomic case facts were added.
    # New conversations use case_facts exclusively, avoiding stale corrected
    # values; old persisted conversations still retain their semantic context.
    if not state.case_facts and state.collected_facts:
        dense_parts.append("；".join(state.collected_facts[-6:]))
    if state.time_info:
        dense_parts.append(state.time_info)
    if state.region:
        dense_parts.append(state.region)
    memories = _active_long_term_memories(state)
    if memories:
        dense_parts.append("；".join(memories))

    if not dense_parts:
        # 三层标准化全空且无口语池：用用户原话兜底，纯向量、不做 domain 过滤，
        # 至少给出语义相关法条，配合 LOW 档保守措辞，避免只回"信息不足"。
        raw_input = "\n".join(
            m.content for m in state.messages if isinstance(m, HumanMessage)
        )[-500:]
        dense_parts = [raw_input or "法律问题咨询"]
        domain = ""
        logger.warning("节点⑤无任何标准化产物，降级为原话全库检索 | chars={}", len(dense_parts[0]))

    question = " ".join(dense_parts)

    # domain="other" 时降级为全库检索：不过滤 domain，让向量语义兜底
    # （LLM 识别失败时不返回 0 条，代价是召回范围变宽）
    effective_domain = domain if domain and domain != "other" else ""

    logger.info(
        "节点⑤检索 | domain={} effective={} tier={} sparse={} dense_chars={}",
        domain, effective_domain or "(全库)", state.confidence_tier,
        sparse_query or "(空,关闭BM25)", len(question),
    )

    from src.agents.legal_knowledge.statute_rag import search_statutes_raw, format_statute_context, _fetch_law_titles
    from src.agents.legal_knowledge.case_rag import search_cases_context

    # 法条检索：拿原始结构化结果，不走生成式 QA
    # HyDE 策略：仅 HIGH 档使用（避免低质量问题被放大偏差）
    use_hyde = (state.confidence_tier == "HIGH")
    _statute_kwargs = dict(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        llm=deps.llm,
        use_hyde=use_hyde,
        use_rrf=bool(sparse_query),
        sparse_query=sparse_query,
    )
    # 双路法条检索：
    #   路径A — domain-filtered（精准，收敛到领域相关法律）
    #   路径B — 全库（semantic，兜底 domain 识别偏差）
    # effective_domain 有值时两路并发后 RRF 融合；
    # effective_domain 为空（domain=other）时只跑全库，不重复请求。
    if effective_domain:
        law_hits_domain_task = search_statutes_raw(domain=effective_domain, skip_rerank=True, **_statute_kwargs)
        law_hits_full_task   = search_statutes_raw(domain="",              skip_rerank=True, **_statute_kwargs)
    else:
        law_hits_domain_task = None
        law_hits_full_task   = search_statutes_raw(domain="", **_statute_kwargs)  # 单路保留内部 rerank

    case_task  = search_cases_context(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        db_session=deps.db_session,
        domain=effective_domain,
        sparse_query=sparse_query,
        llm=deps.llm,
        use_hyde=bool(state.unmatched_issues) and not bool(sparse_query),
    )
    graph_task = query_laws_and_channels(effective_domain, deps.neo4j_driver)

    # 并发检索，添加超时控制（避免慢查询拖垮整体响应）
    retrieval_failures = []

    if effective_domain:
        raw_domain, raw_full, case_result, graph_result = await asyncio.gather(
            asyncio.wait_for(law_hits_domain_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_STATUTE),
            asyncio.wait_for(law_hits_full_task,   timeout=settings.GUIDE_RETRIEVE_TIMEOUT_STATUTE),
            asyncio.wait_for(case_task,  timeout=settings.GUIDE_RETRIEVE_TIMEOUT_CASE),
            asyncio.wait_for(graph_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_GRAPH),
            return_exceptions=True,
        )
        hits_domain = raw_domain if not isinstance(raw_domain, Exception) else []
        hits_full   = raw_full   if not isinstance(raw_full,   Exception) else []
        if isinstance(raw_domain, Exception):
            logger.warning("statute_rag(domain) 失败: {}", raw_domain)
        if isinstance(raw_full, Exception):
            logger.warning("statute_rag(全库) 失败: {}", raw_full)
        if not hits_domain and not hits_full:
            retrieval_failures.append("法条检索")
        # 先 RRF 融合，保留候选与后面的 PG 字面结果一起做一次统一精排。
        law_hits = _rrf_fuse(hits_domain, hits_full, top_n=20)
        logger.info("RRF融合候选 | domain={} full={} fused={}",
                    len(hits_domain), len(hits_full), len(law_hits))
    else:
        raw_full, case_result, graph_result = await asyncio.gather(
            asyncio.wait_for(law_hits_full_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_STATUTE),
            asyncio.wait_for(case_task,  timeout=settings.GUIDE_RETRIEVE_TIMEOUT_CASE),
            asyncio.wait_for(graph_task, timeout=settings.GUIDE_RETRIEVE_TIMEOUT_GRAPH),
            return_exceptions=True,
        )
        if isinstance(raw_full, Exception):
            if isinstance(raw_full, asyncio.TimeoutError):
                logger.warning("statute_rag 超时（>8s），降级跳过")
            else:
                logger.error(f"statute_rag失败: {raw_full}")
            law_hits = []
            retrieval_failures.append("法条检索")
        else:
            law_hits = raw_full or []

    # PG 字面补充：向量有结果也可能语义漂移，始终补充领域内的原文字面命中，
    # 再与向量候选统一精排。只传标准术语，避免口语词污染 LIKE 查询。
    lexical_phrases = list(retrieval_inputs["lexical_phrases"])
    if deps.db_session and effective_domain and lexical_phrases:
        from src.agents.legal_knowledge.statute_rag import search_statutes_pg_fallback
        try:
            pg_hits = await asyncio.wait_for(
                search_statutes_pg_fallback(
                    effective_domain,
                    lexical_phrases,
                    deps.db_session,
                    limit=16,
                ),
                timeout=settings.GUIDE_RETRIEVE_TIMEOUT_AUX,
            )
            if pg_hits:
                combined: list[dict] = []
                seen_refs: set[tuple[str, str]] = set()
                for hit in pg_hits + law_hits:
                    ref = (str(hit.get("law_id", "")), str(hit.get("article_no", "")))
                    if ref not in seen_refs:
                        seen_refs.add(ref)
                        combined.append(hit)
                law_hits = combined
                if "法条检索" in retrieval_failures:
                    retrieval_failures.remove("法条检索")
                logger.info("PG+向量法条候选 | pg={} combined={}", len(pg_hits), len(combined))
        except Exception as pg_err:
            logger.error(f"PG 法条补充失败: {pg_err}")

    if not law_hits and state.followup_basis_refs:
        law_hits = [
            {
                "law_id": str(item.get("law_id") or ""),
                "article_no": str(item.get("article_no") or ""),
                "title": str(item.get("title") or ""),
                "text": str(item.get("text") or "")[:1200],
            }
            for item in state.followup_basis_refs
            if isinstance(item, dict) and item.get("text")
        ][:8]
        if law_hits:
            logger.info("最终法条检索为空，使用追问阶段法条兜底 | refs={}", len(law_hits))
            if "法条检索" in retrieval_failures:
                retrieval_failures.remove("法条检索")

    if effective_domain and law_hits:
        from src.agents.legal_knowledge.reranker import rerank_docs as _rerank
        candidate_count = len(law_hits)
        try:
            law_hits = await asyncio.wait_for(
                _rerank(question, law_hits, top_k=8),
                timeout=settings.GUIDE_RETRIEVE_TIMEOUT_RERANK,
            )
            logger.info("法条统一精排完成 | candidates={} final={}", candidate_count, len(law_hits))
        except Exception as rerank_err:
            logger.warning("法条精排超时或失败，保留融合候选顺序 | err={}", rerank_err)
            law_hits = law_hits[:8]

    fallback_guide = None
    similar_cases = []
    if isinstance(case_result, Exception):
        if isinstance(case_result, asyncio.TimeoutError):
            logger.warning("case_rag 超时（>{}s），降级跳过", settings.GUIDE_RETRIEVE_TIMEOUT_CASE)
        else:
            logger.error(f"case_rag失败: {case_result}")
        case_str = ""
        retrieval_failures.append("案例检索")
    else:
        case_str = case_result.get("context", "")
        similar_cases = case_result.get("cases", [])
        fallback_guide = case_result.get("fallback_guide")
    if isinstance(graph_result, Exception):
        if isinstance(graph_result, asyncio.TimeoutError):
            logger.warning("graph查询 超时（>3s），降级跳过")
        else:
            logger.error(f"graph查询失败: {graph_result}")
        graph_result = {"laws": [], "channels": []}
        retrieval_failures.append("知识图谱查询")

    # 格式化法条上下文（带标题+条号）
    law_titles: dict[str, str] = {}
    if law_hits and deps.db_session:
        try:
            law_titles = await asyncio.wait_for(
                _fetch_law_titles(law_hits, deps.db_session),
                timeout=settings.GUIDE_RETRIEVE_TIMEOUT_AUX,
            )
        except Exception as e:
            logger.warning(f"获取法律标题失败（PostgreSQL不可用），降级显示: {e}")
    # primary_count=5：前5条作为核心法条，确保关键法律依据被充分展示
    law_context_formatted = format_statute_context(law_hits, law_titles, primary_count=5)
    retrieved_law_refs = [
        {
            "law_id": str(hit.get("law_id") or ""),
            "title": law_titles.get(str(hit.get("law_id") or ""), ""),
            "article_no": str(hit.get("article_no") or ""),
            "text": str(hit.get("text") or "")[:1200],
        }
        for hit in law_hits[:8]
    ]

    # 渠道是精确结构化数据：以 PostgreSQL 为主库，按专属渠道、公共法律服务、
    # 12345 兜底分层查询。数据库异常时 Repository 内部返回最小全国渠道。
    try:
        channels = await asyncio.wait_for(
            query_recommended_channels(
                domain=domain,
                region=state.region,
                db=deps.db_session,
                limit=6,
            ),
            timeout=settings.GUIDE_RETRIEVE_TIMEOUT_AUX,
        )
    except Exception as channel_err:
        logger.warning("渠道查询超时或失败，使用空渠道降级 | err={}", channel_err)
        channels = []

    graph_laws = graph_result.get("laws", [])

    # 如果多个检索服务失败，添加降级提示
    retrieval_error_note = ""
    if len(retrieval_failures) >= 2:
        retrieval_error_note = (
            f"\n\n⚠️ **系统提示**：{' 和 '.join(retrieval_failures)} 服务异常，"
            "以下建议基于有限信息。建议稍后重试或直接拨打 **12348** 法律援助热线获取专业指导。"
        )

    updates = {
        "candidate_laws": graph_laws,
        "retrieved_law_refs": retrieved_law_refs,
        "similar_cases": similar_cases,
        "relevant_channels": channels,
        "law_context_str": law_context_formatted or "",
        "case_context_str": case_str or "",
        "retrieval_error_note": retrieval_error_note,
        "fallback_guide": fallback_guide,  # 案例检索兜底指引
        "last_confirmed_count": len(state.confirmed_issues),  # 记录本次检索时的 issue 数量
        "retrieval_completed": True,
        "retrieval_fingerprint": _retrieval_fingerprint(state),
    }

    # 仅 HIGH 档做自省（启发式判断：法条适用性/时效/管辖）
    if state.confidence_tier == "HIGH" and law_context_formatted:
        case_summary = f"法律问题：{'; '.join(state.confirmed_issues)}\n已有证据：{'; '.join(state.evidence_confirmed) or '无'}"
        review_prompt = SELF_REVIEW_PROMPT.format(
            case_summary=case_summary,
            law_context=law_context_formatted[:2000],  # 截取避免过长
        )
        try:
            review_resp = await ainvoke_bounded(
                llm_for_stage(deps.llm, max_tokens=600),
                [SystemMessage(content=review_prompt)],
                timeout=settings.GUIDE_LLM_TIMEOUT_AUDIT,
                stage="retrieval_self_review",
            )
            content = review_resp.content.strip()
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()
            review = json.loads(content)
            if not review.get("ok", True):
                concern = review.get("concern", "法条适用存疑")
                logger.warning("节点⑤自省降档 | HIGH→MID，原因: {}", concern)
                updates["confidence_tier"] = "MEDIUM"
                updates["self_review_note"] = f"\n⚠️ **降档说明**：{concern}"
        except Exception as e:
            logger.warning(f"自省失败，保持原档: {e}")

    return updates


def _scenario_confirmation_plan(report: dict) -> dict:
    """User-facing plain-language confirmation, never a legal-domain picker."""
    question = str(report.get("confirmation_question") or "").strip() or (
        "根据目前信息，您的情况更接近哪一种？"
    )
    options = [
        str(item).strip()
        for item in (report.get("confirmation_options") or [])
        if str(item).strip()
    ]
    options = list(dict.fromkeys([*options, "不清楚/无法确认"]))[:6]
    if len(options) < 2:
        options = ["其他情况", "不清楚/无法确认"]
    return {
        "should_ask": True,
        "plan_kind": "scenario_confirmation",
        "ask_type": "facts",
        "questions": [{
            "field_id": "scenario_confirmation",
            "candidate_id": "scenario_confirmation",
            "question": question,
            "input_type": "single_choice",
            "options": options,
            "placeholder": "",
            "answer_hint": "选择最接近您经历的一项，不确定时直接选择“不清楚/无法确认”。",
            "required": False,
            "decision_effects": ["scenario"],
            "reason": "用于确认最接近的实际场景，避免按错误领域继续追问或检索。",
            "basis_refs": [],
            "official_source": {},
        }],
        "planner_mode": "scenario_confirmation",
    }


async def node_assess_retrieve(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：先做场景再判断，再按已确认场景做领域追问和检索。"""
    emit_guide_progress(
        "evidence_assessment",
        "正在更新证据评估与清单",
        "结合当前事实、已提交材料和检索依据检查证明目标与证据缺口。",
    )
    scenario_report = await assess_scenario(state, _fast_llm_for(deps))
    scenario_updates = {"scenario_analysis": scenario_report.model_dump()}
    if (
        scenario_report.confidence < 0.65
        and scenario_report.discriminating_facts
        and scenario_report.confirmation_options
        and not state.scenario_confirmation_offered
        and not (
            state.force_conclude
            or state.wants_conclude
            or state.supplement_choice == "conclude"
        )
    ):
        logger.info(
            "场景置信度不足，先请用户确认场景 | confidence={} candidates={}",
            scenario_report.confidence,
            scenario_report.competing_scenarios,
        )
        return {
            **scenario_updates,
            "scenario_confirmation_offered": True,
            "followup_plan": _scenario_confirmation_plan(
                scenario_report.model_dump()
            ),
        }
    if (
        scenario_report.confidence >= 0.65
        and scenario_report.primary_domain in DOMAIN_LABELS
        and scenario_report.primary_domain != state.legal_domain
    ):
        scenario_updates["legal_domain"] = scenario_report.primary_domain
        scenario_updates["case_frame"] = (
            scenario_report.primary_frame or state.case_frame
        )
        logger.info(
            "场景再判断调整领域 | old={} new={}",
            state.legal_domain,
            scenario_report.primary_domain,
        )
    state = state.model_copy(update=scenario_updates)
    evidence_report = evaluate_state_evidence(state)
    authority_source = get_domain_followups(
        state.legal_domain or "other"
    ).source
    official_basis_ref = {
        "source_type": "official_process",
        "title": authority_source.title,
        "article_no": "",
        "text": authority_source.usage_note[:500],
        "issuer": authority_source.issuer,
        "url": authority_source.url,
    }
    existing_basis_refs = (
        state.followup_basis_refs or state.retrieved_law_refs or []
    )
    evidence_requirements, requirement_version = merge_evidence_requirements(
        state,
        evidence_report,
        basis_refs=[official_basis_ref, *existing_basis_refs],
    )
    evidence_updates = {
        "evidence_items": [
            item.model_dump() for item in evidence_report.items
        ],
        "proof_targets": [
            item.model_dump() for item in evidence_report.targets
        ],
        "evidence_requirements": evidence_requirements,
        "evidence_requirement_version": requirement_version,
        "evidence_links": [
            item.model_dump() for item in evidence_report.links
        ],
        "evidence_coverage": evidence_report.model_dump(),
    }
    evidence_state = state.model_copy(update=evidence_updates)
    score_updates = await node_score(evidence_state, deps)
    scored_state = evidence_state.model_copy(update=score_updates)
    sufficiency = assess_decision_sufficiency(scored_state)
    assessed_state = scored_state.model_copy(
        update={"decision_sufficiency": sufficiency.model_dump()}
    )
    should_stop, force = should_conclude(
        assessed_state,
        max_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS,
    )
    user_requested_followup = (
        assessed_state.supplement_choice == "continue"
        and assessed_state.allow_extra_followups
    )
    ask_round_limit = (
        settings.GUIDE_MAX_OPT_IN_ASK_ROUNDS
        if user_requested_followup
        else settings.GUIDE_MAX_ASK_ROUNDS
    )
    hard_stop = (
        state.force_conclude
        or force
        or assessed_state.wants_conclude
        or assessed_state.supplement_choice == "conclude"
        or assessed_state.ask_rounds >= ask_round_limit
        or assessed_state.consecutive_low_info_answers >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
    )
    basis_updates: dict = {}
    if hard_stop:
        mode = (
            "decision_sufficient"
            if sufficiency.sufficient_for_definitive_plan and not force
            else "converged"
        )
        followup_plan = {"should_ask": False, "planner_mode": mode}
    else:
        # 两阶段追问：先按领域题库把剩余必问事实作为一张表单一次问完
        # （固定阶段），全部覆盖后再进入动态补充。固定阶段是纯目录驱动，
        # 不触发法条检索；"为什么问"用 rule.why 呈现。
        fixed_plan = await plan_fixed_batch(assessed_state, _fast_llm_for(deps))
        if fixed_plan.get("should_ask"):
            followup_plan = fixed_plan
        else:
            # 追问阶段只检索法条和知识图谱，不检索类案。检索结果既驱动
            # 动态表单，也成为证据需求的可追溯依据。
            batch_capable = all(
                getattr(deps, name, None) is not None
                for name in ("embedding_model", "milvus_client", "neo4j_driver")
            )
            basis_updates = await node_retrieve_followup_basis(assessed_state, deps)
            if basis_updates:
                assessed_state = assessed_state.model_copy(update=basis_updates)
                evidence_requirements, requirement_version = merge_evidence_requirements(
                    assessed_state,
                    evidence_report,
                    basis_refs=[official_basis_ref, *assessed_state.followup_basis_refs],
                )
                evidence_updates.update({
                    "evidence_requirements": evidence_requirements,
                    "evidence_requirement_version": requirement_version,
                })
            followup_plan = (
                await plan_followup_batch(assessed_state, _fast_llm_for(deps))
                if batch_capable
                else await plan_next_followup(assessed_state, _fast_llm_for(deps))
            )
            if (
                batch_capable
                and
                not followup_plan.get("should_ask")
                and evidence_requirements
                and not assessed_state.evidence_collection_offered
            ):
                followup_plan = {
                    "should_ask": True,
                    "plan_kind": "evidence_collection",
                    "ask_type": "evidence_collection",
                    "questions": [],
                    "evidence_checklist": [
                        item for item in evidence_requirements if item.get("active", True)
                    ],
                    "planner_mode": "facts_converged_evidence_collection",
                }

    # 类案、渠道和完整检索只在真正生成方案时执行。
    retrieval_updates: dict = {}
    concluding = hard_stop or not bool(followup_plan.get("should_ask"))
    if concluding:
        if _retrieval_snapshot_reusable(assessed_state):
            logger.info(
                "节点⑤复用检索快照 | fingerprint={}",
                assessed_state.retrieval_fingerprint or "(legacy)",
            )
        else:
            logger.info("节点⑤进入最终收敛，执行完整知识检索")
            retrieval_updates = await node_retrieve(assessed_state, deps)
            assessed_state = assessed_state.model_copy(update=retrieval_updates)
            evidence_requirements, requirement_version = merge_evidence_requirements(
                assessed_state,
                evidence_report,
                basis_refs=[official_basis_ref, *assessed_state.retrieved_law_refs],
            )
            evidence_updates.update({
                "evidence_requirements": evidence_requirements,
                "evidence_requirement_version": requirement_version,
            })
    else:
        logger.info(
            "节点⑤继续动态追问/收集证据，本轮不检索类案 | kind={} questions={}",
            followup_plan.get("plan_kind") or "followup_form",
            len(followup_plan.get("questions") or []),
        )

    trace = followup_plan.get("decision_trace")
    trace_history = list(assessed_state.followup_decision_trace)
    if trace and (not trace_history or trace_history[-1] != trace):
        trace_history = [*trace_history, trace][-50:]
    return {
        **evidence_updates,
        **score_updates,
        **basis_updates,
        **retrieval_updates,
        **scenario_updates,
        "last_confirmed_count": len(assessed_state.confirmed_issues),
        "force_conclude": state.force_conclude or force,
        "followup_plan": followup_plan,
        "followup_decision_trace": trace_history,
        "decision_sufficiency": sufficiency.model_dump(),
        "evidence_collection_offered": (
            True
            if followup_plan.get("plan_kind") == "evidence_collection"
            else state.evidence_collection_offered
        ),
    }


def _followup_authority_hint(state: GuideState, *, ask_type: str, reason: str) -> str:
    reason = _normalized_followup_reason(reason)
    source = get_domain_followups(state.legal_domain).source
    if source.authority_level == "system_guidance":
        return f"追问依据：这是通用案情整理规则，用于{reason}，不是官方固定问卷。"
    label = "事实栏目" if ask_type == "facts" else "证据和材料要素"
    source_link = f"[{source.title}]({source.url})" if source.url else source.title
    return (
        f"追问依据：参考{source.issuer}发布的{source_link}中的{label}整理，"
        f"用于{reason}；不是要求您必须提交的固定材料。"
    )


def _normalized_followup_reason(reason: str) -> str:
    """将题库中的“用于/为了”前缀统一剥离，避免面向用户出现重复介词。"""
    value = str(reason or "").strip().rstrip("。；")
    for prefix in ("为了用于", "用于", "为了"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
            break
    return value or "判断下一步处理方式"


def _display_question(value: str) -> str:
    """Normalize trailing punctuation for every follow-up presentation path."""

    question = " ".join(str(value or "").split()).strip()
    if not question:
        return ""
    if question.endswith(("？", "?")):
        question = re.sub(r"[。；，、：:]+$", "", question[:-1].rstrip())
    else:
        question = re.sub(r"[。；，、：:]+$", "", question)
    return question + "？"


def _user_facing_case_text(value: str) -> str:
    text = " ".join(str(value or "").split()).strip("。；， ")
    text = re.sub(r"^用户(?:声称|称|表示|提到)", "您提到", text)
    text = re.sub(r"^用户", "您", text)
    return text.replace("用户本人", "您本人").replace("将用户", "将您")


def _distinct_case_atoms(state: GuideState) -> list[dict]:
    """Collapse repeated model paraphrases for display without hiding real facts."""
    def semantic_tokens(value: str) -> set[str]:
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(value or ""))
        tokens = {text[index:index + 2] for index in range(max(len(text) - 1, 0))}
        return tokens - {
            "用户", "已经", "现在", "目前", "这个", "那个", "情况", "问题",
            "相关", "进行", "表示", "提到", "现场", "发生", "发现",
        }

    def fact_score(item: dict) -> int:
        structured = sum(bool(item.get(field)) for field in ("subject", "relation", "value"))
        return len(str(item.get("statement") or "")) + 12 * structured + 8 * bool(
            re.search(r"\d", str(item.get("statement") or ""))
        )

    result: list[dict] = []
    for item in active_case_facts(state.case_facts):
        if item.get("category") == "evidence" or item.get("status") != "asserted":
            continue
        value = re.sub(r"\W+", "", str(item.get("value") or "").lower())
        duplicate_index = next(
            (
                index for index, old in enumerate(result)
                if item.get("key") == old.get("key")
                or (
                    value and len(value) >= 2
                    and item.get("category") == old.get("category")
                    and value == re.sub(r"\W+", "", str(old.get("value") or "").lower())
                    and (
                        item.get("subject") == old.get("subject")
                        or item.get("relation") == old.get("relation")
                    )
                )
                or (
                    item.get("category") == old.get("category") == "event"
                    and (
                        str(item.get("key") or "").startswith("legacy.raw.")
                        or str(old.get("key") or "").startswith("legacy.raw.")
                    )
                    and semantic_tokens(item.get("statement", ""))
                    & semantic_tokens(old.get("statement", ""))
                )
            ),
            None,
        )
        if duplicate_index is None:
            result.append(dict(item))
            continue
        old = result[duplicate_index]
        if fact_score(item) > fact_score(old):
            result[duplicate_index] = dict(item)
    return result


def _format_case_summary(state: GuideState) -> str:
    labels = {
        "actor": "相关主体", "relationship": "关系", "event": "经过",
        "claim": "诉求", "amount": "金额", "time": "时间", "location": "地点",
        "procedure": "沟通或处理", "harm": "损失或影响", "uncertainty": "待核实",
    }
    buckets: dict[str, list[str]] = {}
    for item in _distinct_case_atoms(state):
        category = str(item.get("category") or "event")
        label = labels.get(category, "其他事实")
        if category in {"amount", "time", "location"}:
            value = _user_facing_case_text(item.get("value") or item.get("statement", ""))
        else:
            value = _user_facing_case_text(item.get("statement", ""))
        if value and value not in buckets.setdefault(label, []):
            buckets[label].append(value)
    ordered_labels = [
        "经过", "相关主体", "关系", "地点", "金额", "时间",
        "损失或影响", "沟通或处理", "诉求", "待核实", "其他事实",
    ]
    parts = [
        f"{label}：{'、'.join(buckets[label])}"
        for label in ordered_labels if buckets.get(label)
    ]
    return "；".join(parts)


def _fact_assessments_for_prompt(state: GuideState) -> str:
    """Keep stale auxiliary records from contradicting the primary detail store."""

    records = dict(state.fact_records)
    for rule_id, record in list(records.items()):
        if not isinstance(record, dict) or record.get("status") not in {"ambiguous", "conflicted"}:
            continue
        rule = find_fact_followup(state.legal_domain, rule_id)
        if not rule:
            continue
        coverage = candidate_coverage(rule.slot, state)
        if coverage.get("known") and not coverage.get("missing"):
            records.pop(rule_id, None)
    return format_fact_assessments(records)


def _followup_case_anchor(state: GuideState, limit: int = 72) -> str:
    current = [
        item for item in _distinct_case_atoms(state)
        if int(item.get("turn") or 0) == int(state.round or 0)
    ]
    selected = current or _distinct_case_atoms(state)[-3:]
    statements: list[str] = []
    for item in selected[-3:]:
        statement = _user_facing_case_text(item.get("statement", ""))
        if statement and statement not in statements:
            statements.append(statement)
    value = "；".join(statements)
    return value[:limit].rstrip("；，。 ")


def _followup_opening(state: GuideState) -> str:
    if state.supplement_choice == "continue":
        return "好的，我们继续，只补充真正会影响方案的信息。"
    latest_statements = list(dict.fromkeys(
        _user_facing_case_text(item.get("statement", ""))
        for item in active_case_facts(state.case_facts)
        if int(item.get("turn") or 0) == state.round and item.get("statement")
    ))
    if latest_statements:
        visible = latest_statements[:6]
        recorded = "；".join(visible)
        if len(latest_statements) > len(visible):
            recorded += f"等共{len(latest_statements)}项"
        return f"好的，{recorded}，我已经记下。"
    acknowledgement = str(state.followup_plan.get("acknowledgement") or "").strip()
    if acknowledgement:
        return f"{acknowledgement.rstrip('。')}。"
    latest = next(
        (str(message.content).strip() for message in reversed(state.messages) if isinstance(message, HumanMessage)),
        "",
    )
    if latest:
        return "好的，您刚补充的内容我已经记录。"
    issues = "、".join(state.confirmed_issues[:2]) or f"{DOMAIN_LABELS.get(state.legal_domain, '法律')}问题"
    return f"我会继续按“{issues}”帮您梳理。"


def _pending_fraud_warning(state: GuideState) -> bool:
    return bool(state.fraud_stop_loss_warning and not state.fraud_stop_loss_offered)


def _prepend_fraud_warning(state: GuideState, reply: str) -> str:
    if not _pending_fraud_warning(state):
        return reply
    return f"{state.fraud_stop_loss_warning}\n\n---\n\n{reply}"


def _fraud_warning_display_updates(state: GuideState) -> dict:
    if not _pending_fraud_warning(state):
        return {}
    return {
        "fraud_stop_loss_relevant": True,
        "fraud_stop_loss_warning": "",
        "fraud_stop_loss_offered": True,
    }


def _format_followup_reply(
    state: GuideState,
    question: str,
    *,
    ask_type: str,
    reason: str,
    answer_hint: str = "",
    rule_id: str = "",
) -> str:
    """每轮只问一个关键问题，后台评估不增加用户的表单负担。"""
    question = _display_question(question)
    reason = _normalized_followup_reason(reason)
    contextual_reason = str(state.followup_plan.get("contextual_reason") or "").strip().rstrip("。；")
    if contextual_reason:
        purpose = f"{contextual_reason}。"
    elif ask_type == "evidence":
        purpose = f"这项材料主要用于{reason}。"
    else:
        purpose = f"再确认这一点是为了{reason}。"
    if ask_type == "evidence":
        hint = "没有、暂时找不到或不确定都可以直接说，我会同时给出替代办法。"
    else:
        hint = answer_hint or "不清楚时可以说大概情况或“不知道”。"
    if state.followup_plan:
        authority = format_followup_authority(state.followup_plan)
    else:
        authority = _followup_authority_hint(state, ask_type=ask_type, reason=reason)
    authority = re.sub(r"^追问依据[：:]\s*", "", authority).strip()
    reply = "\n\n".join([
        "### 已记录",
        _followup_opening(state),
        "### 请确认",
        f"> **{question}**",
        f"**回答提示：** {hint}",
        "### 为什么要问",
        f"- **用途：** {purpose}\n- **追问依据：** {authority}",
        "---",
        "暂时不方便补充时，直接回复 **“现在生成方案”**，我会按现有信息给出建议。",
    ])
    return _prepend_fraud_warning(state, reply)


def _format_followup_batch_reply(
    state: GuideState,
    questions: list[dict],
    *,
    planner_mode: str = "",
) -> str:
    is_fixed = str(planner_mode).startswith("fixed_")
    if is_fixed:
        stage_heading = (
            f"### 请补充目前尚未明确的信息\n\n"
            f"我们先按【{DOMAIN_LABELS.get(state.legal_domain, '您的纠纷')}】常见情形核对以下 "
            f"**{len(questions)} 个关键信息**，请尽量逐项填写，也可以在聊天框一次性回答。"
        )
    else:
        stage_heading = (
            "### 请补充目前尚未明确的信息\n\n"
            f"本轮根据当前细节库和检索依据生成 **{len(questions)} 个问题**。"
            "您可以填写下方表单，也可以直接在聊天框一次性回答。"
        )
    lines = [
        "### 已记录",
        _followup_opening(state),
        "",
        stage_heading,
    ]
    input_labels = {
        "short_text": "简短填写",
        "long_text": "详细说明",
        "single_choice": "单选",
        "multi_choice": "可多选",
    }
    for index, item in enumerate(questions, start=1):
        question = _display_question(str(item.get("question") or ""))
        lines.extend(["", f"#### {index}. {question}"])
        options = [str(value) for value in (item.get("options") or []) if value]
        if options:
            marker = "[ ]" if item.get("input_type") == "multi_choice" else "○"
            lines.append(" ".join(f"{marker} {value}" for value in options))
        lines.append(
            f"- **填写方式：** {input_labels.get(str(item.get('input_type') or ''), '自由填写')}"
        )
        if item.get("answer_hint"):
            lines.append(f"- **回答提示：** {item['answer_hint']}")
        lines.append(f"- **影响判断：** {item.get('reason') or '下一步处理方式'}")
        basis = [row for row in (item.get("basis_refs") or []) if isinstance(row, dict)]
        if basis:
            lines.append("- **本轮依据：**")
            for row in basis[:2]:
                label = "《{}》{}".format(
                    row.get("title") or "本轮检索法律",
                    row.get("article_no") or "相关规定",
                )
                lines.append(f"  - **{label}**")
                basis_text = " ".join(str(row.get("text") or "").split())[:360]
                if basis_text:
                    content_label = (
                        "条文内容"
                        if row.get("source_type") == "statute"
                        else "依据内容"
                    )
                    lines.append(f"    - **{content_label}：** {basis_text}")
    lines.extend([
        "",
        "---",
        "每项都可以填写 **“不清楚”**。如不想继续补充，可选择 **“按现有信息生成方案”**。",
    ])
    return _prepend_fraud_warning(state, "\n".join(lines))


def _reference_template_note(state: GuideState) -> str:
    """已上传但被识别为空白模板/参考资料的文件，提示未计入已提交证据。"""
    names: list[str] = []
    for record in (state.evidence_assessments or {}).values():
        if not isinstance(record, dict):
            continue
        if str(record.get("availability") or "") != "uploaded_copy":
            continue
        if str(record.get("case_specificity") or "") != "blank_or_reference":
            continue
        name = str(record.get("canonical_item") or "").strip()
        if name:
            names.append(name)
    if not names:
        return ""
    unique = dict.fromkeys(names)
    return (
        "\n> **提示：** 以下上传文件被识别为空白模板或参考资料，未计入已提交证据："
        + "、".join(unique)
        + "\n> 请补充包含本案具体主体、时间、金额或处理结果的实际记录。"
    )


def _format_evidence_collection_reply(state: GuideState, requirements: list[dict]) -> str:
    status_labels = {
        "preliminarily_covered": "已初步覆盖",
        "partially_covered": "部分覆盖",
        "known_missing": "目前缺失",
        "conflicted": "状态有冲突",
        "submitted": "已提交（待核验）",
        "unresolved": "待提交/确认",
    }
    lines = [
        "### 事实阶段已自然收敛",
        "目前没有足以明显改变方案的高价值事实问题。证据清单已根据细节库增量整理，您现在可以集中提交材料。",
        "",
        f"### 证据准备清单（版本 {state.evidence_requirement_version}）",
    ]
    decay_banner = evidence_decay_banner(requirements)
    if decay_banner:
        lines.extend(["", decay_banner])
    for index, item in enumerate(requirements, start=1):
        is_retrieve = item.get("collect_mode") == "retrieve"
        raw_status = str(item.get("status") or "")
        status_label = (
            "待提供线索"
            if is_retrieve and raw_status == "unresolved"
            else status_labels.get(raw_status, item.get("status") or "待确认")
        )
        lines.extend([
            "",
            f"#### {index}. {item.get('label') or '相关材料'}",
            f"- **证明目标：** {item.get('proof_target') or '核对相关案件事实'}",
            f"- **当前状态：** {status_label}",
        ])
        alternatives = [str(value) for value in (item.get("alternatives") or []) if value]
        if alternatives:
            lines.append(f"- **替代材料：** {'、'.join(alternatives[:4])}")
        if item.get("next_action"):
            if is_retrieve:
                lines.append(f"- **需提供的调取线索：** {item['next_action']}")
            else:
                lines.append(f"- **提交建议：** {item['next_action']}")
    reference_note = _reference_template_note(state)
    if reference_note:
        lines.append(reference_note)
    lines.extend([
        "",
        "### 下一步",
        "标注「提供线索」的项目（如公共区域监控）无需上传视频，在对应项填写事发时间与位置即可；「统一上传材料」适用于您可自行持有的材料。",
        "已提交的材料可在证据中心重新上传以更新版本，也可继续补充其他材料；每个清单项旁都有对应的“上传此项”提交口。支持 PDF、DOCX、TXT 和常见图片。",
        "上传后系统会评估材料可能用途、覆盖范围、完整性、主体/时间可见性和补强方向；不直接认定真实性或可采性。",
        "",
        "如果暂时不提交证据，可选择 **“按现有信息生成方案”**，方案会明确标注未核验和证据缺口。",
    ])
    return _prepend_fraud_warning(state, "\n".join(lines))


async def node_ask_facts(state: GuideState, deps: GuideDeps) -> dict:
    """Compatibility wrapper around the single dynamic follow-up planner."""
    return await _ask_from_dynamic_plan(state, deps, preferred_type="facts")


async def node_ask_evidence(state: GuideState, deps: GuideDeps) -> dict:
    """Compatibility wrapper around the single dynamic follow-up planner."""
    return await _ask_from_dynamic_plan(state, deps, preferred_type="evidence")


async def _ask_from_dynamic_plan(
    state: GuideState,
    deps: GuideDeps,
    *,
    preferred_type: str = "",
) -> dict:
    if state.followup_plan:
        plan = state.followup_plan
    else:
        # Direct callers and persisted pre-batch states may not carry the
        # retrieval dependencies or the new decision-sufficiency snapshot.
        # Keep their established single-question planner path intact; normal
        # graph execution prepares both before reaching this node.
        batch_capable = all(
            getattr(deps, name, None) is not None
            for name in ("embedding_model", "milvus_client", "neo4j_driver")
        ) and bool(state.decision_sufficiency)
        plan = (
            await plan_followup_batch(state, _fast_llm_for(deps))
            if batch_capable
            else await plan_next_followup(state, _fast_llm_for(deps))
        )
    if not plan.get("should_ask"):
        return {}
    plan_kind = str(plan.get("plan_kind") or "followup_form")
    if plan_kind == "evidence_collection":
        requirements = [
            item for item in (plan.get("evidence_checklist") or state.evidence_requirements)
            if isinstance(item, dict) and item.get("active", True)
        ]
        reply = _format_evidence_collection_reply(state, requirements)
        return {
            **_fraud_warning_display_updates(state),
            "phase": GuidePhase.DETAIL_GATHER,
            "evidence_rounds": state.evidence_rounds + 1,
            "pending_ask_details": ["请按证据清单提交材料，或说明暂不提交"],
            "pending_ask_type": "evidence_collection",
            "pending_followup_ids": [],
            "followup_plan": plan,
            "evidence_collection_offered": True,
            "messages": [AIMessage(content=reply)],
        }

    questions = [
        item for item in (plan.get("questions") or [])
        if isinstance(item, dict) and item.get("question")
    ]
    if questions:
        planner_mode = str(plan.get("planner_mode") or "")
        reply = _format_followup_batch_reply(state, questions, planner_mode=planner_mode)
        question_texts = [str(item["question"]).strip() for item in questions]
        candidate_ids = [
            str(item.get("candidate_id") or "").strip()
            for item in questions if item.get("candidate_id")
        ]
        decision_keys = [
            str(item.get("field_id") or item.get("candidate_id") or "").strip()
            for item in questions
            if item.get("field_id") or item.get("candidate_id")
        ]
        logger.info(
            "节点⑥动态批量追问 | questions={} mode={}",
            len(questions), planner_mode,
        )
        # 固定阶段是一次性覆盖：展示过即视为已问（与动态批次的"显示≠已回答"
        # 语义区分开，固定剧本不会重新生成）。pending_followup_ids 仍保留，
        # 供 parse_details 解析答案写入 fact_records。
        fixed_marked_asked = (
            _merge_unique(state.asked_followup_ids, candidate_ids)
            if planner_mode.startswith("fixed_")
            else state.asked_followup_ids
        )
        return {
            **_fraud_warning_display_updates(state),
            "phase": GuidePhase.DETAIL_GATHER,
            "ask_rounds": state.ask_rounds + 1,
            "facts_rounds": state.facts_rounds + 1,
            # A displayed field is not treated as answered.  Only fields the
            # user actually submits are added to the cross-turn dedupe store.
            "asked_details": state.asked_details,
            "pending_ask_details": question_texts,
            "pending_ask_type": "facts",
            "asked_followup_ids": fixed_marked_asked,
            "pending_followup_ids": candidate_ids,
            "asked_decision_keys": state.asked_decision_keys,
            "followup_plan": plan,
            "messages": [AIMessage(content=reply)],
        }

    # Backward-compatible single-question plan for persisted states/tests.
    ask_type = str(plan.get("ask_type") or preferred_type or "facts")
    question = str(plan.get("question") or "").strip()
    if not question:
        return {}
    planned_state = state.model_copy(update={"followup_plan": plan})
    reply = _format_followup_reply(
        planned_state,
        question,
        ask_type=ask_type,
        reason=str(plan.get("reason") or "判断下一步处理方式"),
        answer_hint=str(plan.get("answer_hint") or ""),
        rule_id=str(plan.get("candidate_id") or ""),
    )
    candidate_id = str(plan.get("candidate_id") or "").strip()
    decision_key = str(plan.get("decision_key") or candidate_id).strip()
    pending_ids = [candidate_id] if candidate_id else []
    logger.info(
        "节点⑥动态追问 | type={} decision={} candidate={} gain={} burden={} mode={}",
        ask_type, decision_key, candidate_id, plan.get("information_gain"),
        plan.get("user_burden"), plan.get("planner_mode"),
    )
    return {
        **_fraud_warning_display_updates(state),
        "phase": GuidePhase.DETAIL_GATHER,
        "ask_rounds": state.ask_rounds + 1,
        "facts_rounds": state.facts_rounds + (1 if ask_type == "facts" else 0),
        "evidence_rounds": state.evidence_rounds + (1 if ask_type == "evidence" else 0),
        "asked_details": _merge_unique(state.asked_details, [question]),
        "pending_ask_details": [question],
        "pending_ask_type": ask_type,
        "asked_followup_ids": _merge_unique(state.asked_followup_ids, pending_ids),
        "pending_followup_ids": pending_ids,
        "asked_decision_keys": _merge_unique(state.asked_decision_keys, [decision_key]),
        "followup_plan": plan,
        "messages": [AIMessage(content=reply)],
    }


async def node_ask_followup(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑥：展示动态批量表单或事实收敛后的证据清单。"""
    return await _ask_from_dynamic_plan(state, deps)


async def _answer_counter_question(
    state: GuideState,
    deps: GuideDeps,
    user_question: str,
) -> str:
    """Answer a user's interruption before restoring the pending question."""
    if not user_question:
        return "您的疑问我看到了，但根据目前信息还不能确定具体答案。"
    prompt = COUNTER_QUESTION_RESPONSE_PROMPT.format(
        user_question=user_question,
        case_context=format_case_context(state.case_facts) or "（当前案情仍在整理）",
        law_context=state.law_context_str or "（当前没有足够的已检索法律依据）",
    )
    try:
        response = await ainvoke_bounded(
            llm_for_stage(_fast_llm_for(deps), max_tokens=600),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_FOLLOWUP,
            stage="counter_question",
        )
        answer = " ".join(str(response.content or "").split())
        if answer:
            return answer[:500]
    except Exception as exc:
        logger.warning("回答用户反问失败，使用安全降级说明 | err={}", exc)
    return "根据目前已经确认的信息和法律依据，这个问题还不能可靠确定，我会在最终方案中标明判断条件。"


async def node_parse_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦：解析用户对追问的回答，提取证据/地区/时间信息。

    若用户本轮没有回答而是反问，则不抽取任何信息、保留 pending_ask_details，
    把反问记入 deferred_questions，并原样重述待答问题（不消耗 ask_rounds）。
    """
    emit_guide_progress(
        "detail_update",
        "正在更新事实细节库",
        "只解析用户本轮填写或明确补充的内容，并检查是否与已有事实冲突。",
    )
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if not last_msg or not state.pending_ask_details:
        return {}
    if state.wants_conclude and not _current_turn_contains_case_details(state, last_msg):
        # A pure flow-control command changes routing only. It is not a fact,
        # evidence item or document statement, and does not need an LLM parse.
        return {
            "pending_ask_details": [],
            "pending_ask_type": "",
            "pending_followup_ids": [],
            "followup_plan": {},
            "phase": GuidePhase.ISSUE_SEARCH,
        }
    answer_narrative, uploaded_observations = split_uploaded_evidence_blocks(
        last_msg
    )
    structured_answers = _structured_followup_answers(answer_narrative)
    assessment_narrative = (
        "\n".join(structured_answers.values())
        if structured_answers
        else answer_narrative
    ).strip()
    evidence_only_turn = bool(uploaded_observations) and not answer_narrative.strip()
    attachment_inventory = "\n".join(
        f"- {item['name']}（系统已收到副本）"
        for item in uploaded_observations
    )
    answer_for_parse = (
        "\n".join(structured_answers.values())
        if structured_answers
        else answer_narrative
        or ("用户本轮仅提交了附件" if uploaded_observations else last_msg)
    )
    if attachment_inventory:
        answer_for_parse += (
            "\n\n本轮附件清单：\n"
            + attachment_inventory
            + "\n附件全文由证据子系统单独处理，不得直接写成用户确认的案情事实。"
        )
    inspection_task = asyncio.create_task(
        inspect_uploaded_evidence_blocks(last_msg, deps.llm)
    )
    prompt_questions = [
        f"- [{item.get('field_id')}] {item.get('question')}"
        for item in (state.followup_plan.get("questions") or [])
        if isinstance(item, dict) and item.get("field_id") and item.get("question")
    ]
    prompt = PARSE_DETAILS_PROMPT.format(
        asked_details=(
            "\n".join(prompt_questions)
            if prompt_questions
            else "\n".join(f"- {q}" for q in state.pending_ask_details)
        ),
        user_answer=answer_for_parse,
        case_context=format_case_context(state.case_facts),
    )
    raw_content = ""
    try:
        response = await ainvoke_bounded(
            llm_for_stage(_fast_llm_for(deps), max_tokens=1200),
            [SystemMessage(content=prompt)],
            timeout=settings.GUIDE_LLM_TIMEOUT_PARSE,
            stage="parse_followup_answer",
        )
        raw_content = str(response.content or "")
        content = raw_content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
    except Exception as e:
        # 不让模型超时丢掉用户本轮输入。一般陈述按待核实事实保留；
        # 疑问句仍作为反问处理，后续安全回答后恢复原追问。
        looks_like_question = (
            False if structured_answers
            else _looks_like_user_question(assessment_narrative)
        )
        parsed = {
            "is_answer": not looks_like_question,
            "answers_asked_question": not looks_like_question,
            "user_question": answer_narrative if looks_like_question else "",
            "collected_facts": (
                [] if looks_like_question or not assessment_narrative
                else [assessment_narrative]
            ),
            "case_updates": (
                []
                if looks_like_question or not assessment_narrative
                else legacy_fact_updates(
                    [assessment_narrative],
                    user_text=assessment_narrative,
                )
            ),
            "evidence": [],
            "evidence_details": [],
            "evidence_unavailable": [],
            "adverse_facts": [],
        }
        logger.warning(
            "节点⑦解析追问回答失败，改用通用语义形态降级 | err={} raw={}",
            e,
            raw_content[:200],
        )
    inspected_evidence_observations = await inspection_task

    user_question = (parsed.get("user_question") or "").strip()
    is_answer = parsed.get("is_answer", True)
    answers_asked_question = parsed.get("answers_asked_question", is_answer)
    if structured_answers:
        is_answer = True
        answers_asked_question = True
        user_question = ""
    planned_questions = [
        item for item in (state.followup_plan.get("questions") or [])
        if isinstance(item, dict) and item.get("field_id")
    ]
    submitted_field_ids = set(structured_answers)
    submitted_field_ids.update(
        str(item) for item in (parsed.get("answered_question_ids") or []) if item
    )
    if planned_questions and answers_asked_question:
        answered_questions = [
            item for item in planned_questions
            if str(item.get("field_id") or "") in submitted_field_ids
        ]
    else:
        answered_questions = []
    answered_rule_ids = [
        str(item.get("candidate_id") or "")
        for item in answered_questions if item.get("candidate_id")
    ]
    answer_by_rule_id = {
        str(item.get("candidate_id") or ""): structured_answers.get(
            str(item.get("field_id") or ""),
            assessment_narrative,
        )
        for item in answered_questions
        if item.get("candidate_id")
    }
    if evidence_only_turn:
        # The extraction wrapper and document text are evidence metadata, not
        # user-confirmed case facts.  Keep them exclusively in the evidence
        # store even if the language model tries to echo them as a fact.
        parsed["collected_facts"] = []
        parsed["case_updates"] = []
        parsed["new_issues"] = []
    parser_missed_declarative_detail = (
        not evidence_only_turn
        and not is_answer
        and not _looks_like_user_question(assessment_narrative)
    )
    if parser_missed_declarative_detail:
        logger.info("节点⑦将非疑问陈述按主动补充处理 | text={}", last_msg[:120])
        is_answer = True
        answers_asked_question = False
        user_question = ""
        if not parsed.get("collected_facts"):
            parsed["collected_facts"] = [assessment_narrative]
    if state.wants_conclude and (
        state.turn_control_intent == "conclude_now"
        or any(
            phrase in last_msg
            for phrase in ("现在生成方案", "生成方案", "给方案", "按现有信息", "不要再问", "别再问")
        )
    ):
        # 这是流程控制指令，不是需要在结论中回答的法律问题。
        user_question = ""

    if (
        not submitted_field_ids
        and _looks_like_question_repetition(last_msg, state.pending_ask_details)
    ):
        fact_records = dict(state.fact_records)
        for rule_id in state.pending_followup_ids:
            if state.pending_ask_type != "facts":
                continue
            rule = find_fact_followup(state.legal_domain, rule_id)
            if rule:
                record = assess_fact_answer(rule, last_msg, fact_records.get(rule_id))
                record["status"] = "ambiguous"
                fact_records[rule_id] = record
        low_info_count = state.consecutive_low_info_answers + 1
        stalled = low_info_count >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
        if stalled:
            return {
                "fact_records": fact_records,
                "consecutive_low_info_answers": low_info_count,
                "pending_ask_details": [],
                "pending_ask_type": "",
                "pending_followup_ids": [],
                "force_conclude": True,
                "phase": GuidePhase.ISSUE_SEARCH,
            }
        question = state.pending_ask_details[0]
        choice_note = (
            "\n如果这个问题暂时不方便回答，可以直接说“不清楚”，我会记录为未知并换到下一个关键点；"
            "也可以回复“现在生成方案”。"
            if low_info_count >= settings.GUIDE_NO_PROGRESS_CHOICE_ROUNDS
            else ""
        )
        clarification = (
            "我看到这句话更像是把问题重复了一遍，还不能确定您的答案。\n"
            f"请直接回答这个问题：{question}\n"
            "可以用一句很短的话回答，例如“有”“没有”“大概是……”或“不清楚”。"
            + choice_note
        )
        return {
            "fact_records": fact_records,
            "consecutive_low_info_answers": low_info_count,
            "pending_ask_details": state.pending_ask_details,
            "pending_ask_type": state.pending_ask_type,
            "pending_followup_ids": state.pending_followup_ids,
            "messages": [AIMessage(content=clarification)],
        }

    # 用户只是反问，没有回答 → 保留待答问题，不污染证据
    if not is_answer:
        pending = state.pending_ask_details
        logger.info("节点⑦用户反问未作答，保留待答项 | question={} pending={}", user_question, pending)
        deferred = state.deferred_questions + ([user_question] if user_question else [])
        counter_questions = state.consecutive_counter_questions + 1
        stalled = counter_questions >= settings.GUIDE_MAX_COUNTER_QUESTIONS
        if state.force_conclude or state.wants_conclude or stalled:
            return {
                "deferred_questions": deferred,
                "consecutive_counter_questions": counter_questions,
                "pending_ask_details": [],
                "pending_ask_type": "",
                "pending_followup_ids": [],
                "force_conclude": state.force_conclude or stalled,
                "phase": GuidePhase.ISSUE_SEARCH,
            }
        direct_answer = await _answer_counter_question(state, deps, user_question)
        acknowledgement = f"先回答您刚才的问题：{direct_answer}\n"
        reask = acknowledgement + "回到您的案件，为避免方案失准，当前还需要确认：\n" + \
                "\n".join(f"- {q}" for q in pending)
        return {
            "deferred_questions": state.deferred_questions,
            "consecutive_counter_questions": counter_questions,
            "messages": [AIMessage(content=reask)],
            # 不动 pending_ask_details / ask_rounds / asked_details
        }

    def _merge(old: list[str], new: list[str]) -> list[str]:
        seen: set[str] = set()
        return [item for item in old + new if item and not (item in seen or seen.add(item))]

    is_multimodal_evidence = last_msg.startswith("【图片证据补充（视觉模型识别")
    # This node parses answers into facts and evidence only. Promoting a legal
    # classification here allowed an unverified model inference to change the
    # whole case track, so issue normalization remains the sole owner.
    new_issues = list(state.confirmed_issues)
    parsed_facts = [
        item for item in (parsed.get("collected_facts") or [])
        if _is_usable_case_fact(item)
    ]
    if is_multimodal_evidence:
        possession_verbs = ("持有", "保留", "手中有", "另有", "带走", "拍摄")
        evidence_nouns = ("实物", "照片", "录音", "原件", "合同", "票据", "凭证")
        parsed_facts = [
            (
                f"待核验线索（图片文字转述，本次未直接展示）：{item}"
                if any(verb in item for verb in possession_verbs)
                and any(noun in item for noun in evidence_nouns)
                else item
            )
            for item in parsed_facts
        ]
    current_turn_keys = {
        str(item.get("key") or "")
        for item in latest_case_facts(state.case_facts, state.round)
        if item.get("key")
    }
    parsed_case_updates = parsed.get("case_updates") or []
    form_case_updates = _structured_answer_case_updates(
        answered_questions,
        structured_answers,
        domain=state.legal_domain,
    )
    if evidence_only_turn:
        raw_case_updates = []
    elif parsed_case_updates:
        raw_case_updates = [
            item for item in parsed_case_updates
            if str(item.get("key") or "") not in current_turn_keys
            and not _is_transport_wrapper_fact(item.get("statement", ""))
            and not (
                structured_answers
                and str(item.get("key") or "").startswith("legacy.raw.")
            )
        ]
        # When the parser produced grounded atomic facts, do not also persist
        # the complete form answer under a broad field slot.  A compound answer
        # such as "time, place and person" otherwise becomes a misleading
        # single time or procedure fact in the final case reconstruction.
        if not raw_case_updates and form_case_updates:
            raw_case_updates = form_case_updates
    elif form_case_updates:
        raw_case_updates = form_case_updates
    elif current_turn_keys:
        raw_case_updates = []
    else:
        raw_case_updates = legacy_fact_updates(parsed_facts, user_text=answer_for_parse)
    case_facts = reduce_case_facts(
        state.case_facts,
        raw_case_updates,
        user_text=answer_for_parse,
        turn=state.round,
    )
    if parser_missed_declarative_detail and not latest_case_facts(case_facts, state.round):
        case_facts = reduce_case_facts(
            state.case_facts,
            legacy_fact_updates(parsed_facts, user_text=answer_for_parse),
            user_text=answer_for_parse,
            turn=state.round,
        )
    active_atoms = active_case_facts(case_facts)
    atomic_statements = [
        item["statement"] for item in active_atoms
        if item.get("category") != "evidence" and item.get("statement")
    ]
    new_facts = (
        atomic_statements
        if case_facts
        else _merge(state.collected_facts, parsed_facts)
    )
    current_turn_atoms = latest_case_facts(case_facts, state.round)
    atom_evidence, atom_unavailable = evidence_from_case_facts(current_turn_atoms)
    parsed_evidence = parsed.get("evidence") or []
    evidence_observations = normalize_evidence_observations(
        parsed.get("evidence_details"),
        user_text=last_msg,
    )
    evidence_observations.extend(inspected_evidence_observations)
    evidence_observations.extend(uploaded_observations)
    evidence_denial_markers = (
        "没有", "没拍", "没留", "没保存", "未拍", "未留", "未保存",
        "找不到", "拿不出", "丢了", "遗失", "无法提供",
    )
    explicit_evidence_denial = any(
        marker in assessment_narrative for marker in evidence_denial_markers
    )
    if evidence_observations:
        atom_evidence = _merge(
            atom_evidence,
            [item["name"] for item in evidence_observations if item.get("uploaded_copy")],
        )
    observation_leads = [
        item["name"] for item in evidence_observations
        if not item.get("uploaded_copy") and item.get("name")
    ]
    if is_multimodal_evidence:
        type_match = re.search(r"【证据类型】\s*([^\n]+)", last_msg)
        evidence_type = type_match.group(1).strip(" *：:") if type_match else "图片证据"
        present_evidence = [f"已上传图片：{evidence_type}"]
        unverified_evidence = [
            item for item in parsed_evidence
            if item and not any(token in item for token in (evidence_type, "聊天记录截图", "图片证据"))
        ]
    elif evidence_observations:
        present_evidence = list(atom_evidence)
        unverified_evidence = _merge(list(parsed_evidence), observation_leads)
    else:
        # The legacy evidence list has no possession status. Let a grounded
        # case atom (or an uploaded copy) establish "held"; otherwise retain
        # each list item as a lead for later acquisition or verification.
        if explicit_evidence_denial:
            present_evidence = []
            unverified_evidence = [item for item in parsed_evidence if item]
        else:
            present_evidence = _merge(
                atom_evidence,
                [
                    item for item in parsed_evidence
                    if item and item in assessment_narrative
                ],
            )
            unverified_evidence = [
                item for item in parsed_evidence
                if item and item not in present_evidence
            ]
    if len(present_evidence) != len(parsed_evidence):
        logger.info("节点⑦未核验证据线索不计入置信度 | evidence={}", parsed_evidence)
    present_evidence = _merge(present_evidence, atom_evidence)
    present_evidence = _split_evidence_names(present_evidence)
    unverified_evidence = _split_evidence_names(unverified_evidence)
    new_evidence = _merge(state.evidence_confirmed, present_evidence)
    new_unverified = _merge(state.evidence_unverified, unverified_evidence)
    parsed_unavailable = (parsed.get("evidence_unavailable") or []) if explicit_evidence_denial else []
    unavailable = _merge(
        state.evidence_unavailable,
        _merge(parsed_unavailable, atom_unavailable),
    )
    new_adverse = _merge(state.adverse_facts, parsed.get("adverse_facts") or [])

    fact_records = dict(state.fact_records)
    evidence_assessments = assess_initial_evidence(
        [item for item in present_evidence if item not in state.evidence_confirmed],
        state.evidence_assessments,
    )
    evidence_assessments = merge_evidence_observations(
        evidence_assessments,
        evidence_observations,
        domain=state.legal_domain,
    )
    low_info_answer = False
    pending_fact_statuses: list[str] = []
    answer_is_negative = explicit_evidence_denial
    rule_ids_to_assess = (
        answered_rule_ids
        if planned_questions
        else state.pending_followup_ids if answers_asked_question else []
    )
    for rule_id in rule_ids_to_assess:
        if state.pending_ask_type == "facts":
            rule = find_fact_followup(state.legal_domain, rule_id)
            if rule:
                record = assess_fact_answer(
                    rule,
                    answer_by_rule_id.get(rule_id, assessment_narrative),
                    fact_records.get(rule_id),
                )
                fact_records[rule_id] = record
                pending_fact_statuses.append(record["status"])
                # “不知道”是对信息可得性的有效回答；只有含义不清或冲突才算未推进。
                low_info_answer = low_info_answer or record["status"] in {"ambiguous", "conflicted"}
        elif state.pending_ask_type == "evidence":
            rule = find_evidence_followup(state.legal_domain, rule_id)
            if not rule:
                continue
            unavailable_items = parsed_unavailable
            explicitly_unavailable = answer_is_negative or any(
                item in rule.item or rule.item in item or any(keyword in item for keyword in rule.match_keywords)
                for item in unavailable_items
            )
            positive_markers = ("有", "保存", "留着", "在手里", "能找到", "可以提供", "能提供")
            mentioned_present = (
                (bool(present_evidence) or any(marker in assessment_narrative for marker in positive_markers))
                and not explicitly_unavailable
            )
            record = assess_evidence_answer(
                rule,
                assessment_narrative or last_msg,
                unavailable=explicitly_unavailable,
                uploaded=(
                    bool(uploaded_observations) or is_multimodal_evidence
                ) and mentioned_present,
                mentioned_as_present=mentioned_present,
                previous=evidence_assessments.get(rule_id),
            )
            evidence_assessments[rule_id] = record
            if record["availability"] == "unavailable":
                unavailable = _merge(unavailable, [rule.item])
            elif record["availability"] in {"uploaded_copy", "user_claimed_present", "conflicted"}:
                # Prefer the user's concrete material name (for example
                # "付款记录") over the catalog umbrella label. Add the
                # umbrella only when the parser found no specific material.
                if not present_evidence:
                    new_evidence = _merge(new_evidence, [rule.item])
            # 明确没有某项证据会改变证据策略，属于有效进展；unclear/conflicted 才未推进。
            low_info_answer = low_info_answer or record["availability"] in {"unclear", "conflicted"}

    evidence_assessments = merge_evidence_observations(
        evidence_assessments,
        evidence_observations,
        domain=state.legal_domain,
    )
    if not state.pending_followup_ids:
        # 动态问题没有题库 ID 时，解析器确认其回答了当前问题就视为有效推进；
        # 明确否定或不知道仍然是可用于后续决策的信息。
        low_info_answer = not bool(answers_asked_question)
    draft_candidates = [
        item["statement"] for item in active_atoms
        if item.get("category") != "evidence"
        and item.get("status") == "asserted"
        and item.get("statement")
    ]
    if not draft_candidates:
        draft_candidates = [item for item in parsed_facts if _is_draftable_fact(item)]
    if any(status in {"ambiguous", "conflicted", "unknown"} for status in pending_fact_statuses):
        draft_candidates = []
    new_draftable_facts = (
        draft_candidates
        if case_facts
        else _merge(state.draftable_facts, draft_candidates)
    )
    consecutive_low_info = state.consecutive_low_info_answers + 1 if low_info_answer else 0
    force_low_info_conclusion = consecutive_low_info >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
    region = (
        _state_region_name(parsed.get("region", ""))
        or _state_region_name(state.region)
    )
    time_info = (parsed.get("time_info") or "").strip() or state.time_info
    logger.info("节点⑦解析结果 | type={} new_issues={} facts={} evidence={} unavailable={} adverse={} region={} time={} deferred={}",
                state.pending_ask_type,
                parsed.get("new_issues"), parsed.get("collected_facts"), parsed.get("evidence"),
                parsed.get("evidence_unavailable"), parsed.get("adverse_facts"), region, time_info, user_question)
    case_conflicts = [
        item for item in active_atoms
        if item.get("status") == "conflicted" and int(item.get("turn") or 0) == state.round
    ]
    needs_fact_confirmation = bool(case_conflicts) or any(
        status in {"ambiguous", "conflicted"}
        for status in pending_fact_statuses
    )
    if force_low_info_conclusion:
        needs_fact_confirmation = False
    confirmation_messages: list[AIMessage] = []
    if needs_fact_confirmation:
        question = state.pending_ask_details[0]
        if case_conflicts:
            conflict_key = case_conflicts[0].get("key", "这项信息")
            alternatives = [
                item.get("statement", "") for item in active_atoms
                if item.get("key") == conflict_key and item.get("statement")
            ]
            question = f"关于{' / '.join(dict.fromkeys(alternatives))}，请确认哪一个说法为准？"
        status_text = "与前面记录不一致" if case_conflicts or "conflicted" in pending_fact_statuses else "仍有两种可能的理解"
        choice_note = (
            "\n如果这项信息暂时无法确认，可以直接回复“不清楚”，我会记录为未知并换到下一个关键点；"
            "也可以回复“现在生成方案”。"
            if consecutive_low_info >= settings.GUIDE_NO_PROGRESS_CHOICE_ROUNDS
            else ""
        )
        confirmation_messages = [AIMessage(content=(
            f"我暂时没有把这项内容写成确定事实，因为您的回答{status_text}。\n"
            f"请再明确一次：{question}\n"
            "如果是在更正之前的说法，可以直接以“更正：……”开头；不清楚也可以直接说“不清楚”。"
            + choice_note
        ))]
    updates = {
        "confirmed_issues": new_issues,
        "collected_facts": new_facts,
        "draftable_facts": new_draftable_facts,
        "case_facts": case_facts,
        "evidence_confirmed": new_evidence,
        "evidence_unverified": new_unverified,
        "evidence_unavailable": unavailable,
        "fact_records": fact_records,
        "evidence_assessments": evidence_assessments,
        "adverse_facts": new_adverse,
        "region": region,
        "time_info": time_info,
        "pending_ask_details": [question] if needs_fact_confirmation else [],
        "pending_ask_type": state.pending_ask_type if needs_fact_confirmation else "",
        "pending_followup_ids": state.pending_followup_ids if needs_fact_confirmation else [],
        "consecutive_counter_questions": 0,
        "consecutive_low_info_answers": consecutive_low_info,
        "force_conclude": state.force_conclude or force_low_info_conclusion,
        "phase": GuidePhase.DETAIL_GATHER if needs_fact_confirmation else GuidePhase.ISSUE_SEARCH,
        "followup_plan": {},
        "asked_details": _merge_unique(
            state.asked_details,
            [str(item.get("question") or "") for item in answered_questions],
        ),
        "asked_decision_keys": _merge_unique(
            state.asked_decision_keys,
            [str(item.get("field_id") or "") for item in answered_questions],
        ),
        "asked_followup_ids": _merge_unique(
            state.asked_followup_ids,
            answered_rule_ids,
        ),
        "issue_refresh_needed": bool(
            (not answers_asked_question)
            and (
                parsed_facts
                or parsed.get("case_updates")
                or evidence_observations
                or parsed.get("new_issues")
            )
        ),
        "deferred_questions": state.deferred_questions + ([user_question] if user_question else []),
        "messages": confirmation_messages,
    }
    if uploaded_observations:
        updates["evidence_evaluation_version"] = (
            state.evidence_evaluation_version + 1
        )
    meaningful_optional_supplement = (
        state.allow_extra_followups
        and is_answer
        and not low_info_answer
        and not needs_fact_confirmation
        and not state.force_conclude
        and not state.wants_conclude
    )
    if meaningful_optional_supplement:
        # 用户主动补充一项后，把节奏重新交还给用户：继续补充或按现有信息出方案。
        updates.update({
            "awaiting_supplement_choice": False,
            "supplement_choice_offered": False,
            "supplement_choice": "",
            "supplement_has_details": False,
            "allow_extra_followups": False,
        })
    return updates


_ARTICLE_PATTERN = r"第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?"
_BOOK_CITATION_RE = re.compile(r"《([^》\n]{2,80})》")
_SOURCE_CITATION_RE = re.compile(rf"【(.+?)\s+({_ARTICLE_PATTERN})】")
_LAW_TITLE_SUFFIXES = ("法", "法典", "条例", "规定", "办法", "解释", "规则", "通则", "决定")


def _chinese_number_to_int(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in units:
            unit = units[char]
            total += (current or 1) * unit
            current = 0
        else:
            return None
    return total + current


def _normalize_article(article: str) -> tuple[int | str, int | str | None]:
    match = re.fullmatch(r"第(.+?)条(?:之(.+))?", article)
    if not match:
        return article, None
    main = _chinese_number_to_int(match.group(1))
    sub = _chinese_number_to_int(match.group(2)) if match.group(2) else None
    return main if main is not None else match.group(1), sub


def _source_statute_refs(law_context: str) -> dict[str, set[tuple[int | str, int | str | None]]]:
    refs: dict[str, set[tuple[int | str, int | str | None]]] = {}
    for law_name, article in _SOURCE_CITATION_RE.findall(law_context or ""):
        refs.setdefault(law_name.strip(), set()).add(_normalize_article(article))
    return refs


def _sanitize_statute_citations(reply: str, law_context: str) -> str:
    """只保留本轮检索到的法律名称和条号，确定性阻断生成式法条幻觉。"""
    source_refs = _source_statute_refs(law_context)
    if not reply or not _BOOK_CITATION_RE.search(reply):
        return reply

    safe_lines: list[str] = []
    replacement = "> 注：本段涉及的具体条文未在本轮检索结果中出现，已省略；请拨打 12348 核对后再主张。"
    replacement_added = False
    removed_numbered_items = 0
    for line in reply.splitlines():
        if re.match(r"^\s*\*\*【.+】\*\*\s*$", line):
            removed_numbered_items = 0
        ordinal = re.match(r"^(\s*)(\d+)([.、)])(\s+)", line)
        citations = list(_BOOK_CITATION_RE.finditer(line))
        unsupported = False
        legal_reference_seen = False
        replacements: list[tuple[str, str]] = []
        for index, citation in enumerate(citations):
            name = citation.group(1).strip()
            candidates = [title for title in source_refs if title == name or title.endswith(name) or name.endswith(title)]
            segment_end = citations[index + 1].start() if index + 1 < len(citations) else len(line)
            articles = re.findall(_ARTICLE_PATTERN, line[citation.end():segment_end])
            is_law_title = name.endswith(_LAW_TITLE_SUFFIXES)
            if not candidates and not is_law_title:
                # 《责令改正通知书》《劳动仲裁申请书》等文书名称不是法条引用。
                continue
            legal_reference_seen = True
            if len(candidates) != 1:
                unsupported = True
                break
            canonical = candidates[0]
            if any(_normalize_article(article) not in source_refs[canonical] for article in articles):
                unsupported = True
                break
            if canonical != name:
                replacements.append((f"《{name}》", f"《{canonical}》"))

        bare_article_context = re.search(
            rf"(?:依据|根据|依照|按照|法条).{{0,12}}{_ARTICLE_PATTERN}|"
            rf"{_ARTICLE_PATTERN}.{{0,8}}(?:规定|明确|要求)",
            line,
        )
        if legal_reference_seen or bare_article_context:
            source_articles = {article for articles in source_refs.values() for article in articles}
            if any(_normalize_article(article) not in source_articles for article in re.findall(_ARTICLE_PATTERN, line)):
                unsupported = True

        if unsupported:
            logger.warning("结论引用白名单过滤 | line={}", line[:160])
            if ordinal:
                removed_numbered_items += 1
            if not replacement_added:
                safe_lines.append(replacement)
                replacement_added = True
            continue
        if ordinal and removed_numbered_items:
            number = max(1, int(ordinal.group(2)) - removed_numbered_items)
            line = (
                f"{ordinal.group(1)}{number}{ordinal.group(3)}{ordinal.group(4)}"
                + line[ordinal.end():]
            )
        for old, new in replacements:
            line = line.replace(old, new)
        safe_lines.append(line)
    return "\n".join(safe_lines)


def _grounded_statute_entries(law_context: str, limit: int = 8) -> list[tuple[str, str, str]]:
    """Extract exact statute title, article and source text from formatted retrieval context."""
    pattern = re.compile(
        rf"法条\d+【(?P<title>.+?)\s+(?P<article>{_ARTICLE_PATTERN})】\s*\n"
        r"(?P<text>.*?)(?=\n\n---\n\n|\Z)",
        re.S,
    )
    entries: list[tuple[str, str, str]] = []
    for match in pattern.finditer(law_context or ""):
        text = re.sub(r"\s+", " ", match.group("text")).strip()
        text = re.sub(rf"^{re.escape(match.group('article'))}\s*", "", text)
        if text:
            entries.append((match.group("title").strip(), match.group("article"), text))
        if len(entries) >= limit:
            break
    return entries


def _select_grounded_statute_entries(
    entries: list[tuple[str, str, str]],
    state: GuideState | None,
    limit: int = 2,
) -> list[tuple[str, str, str]]:
    # Retrieval order remains the fallback.  When the issue analyst has linked
    # a statute to an actual dispute point, prefer that explicit connection over
    # a merely nearby retrieval hit.
    analyses = list(getattr(state, "issue_analyses", []) or []) if state else []
    referenced = " ".join(
        str(value)
        for analysis in analyses if isinstance(analysis, dict)
        for value in (analysis.get("legal_basis_refs") or [])
    )
    if referenced:
        linked = [
            entry for entry in entries
            if entry[0] in referenced or entry[1] in referenced
        ]
        if linked:
            return linked[:limit]
    return entries[:limit]


def _ensure_grounded_legal_basis(
    reply: str,
    law_context: str,
    state: GuideState | None = None,
) -> str:
    """Render the legal-basis section directly from this turn's retrieved source text."""
    all_entries = _grounded_statute_entries(law_context)
    if not all_entries:
        return reply

    def _entry_cited(entry: tuple[str, str, str]) -> bool:
        title, article, _text = entry
        article_norm = _normalize_article(article)
        article_variants = {
            str(article),
            str(article_norm[1]) if article_norm[1] else "",
            str(article_norm[0]) if article_norm[0] else "",
        }
        return (
            bool(title and f"《{title}》" in reply)
            or bool(title and title in reply)
            or any(variant and variant in reply for variant in article_variants)
        )

    cited = [entry for entry in all_entries if _entry_cited(entry)]
    if cited:
        selected = cited[:8]
        fallback = _select_grounded_statute_entries(all_entries, state, limit=2)
        for entry in fallback:
            if entry not in selected:
                selected.append(entry)
    else:
        selected = _select_grounded_statute_entries(all_entries, state, limit=2)
    if not selected:
        return reply
    heading = re.search(r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【法律依据】\*{0,2}\s*$", reply)
    block = (
        "> 以下条文直接来自本轮知识库检索原文。\n"
        + "\n".join(f"- 《{name}》{article}：{text}" for name, article, text in selected)
    )
    if not heading:
        section = f"**【法律依据】**\n{block}\n\n"
        next_heading = re.search(r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:维权路径|类似案例)", reply)
        if next_heading:
            return reply[:next_heading.start()].rstrip() + "\n\n" + section + reply[next_heading.start():]
        return section + reply

    next_heading = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【[^】]+】\*{0,2}\s*$",
        reply[heading.end():],
    )
    section_end = heading.end() + next_heading.start() if next_heading else len(reply)
    return reply[:heading.end()] + "\n" + block + "\n\n" + reply[section_end:].lstrip()


def _sanitize_forced_followups(reply: str) -> str:
    """结论阶段移除要求用户继续补充并等待下一版方案的尾段。"""
    phrases = (
        "请补充以下关键信息", "请继续补充", "补充上述信息后", "补充后我将",
        "我将为您生成更精准", "请回答以下问题", "还需要您补充",
        "请务必先回答", "先回答上面",
    )
    lines = reply.splitlines()
    kept: list[str] = []
    removed = False
    skip_supplement_section = False
    for line in lines:
        heading_text = re.sub(r"[#*【】\s]", "", line)
        is_supplement_heading = any(
            marker in heading_text for marker in ("关键缺失信息清单", "强烈建议")
        )
        if is_supplement_heading:
            skip_supplement_section = True
            removed = True
            continue
        if skip_supplement_section:
            is_next_section = (
                line.strip() == "---"
                or bool(re.match(r"\s*#{1,6}\s+", line))
                or "【" in line and "】" in line
            )
            if not is_next_section:
                continue
            skip_supplement_section = False
        if any(phrase in line for phrase in phrases):
            removed = True
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    if not removed:
        return cleaned
    note = (
        "> 当前仍有事实和证据缺口，因此胜算只能作初步判断。"
        "您无需继续回答也可先按行动清单执行，并拨打 12348 核验。"
    )
    marker = "\n\n---\n📄"
    position = cleaned.find(marker)
    if position >= 0:
        return cleaned[:position].rstrip() + "\n\n" + note + cleaned[position:]
    return cleaned.rstrip() + "\n\n" + note


def _ensure_contextual_understanding(reply: str, state: GuideState) -> str:
    """Render the opening from grounded case atoms for every legal domain."""
    atoms = active_case_facts(state.case_facts)
    if not atoms:
        return reply
    def _user_facing(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^用户(?:称|表示|提到)", "您提到", text)
        text = re.sub(r"^用户卡内", "您的卡内", text)
        text = re.sub(r"^用户", "您", text)
        return text.replace("用户本人", "您本人").replace("将用户", "将您")

    asserted = [
        _user_facing(item.get("statement", "")) for item in atoms
        if item.get("category") != "evidence"
        and item.get("status") == "asserted"
        and item.get("statement")
    ][-6:]
    uncertain = [
        _user_facing(item.get("statement", "")) for item in atoms
        if item.get("category") != "evidence"
        and item.get("status") in {"uncertain", "conflicted"}
        and item.get("statement")
    ][-2:]
    if not asserted and not uncertain:
        return reply
    summary_parts = []
    if asserted:
        summary_parts.append("我已按您的陈述记录：" + "；".join(dict.fromkeys(asserted)) + "。")
    if uncertain:
        summary_parts.append("仍需核对：" + "；".join(dict.fromkeys(uncertain)) + "。")
    summary = "".join(summary_parts)

    section_pattern = re.compile(
        r"(?ms)^(?P<header>\s*(?:#{1,6}\s*)?\*{0,2}【理解您的情况】\*{0,2}\s*)$"
        r"\n.*?(?=^\s*(?:#{1,6}\s*)?\*{0,2}【法律依据】\*{0,2}\s*$)"
    )
    if not section_pattern.search(reply):
        return reply
    return section_pattern.sub(lambda match: f"{match.group('header')}\n{summary}\n\n", reply)


def _dedupe_text_items(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip("；，。 ")
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _format_long_dialogue_memory(
    state: GuideState,
    *,
    max_turns: int = 40,
    max_chars: int = 4000,
) -> str:
    """Rebuild an old-turn fact ledger without re-injecting raw dialogue text."""
    facts = active_case_facts(state.case_facts)
    status_labels = {
        "asserted": "已确认",
        "uncertain": "未确认",
        "conflicted": "冲突",
        "denied": "否认",
    }
    by_turn: dict[int, list[str]] = {}
    for item in facts:
        turn = int(item.get("turn") or 0)
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        status = str(item.get("status") or "asserted")
        label = status_labels.get(status, status)
        by_turn.setdefault(turn, []).append(f"{statement}（{label}）")

    lines: list[str] = []
    for turn in sorted(by_turn)[-max_turns:]:
        statements = _dedupe_text_items(by_turn[turn], limit=8)
        if statements:
            lines.append(f"第{turn}轮：" + "；".join(statements))

    held, leads = _evidence_for_plan(state)
    if held:
        lines.append("已确认材料：" + "、".join(_dedupe_text_items(held, limit=6)))
    if leads:
        lines.append("待核验线索：" + "、".join(_dedupe_text_items(leads, limit=6)))
    if state.evidence_unavailable:
        lines.append("明确缺失：" + "、".join(_dedupe_text_items(state.evidence_unavailable, limit=6)))
    actions = _dedupe_text_items(
        [str(item) for item in state.collected_facts if str(item).strip()],
        limit=6,
    )
    if actions:
        lines.append("已采取处理：" + "；".join(actions))

    text = "\n".join(lines).strip()
    if not text:
        return "（暂无跨轮次沉淀事实）"
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "……"


def _format_priority_actions(state: GuideState) -> str:
    """Place the analyst's most useful actions before all explanatory sections."""
    actions: list[str] = []
    strategy_plan = state.strategy_plan if isinstance(state.strategy_plan, dict) else {}
    for item in strategy_plan.get("priority_actions") or []:
        if isinstance(item, dict):
            action = str(item.get("action") or "").strip()
            if action:
                detail = str(item.get("purpose") or item.get("why_now") or "").strip()
                actions.append(f"{action}（{detail}）" if detail else action)
    if not actions:
        for analysis in state.issue_analyses or []:
            if not isinstance(analysis, dict):
                continue
            actions.extend(str(item) for item in (analysis.get("recommended_actions") or []))
            actions.extend(str(item) for item in (analysis.get("evidence_actions") or []))
    actions = _dedupe_text_items(actions, limit=4)
    if not actions and state.legal_domain == "criminal_public_security":
        procedure_text = "；".join(
            str(item.get("statement") or "")
            for item in active_case_facts(state.case_facts)
            if item.get("category") == "procedure" and item.get("status") == "asserted"
        )
        if any(marker in procedure_text for marker in ("已报警", "报案", "受案")):
            actions = [
                "尽快向已联系或已受理的公安机关补充完整经过、财物信息、监控线索和目击者线索，并索取或保存受案回执、案件编号。",
                "立即联系监控管理方和目击者，说明警方可能需要调取；保留原始录像线索、联系方式和购买凭证。",
            ]
        else:
            actions = [
                "如侵害尚未处理，优先向 110 或就近公安机关报案，并说明人员、地点、财物和监控、目击者线索。",
                "立即保存购买凭证、监控线索和目击者联系方式，避免线索灭失。",
            ]
    if not actions:
        actions = [
            "按时间顺序保存原始材料，并优先完成争点分析中列出的证据和程序动作。",
            "向有管辖权的受理机构核对材料、回执和下一步程序，并保存每次沟通记录。",
        ]
    return "**【现在最优先行动】**\n" + "\n".join(
        f"{index}. {action}" for index, action in enumerate(actions, start=1)
    )


def _evidence_for_plan(state: GuideState) -> tuple[list[str], list[str]]:
    """Return held material and leads without trusting the legacy flat list.

    ``evidence_confirmed`` was historically populated from a loose model list.
    When structured evidence atoms are available, their provenance and status
    are the source of truth for the final plan.
    """
    evidence_atoms = [
        item for item in active_case_facts(state.case_facts)
        if item.get("category") == "evidence"
    ]
    if not evidence_atoms:
        return (
            _dedupe_text_items([str(item) for item in state.evidence_confirmed], limit=8),
            _dedupe_text_items([str(item) for item in state.evidence_unverified], limit=8),
        )
    held, _unavailable = evidence_from_case_facts(evidence_atoms)
    explicit_material_markers = (
        "保留了", "保存了", "已经保存", "已有", "我有", "持有", "掌握", "留存", "提供了",
    )
    leads = [
        str(item.get("statement") or item.get("value") or "")
        for item in evidence_atoms
        if item.get("evidence_status") not in {"obtained", "unavailable"}
        and not any(
            marker in " ".join(str(item.get(field) or "") for field in ("statement", "value", "source_text"))
            for marker in explicit_material_markers
        )
    ]
    return (
        _dedupe_text_items(held, limit=8),
        _dedupe_text_items([*leads, *state.evidence_unverified], limit=8),
    )


def _format_strategy_headline(state: GuideState) -> str:
    plan = state.strategy_plan if isinstance(state.strategy_plan, dict) else {}
    headline = plan.get("headline_assessment") or {}
    if not isinstance(headline, dict):
        return ""
    position = str(headline.get("position") or "").strip()
    reason = str(headline.get("supporting_reason") or "").strip()
    uncertainty = str(headline.get("uncertainty") or "").strip()
    if not position:
        return ""
    lines = ["**【当前策略判断】**", position]
    if reason:
        lines.append(f"- **判断依据：**{reason}")
    if uncertainty:
        lines.append(f"- **会改变判断的因素：**{uncertainty}")
    return "\n".join(lines)


def _ensure_strategy_headline(reply: str, state: GuideState) -> str:
    if "当前策略判断" in reply:
        return reply
    section = _format_strategy_headline(state)
    if not section:
        return reply
    for marker in ("理解您的情况", "案件完整还原", "核心争点分析", "法律依据"):
        position = reply.find(marker)
        if position >= 0:
            line_start = reply.rfind("\n", 0, position) + 1
            return reply[:line_start].rstrip() + "\n\n" + section + "\n\n" + reply[line_start:].lstrip()
    return section + "\n\n" + reply.lstrip()


def _format_strategy_insights(state: GuideState) -> str:
    plan = state.strategy_plan if isinstance(state.strategy_plan, dict) else {}
    sections: list[str] = []
    mappings = (
        ("opponent_arguments", "【可能的反方观点与应对】"),
        ("institution_focus", "【与办理机构沟通重点】"),
        ("conditions_that_change_result", "【策略分支条件】"),
        ("risk_boundaries", "【当前风险边界】"),
    )
    for key, title in mappings:
        values = [str(item).strip() for item in plan.get(key) or [] if str(item).strip()]
        values = _dedupe_text_items(values, limit=6)
        if values:
            sections.append("**" + title + "**\n" + "\n".join(f"- {item}" for item in values))
    return "\n\n".join(sections)


def _ensure_strategy_insights(reply: str, state: GuideState) -> str:
    if isinstance(state.strategy_plan, dict) and state.strategy_plan:
        reply = _strip_legacy_strategy_sections(reply)
    section = _format_strategy_insights(state)
    if not section or "可能的反方观点与应对" in reply or "策略分支条件" in reply:
        return reply
    return _insert_before_document_offer(reply, section)


def _ensure_priority_actions(reply: str, state: GuideState) -> str:
    if "【现在最优先行动】" in reply:
        return reply
    section = _format_priority_actions(state)
    understanding = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【理解您的情况】\*{0,2}\s*$",
        reply,
    )
    if understanding:
        return reply[:understanding.start()].rstrip() + "\n\n" + section + "\n\n" + reply[understanding.start():].lstrip()
    return section + "\n\n" + reply.lstrip()


def _format_case_reconstruction(state: GuideState) -> str:
    """Render a compact, complete factual record from the frozen case packet."""
    packet_facts = (state.case_analysis_packet or {}).get("facts") or active_case_facts(state.case_facts)
    labels = {
        "event": "经过", "actor": "相关主体", "relationship": "双方关系",
        "time": "时间", "location": "地点", "harm": "损失或伤情",
        "procedure": "已采取的处理", "claim": "诉求", "amount": "财物或金额",
        "evidence": "证据线索",
    }
    buckets: dict[str, list[str]] = {}
    for item in packet_facts:
        if not isinstance(item, dict) or item.get("status") != "asserted":
            continue
        category = str(item.get("category") or "event")
        statement = str(item.get("statement") or item.get("value") or "").strip()
        if not statement:
            continue
        # Form fallbacks can preserve a compound answer under the question's
        # slot. Do not display a place/person-only answer as a "time" merely
        # because the original form asked about several things at once.
        if category == "time" and not re.search(
            r"\d{1,4}年|\d{1,2}[月/-]|\d{1,2}[日号]|\d{1,2}[时点分]|今天|昨天|前天|上午|下午|晚上|凌晨|刚才|此前|之后",
            statement,
        ):
            category = "event"
        bucket = labels.get(category, "其他已确认事实")
        buckets.setdefault(bucket, []).append(statement)
    ordered = ["经过", "相关主体", "双方关系", "时间", "地点", "财物或金额", "损失或伤情", "已采取的处理", "证据线索", "诉求", "其他已确认事实"]
    lines: list[str] = []
    for label in ordered:
        values = _dedupe_text_items(buckets.get(label, []), limit=3)
        if values:
            lines.append(f"- **{label}：** {'；'.join(values)}。")
    if state.evidence_confirmed:
        evidence = "；".join(_dedupe_text_items([str(item) for item in state.evidence_confirmed], limit=5))
        if evidence and not any(line.startswith("- **证据线索：") for line in lines):
            lines.append(f"- **已确认的材料或线索：** {evidence}。")
    return "**【案件完整还原】**\n" + "\n".join(lines[:10]) if lines else ""


def _ensure_case_reconstruction(reply: str, state: GuideState) -> str:
    if "【案件完整还原】" in reply:
        return reply
    section = _format_case_reconstruction(state)
    if not section:
        return reply
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:核心争点分析|法律依据)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + section + "\n\n" + reply[marker.start():].lstrip()
    return _insert_before_document_offer(reply, section)


def _format_fact_tensions(state: GuideState) -> str:
    """Render contradictions as decision boundaries, never as a fact finding."""
    packet = state.case_analysis_packet or {}
    tensions = [item for item in (packet.get("fact_tensions") or []) if isinstance(item, dict)]
    facts = {
        str(item.get("key") or ""): str(item.get("statement") or "").strip()
        for item in packet.get("facts") or []
        if isinstance(item, dict)
    }
    blocks: list[str] = []
    for item in tensions[:3]:
        keys = [*item.get("side_a_fact_keys", []), *item.get("side_b_fact_keys", [])]
        statements = _dedupe_text_items([facts.get(str(key), "") for key in keys], limit=4)
        if not statements:
            statements = _dedupe_text_items([str(value) for value in item.get("statements") or []], limit=4)
        if not statements:
            continue
        blocks.append(
            f"- **{str(item.get('title') or '关键事实待核实')}：** {'；'.join(statements)}。\n"
            f"  - **影响：** {str(item.get('why_it_matters') or '该矛盾会影响当前判断。')}\n"
            f"  - **优先核实：** {str(item.get('resolution_action') or '用原始材料或第三方记录核实。')}"
        )
    return "**【关键矛盾与待核实】**\n" + "\n".join(blocks) if blocks else ""


def _ensure_fact_tensions(reply: str, state: GuideState) -> str:
    if "【关键矛盾与待核实】" in reply:
        return reply
    section = _format_fact_tensions(state)
    if not section:
        return reply
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:核心争点分析|法律依据)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + section + "\n\n" + reply[marker.start():].lstrip()
    return _insert_before_document_offer(reply, section)


def _format_optimal_procedure_path(state: GuideState) -> str:
    """Render the next legal route from case status, not a generic channel list."""
    facts = active_case_facts(state.case_facts)
    procedure_text = "；".join(
        str(item.get("statement") or "") for item in facts
        if item.get("category") == "procedure" and item.get("status") == "asserted"
    )
    steps: list[str] = []
    procedure_steps: list[str] = []
    analyst_actions: list[str] = []
    strategy_plan = state.strategy_plan if isinstance(state.strategy_plan, dict) else {}
    for item in strategy_plan.get("procedure_path") or []:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step") or "").strip()
        if not step:
            continue
        trigger = str(item.get("trigger") or "").strip()
        expected = str(item.get("expected_change") or "").strip()
        suffix = "；触发：" + trigger if trigger else ""
        suffix += "；作用：" + expected if expected else ""
        steps.append(step + suffix)
    if steps:
        steps = _dedupe_text_items(steps, limit=6)
    for analysis in state.issue_analyses or []:
        if isinstance(analysis, dict):
            procedure_steps.extend(
                str(item) for item in (analysis.get("procedure_steps") or [])
            )
            analyst_actions.extend(
                str(item) for item in (analysis.get("recommended_actions") or [])
            )

    # The model's issue-specific route is authoritative when it is available.
    # Domain handling below is only a safe fallback for analysis/render failures.
    if not steps and procedure_steps:
        steps = _dedupe_text_items(procedure_steps, limit=5)
    elif not steps and analyst_actions:
        # Backward compatibility for analyses produced before procedure_steps
        # was introduced. New analyses keep this separate from urgent actions.
        steps = _dedupe_text_items(analyst_actions, limit=5)
    elif not steps and state.legal_domain == "criminal_public_security":
        reported = any(marker in procedure_text for marker in ("已报警", "报案", "受案", "派出所"))
        if reported:
            steps = [
                "向已联系或已受理的公安机关补充完整经过、财物信息、监控位置和目击者线索，并取得或查询受案回执、案件编号。",
                "由公安机关依法调取监控、联系证人、核验嫌疑人和追查财物去向；用户同步保存购买凭证和线索来源。",
                "如查获财物，跟进追赃返还；如财物无法返还或存在损失，再根据案件进展核对赔偿或附带民事救济路径。",
            ]
        else:
            steps = [
                "优先向 110 或就近公安机关报案，完整说明案发经过、人员、财物、地点及监控、证人线索。",
                "取得受案回执或案件编号后，向承办机关补充证据并跟进调查进展。",
                "根据侦查和追赃结果，再评估返还、赔偿或其他救济。",
            ]
    elif not steps:
        for analysis in state.issue_analyses or []:
            if isinstance(analysis, dict):
                steps.extend(str(item) for item in (analysis.get("recommended_actions") or []))
        steps = _dedupe_text_items(steps, limit=4)
    if not steps:
        return ""
    return "**【最优程序路径】**\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )


def _ensure_optimal_procedure_path(reply: str, state: GuideState) -> str:
    if "【最优程序路径】" in reply:
        return reply
    section = _format_optimal_procedure_path(state)
    if not section:
        return reply
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:维权路径比较|法律依据)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + section + "\n\n" + reply[marker.start():].lstrip()
    return _insert_before_document_offer(reply, section)


def _format_evidence_strategy(state: GuideState) -> str:
    """Turn analyst evidence tasks into a short, case-specific action map."""
    existing, evidence_leads = _evidence_for_plan(state)
    existing = _dedupe_text_items(existing, limit=6)
    evidence_leads = _dedupe_text_items(evidence_leads, limit=6)
    tasks: list[str] = []
    strategy_plan = state.strategy_plan if isinstance(state.strategy_plan, dict) else {}
    for item in strategy_plan.get("evidence_plan") or []:
        if not isinstance(item, dict):
            continue
        material = str(item.get("item") or "证据材料").strip()
        action = str(item.get("action") or "").strip()
        target = str(item.get("proof_target") or "").strip()
        status = str(item.get("status") or "unknown").strip()
        tasks.append(f"{material}（状态：{status}；证明目标：{target}；下一步：{action}）")
    for analysis in state.issue_analyses or []:
        if isinstance(analysis, dict):
            tasks.extend(str(item) for item in (analysis.get("evidence_actions") or []))
    tasks = _dedupe_text_items(tasks, limit=5)
    if not existing and not evidence_leads and not tasks:
        return ""
    lines: list[str] = []
    if existing:
        lines.append(f"- **目前已确认：** {'；'.join(existing)}。请保留原始载体，不把线索等同于已被机关采信的证据。")
    if evidence_leads:
        lines.append(f"- **目前只是线索、尚未取得：** {'；'.join(evidence_leads)}。应先确认保存主体、覆盖时间和调取方式。")
    if tasks:
        lines.extend(f"- **优先补强：** {task}" for task in tasks)
    return "**【证据作战图】**\n" + "\n".join(lines)


def _ensure_evidence_strategy(reply: str, state: GuideState) -> str:
    if "【证据作战图】" in reply:
        return reply
    section = _format_evidence_strategy(state)
    if not section:
        return reply
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:法律依据|维权路径比较|行动清单)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + section + "\n\n" + reply[marker.start():].lstrip()
    return _insert_before_document_offer(reply, section)


def _uses_accessible_language(state: GuideState) -> bool:
    """Detect an explicit need for a shorter, easier-to-follow answer."""
    user_text = "\n".join(
        str(message.content)
        for message in state.messages
        if isinstance(message, HumanMessage)
    )
    markers = (
        "年纪大", "老人", "老年", "不识字", "文化不高", "看不懂",
        "说不清", "讲不清", "记不清", "脑子不利索",
    )
    return any(marker in user_text for marker in markers)


def _audience_guidance(state: GuideState) -> str:
    if state.force_conclude or state.wants_conclude:
        return (
            "启用收敛模式：用户已要求给方案，或系统已达到追问上限。禁止继续追问。"
            "全文以2200字为上限，不用表格；法律依据最多2条，路径最多2种，行动步骤最多4步。"
            "对尚未回答的流程问题只作一句话解释，避免重复事实缺口和风险提示。"
        )
    if not _uses_accessible_language(state):
        return (
            "使用短段落和直接表达，避免重复同一风险。完整回答尽量控制在2500字以内，"
            "优先保留法律依据、路径、胜算和行动步骤。"
        )
    return (
        "启用易读模式：用户明确表示年纪大、记不清或说不清。使用简单短句，不用表格，"
        "不责备用户，不连续堆砌术语。全文以1800字为上限；法律依据最多选2条，路径最多2种，"
        "行动清单只保留最重要的3步，并优先给出可以电话办理或由家人协助的方式。"
        "避免使用刺激性、责备性或夸大说法。"
    )


def _compact_final_reply(
    reply: str,
    accessible: bool,
    *,
    compact: bool = False,
) -> str:
    """Remove optional repetition while preserving every required result section."""
    limit = 2200 if accessible else (2600 if compact else 3000)
    if len(reply) <= limit and not accessible:
        return reply
    understanding = re.search(r"\*{0,2}【理解您的情况】\*{0,2}", reply)
    if understanding:
        # Keep the mandatory priority-actions block placed before the case
        # summary.  Older compaction kept only the confidence badge here,
        # silently deleting the most useful part of a strategy-first reply.
        leading = reply[:understanding.start()].strip()
        if "【现在最优先行动】" in leading:
            prefix = leading
        else:
            badge = re.match(r"\s*(\*\*📊[^\n]+\*\*)", reply)
            prefix = badge.group(1) if badge else ""
        reply = (prefix + "\n\n" if prefix else "") + reply[understanding.start():]
    optional_section = re.compile(
        r"\n*\*{0,2}(?:（可选）\s*)?【(?:常见误区|关键缺失信息清单)】\*{0,2}.*?(?=\n---|\n\*{0,2}【|\Z)",
        re.S,
    )
    reply = optional_section.sub("", reply)
    if accessible:
        reply = re.sub(
            r"\n\s*[*+-]\s+\*\*(?:一句话解释|直接支持|行动依据)\*\*：[^\n]*",
            "",
            reply,
        )
        reply = re.sub(
            r"\n*---\s*\n(?:---\s*\n)?\*\*请再次注意：\*\*.*?(?=\n>|\n---|\Z)",
            "\n",
            reply,
            flags=re.S,
        )
    if accessible and len(reply) > limit:
        reply = re.sub(
            r"\n*\*{0,2}最后，最重要的建议：?\*{0,2}.*?(?=\n---|\Z)",
            "",
            reply,
            flags=re.S,
        )
    reply = re.sub(
        r"\n*\*{0,2}【(?:维权情况分析|有利因素与风险|因素分析)】\*{0,2}.*?"
        r"(?=\n(?:#{1,6}\s*)?\*{0,2}【|\n---|\Z)",
        "",
        reply,
        flags=re.S,
    )
    reply = re.sub(r"\s*\[[a-z_]+(?:\.[a-z_]+)+\]", "", reply)
    reply = re.sub(r"\n{3,}", "\n\n", reply).strip()
    if len(reply) <= limit:
        return reply

    suffix = ""
    for marker in ("\n\n---\n📄", "\n---\n📄", "\n📄 **需要参考文书"):
        if (position := reply.find(marker)) >= 0:
            suffix = reply[position:].strip()
            reply = reply[:position].rstrip()
            break
    core_limit = max(600, limit - len(suffix) - (2 if suffix else 0))

    # The prompt's word limit is advisory. Enforce a deterministic display
    # budget at section boundaries while preserving the response contract.
    section_pattern = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?P<title>[^】]+)】\*{0,2}\s*$"
    )
    matches = list(section_pattern.finditer(reply))
    if not matches:
        core = reply[: core_limit - 1].rstrip() + "…"
        return core + (f"\n\n{suffix}" if suffix else "")

    prefix = reply[: matches[0].start()].strip()
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reply)
        header = reply[match.start():match.end()].strip()
        body = reply[match.end():end].strip()
        sections.append((match.group("title"), header, body))

    budgets = {
        "理解您的情况": 240 if not accessible else 180,
        "法律依据": 750 if not accessible else 600,
        "类似案例参考": 280 if not accessible else 200,
        "维权路径比较": 550 if not accessible else 400,
        "优势与劣势": 180 if not accessible else 150,
        "行动清单": 650 if not accessible else 450,
    }

    def _shorten(body: str, budget: int) -> str:
        if len(body) <= budget:
            return body
        units = [
            unit.strip()
            for unit in re.split(
                r"(?<=。)\s*|(?<=；)\s*|(?=\n\s*(?:[-*□]|\d+[.、]))",
                body,
            )
            if unit.strip()
        ]
        kept: list[str] = []
        for unit in units:
            projected = len("\n".join(kept + [unit]))
            if kept and projected > budget - 18:
                break
            remaining = budget - len("\n".join(kept)) - 18
            kept.append(unit[:max(20, remaining)])
            if len("\n".join(kept)) >= budget - 18:
                break
        return ("\n".join(kept).rstrip("，；。") + "……")[:budget]

    rendered = [prefix] if prefix else []
    for title, header, body in sections:
        rendered.append(f"{header}\n{_shorten(body, budgets.get(title, 420))}".strip())
    compacted = "\n\n".join(item for item in rendered if item).strip()
    result = compacted + (f"\n\n{suffix}" if suffix else "")
    if len(result) <= limit:
        return result
    notice = "\n\n> 内容已按易读长度压缩。"
    core = compacted[: max(100, core_limit - len(notice) - 2)].rstrip("，；。\n ") + "……" + notice
    return core + (f"\n\n{suffix}" if suffix else "")


def _normalize_required_sections(reply: str) -> str:
    """Keep model wording compatible with the stable user-facing response contract."""
    normalized = reply.replace("【初步方向建议】", "【维权路径比较】")
    normalized = re.sub(
        r"(?m)^\s*#{1,6}\s*(?:检索到的)?(?:相关)?法律依据\s*$",
        "**【法律依据】**",
        normalized,
    )
    return normalized


_GENERIC_BOILERPLATE_PHRASES = (
    "需要结合完整事实、证据和办案机关认定",
    "当前仅作阶段性分析",
    "现有信息可以支持继续采取低风险的证据保全和程序咨询行动",
    "如关键事实、证据或程序状态不同，法律评价可能随之变化",
    "具体以办案机关认定为准",
)


_INTERNAL_KEY_LEAK_RE = re.compile(
    r"(?:followup\.[a-z_]+|user_[a-z_]+|fraudster_[a-z_]+|"
    r"emergency_action_taken|transaction_records|previous_complaints|"
    r"additional_actions_after_discovery|platform_security_measures)"
)


_INTERNAL_KEY_LABELS = {
    "user_awareness_of_fraud_prevention": "用户对防骗知识的了解情况",
    "user_awareness_of_platform_protection_measures": "用户对平台防骗保护措施的了解情况",
    "user_platform_security_measures": "平台安全措施情况",
    "followup.fraudster_identification": "诈骗者身份识别信息",
    "followup.transaction_records": "交易记录情况",
    "user_complaint_to_platform": "用户向平台投诉情况",
    "emergency_action_taken": "是否采取紧急措施",
    "fraudster_contact_method": "诈骗者联系方式",
    "fraudster_promises": "诈骗者承诺内容",
    "followup.previous_complaints": "此前投诉情况",
    "user_additional_actions_after_discovery": "发现被骗后的其他行动",
}


def _generic_boilerplate_issues(reply: str) -> list[dict]:
    issues: list[dict] = []
    for phrase in _GENERIC_BOILERPLATE_PHRASES:
        if phrase in reply:
            issues.append({
                "type": "generic_boilerplate",
                "target": "核心争点分析/优势与劣势",
                "issue": f"出现泛化模板句式：{phrase}",
                "fix": "结合本案具体事实、法律要件和证据状态改写，禁止保留该句式",
            })
    return issues[:5]


def _force_generic_boilerplate_revision(critique: dict, draft: str) -> dict:
    issues = _generic_boilerplate_issues(draft)
    if not issues:
        return critique
    return {
        **critique,
        "verdict": "revise",
        "issues": [*(critique.get("issues") or []), *issues],
    }


def _strip_generic_boilerplate(reply: str) -> str:
    lines = [
        line
        for line in reply.splitlines()
        if not any(phrase in line for phrase in _GENERIC_BOILERPLATE_PHRASES)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _deterministic_draft_problems(reply: str) -> list[dict]:
    issues = list(_generic_boilerplate_issues(reply))
    if _INTERNAL_KEY_LEAK_RE.search(reply):
        issues.append({
            "type": "internal_key_leak",
            "target": "法条要件核对/证据状态",
            "issue": "出现内部字段ID或语义键，例如 user_、followup.、fraudster_",
            "fix": "改用自然语言事实和证据名称",
        })
    forbidden = (
        "【关键缺失信息清单】",
        "【强烈建议】",
        "请补充上述信息",
        "【动态追问表单回答】",
    )
    if any(marker in reply for marker in forbidden):
        issues.append({
            "type": "forbidden_output",
            "target": "最终方案",
            "issue": "包含违禁栏目或内部格式",
            "fix": "删除违禁栏目，用优势与劣势或决策边界说明",
        })
    return issues[:5]


def _enforce_final_output_contract(reply: str) -> str:
    """Narrow deterministic cleanup for sections the final prompt explicitly forbids."""
    forbidden_section = re.compile(
        r"\n*\*{0,2}(?:（可选）\s*)?【(?:关键缺失信息清单|强烈建议)】\*{0,2}.*?"
        r"(?=\n---|\n\*{0,2}【|\Z)",
        re.S,
    )
    reply = forbidden_section.sub("", reply or "")
    for phrase in ("请补充上述信息", "补充上述信息后", "我将重新为您分析", "请补充关键信息"):
        reply = re.sub(rf"[^\n]*{re.escape(phrase)}[^\n]*\n?", "", reply)
    if "【动态追问表单回答】" in reply:
        reply = re.sub(r"(?ms)^.*?【动态追问表单回答】.*?(?=\n\s*【|\Z)", "", reply)
    return re.sub(r"\n{3,}", "\n\n", reply).strip()


def _insert_before_document_offer(reply: str, section: str) -> str:
    """Insert a required result section before the optional document offer."""
    markers = ("\n\n---\n📄", "\n---\n📄", "\n📄 **需要参考文书")
    positions = [position for marker in markers if (position := reply.find(marker)) >= 0]
    if not positions:
        return reply.rstrip() + "\n\n" + section.strip()
    position = min(positions)
    return reply[:position].rstrip() + "\n\n" + section.strip() + "\n\n" + reply[position:].lstrip()


def _channel_summary_lines(state: GuideState, *, limit: int = 2) -> list[str]:
    """Build fallback routing text from retrieved channel records, not domains."""
    lines: list[str] = []
    for channel in state.relevant_channels:
        if not isinstance(channel, dict):
            continue
        name = str(channel.get("name") or "").strip()
        if not name:
            continue
        contacts = [
            str(value).strip()
            for value in (channel.get("phone"), channel.get("url"))
            if str(value or "").strip()
        ]
        suffix = f"（{'；'.join(contacts)}）" if contacts else ""
        lines.append(f"- **{name}**{suffix}：具体受理范围和材料以该机构答复为准。")
        if len(lines) >= limit:
            break
    return lines


def _ensure_action_checklist(reply: str, state: GuideState) -> str:
    """Restore a generic checklist from structured state when the model omits it."""
    if "【行动清单】" in reply:
        return reply
    channel = "方案中列明的受理机构"
    if isinstance(state.strategy_plan, dict) and state.strategy_plan.get("priority_actions"):
        return reply
    for item in state.relevant_channels:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        contacts = [
            str(value).strip()
            for value in (item.get("phone"), item.get("url"))
            if str(value or "").strip()
        ]
        channel = f"{name}（{'；'.join(contacts)}）" if contacts else name
        break
    steps = (
        "1. 备份原始材料，保留原始载体，并按时间顺序整理已经确认的事实。\n"
        f"2. 联系{channel}核对受理范围、管辖和所需材料；提交后保存回执或受理编号。\n"
        "3. 对时效、费用或程序仍不清楚时，可拨打 12348 进一步核验。"
    )
    return _insert_before_document_offer(reply, f"**【行动清单】**\n{steps}")


def _ensure_legacy_required_plan_sections(reply: str, state: GuideState) -> str:
    """最终压缩后再次核对稳定输出合同，避免模型随机漏段。"""
    if "【维权路径" not in reply:
        routes = _channel_summary_lines(state)
        if not routes:
            routes = [
                "- 当前没有可核验的本地渠道记录，请先拨打 12348 核对受理机构、管辖和程序。"
            ]
        fallback = "**【维权路径比较】**\n" + "\n".join(routes)
        reply = _insert_before_document_offer(reply, fallback)
    if "【优势与劣势】" not in reply:
        _pros_lines = [
            "- 当前检索到的法律依据可作为维权依据，但具体适用仍需结合原始证据核验。"
        ]
        _cons_lines = ["- 当前信息可能仍不完整，结果受事实、证据和对方抗辩影响，不宜据此认定责任或结果。"]
        for item in state.adverse_facts:
            if str(item).strip():
                _cons_lines.append(f"- {str(item).strip()}")
        for item in state.evidence_unavailable:
            if str(item).strip():
                _cons_lines.append(f"- 缺少「{item}」，对方可能质疑举证能力。")
        reply = _insert_before_document_offer(
            reply,
            (
                "**【优势与劣势】**\n"
                "**有利因素**：\n" + "\n".join(_pros_lines)
                + "\n**不利因素**：\n" + "\n".join(_cons_lines)
            ),
        )
    empty_recommendation = re.compile(
        r"(?m)(?:#{1,6}\s*)?\*{0,2}【推荐方案】\*{0,2}\s*"
        r"(?=(?:\n#{1,6}\s*)?\*{0,2}【优势与劣势】\*{0,2})"
    )
    if empty_recommendation.search(reply):
        first_channel = next(iter(_channel_summary_lines(state, limit=1)), "")
        recommendation = (
            f"优先核对并使用已检索到的渠道：{first_channel.lstrip('- ')}"
            if first_channel
            else "先保存原始材料，并拨打 12348 核对受理机构、管辖和时效。"
        )
        reply = empty_recommendation.sub(
            f"【推荐方案】\n{recommendation}\n\n",
            reply,
        )
    return _ensure_action_checklist(reply, state)


def _strip_legacy_strategy_sections(reply: str) -> str:
    """Remove fixed legacy appendices once the strategy center has a plan."""
    targets = {"维权路径比较", "优势与劣势", "行动清单", "证据作用与缺口", "决策边界与条件", "继续完善"}
    targets = {
        "\u7ef4\u6743\u8def\u5f84\u6bd4\u8f83", "\u4f18\u52bf\u4e0e\u52a3\u52bf", "\u884c\u52a8\u6e05\u5355",
        "\u8bc1\u636e\u4f5c\u7528\u4e0e\u7f3a\u53e3", "\u51b3\u7b56\u8fb9\u754c\u4e0e\u6761\u4ef6", "\u7ee7\u7eed\u5b8c\u5584",
    }
    targets.add("\u4f1a\u6539\u53d8\u7ed3\u8bba\u7684\u6761\u4ef6")
    lines = reply.splitlines()
    output: list[str] = []
    skipping = False
    for line in lines:
        normalized = re.sub(r"[#*【】\s:：]", "", line)
        is_heading = bool(normalized) and normalized in targets
        generic_heading = bool(re.match(r"^\s*(?:#{1,6}\s*)?\*{0,2}【?.{1,48}】?\*{0,2}\s*$", line))
        normalized = re.sub(r"[#*\u3010\u3011\s:\uff1a]", "", line)
        is_heading = bool(normalized) and normalized in targets
        generic_heading = generic_heading or bool(re.match(r"^\s*(?:#{1,6}\s*)?\*{0,2}\u3010.{1,48}\u3011\*{0,2}\s*$", line))
        if is_heading:
            skipping = True
            continue
        if skipping and (generic_heading or line.strip() == "---"):
            skipping = False
        if not skipping:
            output.append(line)
    cleaned = "\n".join(output).strip()
    for marker in ("\u7ee7\u7eed\u5b8c\u5584：", "\u7ee7\u7eed\u5b8c\u5584:"):
        position = cleaned.find(marker)
        if position >= 0:
            cleaned = cleaned[:position].rstrip()
    return cleaned


def _ensure_required_plan_sections(reply: str, state: GuideState) -> str:
    """Use the strategy center as the final contract; legacy filling is fallback only."""
    if isinstance(state.strategy_plan, dict) and state.strategy_plan:
        reply = _strip_legacy_strategy_sections(reply)
        reply = _ensure_strategy_headline(reply, state)
        reply = _ensure_priority_actions(reply, state)
        reply = _ensure_optimal_procedure_path(reply, state)
        reply = _ensure_evidence_strategy(reply, state)
        return _ensure_strategy_insights(reply, state)
    return _ensure_legacy_required_plan_sections(reply, state)


def _sanitize_unverified_evidence_assertions(
    reply: str,
    unverified_evidence: list[str],
) -> str:
    """Prevent unverified evidence leads from being restated as materials in hand."""
    for item in unverified_evidence:
        if not item:
            continue
        escaped = re.escape(item)
        note = f"截图文字提到“{item}”，但本次未直接展示；如确实留存，请补充核验"
        reply = re.sub(
            rf"{escaped}是[^。\n]*(?:证据|材料)[。.]?",
            note,
            reply,
        )
        for prefix in ("您手中握有", "您手中持有", "您持有", "您有", "您保留了"):
            reply = reply.replace(f"{prefix}{item}", note)
    return reply


def _ensure_post_conclusion_options(reply: str, state: GuideState | None = None) -> str:
    """Expose supported same-case continuation after a plan has been generated."""
    if "回复「继续补充」" in reply or "已达到主动追问上限" in reply:
        return reply
    if state is not None and isinstance(state.strategy_plan, dict) and state.strategy_plan:
        return reply
    reached_absolute_limit = bool(
        state
        and (
            state.ask_rounds >= settings.GUIDE_MAX_OPT_IN_ASK_ROUNDS
            or state.total_rounds >= settings.GUIDE_MAX_TOTAL_ROUNDS
        )
    )
    if reached_absolute_limit:
        return (
            reply.rstrip()
            + "\n\n🔄 **还想完善方案？** 当前已达到主动追问上限，我不会继续盘问；"
            "您仍可以直接发送新的事实或证据，我会在同一案件中重新评估、更新方案。"
        )
    return (
        reply.rstrip()
        + "\n\n🔄 **还想完善方案？** 您可以直接发送新的事实或证据；"
        "也可以回复「继续补充」，我会接着只问一个最关键的问题，"
        "并在同一案件中重新评估、更新方案。"
    )


def _ensure_case_reference(
    reply: str,
    similar_cases: list[dict],
    case_context: str = "",
    state: GuideState | None = None,
) -> str:
    """Render case references only from structured retrieval results."""
    cases = list(similar_cases)
    if not cases and case_context:
        title_match = re.search(r"案例\d+【([^】]+)】", case_context)
        number_match = re.search(r"基本信息：([^｜\n]+)", case_context)
        gist_match = re.search(r"案情摘要：(.+?)(?:\n法院认为：|\n裁判结果：|\n法律依据：|\n原始链接：|\n\n---|\Z)", case_context, re.S)
        if title_match:
            cases.append({
                "title": title_match.group(1).strip(),
                "case_number": number_match.group(1).strip() if number_match else "",
                "gist": gist_match.group(1).strip() if gist_match else "",
                "text": "",
            })
    grounded_cases: list[dict] = []
    seen_titles: set[str] = set()
    for case in cases:
        title = str(case.get("title") or "").strip()
        summary = str(case.get("gist") or case.get("text") or "").strip()
        if not title or not summary or title in seen_titles:
            continue
        seen_titles.add(title)
        grounded_cases.append(case)
        if len(grounded_cases) >= 2:
            break

    section_pattern = re.compile(
        r"\n*\*{0,2}【类似案例(?:参考)?】\*{0,2}.*?"
        r"(?=\n(?:#{1,6}\s*)?\*{0,2}【|\n---|\Z)",
        re.S,
    )
    reply = section_pattern.sub("", reply).strip()
    if not grounded_cases:
        return reply

    lines = ["**【类似案例参考】**"]
    for case in grounded_cases:
        title = str(case.get("title") or "相似案件").strip()
        case_number = str(case.get("case_number") or "").strip()
        summary = re.sub(
            r"\s+",
            " ",
            str(case.get("gist") or case.get("text") or ""),
        ).strip()
        if len(summary) > 180:
            shortened = summary[:180]
            sentence_end = max(shortened.rfind("。"), shortened.rfind("；"))
            summary = (shortened[: sentence_end + 1] if sentence_end >= 90 else shortened.rstrip("，；。")) + "……"
        label = f"{title}（{case_number}）" if case_number else title
        original_url = str(case.get("original_url") or "").strip()
        source_link = f" [查看原始链接]({original_url})" if original_url else ""
        lines.append(f"- **{label}**：{summary}{source_link}")
    lines.append("- 类案仅用于说明裁判思路，不能替代对您本人证据和事实的判断。")
    block = "\n".join(lines)
    next_section = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【维权路径(?:比较)?】\*{0,2}\s*$",
        reply,
    )
    if next_section:
        return (
            reply[:next_section.start()].rstrip()
            + "\n\n"
            + block
            + "\n\n"
            + reply[next_section.start():]
        )
    return reply.rstrip() + "\n\n" + block


_PROS_CONS_HEADING = re.compile(r"\*{0,2}【优势与劣势】\*{0,2}")


def _ensure_pros_cons(reply: str, state: GuideState) -> str:
    """LLM 漏写不利因素时，用既有结构化数据补齐（不做关键词挖掘）。

    只做结构性补全：若【优势与劣势】段内完全没有"不利/劣势/风险/前提"内容，
    把系统已提取的 adverse_facts 与缺失证据渲染为"**不利因素**"子段。与
    _ensure_decision_uncertainties 同哲学，渲染的是结构化字段而非规则推导。
    """
    if isinstance(state.strategy_plan, dict) and state.strategy_plan:
        return reply
    match = _PROS_CONS_HEADING.search(reply)
    if not match:
        return reply
    section_start = match.start()
    heading_line_end = reply.find("\n", section_start)
    if heading_line_end < 0:
        return reply
    content_start = heading_line_end + 1
    next_heading = re.search(r"\n\*{0,2}【", reply[content_start:])
    section_end = (
        content_start + next_heading.start() if next_heading else len(reply)
    )
    body = reply[content_start:section_end]
    # 只扫标题行之后的正文：标题“优势与劣势”本身含“劣势”二字，扫整段会永远命中。
    if any(keyword in body for keyword in ("不利", "劣势", "风险", "前提")):
        return reply
    cons = [f"- {str(item).strip()}" for item in state.adverse_facts if str(item).strip()]
    cons += [
        f"- 缺少「{item}」，对方可能质疑举证能力。"
        for item in state.evidence_unavailable
        if str(item).strip()
    ]
    if not cons:
        return reply
    block = "\n**不利因素**：\n" + "\n".join(cons)
    return reply[:section_end].rstrip() + "\n" + block + "\n" + reply[section_end:].lstrip()


def _ensure_decision_uncertainties(reply: str, state: GuideState) -> str:
    """Attach application-owned decision limits to every non-definitive plan."""

    if isinstance(state.strategy_plan, dict) and state.strategy_plan:
        return reply
    raw = state.decision_sufficiency or {}
    report = (
        DecisionSufficiencyReport.model_validate(raw)
        if raw
        else assess_decision_sufficiency(state)
    )
    if report.sufficient_for_definitive_plan:
        return reply
    missing = unresolved_decision_summary(report)
    if not missing:
        return reply
    lines = [
        "## 决策边界与条件",
        "",
        "以下信息缺口不会阻止您先采取保全证据、记录沟通等低风险行动，"
        "但涉及责任、金额、期限或受理机构的判断应按条件理解：",
        *[f"- {item}。" for item in missing[:6]],
        "- 在上述信息核实前，不宜把当前方案理解为责任已经成立或结果已经确定。",
    ]
    block = "\n".join(lines)
    if "## 决策边界与条件" in reply:
        start = reply.index("## 决策边界与条件")
        next_heading = reply.find("\n## ", start + 3)
        if next_heading < 0:
            return reply[:start].rstrip() + "\n\n" + block
        return reply[:start].rstrip() + "\n\n" + block + "\n\n" + reply[next_heading:].lstrip()
    document_offer = reply.find("\n---\n📄")
    if document_offer >= 0:
        return (
            reply[:document_offer].rstrip()
            + "\n\n"
            + block
            + "\n"
            + reply[document_offer:]
        )
    return reply.rstrip() + "\n\n" + block


_FACT_SLOT_LABELS = {
    "event_time": "发生时间",
    "procedure": "处理经过",
    "legal_relationship": "纠纷类型或双方关系",
    "current_safety": "当前安全状况",
    "employment_status": "劳动关系状态",
    "claim": "希望实现的结果",
    "transaction": "交易内容与金额",
    "event": "事件经过",
    "harm": "损失或影响",
    "children": "子女情况",
    "property_and_safety": "财产与人身安全",
    "event_and_liability": "事故与责任划分",
    "insurance_and_claim": "保险与赔偿",
    "administrative_action": "行政行为",
    "right_type": "权利类型",
    "infringement": "侵权事实",
    "source_and_harm": "污染来源与影响",
    "agreement": "协议签订情况",
    "event_and_urgency": "事件与紧急程度",
    "region": "发生地点",
}


def _format_unknown_fact_notes(state: GuideState) -> str:
    """把用户明确表示“不清楚”的事实渲染为醒目章节，避免方案被当成已确认事实。

    答案终结性保证这类问题不会再被追问，因此必须显性带进最终方案：标出
    该关键点未确认，并说明它影响哪类判断。
    """
    domain = state.legal_domain or "other"
    valid_ids = {rule.id for rule in fact_followups(domain)}
    unknown: list[tuple[str, str]] = []
    for rule_id, record in (state.fact_records or {}).items():
        if str((record or {}).get("status")) != "unknown":
            continue
        if rule_id not in valid_ids:
            continue
        rule = find_fact_followup(domain, rule_id)
        slot = str((record or {}).get("slot") or getattr(rule, "slot", "") or "")
        label = _FACT_SLOT_LABELS.get(slot, slot or "关键信息")
        why = " ".join(
            str((record or {}).get("why") or (getattr(rule, "why", "") if rule else "")).split()
        )
        for prefix in ("为了用于", "用于", "为了"):
            if why.startswith(prefix):
                why = why[len(prefix):].strip()
                break
        unknown.append((label, why))
    if not unknown:
        return ""
    lines = [
        "## ⚠️ 用户未确认的关键点",
        "",
        "以下信息您表示暂不清楚，本方案已按“未确认”处理，涉及责任、金额、期限或程序"
        "的判断请不要视为已确认事实；核实前建议先拨打 12348 或联系当地法律服务机构：",
    ]
    for label, why in unknown:
        lines.append(f"- **{label}**：用户表示不清楚 → {why or '影响相关法律判断'}")
    return "\n".join(lines)


def _ensure_evidence_coverage_section(reply: str, state: GuideState) -> str:
    """Render application-owned proof coverage instead of model certainty."""

    if isinstance(state.strategy_plan, dict) and state.strategy_plan:
        return reply
    report = state.evidence_coverage or evaluate_state_evidence(state)
    content = format_evidence_coverage(report, max_targets=4)
    if content.startswith("（"):
        return reply
    basis_labels: list[str] = []
    for requirement in state.evidence_requirements:
        if not isinstance(requirement, dict) or not requirement.get("active", True):
            continue
        for basis in requirement.get("basis_refs") or []:
            if not isinstance(basis, dict):
                continue
            label = "《{}》{}".format(
                basis.get("title") or "本轮检索法律",
                basis.get("article_no") or "相关规定",
            )
            if label not in basis_labels:
                basis_labels.append(label)
    basis_note = (
        "\n\n**评估依据：** "
        + "；".join(basis_labels[:4])
        + "。上述依据用于确定待证明事项和材料用途，不代表受理机关已认可材料真实性或证明力。"
        if basis_labels else ""
    )
    block = "## 证据作用与缺口\n\n" + content + basis_note
    pattern = re.compile(
        r"\n*## 证据作用与缺口\s*\n.*?(?=\n## |\n---\n📄|\Z)",
        re.S,
    )
    reply = pattern.sub("", reply).rstrip()
    return _insert_before_document_offer(reply, block)


def _format_issue_analysis_section(state: GuideState) -> str:
    """Render the analyst output as a required user-facing plan section.

    The final prose model is allowed to improve language, but it must not be
    able to silently discard the issue-by-issue reasoning produced earlier in
    the conclusion pipeline.  This also makes the new workflow observable in
    the UI and useful when the final renderer falls back to an older template.
    """
    analyses = [item for item in (state.issue_analyses or []) if isinstance(item, dict)]
    if not analyses:
        return ""
    blocks: list[str] = []
    for item in analyses[:5]:
        title = str(item.get("title") or "核心法律问题").strip()
        current = str(item.get("current_view") or "需结合现有证据作阶段性判断").strip()
        application = str(item.get("application_analysis") or "").strip()
        branch = str(item.get("conditional_branch") or "").strip()
        verify = [str(value).strip() for value in (item.get("facts_to_verify") or []) if str(value).strip()]
        actions = [str(value).strip() for value in (item.get("recommended_actions") or []) if str(value).strip()]
        evidence_actions = [str(value).strip() for value in (item.get("evidence_actions") or []) if str(value).strip()]
        lines = [f"**{title}**", f"- **当前判断：** {current}"]
        if application:
            lines.append(f"- **结合本案：** {application}")
        if branch:
            lines.append(f"- **判断会变化的条件：** {branch}")
        if verify:
            lines.append(f"- **仍需核实：** {'；'.join(verify[:3])}")
        next_actions = actions or evidence_actions
        if next_actions:
            lines.append(f"- **对应行动：** {'；'.join(next_actions[:3])}")
        blocks.append("\n".join(lines))
    return "**【核心争点分析】**\n\n" + "\n\n".join(blocks)


def _format_legal_element_matrix(issue_analyses: list[dict]) -> str:
    """Render the structured legal-element matrix into a deterministic block."""
    lines = ["**【法条要件核对】**"]
    added = False
    for analysis in issue_analyses:
        matrix = analysis.get("legal_element_matrix") or []
        if not matrix:
            continue
        title = str(analysis.get("title") or analysis.get("issue_id") or "争点")
        lines.append(f"**{title}**")
        for item in matrix[:6]:
            ref = str(item.get("legal_basis_ref") or "检索法条")
            element = str(item.get("element") or "适用要件")
            status = str(item.get("status") or "unknown")
            facts = [
                _INTERNAL_KEY_LABELS.get(str(x).strip(), str(x))
                for x in (item.get("supporting_facts") or [])
                if str(x)
            ]
            evidence = [
                _INTERNAL_KEY_LABELS.get(str(x).strip(), str(x))
                for x in (item.get("evidence_items") or [])
                if str(x)
            ]
            why = str(item.get("why") or "").strip()
            change = str(item.get("what_would_change") or "").strip()
            lines.append(f"- **{ref}** · {element}")
            lines.append(f"  当前事实：{'；'.join(facts[:3]) or '无对应事实'}")
            lines.append(f"  证据状态：{'；'.join(evidence[:3]) or '无对应证据'}")
            lines.append(f"  要件状态：{status}")
            if why:
                lines.append(f"  判断理由：{why}")
            if change:
                lines.append(f"  会改变结果：{change}")
            added = True
    if not added:
        return ""
    return "\n".join(lines)


def _ensure_legal_element_review(reply: str, block: str) -> str:
    if not block:
        return reply
    section_pattern = re.compile(
        r"\n*\*{0,2}【法条要件核对】\*{0,2}.*?(?=\n(?:#{1,6}\s*)?\*{0,2}【|\Z)",
        re.S,
    )
    reply = section_pattern.sub("", reply or "").strip()
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:维权路径比较|行动清单|优势与劣势)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + block + "\n\n" + reply[marker.start():].lstrip()
    return reply.rstrip() + "\n\n" + block


def _format_adversarial_review(review: dict) -> str:
    """Render the adversarial review into a user-facing pressure-test block."""
    if not review:
        return ""
    lines = ["**【反方压力测试】**"]
    added = False
    for item in (review.get("opponent_arguments") or [])[:6]:
        if isinstance(item, str) and str(item).strip():
            lines.append(f"- **对方/平台可能反驳：** {item}")
            added = True
            continue
        if not isinstance(item, dict):
            continue
        lines.append(f"- **对方/平台可能反驳：** {item.get('argument')}")
        lines.append(f"  - **用户应如何回应：** {item.get('response')}")
        if item.get("evidence_needed"):
            lines.append(f"  - **需要补什么材料：** {item.get('evidence_needed')}")
        added = True
    for item in (review.get("adverse_points") or [])[:6]:
        if isinstance(item, str) and str(item).strip():
            lines.append(f"- **对用户不利：** {item}")
            added = True
            continue
        if not isinstance(item, dict):
            continue
        lines.append(f"- **对用户不利：** {item.get('point')}")
        if item.get("impact"):
            lines.append(f"  - **影响：** {item.get('impact')}")
        if item.get("countermeasure"):
            lines.append(f"  - **应对：** {item.get('countermeasure')}")
        added = True
    for item in (review.get("evidence_weaknesses") or [])[:5]:
        if isinstance(item, str) and str(item).strip():
            lines.append(f"- **证据弱点：** {item}")
            added = True
            continue
        if not isinstance(item, dict):
            continue
        lines.append(f"- **证据弱点：** {item.get('item')}（{item.get('why')}）")
        if item.get("remedy"):
            lines.append(f"  - **补强：** {item.get('remedy')}")
        added = True
    for item in (review.get("unmet_legal_elements") or [])[:5]:
        if isinstance(item, str) and str(item).strip():
            lines.append(f"- **未满足要件：** {item}")
            added = True
            continue
        if not isinstance(item, dict):
            continue
        lines.append(f"- **未满足要件：** {item.get('element')}（{item.get('law')}）")
        if item.get("what_changes_it"):
            lines.append(f"  - **改变条件：** {item.get('what_changes_it')}")
        added = True
    if review.get("procedure_risks"):
        lines.append("- **程序风险：** " + "；".join(str(x) for x in review["procedure_risks"][:4]))
        added = True
    if review.get("premise_risks"):
        lines.append("- **方案依赖前提：** " + "；".join(str(x) for x in review["premise_risks"][:4]))
        added = True
    if review.get("must_disclose"):
        lines.append("- **必须披露：** " + "；".join(str(x) for x in review["must_disclose"][:4]))
        added = True
    if review.get("current_procedure_stage"):
        lines.append(f"- **当前程序阶段：** {review.get('current_procedure_stage')}")
        added = True
    if review.get("next_procedure_stage"):
        next_line = f"- **下一阶段：** {review.get('next_procedure_stage')}"
        if review.get("next_stage_trigger"):
            next_line += f"（触发：{review.get('next_stage_trigger')}）"
        lines.append(next_line)
        added = True
    for item in (review.get("conditional_paths") or [])[:5]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- **条件分支：** 若{item.get('condition')} → {item.get('path')}"
            f"；若不成立：{item.get('if_false')}"
        )
        added = True
    return "\n".join(lines) if added else ""


def _ensure_adversarial_review(reply: str, block: str) -> str:
    if not block:
        return reply
    section_pattern = re.compile(
        r"\n*\*{0,2}【反方压力测试】\*{0,2}.*?(?=\n(?:#{1,6}\s*)?\*{0,2}【|\Z)",
        re.S,
    )
    reply = section_pattern.sub("", reply or "").strip()
    marker = re.search(
        r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【(?:维权路径比较|行动清单|优势与劣势)】\*{0,2}\s*$",
        reply,
    )
    if marker:
        return reply[:marker.start()].rstrip() + "\n\n" + block + "\n\n" + reply[marker.start():].lstrip()
    return reply.rstrip() + "\n\n" + block


def _ensure_issue_analysis_section(reply: str, state: GuideState) -> str:
    """Insert structured legal application before the legal-basis section."""
    if "【核心争点分析】" in reply:
        return reply
    section = _format_issue_analysis_section(state)
    if not section:
        return reply
    law_marker = re.search(r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【法律依据】\*{0,2}\s*$", reply)
    if law_marker:
        return reply[:law_marker.start()].rstrip() + "\n\n" + section + "\n\n" + reply[law_marker.start():].lstrip()
    return _insert_before_document_offer(reply, section)


def _build_case_analysis_packet(state: GuideState) -> dict:
    """Freeze all currently usable case information before conclusion analysis.

    The old conclusion prompt mixed a short recent-dialogue slice with several
    legacy projections.  This packet is the single input for issue spotting,
    final grounding and rendering, so a fact already collected in an earlier
    round remains available to the analyst.
    """
    active = active_case_facts(state.case_facts)
    if not active and state.collected_facts:
        facts = [
            {"key": f"legacy.collected.{index}", "statement": str(value),
             "status": "asserted", "turn": 0}
            for index, value in enumerate(state.collected_facts)
            if str(value).strip()
        ]
    else:
        facts = active
    held_evidence, evidence_leads = _evidence_for_plan(state)
    return {
        "case_id": state.case_id,
        "domain": state.legal_domain,
        "region": state.region,
        "time_info": state.time_info,
        "user_goal": list(state.confirmed_issues or []) + list(state.unmatched_issues or []),
        "facts": facts[:120],
        "fact_context": format_case_context(facts, limit=120),
        "confirmed_issues": list(state.confirmed_issues or []),
        "unmatched_issues": list(state.unmatched_issues or []),
        "evidence_confirmed": held_evidence,
        "evidence_unverified": evidence_leads,
        "evidence_unavailable": list(state.evidence_unavailable or []),
        "evidence_items": list(state.evidence_items or []),
        "procedure_status": state.followup_plan.get("procedure_status", "")
        if isinstance(state.followup_plan, dict) else "",
        "current_safety": state.current_safety_status,
        "actions_taken": list(state.collected_facts or [])[:20],
        "adverse_facts": list(state.adverse_facts or []),
        "unknown_facts": [
            item.get("statement", "") for item in facts
            if item.get("status") in {"uncertain", "conflicted"}
        ],
    }


def _analysis_json(value: object, *, key: str) -> list[dict]:
    """Parse a model JSON response without letting one bad call break the turn."""
    try:
        parsed = value if isinstance(value, (dict, list)) else _json_content(str(value or ""))
    except Exception:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get(key, [])
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _analysis_object(value: object, *, key: str) -> dict:
    """Parse one JSON object returned by a synthesis stage."""
    try:
        parsed = value if isinstance(value, dict) else _json_content(str(value or ""))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    candidate = parsed.get(key, parsed)
    return dict(candidate) if isinstance(candidate, dict) else {}


def _dedupe_issue_analyses(items: list[dict]) -> list[dict]:
    """Merge repeated model analyses for the same dispute point."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    list_fields = (
        "supporting_facts", "adverse_facts", "legal_basis_refs",
        "facts_to_verify", "evidence_actions", "recommended_actions", "procedure_steps",
        "legal_element_matrix", "opponent_counterarguments",
    )
    for item in items:
        if not isinstance(item, dict):
            continue
        title_key = re.sub(r"\s+", "", str(item.get("title") or "").strip())
        key = title_key or str(item.get("issue_id") or f"issue_{len(order) + 1}")
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        current = merged[key]
        for field in list_fields:
            values = list(current.get(field) or [])
            for value in item.get(field) or []:
                if value not in values:
                    values.append(value)
            current[field] = values[:12]
        for field in ("title", "current_view", "application_analysis", "conditional_branch"):
            candidate = str(item.get(field) or "").strip()
            if len(candidate) > len(str(current.get(field) or "")):
                current[field] = candidate
    return [merged[key] for key in order][:10]


def _fallback_strategy_plan(
    state: GuideState,
    issue_map: list[dict],
    issue_analyses: list[dict],
    legal_basis: dict,
) -> dict:
    """Create a source-bounded strategy when the synthesis call is unavailable."""
    actions: list[str] = []
    procedure: list[dict] = []
    evidence_plan: list[dict] = []
    source_issue_ids = [str(item.get("issue_id")) for item in issue_map if item.get("issue_id")]
    source_law_refs: list[str] = []
    for authority in legal_basis.get("primary_authorities") or []:
        if isinstance(authority, dict):
            label = f"{authority.get('title') or '检索法条'}第{authority.get('article_no') or '相关条'}"
            if label not in source_law_refs:
                source_law_refs.append(label)
    for analysis in issue_analyses:
        if not isinstance(analysis, dict):
            continue
        actions.extend(str(item) for item in (analysis.get("recommended_actions") or []))
        actions.extend(str(item) for item in (analysis.get("evidence_actions") or []))
        for step in analysis.get("procedure_steps") or []:
            procedure.append({
                "order": len(procedure) + 1,
                "step": str(step),
                "trigger": "前一步完成或相关事实得到核实",
                "expected_change": "推进程序并补强对应争点",
            })
    for item in active_case_facts(state.case_facts):
        if item.get("category") != "evidence":
            continue
        status = str(item.get("evidence_status") or "lead")
        material_text = " ".join(
            str(item.get(field) or "") for field in ("statement", "value", "source_text")
        )
        if status in {"lead", "unknown", ""} and any(
            marker in material_text
            for marker in ("保留了", "保存了", "已经保存", "已有", "我有", "持有", "掌握", "留存", "提供了")
        ):
            status = "obtained"
        evidence_plan.append({
            "item": str(item.get("statement") or item.get("value") or "证据材料"),
            "status": status if status in {"obtained", "lead", "unavailable", "unknown"} else "lead",
            "proof_target": "还原事实、主体、损害结果或程序状态",
            "action": "保留原始载体并核验来源、完整性和关联性",
            "priority": "high" if status in {"obtained", "lead"} else "medium",
            "why": "该材料可能影响争点判断和后续程序",
        })
    primary_issue = next(
        (item for item in issue_map if isinstance(item, dict) and item.get("title")),
        {},
    )
    primary_title = str(primary_issue.get("title") or "当前争点")
    issue_reason = str(primary_issue.get("reason") or "").strip()
    headline = (
        f"当前应围绕“{primary_title}”推进事实核实、证据固定和程序动作，"
        "现有材料尚不足以直接确定责任或结果。"
    )
    if issue_reason:
        headline += f"识别该争点的依据是：{issue_reason}。"
    unknowns = []
    for item in (state.case_analysis_packet or {}).get("facts") or []:
        if item.get("status") in {"uncertain", "conflicted"} and item.get("statement"):
            unknowns.append(str(item.get("statement")))
    return {
        "headline_assessment": {
            "position": headline,
            "supporting_reason": "该判断来自当前案件快照、已识别争点、现有证据状态和最终检索法条。",
            "uncertainty": "；".join(unknowns[:6]) or "仍需以办案机关核实结果和原始材料为准。",
        },
        "priority_actions": [
            {"action": item, "object": "相关机构或证据持有人", "purpose": "推进当前争点", "why_now": "避免证据或程序节点继续流失", "risk": "未及时处理可能降低后续证明能力"}
            for item in action_texts
        ],
        "procedure_path": procedure[:6],
        "evidence_plan": evidence_plan[:10],
        "opponent_arguments": [],
        "institution_focus": [],
        "risk_boundaries": unknowns[:6],
        "conditions_that_change_result": [
            str(item.get("facts_that_change_result"))
            for item in issue_map
            if item.get("facts_that_change_result")
        ][:8],
        "source_issue_ids": source_issue_ids[:10],
        "source_law_refs": source_law_refs[:10],
    }


def _validate_strategy_plan(
    plan: dict,
    issue_map: list[dict],
    legal_basis: dict,
) -> dict:
    """Validate structure and references without rewriting the model's reasoning."""
    issue_ids = {str(item.get("issue_id")) for item in issue_map if item.get("issue_id")}
    law_labels = {
        f"{item.get('title') or ''}第{item.get('article_no') or ''}"
        for item in legal_basis.get("primary_authorities") or []
        if isinstance(item, dict)
    }
    refs = [str(item) for item in plan.get("source_issue_ids") or []]
    laws = [str(item) for item in plan.get("source_law_refs") or []]
    unknown_issues = [item for item in refs if issue_ids and item not in issue_ids]
    unsupported_laws = [item for item in laws if law_labels and not any(item in label or label in item for label in law_labels)]
    required_lists = ("priority_actions", "procedure_path", "evidence_plan", "conditions_that_change_result")
    malformed = [key for key in required_lists if not isinstance(plan.get(key, []), list)]
    return {
        "status": "ok" if not unknown_issues and not unsupported_laws and not malformed else "needs_review",
        "unknown_issue_refs": unknown_issues[:10],
        "unsupported_law_refs": unsupported_laws[:10],
        "malformed_fields": malformed,
    }


async def _analyze_strategy_plan(
    state: GuideState,
    deps: GuideDeps,
    packet: dict,
    issue_map: list[dict],
    issue_analyses: list[dict],
    legal_basis: dict,
) -> tuple[dict, dict, dict]:
    emit_guide_progress(
        "strategy_risk",
        "正在合成策略与对抗推演",
        "生成行动策略，并从对方、平台和办案机关角度检查不利点。",
    )
    held, leads = _evidence_for_plan(state)
    evidence_summary = {
        "obtained": held,
        "leads": leads,
        "unavailable": list(state.evidence_unavailable or []),
    }
    prompt = STRATEGY_SYNTHESIS_PROMPT.format(
        case_snapshot=json.dumps(packet, ensure_ascii=False, indent=2)[:12000],
        issue_map=json.dumps(issue_map, ensure_ascii=False, indent=2)[:7000],
        issue_analyses=json.dumps(issue_analyses, ensure_ascii=False, indent=2)[:9000],
        legal_basis_packet=json.dumps(legal_basis, ensure_ascii=False, indent=2)[:9000],
        evidence_summary=json.dumps(evidence_summary, ensure_ascii=False, indent=2)[:4000],
    )
    response = await llm_for_stage(deps.fast_llm or deps.llm, max_tokens=2200).ainvoke(
        [SystemMessage(content=prompt)]
    )
    parsed = _json_content(str(response.content or ""))
    if not isinstance(parsed, dict):
        raise ValueError("策略中枢未返回可用 JSON")
    plan = parsed.get("strategy_plan")
    adversarial_review = parsed.get("adversarial_execution_review")
    if not plan or not adversarial_review:
        raise ValueError("策略中枢未返回策略或对抗推演结果")
    validation = _validate_strategy_plan(plan, issue_map, legal_basis)
    return plan, adversarial_review, validation


async def _critique_plan(
    state: GuideState,
    deps: GuideDeps,
    packet: dict,
    issue_map: list[dict],
    issue_analyses: list[dict],
    legal_basis: dict,
    adversarial_review: dict,
    draft: str,
) -> dict:
    """Critique a plan draft before it can become the final user-facing answer."""
    emit_guide_progress(
        "plan_critique",
        "正在检查方案质量",
        "检查初稿是否回避不利点、要件是否完整、行动是否可执行。",
    )
    prompt = PLAN_CRITIQUE_PROMPT.format(
        case_snapshot=json.dumps(packet, ensure_ascii=False, indent=2)[:12000],
        issue_map=json.dumps(issue_map, ensure_ascii=False, indent=2)[:7000],
        issue_analyses=json.dumps(issue_analyses, ensure_ascii=False, indent=2)[:10000],
        legal_basis=json.dumps(legal_basis, ensure_ascii=False, indent=2)[:9000],
        adversarial_execution_review=json.dumps(
            adversarial_review, ensure_ascii=False, indent=2
        )[:10000],
        draft=draft[:12000],
    )
    response = await llm_for_stage(_fast_llm_for(deps), max_tokens=1200).ainvoke(
        [SystemMessage(content=prompt)]
    )
    parsed = _json_content(str(response.content or ""))
    if not isinstance(parsed, dict) or not parsed.get("verdict"):
        raise ValueError("方案批判未返回可用判定")
    return parsed


async def _revise_plan(
    state: GuideState,
    deps: GuideDeps,
    packet: dict,
    issue_map: list[dict],
    issue_analyses: list[dict],
    legal_basis: dict,
    adversarial_review: dict,
    draft: str,
    critique: dict,
) -> str:
    """Revise a draft according to the structured critique."""
    emit_guide_progress(
        "plan_revision",
        "正在修订方案",
        "根据批判结果修正泛化表述、不利点和可执行性问题。",
    )
    prompt = PLAN_REVISION_PROMPT.format(
        case_snapshot=json.dumps(packet, ensure_ascii=False, indent=2)[:12000],
        issue_map=json.dumps(issue_map, ensure_ascii=False, indent=2)[:7000],
        issue_analyses=json.dumps(issue_analyses, ensure_ascii=False, indent=2)[:10000],
        legal_basis=json.dumps(legal_basis, ensure_ascii=False, indent=2)[:9000],
        adversarial_execution_review=json.dumps(
            adversarial_review, ensure_ascii=False, indent=2
        )[:10000],
        draft=draft[:12000],
        critique=json.dumps(critique, ensure_ascii=False, indent=2)[:8000],
    )
    response = await llm_for_stage(deps.llm, max_tokens=3600).ainvoke(
        [SystemMessage(content=prompt)]
    )
    return str(response.content or "").strip()


def _fallback_issue_map(state: GuideState, packet: dict) -> list[dict]:
    """Small non-scenario fallback used only when issue-spotting is unavailable."""
    issues = list(packet.get("confirmed_issues") or [])
    issues.extend(item for item in packet.get("unmatched_issues") or [] if item not in issues)
    if not issues:
        issues = [DOMAIN_LABELS.get(state.legal_domain, "当前法律争议")]
    fact_keys = [str(item.get("key")) for item in packet.get("facts", []) if item.get("key")]
    return [
        {
            "issue_id": f"issue_{index}",
            "title": str(title),
            "importance": "core" if index == 1 else "conditional",
            "reason": "基于当前已确认的法律问题进入基础分析",
            "supporting_fact_keys": fact_keys[:12],
            "retrieval_questions": [str(title)],
            "facts_that_change_result": [],
        }
        for index, title in enumerate(issues[:8], start=1)
    ]


def _record_fact_tensions(facts: list[dict]) -> list[dict]:
    """Expose unresolved same-key corrections without guessing who is right."""
    grouped: dict[str, list[dict]] = {}
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("status") != "conflicted":
            continue
        key = str(fact.get("key") or "")
        if key:
            grouped.setdefault(key, []).append(fact)
    tensions: list[dict] = []
    for key, records in grouped.items():
        statements = [str(item.get("statement") or "").strip() for item in records]
        statements = [item for item in statements if item]
        if len(statements) < 2:
            continue
        tensions.append({
            "title": "同一事实存在相互矛盾的陈述",
            "side_a_fact_keys": [key],
            "side_b_fact_keys": [key],
            "why_it_matters": "在该事实核实前，不能将相关法律判断写成确定结论。",
            "resolution_action": "保留两种陈述，并优先用原始记录、第三方材料或时间线核实。",
            "statements": statements[:4],
        })
    return tensions[:6]


def _valid_fact_tensions(raw_tensions: object, facts: list[dict]) -> list[dict]:
    fact_keys = {str(item.get("key") or "") for item in facts if isinstance(item, dict)}
    result: list[dict] = []
    for raw in raw_tensions or []:
        if not isinstance(raw, dict):
            continue
        side_a = [str(item) for item in raw.get("side_a_fact_keys") or [] if str(item) in fact_keys]
        side_b = [str(item) for item in raw.get("side_b_fact_keys") or [] if str(item) in fact_keys]
        if not side_a or not side_b:
            continue
        item = {
            "title": str(raw.get("title") or "关键事实存在待核实的矛盾")[:100],
            "side_a_fact_keys": side_a[:5],
            "side_b_fact_keys": side_b[:5],
            "why_it_matters": str(raw.get("why_it_matters") or "该矛盾会影响当前法律判断。")[:300],
            "resolution_action": str(raw.get("resolution_action") or "优先用原始材料或第三方记录核实。")[:300],
        }
        if item not in result:
            result.append(item)
    return result[:6]


async def _identify_issue_map(state: GuideState, deps: GuideDeps, packet: dict) -> list[dict]:
    """Let the model identify issues from the complete packet, not a keyword rule."""
    emit_guide_progress(
        "issue_identification",
        "正在识别法律争点",
        "基于完整案情判断核心争点、条件争点和程序争点。",
    )
    prompt = ISSUE_MAP_PROMPT.format(
        case_snapshot=json.dumps(packet, ensure_ascii=False, indent=2)[:18000],
    )
    response = await llm_for_stage(deps.fast_llm or deps.llm, max_tokens=1500).ainvoke(
        [SystemMessage(content=prompt)]
    )
    parsed = _json_content(str(response.content or ""))
    if isinstance(parsed, dict):
        model_tensions = _valid_fact_tensions(parsed.get("fact_tensions"), packet.get("facts") or [])
        packet["fact_tensions"] = model_tensions or _record_fact_tensions(packet.get("facts") or [])
    issues = _analysis_json(parsed, key="issues")
    if not issues:
        raise ValueError("争点识别未返回可用争点")
    return issues[:10]


def _law_ref_key(item: dict) -> tuple[str, str]:
    """Stable identity for merging statute references across retrieval stages."""
    law_id = str(item.get("law_id") or "").strip()
    article_no = str(item.get("article_no") or "").strip()
    if law_id and article_no:
        return law_id, article_no
    return str(item.get("title") or "").strip(), article_no


def _merge_law_refs(pools: list[list[dict]], *, limit: int = 16) -> list[dict]:
    """Merge final, follow-up and issue-specific statute refs without losing issue links."""
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for pool in pools:
        for raw in pool:
            if not isinstance(raw, dict):
                continue
            key = _law_ref_key(raw)
            if not any(key):
                continue
            if key not in merged:
                merged[key] = {
                    "law_id": str(raw.get("law_id") or ""),
                    "title": str(raw.get("title") or ""),
                    "article_no": str(raw.get("article_no") or ""),
                    "text": str(raw.get("text") or "")[:1200],
                    "source_type": str(raw.get("source_type") or ""),
                    "source_issue_ids": [
                        str(item) for item in (raw.get("source_issue_ids") or []) if str(item)
                    ],
                }
                order.append(key)
                continue
            current = merged[key]
            if str(raw.get("title") or ""):
                current["title"] = str(raw["title"])
            if str(raw.get("text") or "") and not current["text"]:
                current["text"] = str(raw["text"])[:1200]
            if str(raw.get("source_type") or "") and not current["source_type"]:
                current["source_type"] = str(raw["source_type"])
            for issue_id in raw.get("source_issue_ids") or []:
                issue_id = str(issue_id).strip()
                if issue_id and issue_id not in current["source_issue_ids"]:
                    current["source_issue_ids"].append(issue_id)
    return [merged[key] for key in order][:limit]


async def _supplement_strategy_law_retrieval(
    state: GuideState,
    deps: GuideDeps,
    issue_map: list[dict],
) -> tuple[list[dict], str]:
    """Re-query statutes using AI-discovered issues before final synthesis."""
    issue_queries: list[dict] = []
    for issue in issue_map:
        if not isinstance(issue, dict):
            continue
        issue_id = str(issue.get("issue_id") or "").strip()
        title = str(issue.get("title") or "").strip()
        reason = str(issue.get("reason") or "").strip()
        retrieval_questions = [str(item).strip() for item in issue.get("retrieval_questions") or [] if str(item).strip()]
        query_parts = [title, *retrieval_questions]
        if reason:
            query_parts.append(reason)
        query_parts = [part for part in query_parts if part]
        if issue_id and query_parts:
            issue_queries.append({
                "issue_id": issue_id,
                "question": "；".join(query_parts),
            })
    existing = [dict(item) for item in state.retrieved_law_refs or [] if isinstance(item, dict)]
    followup = [dict(item) for item in state.followup_basis_refs or [] if isinstance(item, dict)]
    if not issue_queries:
        return _merge_law_refs([existing, followup]), state.law_context_str or ""
    try:
        from src.agents.legal_knowledge.statute_rag import (
            _fetch_law_titles,
            format_statute_context,
            search_statutes_raw,
        )
        domain = state.legal_domain if state.legal_domain != "other" else ""

        async def _retrieve_for_issue(query: dict) -> list[dict]:
            hits = await asyncio.wait_for(
                search_statutes_raw(
                    question=query["question"],
                    embedding_model=deps.embedding_model,
                    milvus_client=deps.milvus_client,
                    llm=deps.llm,
                    use_hyde=False,
                    use_rrf=False,
                    sparse_query="",
                    domain=domain,
                    top_k=10,
                    rerank_top_k=3,
                ),
                timeout=settings.GUIDE_RETRIEVE_TIMEOUT_STATUTE,
            )
            for hit in hits:
                hit["source_issue_ids"] = [query["issue_id"]]
            return hits

        results = await asyncio.gather(
            *[_retrieve_for_issue(query) for query in issue_queries],
            return_exceptions=True,
        )
        issue_hits: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            issue_hits.extend(result)

        merged = _merge_law_refs(
            [existing, followup, issue_hits],
            limit=16,
        )
        if not merged:
            return existing, state.law_context_str or ""
        titles: dict[str, str] = {}
        if deps.db_session:
            try:
                titles = await asyncio.wait_for(
                    _fetch_law_titles(merged, deps.db_session),
                    timeout=settings.GUIDE_RETRIEVE_TIMEOUT_AUX,
                )
            except Exception as exc:
                logger.warning("策略补充检索获取法条标题失败: {}", exc)
        for item in merged:
            title = titles.get(str(item.get("law_id") or ""))
            if title:
                item["title"] = title
        formatted = format_statute_context(merged, titles, primary_count=min(5, len(merged)))
        return merged[:16], formatted or state.law_context_str or ""
    except Exception as exc:
        logger.warning("策略中枢补充法条检索失败，沿用已有检索结果: {}", exc)
        return _merge_law_refs([existing, followup]), state.law_context_str or ""


def _build_final_legal_basis(state: GuideState, issue_map: list[dict]) -> dict:
    """Build an issue-linked legal-basis packet from the merged retrieval pool.

    ``_supplement_strategy_law_retrieval`` attaches ``source_issue_ids`` to each
    issue-specific hit.  This function uses that mapping instead of taking the
    first eight refs globally, so every dispute point can keep its own statute
    coverage while the final prompt still receives a bounded authority list.
    """
    refs = [dict(item) for item in (state.retrieved_law_refs or []) if isinstance(item, dict)]
    issue_queries: list[str] = []
    issue_authorities: list[dict] = []
    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(ref: dict) -> None:
        key = _law_ref_key(ref)
        if not any(key) or key in seen:
            return
        seen.add(key)
        selected.append(ref)

    for issue in issue_map:
        if not isinstance(issue, dict):
            continue
        issue_id = str(issue.get("issue_id") or "").strip()
        issue_queries.extend(
            str(item) for item in issue.get("retrieval_questions") or [] if str(item).strip()
        )
        mapped = [
            ref for ref in refs
            if issue_id in {str(item) for item in (ref.get("source_issue_ids") or [])}
        ]
        chosen = mapped[:2]
        if not chosen:
            chosen = [ref for ref in refs if _law_ref_key(ref) not in seen][:2]
        issue_authorities.append({
            "issue_id": issue_id,
            "title": str(issue.get("title") or ""),
            "authorities": chosen,
        })
        for ref in chosen:
            _add(ref)

    for ref in refs:
        if len(selected) >= 10:
            break
        _add(ref)

    primary = selected[:10]
    procedural = [ref for ref in refs if _law_ref_key(ref) not in seen][:4]
    return {
        "retrieval_mode": "latest_snapshot_issue_linking",
        "issue_queries": issue_queries[:24],
        "primary_authorities": primary,
        "conditional_authorities": [],
        "procedural_authorities": procedural,
        "issue_authorities": issue_authorities,
        "case_context": state.case_context_str or "",
        "channels": list(state.relevant_channels or []),
        "retrieval_error_note": state.retrieval_error_note or "",
    }


def _fallback_issue_analyses(issue_map: list[dict], legal_basis: dict) -> list[dict]:
    refs = legal_basis.get("primary_authorities") or []
    basis_labels = [
        f"{item.get('title') or '检索到的法律'}第{item.get('article_no') or '相关条'}"
        for item in refs[:4]
    ]
    return [
        {
            "issue_id": item.get("issue_id", f"issue_{index}"),
            "title": item.get("title", "法律争点"),
            "current_view": "需要结合完整事实、证据和办案机关认定，当前仅作阶段性分析。",
            "supporting_facts": list(item.get("supporting_fact_keys") or []),
            "adverse_facts": [],
            "legal_basis_refs": basis_labels[:3],
            "application_analysis": "现有信息可以支持继续采取低风险的证据保全和程序咨询行动，但不能直接认定责任或结果。",
            "conditional_branch": "如关键事实、证据或程序状态不同，法律评价可能随之变化。",
            "facts_to_verify": list(item.get("facts_that_change_result") or []),
            "legal_element_matrix": [],
            "opponent_counterarguments": [],
            "evidence_actions": [],
            "recommended_actions": [],
            "procedure_steps": [],
        }
        for index, item in enumerate(issue_map, start=1)
    ]


async def _analyze_issue_applications(
    state: GuideState,
    deps: GuideDeps,
    packet: dict,
    issue_map: list[dict],
    legal_basis: dict,
) -> list[dict]:
    emit_guide_progress(
        "issue_application",
        "正在分析法律要件",
        "把每个争点与法条要件、事实和证据逐项对应。",
    )
    prompt = ISSUE_APPLICATION_PROMPT.format(
        case_snapshot=json.dumps(packet, ensure_ascii=False, indent=2)[:12000],
        issue_map=json.dumps(issue_map, ensure_ascii=False, indent=2)[:7000],
        legal_basis=json.dumps(legal_basis, ensure_ascii=False, indent=2)[:9000],
    )
    response = await llm_for_stage(deps.fast_llm or deps.llm, max_tokens=1800).ainvoke(
        [SystemMessage(content=prompt)]
    )
    analyses = _dedupe_issue_analyses(_analysis_json(response.content, key="analyses"))
    if not analyses:
        raise ValueError("争点法律适用分析未返回可用分析")
    return analyses[:10]


def _validate_analysis_grounding(
    packet: dict,
    issue_map: list[dict],
    issue_analyses: list[dict],
    legal_basis: dict,
) -> dict:
    fact_keys = {str(item.get("key")) for item in packet.get("facts", []) if item.get("key")}
    law_refs = {
        f"{item.get('title') or ''}第{item.get('article_no') or ''}"
        for item in legal_basis.get("primary_authorities", [])
    }
    unknown_fact_keys: list[str] = []
    for item in issue_analyses:
        for key in item.get("supporting_facts") or []:
            if str(key) in fact_keys:
                continue
            if isinstance(key, str) and key.startswith(("legacy.", "fact_")):
                continue
            if key and str(key) not in unknown_fact_keys:
                unknown_fact_keys.append(str(key))
    referenced_laws = [
        str(ref)
        for item in issue_analyses
        for ref in (item.get("legal_basis_refs") or [])
    ]
    for item in issue_analyses:
        for matrix_item in item.get("legal_element_matrix") or []:
            if isinstance(matrix_item, dict) and str(matrix_item.get("legal_basis_ref") or ""):
                referenced_laws.append(str(matrix_item["legal_basis_ref"]))
    unsupported_laws = [ref for ref in referenced_laws if law_refs and not any(
        ref in label or label in ref for label in law_refs
    )]
    report = {
        "status": "ok" if not unknown_fact_keys and not unsupported_laws else "needs_review",
        "fact_count": len(packet.get("facts") or []),
        "issue_count": len(issue_map),
        "analysis_count": len(issue_analyses),
        "unknown_fact_keys": unknown_fact_keys[:20],
        "unsupported_law_refs": unsupported_laws[:20],
    }
    return report


def _deterministic_conclusion_draft(state: GuideState) -> str:
    """Return a source-bounded, issue-aware plan when an LLM call times out."""

    case_summary = _format_case_summary(state) or "当前案情仍有部分信息待确认"
    law_section = (
        state.law_context_str[:2400]
        if state.law_context_str
        else "当前未检索到可直接引用的具体条文，建议拨打12348进一步核验。"
    )
    channel_lines = _channel_summary_lines(state, limit=3)
    if not channel_lines:
        channel_lines = [
            "- 暂无可核验的本地渠道记录，可先拨打12348核对受理机构、管辖和材料要求。"
        ]
    _pros = ["当前检索到的法律依据中，对您主张有利的条文可作为维权依据。"]
    _cons = [f"当前仍有信息待核验：{case_summary}，结果受事实、证据和对方抗辩影响。"]
    for item in state.adverse_facts:
        if str(item).strip():
            _cons.append(f"- {str(item).strip()}")
    for item in state.evidence_unavailable:
        if str(item).strip():
            _cons.append(f"- 缺少「{item}」，对方可能质疑举证能力。")
    analysis_lines: list[str] = []
    for item in (state.issue_analyses or [])[:6]:
        title = str(item.get("title") or "核心争点").strip()
        current = str(item.get("current_view") or "需结合证据进一步判断").strip()
        application = str(item.get("application_analysis") or "").strip()
        branch = str(item.get("conditional_branch") or "").strip()
        block = f"**{title}**：{current}"
        if application:
            block += f"理由：{application}"
        if branch:
            block += f" 条件变化：{branch}"
        analysis_lines.append(f"- {block[:900]}")
    issue_section = (
        "\n\n**【核心争点分析】**\n" + "\n".join(analysis_lines)
        if analysis_lines else
        "\n\n**【核心争点分析】**\n- 当前检索和事实仅足以形成条件式判断，需结合证据推进核实。"
    )
    return (
        "**【理解您的情况】**\n"
        f"根据您目前的陈述：{case_summary}。\n\n"
        + issue_section
        + "**【法律依据】**\n"
        f"{law_section}\n\n"
        "**【维权路径比较】**\n"
        + "\n".join(channel_lines)
        + "\n\n**【优势与劣势】**\n"
        "**有利因素**：\n"
        + "\n".join(f"- {item}" for item in _pros)
        + "\n**不利因素**：\n"
        + "\n".join(f"- {item}" for item in _cons)
        + "\n\n**【行动清单】**\n"
        "1. 立即备份原始材料和原始载体，按时间顺序整理事实。\n"
        "2. 核对证据作用与缺口，优先补强会影响责任、金额或程序的材料。\n"
        "3. 联系已列明渠道核对受理范围、管辖和材料要求，并保存回执。\n"
        "4. 对关键期限或重大决定仍不确定时，拨打12348咨询专业律师。"
    )


async def node_conclude(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑧：冻结完整事实并完成争点化、依据化和行动化分析。"""
    emit_guide_progress(
        "solution_generation",
        "正在生成或更新维权方案",
        "把最新事实、证据评估、法条、类案和办理渠道合并为可执行建议。",
    )
    logger.info("节点⑧生成结论 | domain={} tier={}", state.legal_domain, state.confidence_tier)

    # ── 结论前分析子流程 ───────────────────────────────────────────────────
    # Existing retrieval already runs after the latest fact turn.  We preserve
    # that knowledge-base snapshot, but make the final model calls consume a
    # complete case packet and an issue-linked legal-basis packet.
    case_analysis_packet = _build_case_analysis_packet(state)
    issue_map = await _identify_issue_map(state, deps, case_analysis_packet)
    supplemented_refs, supplemented_law_context = await _supplement_strategy_law_retrieval(
        state, deps, issue_map
    )
    if supplemented_refs:
        state = state.model_copy(update={
            "retrieved_law_refs": supplemented_refs,
            "law_context_str": supplemented_law_context or state.law_context_str,
        })
    legal_basis_packet = _build_final_legal_basis(state, issue_map)
    issue_analyses = await _analyze_issue_applications(
        state, deps, case_analysis_packet, issue_map, legal_basis_packet
    )
    analysis_validation = _validate_analysis_grounding(
        case_analysis_packet, issue_map, issue_analyses, legal_basis_packet
    )
    strategy_result, situation_result = await asyncio.gather(
        _analyze_strategy_plan(
            state, deps, case_analysis_packet, issue_map, issue_analyses, legal_basis_packet
        ),
        assess_user_situation(state, deps.fast_llm or deps.llm),
    )
    strategy_plan, adversarial_review, strategy_validation = strategy_result
    situation = situation_result
    strategy_plan = {
        **strategy_plan,
        "adversarial_execution_review": adversarial_review,
    }
    state = state.model_copy(update={
        "case_analysis_packet": case_analysis_packet,
        "issue_map": issue_map,
        "legal_basis_packet": legal_basis_packet,
        "issue_analyses": issue_analyses,
        "analysis_validation": analysis_validation,
        "analysis_stage": "validated" if analysis_validation.get("status") == "ok" else "needs_review",
        "strategy_plan": strategy_plan,
        "strategy_plan_version": state.strategy_plan_version + 1,
        "strategy_stage": "validated" if strategy_validation.get("status") == "ok" else "needs_review",
        "strategy_validation": strategy_validation,
    })
    logger.info(
        "结论分析子流程完成 | facts={} issues={} analyses={} validation={}",
        len(case_analysis_packet.get("facts") or []), len(issue_map),
        len(issue_analyses), analysis_validation.get("status"),
    )
    domain = state.legal_domain
    region = state.region or "全国"
    evidence_rule = resolve_state_evidence_checklist(state)
    evidence_checklist = fmt_evidence_checklist(evidence_rule)
    evidence_source = format_evidence_source(evidence_rule)
    channels_str = fmt_channels(state.relevant_channels)
    force_note = (
        "\n> **强制收敛要求**：本轮必须按现有信息给出完整可执行方案，禁止继续追问、"
        "禁止要求用户补充后再回复。可以陈述信息缺口及其风险，但必须使用陈述句。"
        "由于信息有限，建议拨打 **12348** 咨询专业律师。"
        if state.force_conclude else ""
    )
    self_review_str = state.self_review_note if state.self_review_note else ""
    plan_held_evidence, plan_evidence_leads = _evidence_for_plan(state)
    # 追问期间用户反问过、当时答应"等下一起说清楚"的问题，必须在结论里兑现
    deferred_str = (
        "\n## 用户在梳理过程中问过、还没答复的问题（必须在方案中一并回答）\n"
        + "\n".join(f"- {q}" for q in state.deferred_questions)
        if state.deferred_questions else ""
    )

    # 用户不利事实（被对方援引的风险因素）+ 明确缺失的证据
    _adverse_items = list(state.adverse_facts)
    for ev in state.evidence_unavailable:
        _adverse_items.append(f"缺少「{ev}」，对方可能质疑举证能力")
    adverse_facts_section = (
        "\n".join(f"- {f}" for f in _adverse_items)
        if _adverse_items else "（暂未识别到明显不利因素）"
    )
    # 近期对话片段（最近6条，帮助LLM理解完整Q&A上下文）
    recent_msgs = state.messages[-6:]
    snippet_questions = [
        item
        for item in (state.followup_plan or {}).get("questions") or []
        if isinstance(item, dict)
    ]
    dialogue_lines: list[str] = []
    for m in recent_msgs:
        is_human = getattr(m, "type", "") == "human"
        content = (
            _clean_dialogue_message(str(m.content or ""), snippet_questions)
            if is_human
            else str(m.content or "")
        )
        content = content[:300] if content else "（无文本）"
        dialogue_lines.append(f"{'用户' if is_human else '助手'}：{content}")
    dialogue_snippet = "\n".join(dialogue_lines) or "（无近期对话记录）"
    long_dialogue_memory = _format_long_dialogue_memory(state)
    legal_element_review = _format_legal_element_matrix(issue_analyses)
    adversarial_review_block = _format_adversarial_review(adversarial_review)

    # 用户处境审视已与策略/对抗推演并行完成。

    prompt = CONCLUDE_PROMPT.format(
        deferred_questions=deferred_str,
        confidence_guidance=tier_guidance(state.confidence_tier),
        situation_guidance=situation_guidance(situation),
        audience_guidance=_audience_guidance(state),
        confirmed_issues="、".join(state.confirmed_issues) or "法律问题",
        legal_domain=DOMAIN_LABELS.get(domain, domain or "法律"),
        region=region,
        time_info=state.time_info or "暂未确认",
        collected_facts="；".join(state.collected_facts) or "暂未确认",
        long_term_memories="；".join(_active_long_term_memories(state)) or "（无相关长期记忆）",
        evidence_confirmed="、".join(plan_held_evidence) or "暂未确认",
        evidence_unverified="、".join(plan_evidence_leads) or "（无）",
        evidence_unavailable="、".join(state.evidence_unavailable) or "（无）",
        fact_assessments=_fact_assessments_for_prompt(state),
        evidence_assessments=format_evidence_assessments(state.evidence_assessments),
        evidence_coverage=format_evidence_coverage(
            state.evidence_coverage or evaluate_state_evidence(state)
        ),
        time_warning=state.time_warning,
        self_review_note=self_review_str,
        adverse_facts_section=adverse_facts_section,
        dialogue_snippet=dialogue_snippet,
        long_dialogue_memory=long_dialogue_memory,
        law_context=state.law_context_str or "（未检索到具体条文，请参考适用法律原则）",
        case_context=state.case_context_str or "（暂无类案数据）",
        channels=channels_str,
        evidence_checklist=evidence_checklist,
        evidence_source=evidence_source,
        followup_authority=format_domain_authority_summary(domain),
        case_analysis_packet=json.dumps(
            state.case_analysis_packet or {}, ensure_ascii=False, indent=2
        )[:14000],
        issue_map=json.dumps(
            state.issue_map or [], ensure_ascii=False, indent=2
        )[:8000],
        issue_analyses=json.dumps(
            state.issue_analyses or [], ensure_ascii=False, indent=2
        )[:10000],
        legal_basis_packet=json.dumps(
            state.legal_basis_packet or {}, ensure_ascii=False, indent=2
        )[:10000],
        strategy_plan=json.dumps(
            state.strategy_plan or {}, ensure_ascii=False, indent=2
        )[:10000],
        adversarial_execution_review=json.dumps(
            adversarial_review or {}, ensure_ascii=False, indent=2
        )[:8000],
        legal_element_review=legal_element_review or "（当前未生成可渲染的要件矩阵）",
        adversarial_review_block=adversarial_review_block or "（当前未生成可渲染的反方压力测试）",
        force_conclude_note=force_note,
    )
    response = await llm_for_stage(deps.llm, max_tokens=3600).ainvoke(
        [SystemMessage(content=prompt)]
    )
    draft = str(response.content or "").strip()
    final_reply = draft
    deterministic_problems = _deterministic_draft_problems(draft)
    if deterministic_problems or analysis_validation.get("status") != "ok":
        critique = (
            {"verdict": "revise", "issues": deterministic_problems}
            if deterministic_problems
            else await _critique_plan(
                state,
                deps,
                case_analysis_packet,
                issue_map,
                issue_analyses,
                legal_basis_packet,
                adversarial_review,
                draft,
            )
        )
        critique = _force_generic_boilerplate_revision(critique, draft)
        for _attempt in range(2):
            if critique.get("verdict") != "revise":
                break
            final_reply = await _revise_plan(
                state,
                deps,
                case_analysis_packet,
                issue_map,
                issue_analyses,
                legal_basis_packet,
                adversarial_review,
                final_reply,
                critique,
            )
            if not _generic_boilerplate_issues(final_reply):
                break
            critique = _force_generic_boilerplate_revision(critique, final_reply)
    else:
        final_reply = _strip_generic_boilerplate(final_reply)
    final_reply = _sanitize_statute_citations(final_reply, state.law_context_str)
    final_reply = _ensure_grounded_legal_basis(final_reply, state.law_context_str, state)
    final_reply = _enforce_final_output_contract(final_reply)

    # 自动保存关键信息到长期记忆
    user_id = state.user_context.get("user_id")
    if user_id:
        try:
            from src.infra.milvus_store import get_milvus_store
            store = get_milvus_store()

            if region and region != "全国":
                memory_text = f"用户所在地区：{region}"
                await store.aput(
                    namespace=("users", user_id, "memories"),
                    key=f"region_{region}",
                    value={"content": memory_text, "type": "user_profile"},
                )

            summary_parts = [
                f"领域：{DOMAIN_LABELS.get(domain, domain or '法律')}",
                f"法律问题：{'、'.join(state.confirmed_issues) or '未明确'}",
                f"案情事实：{'；'.join(state.collected_facts) or '未补充'}",
                f"时间：{state.time_info or '未确认'}",
                f"已有证据：{'、'.join(state.evidence_confirmed) or '未确认'}",
            ]
            case_summary = "法律咨询摘要：" + "；".join(summary_parts)
            session_key = (state.session_id or "unknown").replace(":", "_")[-120:]
            await store.aput(
                namespace=("users", user_id, "memories"),
                key=f"guide_{session_key}",
                value={"content": case_summary, "type": "legal_case_summary"},
            )
            logger.info("已保存法律咨询长期记忆 | user={} session={}", user_id, state.session_id)
        except Exception as e:
            logger.warning(f"保存长期记忆失败: {e}")

    return {
        **_fraud_warning_display_updates(state),
        "phase": GuidePhase.CONCLUDE,
        "case_analysis_packet": case_analysis_packet,
        "issue_map": issue_map,
        "legal_basis_packet": legal_basis_packet,
        "issue_analyses": issue_analyses,
        "analysis_validation": analysis_validation,
        "analysis_stage": "validated" if analysis_validation.get("status") == "ok" else "needs_review",
        "strategy_plan": strategy_plan,
        "strategy_plan_version": state.strategy_plan_version,
        "strategy_stage": "validated" if strategy_validation.get("status") == "ok" else "needs_review",
        "strategy_validation": strategy_validation,
        "solution_version": state.solution_version + 1,
        "solution_evidence_version": state.evidence_evaluation_version,
        "latest_plan_text": final_reply,
        "messages": [AIMessage(content=final_reply)],
    }


async def node_save_record(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑨：保存咨询记录到 PostgreSQL。"""
    emit_guide_progress(
        "finalizing",
        "正在整理本轮结果",
        "保存案件状态和证据清单，确保后续可以继续补交并更新方案。",
    )
    user_id = state.user_context.get("user_id")
    logger.info("节点⑨保存记录 | session={} domain={}", state.session_id, state.legal_domain)
    try:
        await save_guide_record(
            user_id=user_id,
            session_id=state.session_id,
            domain=state.legal_domain,
            issues=state.confirmed_issues,
            db=deps.db_session,
        )
    except Exception as exc:
        # 持久化失败不能吞掉已经生成的法律指引回复。
        logger.error("保存法律咨询记录失败 | session={} error={}", state.session_id, exc)
    return {"phase": GuidePhase.END}


# ════════════════════════════════════════════════════════════════════════
# 路由函数
# ════════════════════════════════════════════════════════════════════════

def _needs_clarify(state: GuideState) -> bool:
    """澄清门控：低信息消息或没有法律问题/口语问题/案件事实时，领域不能单独绕过澄清。"""
    from src.core.config import get_settings
    settings = get_settings()
    last_msg = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        "",
    )
    if state.messages and _is_low_information_message(last_msg):
        return state.clarify_rounds < settings.GUIDE_MAX_CLARIFY_ROUNDS
    has_any_issue = bool(state.confirmed_issues or state.unmatched_issues)
    has_case_substance = bool(
        state.case_facts
        or state.collected_facts
        or state.draftable_facts
    )
    has_domain = bool(state.legal_domain and state.legal_domain != "other")
    if has_any_issue or (has_domain and has_case_substance):
        return False
    return state.clarify_rounds < settings.GUIDE_MAX_CLARIFY_ROUNDS


def route_after_urgency(state: GuideState) -> str:
    """高危直接熔断；等待追问回答时先解析，否则提取法律问题。"""
    if state.phase == GuidePhase.END:
        return END
    if state.awaiting_supplement_choice and not state.supplement_choice:
        return "ask_followup"
    if state.supplement_choice in {"continue", "conclude"}:
        if state.supplement_has_details:
            return "parse_details" if state.pending_ask_details else "extract_issues"
        return "assess_retrieve"
    if state.pending_ask_details:
        return "parse_details"
    if (
        state.wants_conclude
        and state.confirmed_issues
        and not (
            _current_turn_contains_case_details(
                state,
                next(
                    (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
                    "",
                ),
            )
        )
    ):
        return "assess_retrieve"
    return "extract_issues"


def route_after_extract(state: GuideState) -> str:
    """完全无法识别时澄清；其余情况均进入评分与检索。"""
    if _needs_clarify(state):
        return "clarify"
    return "assess_retrieve"


def route_after_parse(state: GuideState) -> str:
    """反问时继续等待；出现新法律问题则重新标准化，否则重新评分检索。"""
    if state.pending_ask_details:
        return END
    if state.wants_conclude and not state.turn_contains_case_details:
        # Pure workflow controls such as “现在生成方案” must bypass issue/fact
        # extraction even when an older state's issue counters are stale.
        return "assess_retrieve"
    if state.issue_refresh_needed:
        logger.info("路由：用户补充超出原追问范围，重新识别法律问题和领域")
        return "extract_issues"
    if len(state.confirmed_issues) > state.last_confirmed_count:
        logger.info("路由：检测到新法律问题（{}→{}），重新标准化+检索",
                    state.last_confirmed_count, len(state.confirmed_issues))
        return "extract_issues"
    if (
        not state.confirmed_issues
        and not state.unmatched_issues
        and (not state.legal_domain or state.legal_domain == "other")
    ):
        logger.info("路由：澄清答案已结构化但法律问题仍未识别，重新执行语义标准化")
        return "extract_issues"
    return "assess_retrieve"


def route_after_assess_retrieve(state: GuideState) -> str:
    """Route from the planner decision without checking fixed field completion."""
    if state.force_conclude or state.wants_conclude or state.supplement_choice == "conclude":
        return "conclude"
    if state.followup_plan.get("should_ask"):
        return "ask_followup"
    return "conclude"


# ════════════════════════════════════════════════════════════════════════
# 图的组装
# ════════════════════════════════════════════════════════════════════════

def build_guide_graph(deps: GuideDeps):
    """构建九节点法律指引状态图，deps 通过闭包注入。"""
    async def _run_grouped(node, state: GuideState) -> dict:
        """Keep legacy node results compatible with grouped GuideState channels."""

        updates = await node(state, deps)
        return state.group_updates(updates)

    async def _prepare_turn(s):    return await _run_grouped(node_prepare_turn, s)
    async def _check_urgency(s):   return await _run_grouped(node_check_urgency, s)
    async def _extract_issues(s):  return await _run_grouped(node_extract_issues, s)
    async def _clarify(s):         return await _run_grouped(node_clarify, s)
    async def _assess_retrieve(s): return await _run_grouped(node_assess_retrieve, s)
    async def _ask_followup(s):    return await _run_grouped(node_ask_followup, s)
    async def _parse_details(s):   return await _run_grouped(node_parse_details, s)
    async def _conclude(s):        return await _run_grouped(node_conclude, s)
    async def _save_record(s):     return await _run_grouped(node_save_record, s)

    graph = StateGraph(GuideState)
    graph.add_node("prepare_turn",    _prepare_turn)
    graph.add_node("check_urgency",  _check_urgency)
    graph.add_node("extract_issues", _extract_issues)
    graph.add_node("clarify",        _clarify)
    graph.add_node("assess_retrieve", _assess_retrieve)
    graph.add_node("ask_followup",    _ask_followup)
    graph.add_node("parse_details",  _parse_details)
    graph.add_node("conclude",       _conclude)
    graph.add_node("save_record",    _save_record)

    graph.set_entry_point("prepare_turn")
    graph.add_edge("prepare_turn", "check_urgency")
    graph.add_edge("clarify",      END)
    graph.add_edge("ask_followup", END)
    graph.add_edge("conclude",     "save_record")
    graph.add_edge("save_record",  END)

    graph.add_conditional_edges("check_urgency",  route_after_urgency,
        {
            "parse_details": "parse_details",
            "extract_issues": "extract_issues",
            "assess_retrieve": "assess_retrieve",
            "ask_followup": "ask_followup",
            END: END,
        })
    graph.add_conditional_edges("extract_issues", route_after_extract,
        {"clarify": "clarify", "assess_retrieve": "assess_retrieve"})
    graph.add_conditional_edges("parse_details",  route_after_parse,
        {"extract_issues": "extract_issues", "assess_retrieve": "assess_retrieve", END: END})
    graph.add_conditional_edges("assess_retrieve", route_after_assess_retrieve,
        {"ask_followup": "ask_followup", "conclude": "conclude"})
    return graph.compile()


# ════════════════════════════════════════════════════════════════════════
# 对外接口
# ════════════════════════════════════════════════════════════════════════

async def run_guide(
    user_message: str,
    thread_id: str,
    deps: GuideDeps,
    existing_state: GuideState | None = None,
    user_id: str | None = None,
    long_term_memories: list[str] | None = None,
) -> tuple[str, GuideState]:
    """
    执行一轮法律指引对话。

    Args:
        user_message      : 用户本轮输入
        thread_id         : 会话ID（关联Redis + PostgreSQL）
        deps              : 依赖注入容器
        existing_state    : 上一轮状态（多轮对话时传入）
        user_id           : 用户ID，贯穿整个流程
        long_term_memories: Supervisor检索到的长期记忆摘要

    Returns:
        (assistant_reply, new_state)
    """
    emit_guide_progress(
        "received",
        "已收到本轮信息",
        "正在进入案件处理流程。",
    )
    graph = build_guide_graph(deps)

    if existing_state is None:
        state = GuideState(
            session_id=thread_id,
            user_context={"user_id": user_id, "long_term_memories": long_term_memories or []},
        )
    else:
        state = existing_state

    state.messages.append(HumanMessage(content=user_message))

    logger.info("run_guide start | session={} round={} user_id={}", thread_id, state.round, user_id)

    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(state, config=config)

    new_state = GuideState(**result) if isinstance(result, dict) else result

    reply = ""
    for msg in reversed(new_state.messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    logger.info("run_guide complete | session={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    return reply, new_state


def build_guide_deps(db_session=None) -> GuideDeps:
    """构建法律指引依赖注入容器。供 guide_agent 工具和 API 路由共用。"""
    llm = build_chat_llm(temperature=0.3)
    from src.infra.embedding import get_embedding_model
    embedding_model = get_embedding_model()
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return GuideDeps(
        llm=llm,
        fast_llm=build_chat_llm(
            temperature=0.2,
            model=settings.CHAT_MODEL_FAST or settings.CHAT_MODEL,
        ),
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
    )
