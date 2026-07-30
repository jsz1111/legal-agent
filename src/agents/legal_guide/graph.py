"""公民法律指引 LangGraph 状态机。"""
from __future__ import annotations

import asyncio
import json
import re
from difflib import SequenceMatcher
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
from pymilvus import MilvusClient

from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings
from src.agents.legal_guide.state import GuideState, GuidePhase
from src.agents.legal_guide.issue_normalizer import normalize_legal_issues
from src.agents.legal_guide.neo4j_queries import query_laws_and_channels
from src.agents.legal_guide.convergence import should_conclude
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
    active_case_facts,
    evidence_from_case_facts,
    format_case_context,
    legacy_fact_updates,
    latest_case_facts,
    reduce_case_facts,
)
from src.agents.legal_guide.followup_planner import (
    format_followup_authority,
    plan_next_followup,
)
from src.agents.legal_guide.retrieval_query import build_case_retrieval_inputs
from src.agents.legal_guide.authority_registry import format_domain_authority_summary
from src.agents.legal_guide.evidence_rules import (
    format_evidence_source,
    resolve_state_evidence_checklist,
)
from src.agents.legal_guide.followup_catalog import (
    assess_evidence_answer,
    assess_fact_answer,
    assess_initial_evidence,
    assess_initial_facts,
    evidence_effective_count,
    find_evidence_followup,
    find_fact_followup,
    format_evidence_assessments,
    format_fact_assessments,
    get_domain_followups,
)
from src.agents.legal_guide.prompts import (
    URGENCY_CHECK_PROMPT, CLARIFY_PROMPT,
    PARSE_DETAILS_PROMPT, CONCLUDE_PROMPT, PLAN_AUDIT_PROMPT, SELF_REVIEW_PROMPT,
    DOC_TYPE_MAP,
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

请先确保安全，法律维权可在安全后继续。我随时可以帮您梳理具体步骤。"""


class GuideDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session=None):
        self.llm = llm
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.db_session = db_session


def _long_term_memories(state: GuideState, limit: int = 5) -> list[str]:
    """返回已由 Supervisor/Worker 检索出的相关长期记忆，限制长度避免污染提示词。"""
    memories = state.user_context.get("long_term_memories") or []
    return [str(item).strip()[:300] for item in memories[:limit] if str(item).strip()]


_MEMORY_RECALL_MARKERS = (
    "之前说", "以前说", "上次说", "前面说", "还记得", "记得我", "我说过",
    "之前的", "上次的", "以前的",
)


def _active_long_term_memories(state: GuideState) -> list[str]:
    """Historical case facts enter reasoning only when the user explicitly recalls them."""
    user_messages = [
        str(message.content)
        for message in state.messages
        if isinstance(message, HumanMessage)
    ]
    if not user_messages:
        return _long_term_memories(state)
    if not any(marker in text for text in user_messages for marker in _MEMORY_RECALL_MARKERS):
        return []
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
    supplement_has_details = False
    awaiting_supplement_choice = state.awaiting_supplement_choice
    allow_extra_followups = state.allow_extra_followups
    wants_conclude = state.wants_conclude or any(p in last_msg for p in conclude_phrases)
    if state.awaiting_supplement_choice:
        supplement_choice = _supplement_choice_from_text(last_msg)
        supplement_has_details = _supplement_contains_case_details(last_msg)
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
    prompt = URGENCY_CHECK_PROMPT.format(
        user_input=last_msg,
        recent_dialogue=recent_dialogue or f"- {last_msg}",
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        urgency = result.get("urgency", "NORMAL")
        time_clue = result.get("time_clue", "")
        recent_text = "\n".join(recent_human_messages)
        explicitly_safe = any(
            marker in recent_text
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
        allowed_safety_statuses = {"danger", "safe", "unknown", "not_applicable"}
        model_safety_status = str(result.get("safety_status") or "").lower()
        safety_relevant = bool(result.get("safety_relevant"))
        safety_contract_present = (
            "safety_relevant" in result or "safety_status" in result
        )
        if deterministic_current_danger:
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
                if model_safety_status in allowed_safety_statuses - {"not_applicable", "safe"}
                else "unknown"
            )
        else:
            safety_status = "not_applicable"
        current_danger = safety_status == "danger"
        if urgency == "CRITICAL" and not current_danger and explicitly_safe:
            logger.info("近期已明确当前安全且没有新增危险，继续法律梳理而不触发紧急终止")
            return {
                "urgency_level": "normal",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
            }
        if urgency == "CRITICAL" and not current_danger:
            logger.info("涉及人身安全但当前危险未确认，转入单问题安全确认")
            return {
                "urgency_level": "normal",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
            }
        if urgency == "CRITICAL" and current_danger:
            logger.warning("节点②检测到CRITICAL紧急情形")
            return {
                "urgency_level": "critical",
                "safety_relevant": True,
                "current_safety_status": "danger",
                "phase": GuidePhase.END,
                "messages": [AIMessage(content=URGENCY_CRITICAL_RESPONSE)],
            }
        if urgency == "TIME" and time_clue:
            warning = f'\n⚠️ **时效提醒**：您提到"{time_clue}"，请注意维权时效（劳动仲裁1年、一般民事3年），建议尽快行动。'
            logger.info("节点②检测到时效紧迫: {}", time_clue)
            return {
                "urgency_level": "time",
                "safety_relevant": safety_relevant,
                "current_safety_status": safety_status,
                "time_warning": warning,
            }
        return {
            "urgency_level": "normal",
            "safety_relevant": safety_relevant,
            "current_safety_status": safety_status,
        }
    except Exception as e:
        logger.warning(f"紧急检测解析失败: {e}")
    return {
        "urgency_level": "normal",
        "safety_relevant": state.safety_relevant,
        "current_safety_status": state.current_safety_status,
    }


async def node_extract_issues(state: GuideState, deps: GuideDeps) -> dict:
    """节点③：标准化法律问题，并把当前消息归入通用原子案情。"""
    human_msgs = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {}
    combined_input = "\n".join(human_msgs[-3:])
    memories = _active_long_term_memories(state)
    prior_messages = human_msgs[-3:-1]
    normalizer_input = (
        "[近期对话，仅用于理解语境]\n"
        + ("\n".join(prior_messages) or "无")
        + "\n\n[当前用户消息]\n"
        + human_msgs[-1]
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
    if memories:
        normalizer_input += (
            "\n\n[相关长期记忆，仅作补充；与本轮冲突时以本轮为准]\n"
            + "\n".join(f"- {item}" for item in memories)
        )
    logger.info("节点③提取法律问题 | round={}", state.round)
    result = await normalize_legal_issues(
        user_input=normalizer_input,
        llm=deps.llm,
        neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )
    # 两个池分别累积，跨轮保序去重（不用 set，避免检索 query 每轮字符串顺序漂移）
    def _merge(old: list[str], new: list[str]) -> list[str]:
        seen: set[str] = set()
        return [x for x in old + new if not (x in seen or seen.add(x))]

    latest_user_text = human_msgs[-1]
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
    # The primary legal domain is stable for one case. New facts can broaden
    # retrieval, but an inferred label must not silently replace the workflow.
    domain = state.legal_domain or proposed_domain
    new_term_map = {**state.term_map, **result_term_map}
    raw_case_updates = result.get("case_updates") or legacy_fact_updates(
        result.get("collected_facts") or [],
        user_text=latest_user_text,
    )
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
    atom_evidence, atom_unavailable = evidence_from_case_facts(case_facts)
    new_evidence = _merge(state.evidence_confirmed, atom_evidence)
    new_unavailable = _merge(state.evidence_unavailable, atom_unavailable)
    region_extracted = (
        _state_region_name(state.region)
        or _state_region_name(result.get("region", ""))
    )
    time_info = state.time_info or result.get("time_info", "")

    logger.info(
        "节点③结果 | standard={} colloquial={} domain={} evidence={} region={} time_info={}",
        new_confirmed, new_unmatched, domain, new_evidence, region_extracted, time_info,
    )

    updates = {
        "unmatched_issues": new_unmatched,
        "term_map": new_term_map,
        "collected_facts": new_facts,
        "draftable_facts": new_draftable_facts,
        "case_facts": case_facts,
        "fact_records": fact_records,
        "evidence_unavailable": new_unavailable,
        "legal_domain": domain,
    }

    if new_evidence != state.evidence_confirmed:
        updates["evidence_confirmed"] = new_evidence
        newly_found = [item for item in new_evidence if item not in state.evidence_confirmed]
        updates["evidence_assessments"] = assess_initial_evidence(
            newly_found,
            state.evidence_assessments,
        )

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
    prompt = CLARIFY_PROMPT.format(user_input=last_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {
        "clarify_rounds": state.clarify_rounds + 1,
        "phase": GuidePhase.CLARIFY,
        "messages": [AIMessage(content=response.content)],
    }


async def node_score(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：纯规则打分（打分前置、零 I/O），决定是否值得深度检索。"""
    domain = state.legal_domain
    evidence_total = len(resolve_state_evidence_checklist(state).items)
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


async def node_retrieve(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤内部检索：所有档位检索 statute+case+graph，HIGH 档额外自省。

    法条检索策略：
    - effective_domain 不为空 → domain-filtered + 全库 双路并发，RRF 融合
    - effective_domain 为空（domain=other）→ 仅全库向量检索
    避免 domain 识别错误时返回 0 条法律。
    """
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
            pg_hits = await search_statutes_pg_fallback(
                effective_domain,
                lexical_phrases,
                deps.db_session,
                limit=16,
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

    if effective_domain and law_hits:
        from src.agents.legal_knowledge.reranker import rerank_docs as _rerank
        candidate_count = len(law_hits)
        law_hits = await _rerank(question, law_hits, top_k=8)
        logger.info("法条统一精排完成 | candidates={} final={}", candidate_count, len(law_hits))

    fallback_guide = None
    similar_cases = []
    if isinstance(case_result, Exception):
        if isinstance(case_result, asyncio.TimeoutError):
            logger.warning("case_rag 超时（>5s），降级跳过")
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
            law_titles = await _fetch_law_titles(law_hits, deps.db_session)
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
    channels = await query_recommended_channels(
        domain=domain,
        region=state.region,
        db=deps.db_session,
        limit=6,
    )

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
    }

    # 仅 HIGH 档做自省（启发式判断：法条适用性/时效/管辖）
    if state.confidence_tier == "HIGH" and law_context_formatted:
        case_summary = f"法律问题：{'; '.join(state.confirmed_issues)}\n已有证据：{'; '.join(state.evidence_confirmed) or '无'}"
        review_prompt = SELF_REVIEW_PROMPT.format(
            case_summary=case_summary,
            law_context=law_context_formatted[:2000],  # 截取避免过长
        )
        try:
            review_resp = await deps.llm.ainvoke([SystemMessage(content=review_prompt)])
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


async def node_assess_retrieve(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：规则评分与法条、类案、渠道检索合并为一个原子业务步骤。"""
    score_updates = await node_score(state, deps)
    scored_state = state.model_copy(update=score_updates)
    has_current_turn_facts = bool(
        latest_case_facts(scored_state.case_facts, scored_state.round)
    )
    if (
        scored_state.retrieval_completed
        and (scored_state.wants_conclude or scored_state.supplement_choice in {"continue", "conclude"})
        and not scored_state.supplement_has_details
        and not has_current_turn_facts
    ):
        logger.info("节点⑤复用上一轮检索结果 | 用户已完成补充选择")
        retrieval_updates = {}
    else:
        retrieval_updates = await node_retrieve(scored_state, deps)
    assessed_state = scored_state.model_copy(update=retrieval_updates)
    _, force = should_conclude(
        assessed_state,
        max_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS,
    )
    hard_stop = (
        state.force_conclude
        or force
        or assessed_state.wants_conclude
        or assessed_state.supplement_choice == "conclude"
        or assessed_state.ask_rounds >= settings.GUIDE_MAX_ASK_ROUNDS
        or assessed_state.consecutive_low_info_answers >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
    )
    if hard_stop:
        followup_plan = {"should_ask": False, "planner_mode": "converged"}
    else:
        followup_plan = await plan_next_followup(assessed_state, deps.llm)
    return {
        **score_updates,
        **retrieval_updates,
        "force_conclude": state.force_conclude or force,
        "followup_plan": followup_plan,
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


def _user_facing_case_text(value: str) -> str:
    text = " ".join(str(value or "").split()).strip("。；， ")
    text = re.sub(r"^用户(?:称|表示|提到)", "您提到", text)
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


def _contextualize_followup_question(state: GuideState, question: str) -> str:
    anchor = _followup_case_anchor(state)
    if not anchor:
        return question
    compact_question = re.sub(r"\W+", "", question)
    atoms = [
        re.sub(r"\W+", "", str(item.get(field) or ""))
        for item in _distinct_case_atoms(state)[-4:]
        for field in ("value", "subject")
    ]
    if any(len(atom) >= 2 and atom in compact_question for atom in atoms):
        return question
    return f"结合您说的“{anchor}”，{question}"


def _followup_opening(state: GuideState) -> str:
    if state.supplement_choice == "continue":
        return "好的，我们继续，只补充真正会影响方案的信息。"
    acknowledgement = str(state.followup_plan.get("acknowledgement") or "").strip()
    if acknowledgement:
        return f"{acknowledgement.rstrip('。')}。"
    latest_statements = [
        item.get("statement", "")
        for item in active_case_facts(state.case_facts)
        if int(item.get("turn") or 0) == state.round and item.get("statement")
    ][:2]
    if latest_statements:
        recorded = "；".join(_user_facing_case_text(item) for item in latest_statements)
        return f"好的，{recorded}，我已经记下。"
    latest = next(
        (str(message.content).strip() for message in reversed(state.messages) if isinstance(message, HumanMessage)),
        "",
    )
    if latest:
        return "好的，您刚补充的内容我已经记录。"
    issues = "、".join(state.confirmed_issues[:2]) or f"{DOMAIN_LABELS.get(state.legal_domain, '法律')}问题"
    return f"我会继续按“{issues}”帮您梳理。"


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
    question = _contextualize_followup_question(state, question.strip())
    if "？" not in question and "?" not in question:
        question += "？"
    reason = _normalized_followup_reason(reason)
    contextual_reason = str(state.followup_plan.get("contextual_reason") or "").strip().rstrip("。；")
    lines = [_followup_opening(state)]
    if contextual_reason:
        lines.append(f"{contextual_reason}。")
    elif ask_type == "evidence":
        anchor = _followup_case_anchor(state)
        prefix = f"针对您说的“{anchor}”，" if anchor else ""
        lines.append(f"{prefix}这项材料主要用于{reason}。")
    else:
        anchor = _followup_case_anchor(state)
        prefix = f"结合您目前说清的“{anchor}”，" if anchor else ""
        lines.append(f"{prefix}再确认这一点是为了{reason}。")
    lines.append(question)
    if ask_type == "evidence":
        lines.append("没有或不确定都可以直接说，我会给替代办法。")
    else:
        lines.append(answer_hint or "不清楚时可以说大概情况或“不知道”。")
    if state.followup_plan:
        lines.append(format_followup_authority(state.followup_plan))
    else:
        lines.append(_followup_authority_hint(state, ask_type=ask_type, reason=reason))
    lines.append("如果不方便补充，直接回复“现在生成方案”，我会按现有信息给出建议。")
    return "\n".join(lines)


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
    plan = state.followup_plan or await plan_next_followup(state, deps.llm)
    if not plan.get("should_ask"):
        return {}
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
    """节点⑥：展示节点⑤按信息增益动态选出的唯一追问。"""
    return await _ask_from_dynamic_plan(state, deps)


async def node_parse_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦：解析用户对追问的回答，提取证据/地区/时间信息。

    若用户本轮没有回答而是反问，则不抽取任何信息、保留 pending_ask_details，
    把反问记入 deferred_questions，并原样重述待答问题（不消耗 ask_rounds）。
    """
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if not last_msg or not state.pending_ask_details:
        return {}
    if state.wants_conclude and not _supplement_contains_case_details(last_msg):
        # A pure flow-control command changes routing only. It is not a fact,
        # evidence item or document statement, and does not need an LLM parse.
        return {
            "pending_ask_details": [],
            "pending_ask_type": "",
            "pending_followup_ids": [],
            "followup_plan": {},
            "phase": GuidePhase.ISSUE_SEARCH,
        }
    prompt = PARSE_DETAILS_PROMPT.format(
        asked_details="\n".join(f"- {q}" for q in state.pending_ask_details),
        user_answer=last_msg,
        case_context=format_case_context(state.case_facts),
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
    except Exception as e:
        # 解析失败：清空 pending 避免卡在重述死循环，代价是本轮回答丢失
        logger.warning("节点⑦解析追问回答失败，丢弃本轮抽取 | err={} raw={}", e, response.content[:200])
        return {"pending_ask_details": [], "pending_ask_type": "", "pending_followup_ids": []}

    user_question = (parsed.get("user_question") or "").strip()
    is_answer = parsed.get("is_answer", True)
    answers_asked_question = parsed.get("answers_asked_question", is_answer)
    parser_missed_declarative_detail = not is_answer and not _looks_like_user_question(last_msg)
    if parser_missed_declarative_detail:
        logger.info("节点⑦将非疑问陈述按主动补充处理 | text={}", last_msg[:120])
        is_answer = True
        answers_asked_question = False
        user_question = ""
        if not parsed.get("collected_facts"):
            parsed["collected_facts"] = [last_msg]
    if state.wants_conclude and any(
        phrase in last_msg
        for phrase in ("现在生成方案", "生成方案", "给方案", "按现有信息", "不要再问", "别再问")
    ):
        # 这是流程控制指令，不是需要在结论中回答的法律问题。
        user_question = ""

    if _looks_like_question_repetition(last_msg, state.pending_ask_details):
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
        clarification = (
            "我看到这句话更像是把问题重复了一遍，还不能确定您的答案。\n"
            f"请直接回答这个问题：{question}\n"
            "可以用一句很短的话回答，例如“有”“没有”“大概是……”或“不清楚”。"
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
        acknowledgement = (
            f"您问的“{user_question}”我已记下，会在最终方案中一并回答。\n"
            if user_question else "您的疑问我已记下，会在最终方案中一并回答。\n"
        )
        reask = acknowledgement + "为避免方案失准，当前还需要确认：\n" + \
                "\n".join(f"- {q}" for q in pending)
        return {
            "deferred_questions": deferred,
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
    if parsed_case_updates:
        raw_case_updates = [
            item for item in parsed_case_updates
            if str(item.get("key") or "") not in current_turn_keys
        ]
    elif current_turn_keys:
        raw_case_updates = []
    else:
        raw_case_updates = legacy_fact_updates(parsed_facts, user_text=last_msg)
    case_facts = reduce_case_facts(
        state.case_facts,
        raw_case_updates,
        user_text=last_msg,
        turn=state.round,
    )
    if parser_missed_declarative_detail and not latest_case_facts(case_facts, state.round):
        case_facts = reduce_case_facts(
            state.case_facts,
            legacy_fact_updates(parsed_facts, user_text=last_msg),
            user_text=last_msg,
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
    atom_evidence, atom_unavailable = evidence_from_case_facts(case_facts)
    parsed_evidence = parsed.get("evidence") or []
    if is_multimodal_evidence:
        type_match = re.search(r"【证据类型】\s*([^\n]+)", last_msg)
        evidence_type = type_match.group(1).strip(" *：:") if type_match else "图片证据"
        present_evidence = [f"已上传图片：{evidence_type}"]
        unverified_evidence = [
            item for item in parsed_evidence
            if item and not any(token in item for token in (evidence_type, "聊天记录截图", "图片证据"))
        ]
    else:
        unverified_markers = ("声称", "自述持有", "提及持有", "未在本图", "未显示", "待核验", "疑似持有")
        present_evidence = [
            item for item in parsed_evidence
            if item and not any(marker in item for marker in unverified_markers)
        ]
        unverified_evidence = [
            item for item in parsed_evidence
            if item and item not in present_evidence
        ]
    if len(present_evidence) != len(parsed_evidence):
        logger.info("节点⑦未核验证据线索不计入置信度 | evidence={}", parsed_evidence)
    present_evidence = _merge(present_evidence, atom_evidence)
    new_evidence = _merge(state.evidence_confirmed, present_evidence)
    new_unverified = _merge(state.evidence_unverified, unverified_evidence)
    evidence_denial_markers = (
        "没有", "没拍", "没留", "没保存", "未拍", "未留", "未保存",
        "找不到", "拿不出", "丢了", "遗失", "无法提供",
    )
    explicit_evidence_denial = any(marker in last_msg for marker in evidence_denial_markers)
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
    low_info_answer = False
    pending_fact_statuses: list[str] = []
    answer_is_negative = explicit_evidence_denial
    for rule_id in (state.pending_followup_ids if answers_asked_question else []):
        if state.pending_ask_type == "facts":
            rule = find_fact_followup(state.legal_domain, rule_id)
            if rule:
                record = assess_fact_answer(rule, last_msg, fact_records.get(rule_id))
                fact_records[rule_id] = record
                pending_fact_statuses.append(record["status"])
                low_info_answer = low_info_answer or record["status"] in {"unknown", "ambiguous"}
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
                (bool(present_evidence) or any(marker in last_msg for marker in positive_markers))
                and not explicitly_unavailable
            )
            record = assess_evidence_answer(
                rule,
                last_msg,
                unavailable=explicitly_unavailable,
                uploaded=is_multimodal_evidence and mentioned_present,
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
            low_info_answer = low_info_answer or record["availability"] in {"unavailable", "unclear"}

    if not state.pending_followup_ids:
        if state.pending_ask_type == "facts":
            low_info_answer = any(marker in last_msg for marker in ("不知道", "不清楚", "不记得", "记不清"))
        elif state.pending_ask_type == "evidence":
            low_info_answer = answer_is_negative
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
        confirmation_messages = [AIMessage(content=(
            f"我暂时没有把这项内容写成确定事实，因为您的回答{status_text}。\n"
            f"请再明确一次：{question}\n"
            "如果是在更正之前的说法，可以直接以“更正：……”开头；不清楚也可以直接说“不清楚”。"
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
        "deferred_questions": state.deferred_questions + ([user_question] if user_question else []),
        "messages": confirmation_messages,
    }
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
    # Retrieval already performs domain/full-corpus fusion and semantic reranking.
    # Preserve that order instead of adding scenario-specific article priorities.
    return entries[:limit]


def _ensure_grounded_legal_basis(
    reply: str,
    law_context: str,
    state: GuideState | None = None,
) -> str:
    """Render the legal-basis section directly from this turn's retrieved source text."""
    entries = _select_grounded_statute_entries(
        _grounded_statute_entries(law_context),
        state,
    )
    if not entries:
        return reply
    heading = re.search(r"(?m)^\s*(?:#{1,6}\s*)?\*{0,2}【法律依据】\*{0,2}\s*$", reply)
    block = (
        "> 以下条文直接来自本轮知识库检索原文。\n"
        + "\n".join(f"- 《{name}》{article}：{text}" for name, article, text in entries)
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
        badge = re.match(r"\s*(\*\*📊[^\n]+\*\*)", reply)
        prefix = f"{badge.group(1)}\n\n" if badge else ""
        reply = prefix + reply[understanding.start():]
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
        "维权胜算评估": 180 if not accessible else 150,
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


def _ensure_required_plan_sections(reply: str, state: GuideState) -> str:
    """最终压缩后再次核对稳定输出合同，避免模型随机漏段。"""
    if "【维权路径" not in reply:
        routes = _channel_summary_lines(state)
        if not routes:
            routes = [
                "- 当前没有可核验的本地渠道记录，请先拨打 12348 核对受理机构、管辖和程序。"
            ]
        fallback = "**【维权路径比较】**\n" + "\n".join(routes)
        reply = _insert_before_document_offer(reply, fallback)
    if "【维权胜算评估】" not in reply:
        tier_label = {"HIGH": "较充分", "MEDIUM": "一般", "LOW": "有限"}.get(
            state.confidence_tier, "一般"
        )
        reply = _insert_before_document_offer(
            reply,
            (
                "**【维权胜算评估】**\n"
                f"- **当前信息充分度：{tier_label}**。系统不据此计算胜诉率；"
                "最终结果仍取决于事实核验、原始证据、法律适用、对方抗辩及履行能力。"
            ),
        )
    empty_recommendation = re.compile(
        r"(?m)(?:#{1,6}\s*)?\*{0,2}【推荐方案】\*{0,2}\s*"
        r"(?=(?:\n#{1,6}\s*)?\*{0,2}【维权胜算评估】\*{0,2})"
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


async def node_conclude(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑧：生成五段式行动方案（理解+法条+类案+路径+行动清单）。"""
    logger.info("节点⑧生成结论 | domain={} tier={}", state.legal_domain, state.confidence_tier)
    domain = state.legal_domain
    accessible_mode = _uses_accessible_language(state)
    compact_mode = accessible_mode or state.force_conclude or state.wants_conclude
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
    dialogue_snippet = "\n".join(
        f"{'用户' if getattr(m, 'type', '') == 'human' else '助手'}：{str(m.content)[:300]}"
        for m in recent_msgs
    ) or "（无近期对话记录）"

    prompt = CONCLUDE_PROMPT.format(
        deferred_questions=deferred_str,
        confidence_guidance=tier_guidance(state.confidence_tier),
        audience_guidance=_audience_guidance(state),
        confirmed_issues="、".join(state.confirmed_issues) or "法律问题",
        legal_domain=DOMAIN_LABELS.get(domain, domain or "法律"),
        region=region,
        time_info=state.time_info or "暂未确认",
        collected_facts="；".join(state.collected_facts) or "暂未确认",
        long_term_memories="；".join(_active_long_term_memories(state)) or "（无相关长期记忆）",
        evidence_confirmed="、".join(state.evidence_confirmed) or "暂未确认",
        evidence_unverified="、".join(state.evidence_unverified) or "（无）",
        evidence_unavailable="、".join(state.evidence_unavailable) or "（无）",
        fact_assessments=format_fact_assessments(state.fact_records),
        evidence_assessments=format_evidence_assessments(state.evidence_assessments),
        time_warning=state.time_warning,
        self_review_note=self_review_str,
        adverse_facts_section=adverse_facts_section,
        dialogue_snippet=dialogue_snippet,
        law_context=state.law_context_str or "（未检索到具体条文，请参考适用法律原则）",
        case_context=state.case_context_str or "（暂无类案数据）",
        channels=channels_str,
        evidence_checklist=evidence_checklist,
        evidence_source=evidence_source,
        followup_authority=format_domain_authority_summary(domain),
        force_conclude_note=force_note,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    audited_content = response.content
    try:
        audit_prompt = PLAN_AUDIT_PROMPT.format(
            case_context=format_case_context(state.case_facts),
            evidence_confirmed="、".join(state.evidence_confirmed) or "暂未确认",
            law_context=state.law_context_str or "（本轮未检索到可引用条文）",
            draft=response.content,
        )
        audit_response = await deps.llm.ainvoke([SystemMessage(content=audit_prompt)])
        if str(audit_response.content or "").strip():
            audited_content = audit_response.content
    except Exception as exc:
        logger.warning("行动方案动态审校失败，继续执行确定性来源校验: {}", exc)

    # 在回复末尾添加检索错误降级提示（如果有）
    final_reply = _normalize_required_sections(audited_content)
    final_reply = _sanitize_statute_citations(final_reply, state.law_context_str)
    final_reply = _ensure_grounded_legal_basis(final_reply, state.law_context_str, state)
    final_reply = _sanitize_unverified_evidence_assertions(
        final_reply,
        state.evidence_unverified,
    )
    final_reply = _ensure_contextual_understanding(final_reply, state)
    final_reply = _sanitize_forced_followups(final_reply)
    final_reply = _ensure_action_checklist(final_reply, state)
    final_reply = _ensure_case_reference(
        final_reply,
        state.similar_cases,
        state.case_context_str,
        state,
    )
    if state.retrieval_error_note:
        final_reply += state.retrieval_error_note

    # ── 置信档位用户提示（前置标签，让用户知道回答可信度） ──────────────────
    _tier_badge = {
        "HIGH":   "**📊 当前事实和法律依据较充分，可作为行动参考；原始证据仍需核对。**",
        "MEDIUM": "**📊 基本法律依据已找到，但信息尚有缺口，请结合实际情况判断。**",
        "LOW":    "**📊 已找到相关法律依据，但部分事实或证据仍待核验；以下方案供初步行动参考，重要决定前可拨打 12348 咨询专业律师。**",
    }
    badge = _tier_badge.get(state.confidence_tier or "", "")
    if badge:
        final_reply = badge + "\n\n" + final_reply

    final_reply = _compact_final_reply(
        final_reply,
        accessible_mode,
        compact=compact_mode,
    )
    final_reply = _ensure_required_plan_sections(final_reply, state)

    # ── 案例未命中时，把 fallback 指引拼入回复（告知用户去哪里查案例） ──────
    if (
        not state.case_context_str
        and state.fallback_guide
        and not compact_mode
        and len(final_reply) < 2600
    ):
        fb = state.fallback_guide
        platform = fb.get("platform", "中国裁判文书网")
        url      = fb.get("url", "https://wenshu.court.gov.cn")
        tips     = fb.get("search_tips", "")
        final_reply += (
            f"\n\n---\n📋 **未在案例库中找到相似案例**，您可前往"
            f"[{platform}]({url}) 自行检索参考：\n{tips}"
        )

    evidence_source_url = (
        str((evidence_rule.source or {}).get("source_page_url") or "")
        if evidence_rule.is_officially_grounded
        else ""
    )
    if evidence_source_url and evidence_source_url not in final_reply:
        if compact_mode or len(final_reply) > 2400:
            final_reply += (
                "\n\n> **证据清单依据**：参考国家级诉讼文书示范文本整理，"
                "不是个案必交材料；具体以受理机关要求为准。"
                f"[官方发布页]({evidence_source_url})"
            )
        else:
            final_reply += f"\n\n> **证据清单依据**：{evidence_source}"

    # 只要已识别法律问题，就保留用户主动生成参考文书的能力。
    if state.confirmed_issues or (state.legal_domain and state.legal_domain != "other"):
        doc_type = DOC_TYPE_MAP.get(state.legal_domain, "投诉信/申请书")
        low_note = "当前信息仍有限，生成后请重点补全占位信息并交由专业人士核对。" if state.confidence_tier == "LOW" else ""
        if "生成文书" not in final_reply:
            final_reply += (
                f"\n\n---\n📄 **需要参考文书？** {low_note}"
                f"如需生成{doc_type}草稿，请回复「生成文书」。"
            )

    final_reply = _compact_final_reply(
        final_reply,
        accessible_mode,
        compact=compact_mode,
    )

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
        "phase": GuidePhase.CONCLUDE,
        "messages": [AIMessage(content=final_reply)],
    }


async def node_save_record(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑨：保存咨询记录到 PostgreSQL。"""
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
    """澄清门控：无任何法律问题（标准化+口语）且未达上限。

    改进：即使L3映射失败，只要有口语问题或领域，也应继续检索而非澄清。
    """
    from src.core.config import get_settings
    settings = get_settings()
    # 有标准化问题或口语问题或已锁定领域 → 不需要澄清
    has_any_issue = bool(state.confirmed_issues or state.unmatched_issues)
    has_domain = bool(state.legal_domain and state.legal_domain != "other")
    if has_any_issue or has_domain:
        return False
    # 完全无法提取任何问题 → 需要澄清（但有轮数上限）
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
        and not _supplement_contains_case_details(
            next(
                (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
                "",
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
    if len(state.confirmed_issues) > state.last_confirmed_count:
        logger.info("路由：检测到新法律问题（{}→{}），重新标准化+检索",
                    state.last_confirmed_count, len(state.confirmed_issues))
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
    async def _prepare_turn(s):    return await node_prepare_turn(s, deps)
    async def _check_urgency(s):   return await node_check_urgency(s, deps)
    async def _extract_issues(s):  return await node_extract_issues(s, deps)
    async def _clarify(s):         return await node_clarify(s, deps)
    async def _assess_retrieve(s): return await node_assess_retrieve(s, deps)
    async def _ask_followup(s):    return await node_ask_followup(s, deps)
    async def _parse_details(s):   return await node_parse_details(s, deps)
    async def _conclude(s):        return await node_conclude(s, deps)
    async def _save_record(s):     return await node_save_record(s, deps)

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

    reply = _with_memory_recall_preface(new_state, user_message, reply)

    logger.info("run_guide complete | session={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    return reply, new_state


def build_guide_deps(db_session=None) -> GuideDeps:
    """构建法律指引依赖注入容器。供 guide_agent 工具和 API 路由共用。"""
    llm = ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    from src.infra.embedding import get_embedding_model
    embedding_model = get_embedding_model()
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return GuideDeps(
        llm=llm,
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
    )
