"""公民法律指引 LangGraph 状态机（9节点，对标 inquiry/graph.py）。"""
from __future__ import annotations

import asyncio
import json
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.graph import StateGraph, END
from pymilvus import MilvusClient

from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings
from src.agents.legal_guide.state import GuideState, GuidePhase
from src.agents.legal_guide.issue_normalizer import normalize_legal_issues
from src.agents.legal_guide.neo4j_queries import query_laws_and_channels, query_channels_by_region
from src.agents.legal_guide.convergence import check_convergence
from src.agents.legal_guide.confidence import score_confidence, tier_guidance
from src.agents.legal_guide.db_queries import load_user_context, save_guide_record
from src.agents.legal_guide.prompts import (
    URGENCY_CHECK_PROMPT, CLARIFY_PROMPT, ASK_DETAILS_PROMPT,
    PARSE_DETAILS_PROMPT, CONCLUDE_PROMPT,
    DOMAIN_DETAIL_TEMPLATES, EVIDENCE_TEMPLATES, DOMAIN_LABELS,
)

settings = get_settings()

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


# ── 格式化辅助函数 ─────────────────────────────────────────────────────────

def _fmt_channels(channels: list[dict]) -> str:
    if not channels:
        return "（暂无检索到具体渠道，建议拨打12348法律援助热线）"
    lines = []
    for c in channels[:6]:
        name = c.get("name", "")
        phone = c.get("phone", "")
        url = c.get("url", "")
        parts = [name]
        if phone:
            parts.append(f"电话：{phone}")
        if url:
            parts.append(f"官网：{url}")
        lines.append("· " + "  ".join(parts))
    return "\n".join(lines)


def _fmt_evidence_checklist(domain: str) -> str:
    items = EVIDENCE_TEMPLATES.get(domain, ["相关合同/协议", "通讯记录截图", "付款凭证"])
    return "\n".join(f"  - {item}" for item in items)


# ════════════════════════════════════════════════════════════════════════
# 节点函数
# ════════════════════════════════════════════════════════════════════════

async def node_load_context(state: GuideState, deps: GuideDeps) -> dict:
    """节点①：加载用户历史咨询上下文（仅首轮）。"""
    if state.round > 0:
        return {}
    user_id = state.user_context.get("user_id")
    logger.info("节点①加载上下文 | session={}", state.session_id)
    ctx = await load_user_context(user_id, deps.db_session)
    region = ctx.get("region", "")
    return {"user_context": ctx, "region": region or state.region}


async def node_check_urgency(state: GuideState, deps: GuideDeps) -> dict:
    """节点②：三级紧急分类（每一轮都执行）。CRITICAL → 立即给援助信息+END。

    关键安全设计：不能只在首轮检测。用户可能在多轮对话中途才追加高危案情
    （例：先聊租房纠纷，几轮后才说"对方上门殴打我"），因此每轮都必须重跑。
    """
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if not last_msg:
        return {}
    logger.info("节点②紧急检测 | session={} round={}", state.session_id, state.round)
    prompt = URGENCY_CHECK_PROMPT.format(user_input=last_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        urgency = result.get("urgency", "NORMAL")
        time_clue = result.get("time_clue", "")
        if urgency == "CRITICAL":
            logger.warning("节点②检测到CRITICAL紧急情形")
            return {
                "urgency_level": "critical",
                "phase": GuidePhase.END,
                "messages": [AIMessage(content=URGENCY_CRITICAL_RESPONSE)],
            }
        if urgency == "TIME" and time_clue:
            warning = f'\n⚠️ **时效提醒**：您提到"{time_clue}"，请注意维权时效（劳动仲裁1年、一般民事3年），建议尽快行动。'
            logger.info("节点②检测到时效紧迫: {}", time_clue)
            return {"urgency_level": "time", "time_warning": warning}
    except Exception as e:
        logger.warning(f"紧急检测解析失败: {e}")
    return {"urgency_level": "normal"}


async def node_extract_issues(state: GuideState, deps: GuideDeps) -> dict:
    """节点③：三层法律问题标准化。拼接最近3条人类消息，避免多轮澄清后上下文丢失。"""
    human_msgs = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {}
    # 取最近3条拼接，给LLM更完整的上下文
    combined_input = "\n".join(human_msgs[-3:])
    logger.info("节点③提取法律问题 | round={}", state.round)
    result = await normalize_legal_issues(
        user_input=combined_input,
        llm=deps.llm,
        neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )
    new_confirmed = list(set(state.confirmed_issues) | set(result["confirmed"]))
    new_unmatched = list(set(state.unmatched_issues) | set(result["unmatched"]))
    domain = result["domain"] or state.legal_domain
    logger.info("节点③结果 | confirmed={} domain={}", new_confirmed, domain)
    if new_confirmed:
        return {
            "confirmed_issues": new_confirmed,
            "unmatched_issues": new_unmatched,
            "legal_domain": domain,
            "phase": GuidePhase.ISSUE_SEARCH,
        }
    return {
        "unmatched_issues": new_unmatched,
        "legal_domain": domain,
        "phase": GuidePhase.CLARIFY,
    }


async def node_clarify(state: GuideState, deps: GuideDeps) -> dict:
    """节点④：引导用户描述清楚法律情况。"""
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    logger.info("节点④澄清引导 | round={}", state.round)
    prompt = CLARIFY_PROMPT.format(user_input=last_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {"round": state.round + 1, "messages": [AIMessage(content=response.content)]}


async def node_issue_search(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：并行RAG检索（法条+类案+图谱渠道）。"""
    question = "；".join(state.confirmed_issues) or "法律问题咨询"
    domain = state.legal_domain
    logger.info("节点⑤检索 | domain={} issues={}", domain, state.confirmed_issues)

    from src.agents.legal_knowledge.statute_rag import search_statutes
    from src.agents.legal_knowledge.case_rag import search_cases

    law_task = search_statutes(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        llm=deps.llm,
        db_session=deps.db_session,
        domain=domain,
    )
    case_task = search_cases(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        llm=deps.llm,
        db_session=deps.db_session,
        domain=domain,
    )
    graph_task = query_laws_and_channels(domain, deps.neo4j_driver)

    law_str, case_str, graph_result = await asyncio.gather(
        law_task, case_task, graph_task, return_exceptions=True
    )
    if isinstance(law_str, Exception):
        logger.warning(f"statute_rag失败: {law_str}")
        law_str = ""
    if isinstance(case_str, Exception):
        logger.warning(f"case_rag失败: {case_str}")
        case_str = ""
    if isinstance(graph_result, Exception):
        logger.warning(f"graph查询失败: {graph_result}")
        graph_result = {"laws": [], "channels": []}

    # 有地区时查精确渠道
    channels = graph_result.get("channels", [])
    if state.region:
        region_channels = await query_channels_by_region(domain, state.region, deps.neo4j_driver)
        if region_channels:
            channels = region_channels

    graph_laws = graph_result.get("laws", [])
    # 只有 statute_rag 真正检索到内容（非"未找到"类提示）才算 milvus_hit
    _NO_RESULT = ("未找到", "未检索到", "没有找到", "暂无", "当前法条库")
    milvus_hit = bool(law_str and not any(m in law_str for m in _NO_RESULT))
    found_laws_count = len(graph_laws) + (3 if milvus_hit else 0)
    should_conclude, force_conclude = check_convergence(
        laws=graph_laws,
        domain=domain,
        current_round=state.round,
        milvus_hit=milvus_hit,
    )

    # 置信度打分：决定结论输出的档次（高/中/低）
    case_hit = bool(case_str and not any(m in case_str for m in _NO_RESULT))
    conf = score_confidence(
        confirmed_issues=state.confirmed_issues,
        evidence_confirmed=state.evidence_confirmed,
        candidate_laws=graph_laws,
        milvus_hit=milvus_hit,
        case_hit=case_hit,
        domain_locked=bool(domain),
        region_known=bool(state.region),
    )
    logger.info("节点⑤收敛判断 | found_laws={} should_conclude={} force={} conf={}({})",
                found_laws_count, should_conclude, force_conclude,
                conf["score"], conf["tier"])

    updates = {
        "candidate_laws": graph_laws,
        "relevant_channels": channels,
        "law_context_str": law_str or "",
        "case_context_str": case_str or "",
        "force_conclude": force_conclude,
        "confidence_score": conf["score"],
        "confidence_tier": conf["tier"],
    }
    if should_conclude:
        updates["phase"] = GuidePhase.CONCLUDE
    else:
        updates["phase"] = GuidePhase.DETAIL_GATHER
    return updates


async def node_ask_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑥：按领域模板追问关键细节。"""
    domain = state.legal_domain
    all_questions = DOMAIN_DETAIL_TEMPLATES.get(domain, [
        "事情大概发生在什么时候？",
        "对方是个人还是公司/机构？",
        "目前有哪些证据或记录？",
    ])
    # 过滤已问过的，取前3个
    pending = [q for q in all_questions if q not in state.asked_details][:3]
    if not pending:
        # 已全部问过，直接收敛
        logger.info("节点⑥无新追问，直接结论")
        return {"phase": GuidePhase.CONCLUDE}

    domain_label = DOMAIN_LABELS.get(domain, "法律")
    issues_str = "、".join(state.confirmed_issues[:3]) or "您描述的情况"
    prompt = ASK_DETAILS_PROMPT.format(
        domain_label=domain_label,
        confirmed_issues=issues_str,
        details_to_ask="\n".join(f"- {q}" for q in pending),
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.info("节点⑥追问 | domain={} questions={}", domain, pending)
    return {
        "round": state.round + 1,
        "asked_details": state.asked_details + pending,
        "pending_ask_details": pending,
        "messages": [AIMessage(content=response.content)],
    }


async def node_parse_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦：解析用户对追问的回答，提取证据/地区/时间信息。"""
    last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    if not last_msg or not state.pending_ask_details:
        return {}
    prompt = PARSE_DETAILS_PROMPT.format(
        asked_details="\n".join(f"- {q}" for q in state.pending_ask_details),
        user_answer=last_msg,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
        new_issues = list(set(state.confirmed_issues) | set(parsed.get("new_issues", [])))
        new_evidence = list(set(state.evidence_confirmed) | set(parsed.get("evidence", [])))
        region = parsed.get("region", "") or state.region
        logger.info("节点⑦解析结果 | new_issues={} evidence={} region=",
                    parsed.get("new_issues"), parsed.get("evidence"), region)
        return {
            "confirmed_issues": new_issues,
            "evidence_confirmed": new_evidence,
            "region": region,
            "pending_ask_details": [],
        }
    except Exception as e:
        logger.warning(f"解析追问回答失败: {e}")
        return {"pending_ask_details": []}


async def node_conclude(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑧：生成五段式行动方案（理解+法条+类案+路径+行动清单）。"""
    logger.info("节点⑧生成结论 | domain={} force={}", state.legal_domain, state.force_conclude)
    domain = state.legal_domain
    region = state.region or "全国"
    evidence_checklist = _fmt_evidence_checklist(domain)
    channels_str = _fmt_channels(state.relevant_channels)
    force_note = (
        "\n> 由于信息有限，以上建议供参考。如情况复杂，建议拨打 **12348** 咨询专业律师。"
        if state.force_conclude else ""
    )
    prompt = CONCLUDE_PROMPT.format(
        confidence_guidance=tier_guidance(state.confidence_tier),
        confirmed_issues="、".join(state.confirmed_issues) or "法律问题",
        legal_domain=DOMAIN_LABELS.get(domain, domain or "法律"),
        region=region,
        evidence_confirmed="、".join(state.evidence_confirmed) or "暂未确认",
        time_warning=state.time_warning,
        law_context=state.law_context_str or "（未检索到具体条文，请参考适用法律原则）",
        case_context=state.case_context_str or "（暂无类案数据）",
        channels=channels_str,
        evidence_checklist=evidence_checklist,
        force_conclude_note=force_note,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    return {
        "phase": GuidePhase.CONCLUDE,
        "messages": [AIMessage(content=response.content)],
    }


async def node_save_record(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑨：保存咨询记录到 PostgreSQL。"""
    user_id = state.user_context.get("user_id")
    logger.info("节点⑨保存记录 | session={} domain={}", state.session_id, state.legal_domain)
    await save_guide_record(
        user_id=user_id,
        session_id=state.session_id,
        domain=state.legal_domain,
        issues=state.confirmed_issues,
        db=deps.db_session,
    )
    return {"phase": GuidePhase.END}


# ════════════════════════════════════════════════════════════════════════
# 路由函数
# ════════════════════════════════════════════════════════════════════════

def route_dispatcher(state: GuideState) -> str:
    # 首轮先加载历史上下文；后续轮直接进入紧急检测关卡。
    # 无论哪一轮，最终都会流经 check_urgency（多轮高危熔断的关键）。
    if state.round == 0:
        return "load_context"
    return "check_urgency"


def route_after_urgency(state: GuideState) -> str:
    # CRITICAL 已在节点内置 phase=END，直接熔断。
    if state.phase == GuidePhase.END:
        return END
    # 非首轮且正在等待追问回答 → 先解析回答，再检索。
    if state.pending_ask_details:
        return "parse_details"
    return "extract_issues"


def route_after_extract(state: GuideState) -> str:
    if state.phase == GuidePhase.ISSUE_SEARCH:
        return "issue_search"
    return "clarify"


def route_after_search(state: GuideState) -> str:
    if state.phase == GuidePhase.CONCLUDE:
        return "conclude"
    return "ask_details"


def route_after_ask(state: GuideState) -> str:
    if state.phase == GuidePhase.CONCLUDE:
        return "conclude"
    return END  # 等待用户回答


def route_after_parse(state: GuideState) -> str:
    return "issue_search"


def route_after_conclude(state: GuideState) -> str:
    if state.phase == GuidePhase.END:
        return END
    return "save_record"


# ════════════════════════════════════════════════════════════════════════
# 图的组装
# ════════════════════════════════════════════════════════════════════════

def build_guide_graph(deps: GuideDeps):
    """构建并编译法律指引 StateGraph，deps 通过闭包注入。"""
    async def _load_context(s):    return await node_load_context(s, deps)
    async def _check_urgency(s):   return await node_check_urgency(s, deps)
    async def _extract_issues(s):  return await node_extract_issues(s, deps)
    async def _clarify(s):         return await node_clarify(s, deps)
    async def _issue_search(s):    return await node_issue_search(s, deps)
    async def _ask_details(s):     return await node_ask_details(s, deps)
    async def _parse_details(s):   return await node_parse_details(s, deps)
    async def _conclude(s):        return await node_conclude(s, deps)
    async def _save_record(s):     return await node_save_record(s, deps)

    graph = StateGraph(GuideState)
    graph.add_node("dispatcher",     lambda s: {})
    graph.add_node("load_context",   _load_context)
    graph.add_node("check_urgency",  _check_urgency)
    graph.add_node("extract_issues", _extract_issues)
    graph.add_node("clarify",        _clarify)
    graph.add_node("issue_search",   _issue_search)
    graph.add_node("ask_details",    _ask_details)
    graph.add_node("parse_details",  _parse_details)
    graph.add_node("conclude",       _conclude)
    graph.add_node("save_record",    _save_record)

    graph.set_entry_point("dispatcher")
    graph.add_edge("load_context", "check_urgency")
    graph.add_edge("clarify",      END)
    graph.add_edge("save_record",  END)

    graph.add_conditional_edges("dispatcher",     route_dispatcher,
        {"load_context": "load_context", "check_urgency": "check_urgency"})
    graph.add_conditional_edges("check_urgency",  route_after_urgency,
        {"parse_details": "parse_details", "extract_issues": "extract_issues", END: END})
    graph.add_conditional_edges("extract_issues", route_after_extract,
        {"issue_search": "issue_search", "clarify": "clarify"})
    graph.add_conditional_edges("issue_search",   route_after_search,
        {"conclude": "conclude", "ask_details": "ask_details"})
    graph.add_conditional_edges("ask_details",    route_after_ask,
        {"conclude": "conclude", END: END})
    graph.add_conditional_edges("parse_details",  route_after_parse,
        {"issue_search": "issue_search"})
    graph.add_conditional_edges("conclude",       route_after_conclude,
        {"save_record": "save_record", END: END})
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

    logger.info("▶ run_guide | session={} round={} user_id={}", thread_id, state.round, user_id)

    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(state, config=config)

    new_state = GuideState(**result) if isinstance(result, dict) else result

    reply = ""
    for msg in reversed(new_state.messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    logger.info("◀ run_guide 完成 | session={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    return reply, new_state


def build_guide_deps(db_session=None) -> GuideDeps:
    """构建法律指引依赖注入容器。供 guide_agent 工具和 API 路由共用。"""
    llm = ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
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
