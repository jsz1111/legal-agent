"""公民法律指引 LangGraph 状态机。"""
from __future__ import annotations

import asyncio
import json
import re
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
from src.agents.legal_guide.authority_registry import format_domain_authority_summary
from src.agents.legal_guide.evidence_rules import (
    format_evidence_source,
    resolve_state_evidence_checklist,
)
from src.agents.legal_guide.followup_catalog import (
    EvidenceFollowup,
    FactFollowup,
    assess_evidence_answer,
    assess_fact_answer,
    assess_initial_evidence,
    assess_initial_facts,
    evidence_effective_count,
    evidence_followups,
    evidence_rule_resolved,
    fact_followups,
    fact_rule_resolved,
    find_evidence_followup,
    find_fact_followup,
    format_evidence_assessments,
    format_fact_assessments,
)
from src.agents.legal_guide.prompts import (
    URGENCY_CHECK_PROMPT, CLARIFY_PROMPT,
    PARSE_DETAILS_PROMPT, CONCLUDE_PROMPT, SELF_REVIEW_PROMPT,
    DOC_TYPE_MAP,
    DOMAIN_DETAIL_TEMPLATES, DOMAIN_LABELS,
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


def _long_term_memories(state: GuideState, limit: int = 5) -> list[str]:
    """返回已由 Supervisor/Worker 检索出的相关长期记忆，限制长度避免污染提示词。"""
    memories = state.user_context.get("long_term_memories") or []
    return [str(item).strip()[:300] for item in memories[:limit] if str(item).strip()]


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
        "按现有信息", "按现在这些", "最终建议", "最终方案", "请收敛",
        "只能说这些", "只说这些", "没有更多信息", "没有更多证据", "没更多信息",
    )
    total_rounds = state.total_rounds + 1
    return {
        **context_updates,
        "round": state.round + 1,
        "total_rounds": total_rounds,
        "wants_conclude": state.wants_conclude or any(p in last_msg for p in conclude_phrases),
        "force_conclude": state.force_conclude or total_rounds >= settings.GUIDE_MAX_TOTAL_ROUNDS,
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
    """节点③：三层法律问题标准化。拼接最近3条人类消息，避免多轮澄清后上下文丢失。

    同时提取用户主动提供的证据（首轮优化）：如果用户在描述问题时已经提到证据，直接提取，
    避免后续节点重复追问已有证据。
    """
    human_msgs = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {}
    # 取最近3条拼接，给LLM更完整的上下文
    combined_input = "\n".join(human_msgs[-3:])
    memories = _long_term_memories(state)
    normalizer_input = combined_input
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

    new_confirmed = _merge(state.confirmed_issues, result["standard"])
    # 已升级为标准术语的口语词，从口语池剔除，避免同一件事在两个池里各出现一次
    promoted = set(result["term_map"])
    new_unmatched = [
        x for x in _merge(state.unmatched_issues, result["colloquial"])
        if x not in promoted
    ]
    domain = result["domain"] or state.legal_domain
    new_term_map = {**state.term_map, **result["term_map"]}
    extracted_facts = result.get("collected_facts") or []
    new_facts = _merge(state.collected_facts, extracted_facts)
    fact_records = assess_initial_facts(extracted_facts, state.fact_records)

    # 提取首轮主动提供的证据（避免重复追问）- 使用模糊匹配
    new_evidence = state.evidence_confirmed.copy()
    region_extracted = (
        normalize_region_name(state.region)
        or normalize_region_name(result.get("region", ""))
        or extract_supported_region(combined_input)
    )
    time_info = state.time_info or result.get("time_info", "")

    if state.round <= 1 and len(human_msgs) == 1:  # 仅首轮生效
        # 每组模式对应一个规范证据名称
        evidence_patterns = [
            (["劳动合同", "合同书", "劳动协议", "签合同", "签了合同"], "劳动合同"),
            (["工资条", "工资单", "薪资条", "薪资单", "工资表"], "工资条"),
            (["转账记录", "银行流水", "转账截图", "流水", "银行转账"], "转账记录"),
            (["打卡记录", "考勤", "考勤表", "考勤记录", "打卡"], "考勤记录"),
            (["聊天记录", "微信记录", "聊天截图", "微信聊天"], "聊天记录"),
            (["照片", "图片", "拍照"], "照片"),
            (["视频", "录像"], "视频"),
            (["录音", "通话录音", "电话录音"], "录音"),
            (["收据", "发票", "票据"], "收据"),
            (["工作证", "工牌", "员工卡"], "工作证"),
            (["合同", "协议"], "合同"),
        ]

        for patterns, canonical_name in evidence_patterns:
            affirmed = False
            for pattern in patterns:
                start = combined_input.find(pattern)
                if start < 0:
                    continue
                prefix = combined_input[max(0, start - 4):start]
                if not any(negative in prefix for negative in ("没有", "没", "无", "不存在")):
                    affirmed = True
                    break
            if affirmed:
                if canonical_name not in new_evidence:
                    new_evidence.append(canonical_name)

        # 提取时间信息（判断是否提到时间）
        time_patterns = [
            r"\d+个?月",
            r"\d+年",
            r"(去年|前年|今年|上月|上个月)",
            r"\d{4}年\d{1,2}月",
            r"(一|二|三|四|五|六|七|八|九|十|几)个月",
        ]
        if not time_info:
            for pattern in time_patterns:
                match = re.search(pattern, combined_input)
                if match:
                    time_info = match.group(0)
                    break

        if new_evidence:
            logger.info("节点③提取首轮证据 | evidence={}", new_evidence)

    logger.info(
        "节点③结果 | standard={} colloquial={} domain={} evidence={} region={} time_info={}",
        new_confirmed, new_unmatched, domain, new_evidence, region_extracted, time_info,
    )

    updates = {
        "unmatched_issues": new_unmatched,
        "term_map": new_term_map,
        "collected_facts": new_facts,
        "fact_records": fact_records,
        "legal_domain": domain,
    }

    if new_evidence:
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

    # 改进时间检测：检查用户输入中是否包含时间相关表达
    import re
    combined_input = "\n".join(
        m.content for m in state.messages if isinstance(m, HumanMessage)
    )
    time_patterns = [
        r"\d+个?月",
        r"\d+年",
        r"(去年|前年|今年|上月|上个月|最近)",
        r"\d{4}年\d{1,2}月",
        r"(一|二|三|四|五|六|七|八|九|十|几)个月",
    ]
    time_known = (
        bool(state.time_warning) or
        bool(state.time_info) or
        any(re.search(pattern, combined_input) for pattern in time_patterns)
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
    # ── 双查询构建：Dense 与 Sparse 走不同的料 ────────────────────────────────
    # sparse_query（BM25，字面匹配）：只用标准术语。BM25 对口语零召回，
    #   混入领域标签/事实描述只会稀释 IDF 权重。
    # question（Dense/HyDE，语义匹配）：标准术语 + 口语 + 已确认证据。
    #   向量能容忍语义 gap，信息越具体召回越贴案情。
    domain = state.legal_domain
    sparse_query = "、".join(state.confirmed_issues)

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
    if state.collected_facts:
        dense_parts.append("；".join(state.collected_facts[-6:]))
    if state.time_info:
        dense_parts.append(state.time_info)
    if state.region:
        dense_parts.append(state.region)
    memories = _long_term_memories(state)
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
    if deps.db_session and effective_domain and state.confirmed_issues:
        from src.agents.legal_knowledge.statute_rag import search_statutes_pg_fallback
        try:
            pg_hits = await search_statutes_pg_fallback(
                effective_domain,
                state.confirmed_issues,
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
    if scored_state.wants_conclude and scored_state.retrieval_completed:
        logger.info("节点⑤复用上一轮检索结果 | 用户已要求按现有信息收敛")
        retrieval_updates = {}
    else:
        retrieval_updates = await node_retrieve(scored_state, deps)
    assessed_state = scored_state.model_copy(update=retrieval_updates)
    _, force = should_conclude(
        assessed_state,
        max_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS,
    )
    return {
        **score_updates,
        **retrieval_updates,
        "force_conclude": state.force_conclude or force,
    }


def _remaining_fact_rules(state: GuideState) -> list[FactFollowup]:
    return [
        rule for rule in fact_followups(state.legal_domain)
        if rule.id not in state.asked_followup_ids
        and rule.question not in state.asked_details
        and not fact_rule_resolved(rule, state)
    ]


def _remaining_fact_questions(state: GuideState) -> list[str]:
    return [rule.question for rule in _remaining_fact_rules(state)]


def _remaining_evidence_rules(state: GuideState) -> list[EvidenceFollowup]:
    known = state.evidence_confirmed + state.evidence_unavailable
    return [
        rule for rule in evidence_followups(state.legal_domain)
        if rule.id not in state.asked_followup_ids
        and rule.question not in state.asked_details
        and not evidence_rule_resolved(rule, known)
    ]


def _remaining_evidence_questions(state: GuideState) -> list[str]:
    return [rule.item for rule in _remaining_evidence_rules(state)]


_OPTIONAL_EVIDENCE_MARKERS = ("如有", "若有", "如涉及", "可选", "进入诉讼阶段时")

_EVIDENCE_ALIAS_GROUPS = (
    ("身份证明", "身份证", "主体资格", "营业执照", "工商信息"),
    ("劳动合同", "书面劳动合同", "录用通知", "入职登记", "工作证", "工牌", "劳动关系"),
    ("工资单", "工资条", "银行流水", "工资流水", "转账记录", "社会保险", "社保", "工资标准", "工资支付"),
    ("考勤", "打卡", "排班", "加班"),
    ("聊天记录", "微信", "短信", "邮件", "沟通记录"),
)


def _is_core_followup_evidence(item: str) -> bool:
    """Only ask for generally necessary evidence during the conversation.

    Conditional items still remain in the authoritative final checklist, but
    asking about litigation-stage or injury-only documents in an ordinary wage
    dispute creates unnecessary turns and is misleading to users.
    """
    if any(marker in item for marker in _OPTIONAL_EVIDENCE_MARKERS):
        return False
    # Identity and counterparty registration materials belong in the final
    # filing checklist; they rarely improve the initial merits assessment.
    return not any(marker in item for marker in ("身份证明", "主体资格"))


def _is_relevant_followup_evidence(item: str, state: GuideState) -> bool:
    """Drop channel-specific evidence that does not match the accumulated facts."""
    recent_user_text = "".join(
        str(message.content)[-500:]
        for message in state.messages[-6:]
        if isinstance(message, HumanMessage)
    )
    context = "".join(state.confirmed_issues + state.collected_facts) + recent_user_text
    is_dine_in_food_case = (
        state.legal_domain == "consumer_market"
        and any(keyword in context for keyword in ("餐馆", "餐厅", "饭店", "就餐", "饭里", "菜里"))
    )
    if is_dine_in_food_case and any(
        keyword in item for keyword in ("物流", "签收", "交付和验收", "验收单", "快递")
    ):
        return False
    return True


def _evidence_item_is_resolved(item: str, known_items: list[str]) -> bool:
    """Treat common user wording as covering the matching official checklist row."""
    if item in known_items:
        return True
    for known in known_items:
        if not known:
            continue
        if known in item or item in known:
            return True
        for aliases in _EVIDENCE_ALIAS_GROUPS:
            if any(alias in item for alias in aliases) and any(alias in known for alias in aliases):
                return True
    return False


def _fact_question_is_resolved(question: str, state: GuideState) -> bool:
    """Skip fact prompts already answered elsewhere on the state blackboard."""
    if "距离事件发生多久" in question and state.time_info:
        return True
    user_context = "".join(
        str(message.content)
        for message in state.messages[-8:]
        if isinstance(message, HumanMessage)
    )
    accumulated_context = "".join(state.collected_facts) + user_context
    if "商家是否已经回应" in question:
        has_merchant = any(token in accumulated_context for token in ("商家", "店家", "店里", "老板"))
        has_response = any(
            token in accumulated_context
            for token in ("回应", "回复", "答应", "拒绝", "退钱", "退款", "赔偿", "承认", "不管")
        )
        if has_merchant and has_response:
            return True
    evidence_like = ("合同", "工资单", "工资条", "银行流水", "打卡", "考勤")
    if any(token in question for token in evidence_like):
        known = state.evidence_confirmed + state.evidence_unavailable
        return _evidence_item_is_resolved(question, known)
    return False


def _next_ask_type(state: GuideState) -> str:
    """根据置信档位选择下一类追问，并在任一清单耗尽时自动回退。"""
    if (
        state.ask_rounds >= settings.GUIDE_MAX_ASK_ROUNDS
        or state.ask_rounds >= settings.GUIDE_SOFT_ASK_ROUNDS
        or state.consecutive_low_info_answers >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
    ):
        return ""

    facts_available = (
        state.facts_rounds < settings.GUIDE_MAX_FACT_ROUNDS
        and bool(_remaining_fact_questions(state))
    )
    evidence_available = (
        state.evidence_rounds < settings.GUIDE_MAX_EVIDENCE_ROUNDS
        and bool(_remaining_evidence_questions(state))
    )

    if state.confidence_tier == "LOW" and facts_available and state.facts_rounds < 2:
        return "facts"
    if evidence_available:
        return "evidence"
    if facts_available:
        return "facts"
    if evidence_available:
        return "evidence"
    return ""


def _format_followup_reply(
    state: GuideState,
    question: str,
    *,
    ask_type: str,
    reason: str,
    answer_hint: str = "",
) -> str:
    """每轮只问一个关键问题，后台评估不增加用户的表单负担。"""
    domain_label = DOMAIN_LABELS.get(state.legal_domain, "法律")
    issues = "、".join(state.confirmed_issues[:2]) or f"{domain_label}问题"
    question = question.strip()
    if "？" not in question and "?" not in question:
        question += "？"
    lines = [f"已定位到与“{issues}”相关的法律方向。我只再确认一个关键点：", question]
    if ask_type == "evidence":
        lines.append(f"这项材料主要用于{reason}。没有或不确定都可以直接说，我会给替代办法。")
    else:
        lines.append(f"这是为了{reason}。{answer_hint or '不清楚时可以说大概情况或“不知道”。'}")
    return "\n".join(lines)


async def node_ask_facts(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑥辅助分支：追问时间、金额、合同类型等构成要件。"""
    domain = state.legal_domain
    pending_rules = _remaining_fact_rules(state)
    if not pending_rules:
        logger.info("节点⑥事实分支无细节模板，跳过 | domain={}", domain)
        return {}
    if state.facts_rounds >= settings.GUIDE_MAX_FACT_ROUNDS:
        logger.info("节点⑥事实分支细节已齐 | facts_rounds={}", state.facts_rounds)
        return {}
    rule = pending_rules[0]
    to_ask = [rule.question]
    reply = _format_followup_reply(
        state,
        rule.question,
        ask_type="facts",
        reason=rule.why,
        answer_hint=rule.answer_hint,
    )
    logger.info("节点⑥事实分支追问 | domain={} facts_rounds={} questions={}",
                domain, state.facts_rounds, to_ask)

    return {
        "phase": GuidePhase.DETAIL_GATHER,
        "ask_rounds": state.ask_rounds + 1,
        "facts_rounds": state.facts_rounds + 1,
        "asked_details": state.asked_details + to_ask,
        "pending_ask_details": to_ask,
        "pending_ask_type": "facts",
        "asked_followup_ids": state.asked_followup_ids + [rule.id],
        "pending_followup_ids": [rule.id],
        "messages": [AIMessage(content=reply)],
    }


async def node_ask_evidence(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑥辅助分支：追问聊天记录、合同、转账凭证等证据材料。"""
    domain = state.legal_domain
    pending_rules = _remaining_evidence_rules(state)
    if not pending_rules:
        logger.info("节点⑥证据分支已齐 | evidence_rounds={}", state.evidence_rounds)
        return {}
    if state.evidence_rounds >= settings.GUIDE_MAX_EVIDENCE_ROUNDS:
        return {}
    rule = pending_rules[0]
    to_ask = [rule.question]
    reply = _format_followup_reply(
        state,
        rule.question,
        ask_type="evidence",
        reason=rule.purpose,
    )
    logger.info("节点⑥证据分支追问 | domain={} evidence_rounds={} questions={}",
                domain, state.evidence_rounds, to_ask)

    return {
        "phase": GuidePhase.DETAIL_GATHER,
        "ask_rounds": state.ask_rounds + 1,
        "evidence_rounds": state.evidence_rounds + 1,
        "asked_details": state.asked_details + to_ask,
        "pending_ask_details": to_ask,
        "pending_ask_type": "evidence",
        "asked_followup_ids": state.asked_followup_ids + [rule.id],
        "pending_followup_ids": [rule.id],
        "messages": [AIMessage(content=reply)],
    }


async def node_ask_followup(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑥：LOW 优先补事实，MEDIUM/HIGH 优先补证据。"""
    ask_type = _next_ask_type(state)
    if ask_type == "facts":
        return await node_ask_facts(state, deps)
    if ask_type == "evidence":
        return await node_ask_evidence(state, deps)
    logger.info("统一追问节点无可追问项 | tier={}", state.confidence_tier)
    return {}


async def node_parse_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦：解析用户对追问的回答，提取证据/地区/时间信息。

    若用户本轮没有回答而是反问，则不抽取任何信息、保留 pending_ask_details，
    把反问记入 deferred_questions，并原样重述待答问题（不消耗 ask_rounds）。
    """
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
    except Exception as e:
        # 解析失败：清空 pending 避免卡在重述死循环，代价是本轮回答丢失
        logger.warning("节点⑦解析追问回答失败，丢弃本轮抽取 | err={} raw={}", e, response.content[:200])
        return {"pending_ask_details": [], "pending_ask_type": "", "pending_followup_ids": []}

    user_question = (parsed.get("user_question") or "").strip()
    is_answer = parsed.get("is_answer", True)

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
    parsed_new_issues = parsed.get("new_issues") or []
    if is_multimodal_evidence:
        parsed_new_issues = [
            item for item in parsed_new_issues
            if not any(marker in item for marker in ("可能", "疑似", "或许", "推测"))
        ]
    new_issues = _merge(state.confirmed_issues, parsed_new_issues)
    parsed_facts = parsed.get("collected_facts") or []
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
    new_facts = _merge(state.collected_facts, parsed_facts)
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
    new_evidence = _merge(state.evidence_confirmed, present_evidence)
    new_unverified = _merge(state.evidence_unverified, unverified_evidence)
    unavailable = _merge(state.evidence_unavailable, parsed.get("evidence_unavailable") or [])
    new_adverse = _merge(state.adverse_facts, parsed.get("adverse_facts") or [])

    fact_records = dict(state.fact_records)
    evidence_assessments = assess_initial_evidence(
        [item for item in present_evidence if item not in state.evidence_confirmed],
        state.evidence_assessments,
    )
    low_info_answer = False
    negative_markers = ("没有", "没留", "没保存", "找不到", "拿不出", "不清楚", "不知道", "不记得")
    answer_is_negative = any(marker in last_msg for marker in negative_markers)
    for rule_id in state.pending_followup_ids:
        if state.pending_ask_type == "facts":
            rule = find_fact_followup(state.legal_domain, rule_id)
            if rule:
                record = assess_fact_answer(rule, last_msg, fact_records.get(rule_id))
                fact_records[rule_id] = record
                low_info_answer = low_info_answer or record["status"] == "unknown"
        elif state.pending_ask_type == "evidence":
            rule = find_evidence_followup(state.legal_domain, rule_id)
            if not rule:
                continue
            unavailable_items = parsed.get("evidence_unavailable") or []
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
                new_evidence = _merge(new_evidence, [rule.item])
            low_info_answer = low_info_answer or record["availability"] in {"unavailable", "unclear"}

    if not state.pending_followup_ids:
        if state.pending_ask_type == "facts":
            low_info_answer = any(marker in last_msg for marker in ("不知道", "不清楚", "不记得", "记不清"))
        elif state.pending_ask_type == "evidence":
            low_info_answer = answer_is_negative
    consecutive_low_info = state.consecutive_low_info_answers + 1 if low_info_answer else 0
    force_low_info_conclusion = consecutive_low_info >= settings.GUIDE_MAX_LOW_INFO_ANSWERS
    region = (
        normalize_region_name(parsed.get("region", ""))
        or normalize_region_name(state.region)
        or extract_supported_region(last_msg)
    )
    time_info = (parsed.get("time_info") or "").strip() or state.time_info
    logger.info("节点⑦解析结果 | type={} new_issues={} facts={} evidence={} unavailable={} adverse={} region={} time={} deferred={}",
                state.pending_ask_type,
                parsed.get("new_issues"), parsed.get("collected_facts"), parsed.get("evidence"),
                parsed.get("evidence_unavailable"), parsed.get("adverse_facts"), region, time_info, user_question)
    return {
        "confirmed_issues": new_issues,
        "collected_facts": new_facts,
        "evidence_confirmed": new_evidence,
        "evidence_unverified": new_unverified,
        "evidence_unavailable": unavailable,
        "fact_records": fact_records,
        "evidence_assessments": evidence_assessments,
        "adverse_facts": new_adverse,
        "region": region,
        "time_info": time_info,
        "pending_ask_details": [],
        "pending_ask_type": "",
        "pending_followup_ids": [],
        "consecutive_counter_questions": 0,
        "consecutive_low_info_answers": consecutive_low_info,
        "force_conclude": state.force_conclude or force_low_info_conclusion,
        "phase": GuidePhase.ISSUE_SEARCH,
        "deferred_questions": state.deferred_questions + ([user_question] if user_question else []),
    }


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
            if not safe_lines or safe_lines[-1] != replacement:
                safe_lines.append(replacement)
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


def _sanitize_forced_followups(reply: str) -> str:
    """强制收敛后移除要求用户继续补充并等待下一版方案的尾段。"""
    phrases = (
        "请补充以下关键信息", "请继续补充", "补充上述信息后", "补充后我将",
        "我将为您生成更精准", "请回答以下问题", "还需要您补充",
        "请务必先回答", "先回答上面",
    )
    positions = [reply.find(phrase) for phrase in phrases if phrase in reply]
    if not positions:
        return reply

    position = min(positions)
    line_start = reply.rfind("\n", 0, position) + 1
    line_end = reply.find("\n", position)
    if line_end < 0:
        line_end = len(reply)
    current_line = reply[line_start:line_end]
    line_only = bool(re.match(r"\s*\d+[.、)]", current_line)) and "以下" not in current_line
    if line_only:
        note = (
            "> 当前仍有事实和证据缺口，因此胜算只能作初步判断。"
            "您无需继续回答也可先按行动清单执行，并拨打 12348 核验。"
        )
        return reply[:line_start] + note + reply[line_end:]

    section_starts = [
        reply.rfind(marker, 0, position)
        for marker in ("\n**", "\n##", "\n【")
    ]
    start = max(section_starts)
    if start < 0:
        start = position
    end = reply.find("\n---", position)
    if end < 0:
        end = len(reply)
    note = (
        "\n\n> 当前仍有事实和证据缺口，因此胜算只能作初步判断。"
        "您无需继续回答也可先按上述行动清单保存证据、咨询 12348，并在时效内启动适当程序。"
    )
    return reply[:start].rstrip() + note + reply[end:]


def _sanitize_evidence_overconfidence(reply: str) -> str:
    """Remove deterministic overclaims that remain unsafe at any confidence tier."""
    replacements = (
        ("铁证如山", "证明力仍需结合原始载体核验"),
        ("直接铁证", "较有力但仍需核验的证据"),
        ("铁证", "较有力但仍需核验的证据"),
        ("证据链完整", "现有材料可以相互印证，但仍有核验缺口"),
        ("证据链相对完整", "现有材料覆盖面较广，但仍需核验原件和内容"),
        ("已经足够证明", "可用于初步证明"),
        ("完全可以证明", "可用于初步证明，但仍需结合原始载体核验"),
        ("胜诉希望很大", "具备一定主张基础，结果仍取决于证据核验和受理机关判断"),
        ("可能性非常大", "具备一定主张基础，最终结果仍取决于证据和审理判断"),
        ("几乎等于承认", "可作为对相关事实的承认线索"),
        ("这几乎不可能", "这仍需双方结合证据说明"),
        ("法院通常对消费者较为宽容", "是否获得支持仍以具体证据和审理结果为准"),
        ("法定最高赔偿（1000元）", "符合适用条件时可主张的惩罚性赔偿"),
        ("您已经完成了主要的举证责任", "您现有材料可用于初步举证，但原件和内容仍需核验"),
        (
            "举证责任主要在公司，公司需要证明它“没有拖欠”或“有正当理由拖欠”",
            "公司也应就工资支付情况提供由其掌握的记录，但您仍需先证明劳动关系、工资标准和欠薪事实",
        ),
    )
    for old, new in replacements:
        reply = reply.replace(old, new)
    return reply


def _sanitize_user_facing_tone(reply: str) -> str:
    """Replace unnecessarily harsh wording without weakening the legal risk message."""
    replacements = (
        ("这是非常致命的", "这会明显增加举证难度"),
        ("非常致命", "会明显增加举证难度"),
        ("证明力严重不足", "现阶段证明力有限，仍需结合内容补强"),
        ("完全错误", "并不准确"),
        ("够不够硬", "能否形成初步证明"),
    )
    for old, new in replacements:
        reply = reply.replace(old, new)
    return reply


def _sanitize_labor_procedure_claims(reply: str, state: GuideState) -> str:
    """Keep common labor remedies conditional when the model overstates them."""
    context = "".join(state.confirmed_issues + state.collected_facts)
    if state.legal_domain != "labor_social_security" and not any(
        word in context for word in ("劳动", "工资", "欠薪")
    ):
        return reply

    safe_compensation = (
        "可依法提出有证据支持的劳动争议请求；加付赔偿须先满足劳动行政部门责令限期支付后"
        "仍逾期不支付等法定条件，不能作为劳动仲裁当然支持的请求"
    )
    percentage_pattern = (
        r"(?:50\s*%\s*(?:[-—至~]|到)\s*100\s*%|"
        r"百分之五十以上百分之一百以下|欠薪数额的一半到一倍)"
    )
    lines: list[str] = []
    section = ""
    for line in reply.splitlines():
        if header := re.search(r"【([^】]+)】", line):
            section = header.group(1)
        is_law_section = section == "法律依据"
        if not is_law_section and "加付" in line and "赔偿" in line and (
            "仲裁" in line or "主张" in line or re.search(percentage_pattern, line)
        ):
            line = re.sub(
                rf"(?:可以|可)(?:一并)?(?:主张)?[^。；|\n]{{0,100}}?加付赔偿金?\s*[（(]?{percentage_pattern}[）)]?(?:、[^。；|\n]{{0,40}})?",
                safe_compensation,
                line,
            )
            line = re.sub(
                rf"(?:如果)?[^。；|\n]{{0,50}}?仲裁[^。；|\n]{{0,80}}?{percentage_pattern}[^。；|\n]*?加付赔偿金?",
                safe_compensation,
                line,
            )
            line = re.sub(
                rf"(?:请求|主张)[^。；|\n]{{0,40}}?{percentage_pattern}[^。；|\n]*?赔偿金?",
                "就加付赔偿问题先向劳动监察部门核实是否已具备法定前提",
                line,
            )
            if "不能作为劳动仲裁当然支持" not in line and "法定前提" not in line:
                line = re.sub(
                    rf"[^。；|\n]{{0,100}}?加付赔偿金?[^。；|\n]*",
                    safe_compensation,
                    line,
                    count=1,
                )
        if not is_law_section:
            line = line.replace(
                "除了要回工资，还有可能要到刚才说的那笔额外赔偿金",
                "可依法请求支付工资；其他请求要结合事实和受理规则核验",
            )
        lines.append(line)
    reply = "\n".join(lines)

    reply = re.sub(
        r"（注：这项权益[^）]{0,160}?从您知道权利被侵害之日起算。?）",
        "（注：是否可主张及仲裁时效起算，需结合用工状态、未签合同期间和当地裁判规则核验，不能仅按‘知道权利被侵害之日’概括。）",
        reply,
    )
    reply = re.sub(
        r"（[^（）\n]{0,40}?可主张\s*\d+\s*个月的双倍工资[^）\n]*）",
        "（是否可主张及可支持期间，需结合实际用工起止时间、书面合同签订情况和仲裁时效核验。）",
        reply,
    )
    if any(word in context for word in ("拖欠劳动报酬", "拖欠工资", "欠薪")):
        reply = re.sub(
            r"劳动仲裁的时效是\*{0,2}一年\*{0,2}[，,]?从您知道或应当知道权利被侵害之日起(?:计算)?",
            (
                "劳动争议通常适用一年仲裁时效；但劳动关系存续期间因拖欠劳动报酬发生争议，"
                "不受该一般期间限制，劳动关系终止的应自终止之日起一年内提出"
            ),
            reply,
        )
    return reply


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
        "避免使用‘致命’‘严重不足’‘完全错误’‘够不够硬’等刺激性说法。"
    )


def _compact_final_reply(reply: str, accessible: bool) -> str:
    """Remove optional repetition while preserving every required result section."""
    limit = 2200 if accessible else 3000
    if len(reply) <= limit and not accessible:
        return reply

    # The confidence badge already communicates the caveat, so a generated preamble is redundant.
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
    return re.sub(r"\n{3,}", "\n\n", reply).strip()


def _normalize_required_sections(reply: str) -> str:
    """Keep model wording compatible with the stable user-facing response contract."""
    return reply.replace("【初步方向建议】", "【维权路径比较】")


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


def _sanitize_food_compensation_certainty(reply: str, state: GuideState) -> str:
    """Keep the Food Safety Law minimum-additional-compensation rule conditional."""
    context = "".join(state.confirmed_issues + state.collected_facts)
    if "食品" not in context and "玻璃渣" not in context and "异物" not in context:
        return reply
    replacements = (
        (
            "所以您最低可以索赔 **1000元**",
            "若经核验符合第一百四十八条的适用条件，可主张增加赔偿不足1000元按1000元计算；是否支持仍由处理机关结合证据判断",
        ),
        (
            "所以您最低可以索赔1000元",
            "若经核验符合第一百四十八条的适用条件，可主张增加赔偿不足1000元按1000元计算；是否支持仍由处理机关结合证据判断",
        ),
        ("最低赔偿1000元", "符合第一百四十八条适用条件时的增加赔偿最低额规则"),
        ("经营存在习惯性问题", "曾出现过类似问题的线索，仍需进一步核验"),
    )
    for old, new in replacements:
        reply = reply.replace(old, new)
    return reply


def _ensure_case_reference(
    reply: str,
    similar_cases: list[dict],
    case_context: str = "",
) -> str:
    """模型漏写类案部分时，用结构化检索结果补一条简短摘要。"""
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
    if "类似案例" in reply:
        if cases:
            return reply
        return re.sub(
            r"\n*\*{0,2}【类似案例(?:参考)?】\*{0,2}.*?(?=\n---|\n\*{0,2}【|\Z)",
            "",
            reply,
            flags=re.S,
        ).strip()
    if not cases:
        return reply
    case = cases[0]
    title = str(case.get("title") or "相似案件").strip()
    case_number = str(case.get("case_number") or "").strip()
    summary = str(case.get("gist") or case.get("text") or "").strip()
    summary = re.sub(r"\s+", " ", summary)[:500]
    if not summary:
        return reply
    label = f"{title}（{case_number}）" if case_number else title
    block = (
        "\n\n**【类似案例参考】**\n"
        f"- **{label}**：{summary}\n"
        "- 类案仅用于说明裁判思路，不能替代对您本人证据和事实的判断。"
    )
    return reply.rstrip() + block


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
        long_term_memories="；".join(_long_term_memories(state)) or "（无相关长期记忆）",
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

    # 在回复末尾添加检索错误降级提示（如果有）
    final_reply = _normalize_required_sections(response.content)
    final_reply = _sanitize_statute_citations(final_reply, state.law_context_str)
    final_reply = _sanitize_evidence_overconfidence(final_reply)
    final_reply = _sanitize_user_facing_tone(final_reply)
    final_reply = _sanitize_unverified_evidence_assertions(
        final_reply,
        state.evidence_unverified,
    )
    final_reply = _sanitize_food_compensation_certainty(final_reply, state)
    final_reply = _sanitize_labor_procedure_claims(final_reply, state)
    if state.force_conclude or state.wants_conclude:
        final_reply = _sanitize_forced_followups(final_reply)
    final_reply = _ensure_case_reference(final_reply, state.similar_cases, state.case_context_str)
    if state.retrieval_error_note:
        final_reply += state.retrieval_error_note

    # ── 置信档位用户提示（前置标签，让用户知道回答可信度） ──────────────────
    _tier_badge = {
        "HIGH":   "**📊 当前事实和法律依据较充分，可作为行动参考；原始证据仍需核对。**",
        "MEDIUM": "**📊 基本法律依据已找到，但信息尚有缺口，请结合实际情况判断。**",
        "LOW":    "**📊 知识库检索到的法律依据有限，以下方案仅供初步参考，重要决策前请拨打 12348 咨询专业律师。**",
    }
    badge = _tier_badge.get(state.confidence_tier or "", "")
    if badge:
        final_reply = badge + "\n\n" + final_reply

    final_reply = _compact_final_reply(final_reply, compact_mode)

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
        if not compact_mode or len(final_reply) <= 2000:
            final_reply += (
                f"\n\n---\n📄 **需要参考文书？** {low_note}"
                f"如需生成{doc_type}草稿，请回复「生成文书」。"
            )

    final_reply = _compact_final_reply(final_reply, compact_mode)

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
    if state.pending_ask_details:
        return "parse_details"
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
    """信息足够或达到上限时收敛，否则只输出一条针对性追问。"""
    should_stop, _ = should_conclude(
        state,
        max_rounds=settings.GUIDE_MAX_TOTAL_ROUNDS,
    )
    if state.force_conclude or should_stop:
        return "conclude"
    if _next_ask_type(state):
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
        {"parse_details": "parse_details", "extract_issues": "extract_issues", END: END})
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
