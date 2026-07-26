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
from src.agents.legal_guide.formatters import fmt_channels, fmt_evidence_checklist, is_doc_request
from src.agents.legal_guide.doc_generator import generate_legal_doc
from src.agents.legal_guide.prompts import (
    URGENCY_CHECK_PROMPT, CLARIFY_PROMPT, ASK_DETAILS_PROMPT,
    PARSE_DETAILS_PROMPT, CONCLUDE_PROMPT, SELF_REVIEW_PROMPT,
    DOC_TYPE_MAP,
    DOMAIN_DETAIL_TEMPLATES, EVIDENCE_TEMPLATES, DOMAIN_LABELS, GENERIC_EVIDENCE,
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
    """节点③：三层法律问题标准化。拼接最近3条人类消息，避免多轮澄清后上下文丢失。

    同时提取用户主动提供的证据（首轮优化）：如果用户在描述问题时已经提到证据，直接提取，
    避免后续节点重复追问已有证据。
    """
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

    # 提取首轮主动提供的证据（避免重复追问）- 使用模糊匹配
    new_evidence = state.evidence_confirmed.copy()
    region_extracted = state.region
    time_known = False

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
            (["证明", "证明材料", "证明文件"], "证明"),
            (["买了", "购买", "订购"], "合同"),  # 新增：购买行为暗示有合同
            (["合同", "协议"], "合同"),
        ]

        for patterns, canonical_name in evidence_patterns:
            # 只要任一模式匹配，就记录规范名称
            if any(pattern in combined_input for pattern in patterns):
                if canonical_name not in new_evidence:
                    new_evidence.append(canonical_name)

        # 提取地区信息
        import re
        if not region_extracted:
            region_patterns = [
                r"在(.{2,6}?)(?:工作|上班|生活|居住)",  # 在北京工作
                r"在(.{2,6}?)(?:[,，。\s]|$)",          # 在上海（句尾或标点前）
                r"(.{2,6}?)市",                        # 北京市
                r"(.{2,6}?)(?:区|县)",                  # 朝阳区
                r"(.{2,6}?)的公司",                     # 上海的公司
                r"(.{2,6}?)地区",                       # 华北地区
            ]
            for pattern in region_patterns:
                match = re.search(pattern, combined_input)
                if match:
                    candidate = match.group(1).strip()
                    # 过滤掉太短或明显不是地名的结果
                    if len(candidate) >= 2 and candidate not in ["公司", "我们", "现在", "以前"]:
                        region_extracted = candidate
                        logger.info("节点③提取地区 | region={}", region_extracted)
                        break

        # 提取时间信息（判断是否提到时间）
        time_patterns = [
            r"\d+个?月",
            r"\d+年",
            r"(去年|前年|今年|上月|上个月)",
            r"\d{4}年\d{1,2}月",
            r"(一|二|三|四|五|六|七|八|九|十|几)个月",
        ]
        if any(re.search(pattern, combined_input) for pattern in time_patterns):
            time_known = True

        if new_evidence:
            logger.info("节点③提取首轮证据 | evidence={}", new_evidence)

    logger.info(
        "节点③结果 | standard={} colloquial={} domain={} evidence={} region={} time_known={}",
        new_confirmed, new_unmatched, domain, new_evidence, region_extracted, time_known,
    )

    updates = {
        "unmatched_issues": new_unmatched,
        "term_map": new_term_map,
        "legal_domain": domain,
    }

    if new_evidence:
        updates["evidence_confirmed"] = new_evidence

    if region_extracted and not state.region:
        updates["region"] = region_extracted

    # 注意：GuideState没有time_known字段，需要通过其他方式传递给score节点
    # 这里暂时不更新，在score节点中直接检查confirmed_issues是否包含时间信息

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
        "round": state.round + 1,
        "clarify_rounds": state.clarify_rounds + 1,
        "total_rounds": state.total_rounds + 1,
        "messages": [AIMessage(content=response.content)],
    }


async def node_score(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑤：纯规则打分（打分前置、零 I/O），决定是否值得深度检索。"""
    domain = state.legal_domain
    evidence_total = len(EVIDENCE_TEMPLATES.get(domain) or GENERIC_EVIDENCE)

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
        any(re.search(pattern, combined_input) for pattern in time_patterns)
    )

    conf = score_confidence(
        confirmed_issues=state.confirmed_issues,
        evidence_confirmed=state.evidence_confirmed,
        evidence_total=evidence_total,
        domain_locked=bool(domain),
        region_known=bool(state.region),
        time_known=time_known,
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
    """节点⑥：所有档位都检索（statute+case+graph），HIGH 档额外做自省可降档。

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

    if not dense_parts:
        # 三层标准化全空且无口语池：用用户原话兜底，纯向量、不做 domain 过滤，
        # 至少给出语义相关法条，配合 LOW 档保守措辞，避免只回"信息不足"。
        raw_input = "\n".join(
            m.content for m in state.messages if isinstance(m, HumanMessage)
        )[-500:]
        dense_parts = [raw_input or "法律问题咨询"]
        domain = ""
        logger.warning("节点⑥无任何标准化产物，降级为原话全库检索 | chars={}", len(dense_parts[0]))

    question = " ".join(dense_parts)

    # domain="other" 时降级为全库检索：不过滤 domain，让向量语义兜底
    # （LLM 识别失败时不返回 0 条，代价是召回范围变宽）
    effective_domain = domain if domain and domain != "other" else ""

    logger.info(
        "节点⑥检索 | domain={} effective={} tier={} sparse={} dense_chars={}",
        domain, effective_domain or "(全库)", state.confidence_tier,
        sparse_query or "(空,关闭BM25)", len(question),
    )

    from src.agents.legal_knowledge.statute_rag import search_statutes_raw, format_statute_context, _fetch_law_titles
    from src.agents.legal_knowledge.case_rag import search_cases

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

    case_task  = search_cases(
        question=question,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
        llm=deps.llm,
        db_session=deps.db_session,
        domain=effective_domain,
    )
    graph_task = query_laws_and_channels(effective_domain, deps.neo4j_driver)

    # 并发检索，添加超时控制（避免慢查询拖垮整体响应）
    retrieval_failures = []

    if effective_domain:
        raw_domain, raw_full, case_str, graph_result = await asyncio.gather(
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
        # 先 RRF 融合（保留 top 20 给 reranker，比直接 top 8 给更多候选）
        fused = _rrf_fuse(hits_domain, hits_full, top_n=20)
        # 再统一精排一次（两路候选放在同一上下文里比较，比各自精排后融合更准确）
        from src.agents.knowledge.reranker import rerank_docs as _rerank
        law_hits = await _rerank(question, fused, top_k=8)
        logger.info("RRF融合+精排 | domain={} full={} fused={} final={}",
                    len(hits_domain), len(hits_full), len(fused), len(law_hits))
    else:
        raw_full, case_str, graph_result = await asyncio.gather(
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

    # PG 兜底：Milvus 异常或返回空时，从 PostgreSQL 按领域+关键词检索。
    # 只传标准术语——ilike 是字面匹配，口语词（"被炒鱿鱼"）在法条正文里根本不存在。
    if not law_hits and deps.db_session and effective_domain and state.confirmed_issues:
        from src.agents.legal_knowledge.statute_rag import search_statutes_pg_fallback
        try:
            pg_hits = await search_statutes_pg_fallback(effective_domain, state.confirmed_issues, deps.db_session)
            if pg_hits:
                law_hits = pg_hits
                # PG 兜底成功，撤销失败标记（如有）
                if "法条检索" in retrieval_failures:
                    retrieval_failures.remove("法条检索")
                logger.info("PG 法条兜底成功 | hits={}", len(pg_hits))
        except Exception as pg_err:
            logger.error(f"PG 法条兜底失败: {pg_err}")

    fallback_guide = None
    if isinstance(case_str, Exception):
        if isinstance(case_str, asyncio.TimeoutError):
            logger.warning("case_rag 超时（>5s），降级跳过")
        else:
            logger.error(f"case_rag失败: {case_str}")
        case_str = ""
        retrieval_failures.append("案例检索")
    else:
        # 解析 case_rag 返回的 JSON，提取 fallback_guide
        try:
            import json
            case_data = json.loads(case_str) if isinstance(case_str, str) and case_str.startswith("{") else None
            if case_data and "fallback_guide" in case_data:
                fallback_guide = case_data["fallback_guide"]
                case_str = ""  # 空结果，不用显示 JSON 原文
        except Exception:
            pass  # case_str 是普通文本回答，保持原样
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

    # 有地区时查精确渠道
    channels = graph_result.get("channels", [])
    if state.region:
        region_channels = await query_channels_by_region(domain, state.region, deps.neo4j_driver)
        if region_channels:
            channels = region_channels

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
        "relevant_channels": channels,
        "law_context_str": law_context_formatted or "",
        "case_context_str": case_str or "",
        "retrieval_error_note": retrieval_error_note,
        "fallback_guide": fallback_guide,  # 案例检索兜底指引
        "_last_confirmed_count": len(state.confirmed_issues),  # 记录本次检索时的 issue 数量
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
                logger.warning("节点⑥自省降档 | HIGH→MID，原因: {}", concern)
                updates["confidence_tier"] = "MEDIUM"
                updates["self_review_note"] = f"\n⚠️ **降档说明**：{concern}"
        except Exception as e:
            logger.warning(f"自省失败，保持原档: {e}")

    return updates


async def node_ask_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦：按证据清单追问（改为证据驱动），上限可配置。智能跳过可选证据。

    【已废弃】此节点保留用于向后兼容，实际使用 ask_facts + ask_evidence。
    """
    from src.core.config import get_settings
    settings = get_settings()
    domain = state.legal_domain
    evidence_tpl = EVIDENCE_TEMPLATES.get(domain) or GENERIC_EVIDENCE

    # 过滤已问过的
    pending = [e for e in evidence_tpl if e not in state.asked_details]

    # 区分必需和可选证据
    OPTIONAL_KEYWORDS = ["如有", "若有", "可选"]
    required = [e for e in pending if not any(kw in e for kw in OPTIONAL_KEYWORDS)]
    optional = [e for e in pending if any(kw in e for kw in OPTIONAL_KEYWORDS)]

    # 优先问必需的,可选的只在必需的都问完后一起问（最多3个）
    if required:
        to_ask = required[:3]  # 每次最多问3个必需项
    elif optional and state.ask_rounds < settings.GUIDE_MAX_ASK_ROUNDS:
        to_ask = optional[:3]  # 必需项问完，再问可选项
    else:
        # 已全部问过，直接打分
        logger.info("节点⑦证据已齐，跳过追问")
        return {}

    if not to_ask:
        logger.info("节点⑦证据已齐，跳过追问")
        return {}

    domain_label = DOMAIN_LABELS.get(domain, "法律")
    issues_str = "、".join(state.confirmed_issues[:3]) or "您描述的情况"
    prompt = ASK_DETAILS_PROMPT.format(
        domain_label=domain_label,
        confirmed_issues=issues_str,
        details_to_ask="\n".join(f"- {e}" for e in to_ask),
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.info("节点⑦追问证据 | domain={} ask_rounds={} questions={}", domain, state.ask_rounds, to_ask)
    return {
        "round": state.round + 1,
        "ask_rounds": state.ask_rounds + 1,
        "total_rounds": state.total_rounds + 1,
        "asked_details": state.asked_details + to_ask,
        "pending_ask_details": to_ask,
        "messages": [AIMessage(content=response.content)],
    }


async def node_ask_facts(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦A：追问法律细节（时间/金额/合同类型等构成要件），用于提高检索精度。"""
    from src.core.config import get_settings
    settings = get_settings()
    domain = state.legal_domain

    # 从 DOMAIN_DETAIL_TEMPLATES 获取该领域需要追问的细节
    detail_tpl = DOMAIN_DETAIL_TEMPLATES.get(domain, [])
    if not detail_tpl:
        # 无模板，跳过细节追问
        logger.info("节点⑦A无细节模板，跳过 | domain={}", domain)
        return {}

    # 过滤已问过的
    pending = [d for d in detail_tpl if d not in state.asked_details]

    if not pending or state.facts_rounds >= 5:
        logger.info("节点⑦A法律细节已齐 | facts_rounds={}", state.facts_rounds)
        return {}

    to_ask = pending[:2]  # 每次问 2 个细节

    domain_label = DOMAIN_LABELS.get(domain, "法律")
    issues_str = "、".join(state.confirmed_issues[:3]) or "您描述的情况"

    # 已检索到的相关法条（帮助 LLM 生成法条要件相关的追问）
    law_hint = ""
    if state.law_context_str:
        # 只取前 400 字，避免 prompt 过长
        law_hint = f"\n\n【已检索到的相关法律依据（前400字）】\n{state.law_context_str[:400]}"

    # 构造追问 prompt
    prompt = f"""基于用户的{domain_label}问题（{issues_str}），需要确认以下关键细节：

{chr(10).join(f"- {d}" for d in to_ask)}{law_hint}

请用自然对话方式询问这些细节（不要机械复述清单），结合法律依据帮助用户理解为什么需要这些信息（如时效、举证要求等）。保持语气亲切、专业。"""

    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.info("节点⑦A追问法律细节 | domain={} facts_rounds={} questions={}",
                domain, state.facts_rounds, to_ask)

    return {
        "round": state.round + 1,
        "total_rounds": state.total_rounds + 1,
        "facts_rounds": state.facts_rounds + 1,
        "asked_details": state.asked_details + to_ask,
        "pending_ask_details": to_ask,
        "messages": [AIMessage(content=response.content)],
    }


async def node_ask_evidence(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑦B：追问证据材料（聊天记录、合同、转账凭证等），用于评估维权可行性。"""
    from src.core.config import get_settings
    settings = get_settings()
    domain = state.legal_domain
    evidence_tpl = EVIDENCE_TEMPLATES.get(domain) or GENERIC_EVIDENCE

    # 过滤已问过的
    pending = [e for e in evidence_tpl if e not in state.asked_details]

    # 区分必需和可选证据
    OPTIONAL_KEYWORDS = ["如有", "若有", "可选"]
    required = [e for e in pending if not any(kw in e for kw in OPTIONAL_KEYWORDS)]
    optional = [e for e in pending if any(kw in e for kw in OPTIONAL_KEYWORDS)]

    # 优先问必需的
    if required:
        to_ask = required[:3]
    elif optional and state.evidence_rounds < 5:
        to_ask = optional[:3]
    else:
        logger.info("节点⑦B证据已齐 | evidence_rounds={}", state.evidence_rounds)
        return {}

    if not to_ask:
        return {}

    domain_label = DOMAIN_LABELS.get(domain, "法律")
    issues_str = "、".join(state.confirmed_issues[:3]) or "您描述的情况"

    # 注入已检索到的法律依据（前400字），让追问与法条要件关联
    law_hint = ""
    if state.law_context_str:
        law_hint = f"\n\n【已检索到的相关法律依据（前400字）】\n{state.law_context_str[:400]}"

    prompt = f"""基于用户的{domain_label}问题（{issues_str}），需要确认以下证据材料：
{chr(10).join(f"- {e}" for e in to_ask)}{law_hint}

请用自然对话方式询问这些证据（不要机械复述清单），结合法律依据解释为什么需要这些证据材料（如举证责任、关键要件证明等）。保持语气亲切、专业，200-600字。"""

    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.info("节点⑦B追问证据 | domain={} evidence_rounds={} questions={}",
                domain, state.evidence_rounds, to_ask)

    return {
        "round": state.round + 1,
        "total_rounds": state.total_rounds + 1,
        "evidence_rounds": state.evidence_rounds + 1,
        "asked_details": state.asked_details + to_ask,
        "pending_ask_details": to_ask,
        "messages": [AIMessage(content=response.content)],
    }


async def node_parse_details(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑧：解析用户对追问的回答，提取证据/地区/时间信息。

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
        logger.warning("节点⑧解析追问回答失败，丢弃本轮抽取 | err={} raw={}", e, response.content[:200])
        return {"pending_ask_details": []}

    user_question = (parsed.get("user_question") or "").strip()
    is_answer = parsed.get("is_answer", True)

    # 用户只是反问，没有回答 → 保留待答问题，不污染证据
    if not is_answer:
        pending = state.pending_ask_details
        logger.info("节点⑧用户反问未作答，保留待答项 | question={} pending={}", user_question, pending)
        reask = "这个问题我们等下一起说清楚。先把下面几点确认一下，我才能给出准确的方案：\n" + \
                "\n".join(f"- {q}" for q in pending)
        return {
            "round": state.round + 1,
            "deferred_questions": state.deferred_questions + ([user_question] if user_question else []),
            "messages": [AIMessage(content=reask)],
            # 不动 pending_ask_details / ask_rounds / asked_details
        }

    new_issues = list(set(state.confirmed_issues) | set(parsed.get("new_issues", [])))
    new_evidence = list(set(state.evidence_confirmed) | set(parsed.get("evidence", [])))
    unavailable = list(set(state.evidence_unavailable) | set(parsed.get("evidence_unavailable", [])))
    new_adverse = list(set(state.adverse_facts) | set(parsed.get("adverse_facts", [])))
    region = parsed.get("region", "") or state.region
    logger.info("节点⑧解析结果 | new_issues={} evidence={} unavailable={} adverse={} region={} deferred={}",
                parsed.get("new_issues"), parsed.get("evidence"),
                parsed.get("evidence_unavailable"), parsed.get("adverse_facts"), region, user_question)
    return {
        "confirmed_issues": new_issues,
        "evidence_confirmed": new_evidence,
        "evidence_unavailable": unavailable,
        "adverse_facts": new_adverse,
        "region": region,
        "pending_ask_details": [],
        "deferred_questions": state.deferred_questions + ([user_question] if user_question else []),
    }


async def node_conclude(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑨：生成五段式行动方案（理解+法条+类案+路径+行动清单）。"""
    logger.info("节点⑨生成结论 | domain={} tier={}", state.legal_domain, state.confidence_tier)
    domain = state.legal_domain
    region = state.region or "全国"
    evidence_checklist = fmt_evidence_checklist(domain)
    channels_str = fmt_channels(state.relevant_channels)
    force_note = (
        "\n> 由于信息有限，以上建议供参考。如情况复杂，建议拨打 **12348** 咨询专业律师。"
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
        confirmed_issues="、".join(state.confirmed_issues) or "法律问题",
        legal_domain=DOMAIN_LABELS.get(domain, domain or "法律"),
        region=region,
        evidence_confirmed="、".join(state.evidence_confirmed) or "暂未确认",
        evidence_unavailable="、".join(state.evidence_unavailable) or "（无）",
        time_warning=state.time_warning,
        self_review_note=self_review_str,
        adverse_facts_section=adverse_facts_section,
        dialogue_snippet=dialogue_snippet,
        law_context=state.law_context_str or "（未检索到具体条文，请参考适用法律原则）",
        case_context=state.case_context_str or "（暂无类案数据）",
        channels=channels_str,
        evidence_checklist=evidence_checklist,
        force_conclude_note=force_note,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    # 在回复末尾添加检索错误降级提示（如果有）
    final_reply = response.content
    if state.retrieval_error_note:
        final_reply += state.retrieval_error_note

    # ── 置信档位用户提示（前置标签，让用户知道回答可信度） ──────────────────
    _tier_badge = {
        "HIGH":   "**📊 法律依据充分，方案可直接参考执行。**",
        "MEDIUM": "**📊 基本法律依据已找到，但信息尚有缺口，请结合实际情况判断。**",
        "LOW":    "**📊 知识库检索到的法律依据有限，以下方案仅供初步参考，重要决策前请拨打 12348 咨询专业律师。**",
    }
    badge = _tier_badge.get(state.confidence_tier or "", "")
    if badge:
        final_reply = badge + "\n\n" + final_reply

    # ── 案例未命中时，把 fallback 指引拼入回复（告知用户去哪里查案例） ──────
    if not state.case_context_str and state.fallback_guide:
        fb = state.fallback_guide
        platform = fb.get("platform", "中国裁判文书网")
        url      = fb.get("url", "https://wenshu.court.gov.cn")
        tips     = fb.get("search_tips", "")
        final_reply += (
            f"\n\n---\n📋 **未在案例库中找到相似案例**，您可前往"
            f"[{platform}]({url}) 自行检索参考：\n{tips}"
        )

    # MEDIUM/HIGH 档时提示用户可以生成参考文书
    if state.confidence_tier in ("HIGH", "MEDIUM") and state.confirmed_issues:
        doc_type = DOC_TYPE_MAP.get(state.legal_domain, "投诉信/申请书")
        final_reply += f"\n\n---\n📄 **需要参考文书？** 如需生成{doc_type}草稿，请回复「生成文书」。"

    # 根据置信度决定是否结束对话：
    # - HIGH档 或 强制收敛 → 设置 phase=END（对话结束）
    # - MEDIUM/LOW档 → 保持 CONCLUDE 状态，允许后续继续追问证据（route_after_conclude 会判断）
    should_end = (state.confidence_tier == "HIGH" or state.force_conclude)
    phase_update = GuidePhase.END if should_end else GuidePhase.CONCLUDE

    # 自动保存关键信息到长期记忆
    user_id = state.user_context.get("user_id")
    if user_id and region and region != "全国":
        try:
            from src.infra.milvus_store import get_milvus_store
            from src.infra.embedding import get_embedding_model
            store = get_milvus_store()
            embedding_model = get_embedding_model()

            # 保存用户地区信息
            memory_text = f"用户所在地区：{region}"
            await store.aput(
                namespace=("users", user_id, "memories"),
                key=f"region_{region}",
                value={"text": memory_text, "type": "user_profile"},
                embedding_model=embedding_model,
            )
            logger.info(f"已保存用户地区记忆: {region}")
        except Exception as e:
            logger.warning(f"保存长期记忆失败: {e}")

    return {
        "phase": phase_update,
        "messages": [AIMessage(content=final_reply)],
    }


async def node_generate_doc(state: GuideState, deps: GuideDeps) -> dict:
    """节点⑩：生成参考文书初稿（核心逻辑委托给 doc_generator）。

    注意：此节点只在 phase=END 时由 route_after_urgency 触发，
    因此不需要再次设置 phase=END（已经是 END 状态）。
    """
    logger.info("节点⑩生成文书 | domain={}", state.legal_domain)
    doc_type, doc = await generate_legal_doc(
        legal_domain=state.legal_domain,
        confirmed_issues=state.confirmed_issues,
        region=state.region,
        evidence_confirmed=state.evidence_confirmed,
        law_context_str=state.law_context_str,
        llm=deps.llm,
    )
    logger.info("节点⑩文书生成完成 | doc_type={} len={}", doc_type, len(doc))
    return {
        "doc_draft": doc,
        "messages": [AIMessage(content=doc)],
        # phase 保持原状态（已经是 END），避免再次触发 conclude
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


def _needs_more_evidence(state: GuideState) -> bool:
    """追问门控：证据清单有缺口且未达上限。"""
    from src.core.config import get_settings
    settings = get_settings()
    domain = state.legal_domain
    evidence_tpl = EVIDENCE_TEMPLATES.get(domain) or GENERIC_EVIDENCE
    pending = [e for e in evidence_tpl if e not in state.asked_details]
    return bool(pending) and state.ask_rounds < settings.GUIDE_MAX_ASK_ROUNDS


def route_dispatcher(state: GuideState) -> str:
    """首轮先加载历史上下文；后续轮直接进入紧急检测关卡。
    每轮用户消息都会流经 check_urgency（多轮高危熔断）。"""
    if state.round == 0:
        return "load_context"
    return "check_urgency"


def route_after_urgency(state: GuideState) -> str:
    """CRITICAL 已在节点内置 phase=END，直接熔断。
    phase=END 且用户请求文书 → generate_doc（文书生成入口）。
    非首轮且正在等待追问回答 → 先解析回答，再检索。"""
    if state.phase == GuidePhase.END:
        last_msg = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
        if is_doc_request(last_msg) and state.confirmed_issues:
            # 设置 phase 保持 END，避免文书生成后再次触发 conclude
            return "generate_doc"
        return END
    # 非首轮且正在等待追问回答 → 先解析回答
    if state.pending_ask_details:
        return "parse_details"
    return "extract_issues"


def route_after_extract(state: GuideState) -> str:
    """提取后分流：需澄清 → clarify；澄清已达上限仍无 issue → 仍走检索（原话兜底）；有 issue → 证据校验。"""
    from src.core.config import get_settings
    settings = get_settings()
    if _needs_clarify(state):
        return "clarify"
    # 澄清达上限仍无 issue：不再直接输出"信息不足"，改为打分（落 LOW 档）后仍然检索，
    # 由 node_retrieve 用用户原话做无 domain 过滤的兜底检索，至少给出相关法条
    if not state.confirmed_issues and state.clarify_rounds >= settings.GUIDE_MAX_CLARIFY_ROUNDS:
        return "score"
    # 有 issue，走证据校验
    return route_after_evidence_check(state)


def route_after_parse(state: GuideState) -> str:
    """解析用户回答后，判断是解析的细节还是证据，并分别路由。
    pending 仍在 = 用户本轮只是反问、问题已重述 → 挂起等回答，不推进流程。"""
    if state.pending_ask_details:
        return END

    # 判断当前是在追问法律细节阶段还是证据阶段
    # 如果 facts_rounds > 0 且最近在追问细节 → 走细节路由
    # 否则走证据路由
    if state.facts_rounds > state.evidence_rounds:
        return route_after_parse_facts(state)
    else:
        return route_after_parse_evidence(state)


def route_after_parse_facts(state: GuideState) -> str:
    """解析法律细节后，判断是否需要重新检索。"""
    # 用户补充了新的法律问题 → 重新标准化+检索
    if len(state.confirmed_issues) > state._last_confirmed_count:
        logger.info("路由：检测到新法律问题（{}→{}），重新标准化+检索",
                    state._last_confirmed_count, len(state.confirmed_issues))
        return "extract_issues"

    # 无新问题，重新打分（可能细节补充后置信度提升）
    logger.info("路由：无新法律问题，重新打分")
    return "score"


def route_after_parse_evidence(state: GuideState) -> str:
    """解析证据后的路由：方案A - 直接收敛输出方案。

    用户补充证据后，重新打分+检索，然后输出方案并结束。
    依赖conclude中的引导语提示用户继续补充信息（如需）。
    """
    from src.agents.legal_guide.convergence import should_conclude

    should_stop, force = should_conclude(state, max_rounds=12)

    if should_stop:
        return "conclude"

    # 重新打分+检索，输出更新后的方案
    return "score"


def route_after_evidence_check(state: GuideState) -> str:
    """证据校验共用路由：

    核心原则：只要提取到标准化问题，就必须先检索（不管有没有证据）。

    修改后的逻辑：
    1. 有标准化问题 → 先打分+检索，让用户看到法律依据
    2. 检索后根据置信度决定是否继续追问（在 route_after_conclude 中判断）

    这样确保：
    - 低智商用户（信息少）也能看到相关法律
    - 高智商用户（信息全）直接得到完整方案
    - 避免"问了半天什么都没给"的糟糕体验
    """
    # 只要有标准化问题，就应该打分+检索
    return "score"


def route_after_score(state: GuideState) -> str:
    """打分后，根据置信度和收敛判断分流。"""
    from src.agents.legal_guide.convergence import should_conclude

    # 判断是否应该收敛
    should_stop, force = should_conclude(state, max_rounds=12)

    if should_stop:
        if force:
            # 强制收敛，标记 force_conclude
            state.force_conclude = True
        return "retrieve"  # 所有档位都先检索，再决定

    # 未收敛，继续走检索流程
    return "retrieve"


def route_after_retrieve(state: GuideState) -> str:
    """检索后，始终先输出方案（根据置信度调整语气），然后判断是否继续追问。

    改进逻辑：
    - 所有档位都先 conclude（输出检索结果+方案）
    - conclude 节点根据置信度调整输出：
      - HIGH: 完整方案 → END
      - MEDIUM: 较完整方案，提示可补充证据 → END 或继续追问
      - LOW: 初步指引（明确标注"信息不足"），强烈建议补充证据 → 继续追问（如果未达上限）
    - conclude 后通过新的路由判断是否继续追问
    """
    return "conclude"


def route_after_ask(state: GuideState) -> str:
    """追问后等待用户回答（挂起）。"""
    return END


def route_after_conclude(state: GuideState) -> str:
    """conclude 输出方案后的路由：按置信度决定是否追问。

    - HIGH 或已强制结束 → save_record
    - MEDIUM → ask_evidence（补充证据提升置信度，最多3轮）
    - LOW    → ask_facts（基本事实不清，先问事实，最多3轮）
    - 超出轮次上限 → save_record
    """
    if state.confidence_tier == "HIGH" or state.force_conclude:
        return "save_record"
    if state.confidence_tier == "MEDIUM" and state.evidence_rounds < 3:
        return "ask_evidence"
    if state.confidence_tier == "LOW" and state.facts_rounds < 3:
        return "ask_facts"
    return "save_record"


# ════════════════════════════════════════════════════════════════════════
# 图的组装
# ════════════════════════════════════════════════════════════════════════

def build_guide_graph(deps: GuideDeps):
    """构建并编译法律指引 StateGraph，deps 通过闭包注入。"""
    async def _generate_doc(s):    return await node_generate_doc(s, deps)
    async def _load_context(s):    return await node_load_context(s, deps)
    async def _check_urgency(s):   return await node_check_urgency(s, deps)
    async def _extract_issues(s):  return await node_extract_issues(s, deps)
    async def _clarify(s):         return await node_clarify(s, deps)
    async def _score(s):           return await node_score(s, deps)
    async def _retrieve(s):        return await node_retrieve(s, deps)
    async def _ask_details(s):     return await node_ask_details(s, deps)
    async def _ask_facts(s):       return await node_ask_facts(s, deps)
    async def _ask_evidence(s):    return await node_ask_evidence(s, deps)
    async def _parse_details(s):   return await node_parse_details(s, deps)
    async def _conclude(s):        return await node_conclude(s, deps)
    async def _save_record(s):     return await node_save_record(s, deps)

    graph = StateGraph(GuideState)
    graph.add_node("generate_doc",   _generate_doc)
    graph.add_node("dispatcher",     lambda s: {"round": s.round + 1})
    graph.add_node("load_context",   _load_context)
    graph.add_node("check_urgency",  _check_urgency)
    graph.add_node("extract_issues", _extract_issues)
    graph.add_node("clarify",        _clarify)
    graph.add_node("score",          _score)
    graph.add_node("retrieve",       _retrieve)
    graph.add_node("ask_details",    _ask_details)   # 保留兼容
    graph.add_node("ask_facts",      _ask_facts)     # 新增：追问法律细节
    graph.add_node("ask_evidence",   _ask_evidence)  # 新增：追问证据
    graph.add_node("parse_details",  _parse_details)
    graph.add_node("conclude",       _conclude)
    graph.add_node("save_record",    _save_record)

    graph.set_entry_point("dispatcher")
    graph.add_edge("generate_doc",  END)
    graph.add_edge("load_context",  "check_urgency")
    graph.add_edge("clarify",      END)
    graph.add_edge("save_record",  END)
    graph.add_edge("ask_facts",    END)     # 追问后等待用户回答
    graph.add_edge("ask_evidence", END)     # 追问后等待用户回答

    graph.add_conditional_edges("dispatcher",     route_dispatcher,
        {"load_context": "load_context", "check_urgency": "check_urgency"})
    graph.add_conditional_edges("check_urgency",  route_after_urgency,
        {"parse_details": "parse_details", "extract_issues": "extract_issues",
         "generate_doc": "generate_doc", END: END})
    graph.add_conditional_edges("extract_issues", route_after_extract,
        {"clarify": "clarify", "conclude": "conclude", "ask_details": "ask_details", "score": "score"})
    graph.add_conditional_edges("parse_details",  route_after_parse,
        {"ask_facts": "ask_facts", "ask_evidence": "ask_evidence",
         "extract_issues": "extract_issues", "score": "score", "conclude": "conclude", END: END})
    graph.add_conditional_edges("ask_details",    route_after_ask,
        {END: END})
    graph.add_conditional_edges("score",          route_after_score,
        {"retrieve": "retrieve"})
    graph.add_conditional_edges("retrieve",       route_after_retrieve,
        {"ask_facts": "ask_facts", "ask_evidence": "ask_evidence", "conclude": "conclude"})
    graph.add_conditional_edges("conclude",       route_after_conclude,
        {"save_record": "save_record", "ask_evidence": "ask_evidence",
         "ask_facts": "ask_facts", END: END})
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
