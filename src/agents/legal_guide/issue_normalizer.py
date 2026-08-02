"""三层法律问题标准化：LLM提取 → Neo4j LegalConcept精确匹配 → Milvus语义术语替换。

  第一层：LLM 提取 + 粗标准化（口语→法律问题描述）
  第二层：Neo4j LegalConcept 节点精确匹配（命中即确认为标准术语）
  第三层：legal_term_index 语义兜底（返回 {原词: 标准术语} 映射，完成替换）

依赖 build_legal_concepts.py 先运行以填充 Neo4j 和 Milvus。
"""
from __future__ import annotations

import json
import re
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.agents.legal_guide.prompts import (
    DOMAIN_MAPPING,
    INTAKE_CLASSIFY_PROMPT,
    ISSUE_EXTRACT_PROMPT,
)
from src.agents.legal_guide.case_model import CaseFactUpdate
from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage
from src.core.config import get_settings

# DeepSeek thinking mode不依赖 tool_choice，直接使用严格 JSON 提示词。
_ISSUE_EXTRACT_FALLBACK_PROMPT = ISSUE_EXTRACT_PROMPT


class IssuesOutput(BaseModel):
    """LLM 结构化输出 Schema。"""
    issues: list[str] = Field(description="标准化的法律问题列表，无则空列表")
    domain: str = Field(description="推断的法律领域代码，如 labor_social_security")
    facts: list[str] = Field(default_factory=list, description="用户明确说出的客观案情事实")
    case_updates: list[CaseFactUpdate] = Field(default_factory=list, description="带用户原文锚点的原子案情更新")
    evidence_details: list[dict] = Field(default_factory=list, description="带用户原文锚点的证据基础属性")
    region: str = Field(default="", description="用户明确提到的地区")
    time_info: str = Field(default="", description="用户明确提到的时间信息")
    degraded: bool = Field(default=False, description="是否由非生成式降级路径产生")


_INTAKE_SECTIONS = (
    ("事情经过", "event.summary", "event"),
    ("对方及双方关系", "relationship.counterparty", "relationship"),
    ("时间、地点和金额", "case.time_place_amount", "time"),
    ("希望解决的结果", "claim.requested_outcome", "claim"),
    ("已经沟通或处理的情况", "procedure.current_status", "procedure"),
)


def _parse_intake_sections(user_input: str) -> dict[str, str]:
    """Parse only the application-owned first-turn form contract."""

    if "【首次案件材料包】" not in user_input:
        return {}
    labels = "|".join(re.escape(item[0]) for item in _INTAKE_SECTIONS)
    pattern = re.compile(
        rf"【({labels})】\s*(.+?)(?=\s*【(?:{labels})】|\s*\[本轮附件清单\]|\Z)",
        re.S,
    )
    return {
        label: " ".join(value.split()).strip()
        for label, value in pattern.findall(user_input)
        if " ".join(value.split()).strip()
    }


def _intake_case_updates(sections: dict[str, str]) -> list[CaseFactUpdate]:
    updates: list[CaseFactUpdate] = []
    for label, key, category in _INTAKE_SECTIONS:
        value = sections.get(label, "")
        if not value:
            continue
        updates.append(CaseFactUpdate(
            key=key,
            category=category,
            statement=value,
            value=value,
            certainty="asserted",
            operation="add",
            source_text=value,
        ))
    return updates


async def _extract_structured_intake(
    user_input: str,
    llm: BaseChatModel,
) -> IssuesOutput | None:
    """Use deterministic form facts and a small model call only for routing."""

    sections = _parse_intake_sections(user_input)
    if not sections:
        return None
    case_summary = "\n".join(
        f"{label}：{sections[label]}"
        for label, _, _ in _INTAKE_SECTIONS
        if label in sections
    )
    updates = _intake_case_updates(sections)
    facts = [item.statement for item in updates]
    try:
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=450),
            [SystemMessage(content=INTAKE_CLASSIFY_PROMPT.format(
                case_summary=case_summary,
            ))],
            timeout=min(get_settings().GUIDE_LLM_TIMEOUT_EXTRACT, 8.0),
            stage="intake_classification",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        data = json.loads(content)
        return IssuesOutput(
            issues=[
                str(item).strip()
                for item in data.get("issues", [])
                if str(item).strip()
            ],
            domain=str(data.get("domain") or "other"),
            facts=facts,
            case_updates=updates,
            evidence_details=[],
            region=str(data.get("region") or "").strip(),
            time_info=str(
                data.get("time_info")
                or sections.get("时间、地点和金额", "")
            ).strip(),
        )
    except Exception as exc:
        logger.warning(
            "首轮材料包分类失败，保留表单结构并进入语义降级: {}",
            exc,
        )
        narrative = sections.get("事情经过", "")
        fallback = _deterministic_issue_fallback(narrative)
        return IssuesOutput(
            issues=(
                list(fallback.issues)
                if fallback
                else ([narrative] if narrative else [])
            ),
            domain=fallback.domain if fallback else "other",
            facts=facts,
            case_updates=updates,
            evidence_details=[],
            time_info=sections.get("时间、地点和金额", ""),
            degraded=True,
        )


def _deterministic_issue_fallback(user_input: str) -> IssuesOutput | None:
    """Recover only high-precision intents when structured LLM output is unavailable."""
    normalized = "".join((user_input or "").split())
    if is_high_precision_fraud_report(normalized):
        return IssuesOutput(
            issues=["疑似网络诈骗线索"],
            domain="cyber_data_fraud",
            facts=[user_input.strip()] if user_input.strip() else [],
        )
    wage_pattern = re.compile(
        r"(?:欠|拖欠|没发|不发|不给).{0,8}(?:工资|工钱|薪水|劳动报酬)|"
        r"(?:工资|工钱|薪水|劳动报酬).{0,8}(?:欠|拖欠|没发|不发|不给)"
    )
    if wage_pattern.search(normalized):
        return IssuesOutput(
            issues=["拖欠劳动报酬"],
            domain="labor_social_security",
            facts=[user_input.strip()] if user_input.strip() else [],
        )
    if re.search(r"(?:交通事故|道路事故|相撞|碰撞).{0,30}(?:受伤|医疗费|误工费|护理费|伤残)", normalized):
        return IssuesOutput(
            issues=["交通事故人身损害赔偿争议"],
            domain="traffic_personal_injury",
            facts=[user_input.strip()] if user_input.strip() else [],
        )
    if re.search(r"(?:租赁|租房|房东|承租).{0,40}(?:押金|租金|退租|解除合同)", normalized):
        return IssuesOutput(
            issues=["房屋租赁合同履行与返还争议"],
            domain="contracts_property_housing",
            facts=[user_input.strip()] if user_input.strip() else [],
        )
    contract_nonperformance = bool(re.search(
        r"(?:合同|协议|约定|委托|代购).{0,80}(?:未交付|不交付|未退款|不退款|拒绝退款|没有退款)|"
        r"(?:支付|付款|转账).{0,80}(?:未交付|不交付|未退款|不退款|拒绝退款|没有退款)",
        normalized,
    ))
    if contract_nonperformance:
        return IssuesOutput(
            issues=["合同履行与退款争议"],
            domain="contracts_property_housing",
            facts=[user_input.strip()] if user_input.strip() else [],
        )
    return None


def is_high_precision_fraud_report(user_input: str) -> bool:
    """Recognize explicit fraud reports without deciding that a crime is established."""

    normalized = "".join(str(user_input or "").split())
    if not normalized:
        return False
    direct_markers = (
        "我被骗了", "被骗了", "诈骗了我", "骗我转账", "转账后拉黑",
        "付款后拉黑", "刷单被骗", "杀猪盘", "冒充客服骗", "冒充公检法骗",
    )
    if any(marker in normalized for marker in direct_markers):
        return True
    return bool(re.search(
        r"(?:闲鱼|二手平台|网购|网友|陌生人).{0,20}(?:转账|付款).{0,20}(?:拉黑|失联|不发货)|"
        r"(?:转账|付款).{0,20}(?:后|以后).{0,12}(?:拉黑|失联|不发货)",
        normalized,
    ))


# ── 第一层：LLM 提取 + 粗标准化 ──────────────────────────────────────────────

async def extract_legal_issues(
    user_input: str,
    llm: BaseChatModel,
    *,
    fallback_text: str = "",
) -> IssuesOutput:
    """LLM 提取+标准化法律问题，直接用 JSON prompt（DeepSeek thinking mode 不支持 tool_choice）。"""
    structured_intake = await _extract_structured_intake(user_input, llm)
    if structured_intake is not None:
        return structured_intake
    try:
        prompt = _ISSUE_EXTRACT_FALLBACK_PROMPT.format(user_input=user_input)
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=1400),
            [SystemMessage(content=prompt)],
            timeout=get_settings().GUIDE_LLM_TIMEOUT_EXTRACT,
            stage="issue_extraction",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        data = json.loads(content)
        result = IssuesOutput(
            issues=[i for i in data.get("issues", []) if i],
            domain=data.get("domain", "other") or "other",
            facts=[item for item in data.get("facts", []) if item],
            case_updates=[CaseFactUpdate.model_validate(item) for item in data.get("case_updates", []) if isinstance(item, dict)],
            evidence_details=[
                item for item in data.get("evidence_details", [])
                if isinstance(item, dict)
            ],
            region=(data.get("region") or "").strip(),
            time_info=(data.get("time_info") or "").strip(),
        )
        deterministic = _deterministic_issue_fallback(user_input)
        if deterministic and deterministic.domain == "cyber_data_fraud":
            result.domain = deterministic.domain
            result.issues = list(dict.fromkeys([*deterministic.issues, *result.issues]))
            if not result.facts:
                result.facts = deterministic.facts
        elif not result.issues:
            result = deterministic or result
        logger.debug("提取法律问题: {} domain: {}", result.issues, result.domain)
        return result
    except Exception as e:
        logger.warning(f"法律问题提取失败: {e}")
        semantic_seed = " ".join(str(fallback_text or user_input or "").split()).strip()
        fallback = _deterministic_issue_fallback(semantic_seed)
        if fallback:
            logger.info("法律问题提取启用确定性兜底 | issues={} domain={}", fallback.issues, fallback.domain)
            fallback.degraded = True
            return fallback
        # 模型不可用时不丢弃整段案情。原始对话仅作为语义检索种子，
        # 后续必须经过术语向量库阈值校验；它不是法律结论。
        if semantic_seed:
            return IssuesOutput(
                issues=[semantic_seed],
                domain="other",
                degraded=True,
            )
        return IssuesOutput(issues=[], domain="other", degraded=True)


# ── 第二层：Neo4j LegalConcept 精确匹配 ──────────────────────────────────────

async def confirm_domain_in_neo4j(domain: str, neo4j_driver: AsyncDriver) -> str:
    """确认领域节点在 Neo4j 中存在（节点不存在时仍保留 LLM 推断值）。"""
    if not domain or domain == "other":
        return domain
    try:
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (d:Domain {name: $name}) RETURN d.name AS name LIMIT 1",
                name=domain,
            )
            record = await result.single()
            logger.debug("Neo4j领域确认 {}: {}", domain, "命中" if record else "未找到，保留推断值")
        return domain
    except Exception as e:
        logger.warning(f"Neo4j领域确认失败: {e}")
        return domain


async def match_issues_in_neo4j(
    issues: list[str],
    neo4j_driver: AsyncDriver,
) -> tuple[list[str], list[str]]:
    """第二层：Neo4j LegalConcept 精确匹配。

    返回 (命中的标准术语列表, 未命中列表)。

    命中 = issues 中的词恰好是 LegalConcept 节点的 name。
    未命中词进入第三层语义兜底。
    """
    if not issues:
        return [], []
    try:
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (c:LegalConcept) WHERE c.name IN $names RETURN c.name AS name",
                names=issues,
            )
            records = await result.data()
        matched_set = {r["name"] for r in records}
        matched   = [s for s in issues if s in matched_set]
        unmatched = [s for s in issues if s not in matched_set]
        logger.debug("Neo4j LegalConcept匹配: 命中={} 未命中={}", matched, unmatched)
        return matched, unmatched
    except Exception as e:
        # LegalConcept 节点可能尚未建立（build_legal_concepts.py 未运行），降级保留全部
        logger.warning("Neo4j LegalConcept匹配失败（可能尚未建图），降级: {}", e)
        return [], issues


# ── 第三层：legal_term_index 语义映射（术语替换） ─────────────────────────────

TERM_COLLECTION    = "legal_term_index"
SEMANTIC_THRESHOLD = 0.75  # 低于此值视为真正的图谱外问题
SEMANTIC_TOP_K     = 3     # 取 top3 候选，便于在同领域候选中择优
_NEAR_MISS_MARGIN  = 0.08  # 落在 [阈值-margin, 阈值) 的记日志，供调阈值参考


async def semantic_fallback(
    unmatched_issues: list[str],
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    domain: str = "",
) -> tuple[dict[str, str], list[str], list[str]]:
    """第三层：legal_term_index 语义兜底，把口语描述吸附到最近的标准法律术语。

    一次批量 search（data 传多个查询向量），取 top-K 候选后择优：
    同领域候选优先（同分区语义更可信），否则退回全局 top1。
    这样既避免"劳动口语被映射到刑事术语"，又不会把跨领域的民法典通用术语过滤掉。

    Returns:
        mapped         : {用户原词: 标准法律术语} —— 保证 value 是法条正文用语
        still_unmatched: 无法映射的口语描述（保留原词，仅供 Dense 检索参考）
    """
    if not unmatched_issues:
        return {}, [], []

    try:
        query_embeddings = await embedding_model.aembed_documents(unmatched_issues)
    except Exception as e:
        logger.warning("第三层 embedding 失败，全部降级为口语词: {}", e)
        return {}, list(unmatched_issues), []

    try:
        results = milvus_client.search(
            collection_name=TERM_COLLECTION,
            data=query_embeddings,
            anns_field="embedding",
            limit=SEMANTIC_TOP_K,
            output_fields=["name", "domain"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
    except Exception as e:
        # collection 不存在（未跑 build_legal_concepts.py）或 Milvus 异常：
        # 不再做无意义的 statute_index 可分类性检测，直接全部降级为口语词，
        # 由 node_retrieve 走纯 Dense 检索兜底。
        logger.warning("{} 不可用，第三层整体跳过: {}", TERM_COLLECTION, e)
        return {}, list(unmatched_issues), []

    mapped: dict[str, str] = {}
    still_unmatched: list[str] = []
    mapped_domains: list[str] = []

    for issue, cands in zip(unmatched_issues, results or []):
        hits = [
            {
                "name":   c["entity"]["name"],
                "domain": c["entity"].get("domain", ""),
                "score":  c.get("distance", 0.0),
            }
            for c in (cands or [])
        ]
        passing = [h for h in hits if h["score"] >= SEMANTIC_THRESHOLD]
        if not passing:
            still_unmatched.append(issue)
            if hits:
                best = max(hits, key=lambda h: h["score"])
                if best["score"] >= SEMANTIC_THRESHOLD - _NEAR_MISS_MARGIN:
                    logger.debug(
                        "第三层擦边未采纳: '{}' ~ '{}' (score={:.3f} < {})",
                        issue, best["name"], best["score"], SEMANTIC_THRESHOLD,
                    )
            continue

        # 同领域候选优先，其次全局最高分
        same_domain = [h for h in passing if domain and h["domain"] == domain]
        pick = max(same_domain or passing, key=lambda h: h["score"])
        mapped[issue] = pick["name"]
        picked_domain = str(pick.get("domain") or "")
        if picked_domain and picked_domain not in mapped_domains:
            mapped_domains.append(picked_domain)
        logger.debug(
            "语义映射: '{}' → '{}' (score={:.3f} domain={} 同域优先={})",
            issue, pick["name"], pick["score"], pick["domain"], bool(same_domain),
        )

    return mapped, still_unmatched, mapped_domains


# ── 入口函数 ──────────────────────────────────────────────────────────────────

def _dedup(items: list[str]) -> list[str]:
    """保序去重（不用 set，避免每轮检索 query 字符串顺序漂移）。"""
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


async def normalize_legal_issues(
    user_input: str,
    llm: BaseChatModel,
    neo4j_driver: AsyncDriver,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    *,
    fallback_text: str = "",
) -> dict:
    """完整三层法律问题标准化入口。

    关键约定：standard 与 colloquial 是两个**互不合并**的池。
    合并会让口语词流进 BM25 / PG LIKE，那两个通道是字面匹配，喂口语等于零召回。

    Returns:
        standard   : 法条正文用语（L2 精确命中 + L3 语义映射结果）
                     → 供 BM25 sparse_query、PG ilike 关键词、Dense
        colloquial : 无法映射到标准术语的口语描述
                     → 只供 Dense/HyDE（向量能容忍语义 gap，BM25 不能）
        term_map   : {口语原词: 标准术语}，仅供日志/调试面板展示
        domain     : 确认后的法律领域代码
    """
    # ── 第一层：LLM 提取 + domain 推断 ────────────────────────────────────────
    extracted = await extract_legal_issues(
        user_input,
        llm,
        fallback_text=fallback_text,
    )
    issues = _dedup([i.strip() for i in extracted.issues if i.strip()])
    raw_domain = extracted.domain
    mapped_domain = DOMAIN_MAPPING.get(raw_domain, raw_domain)
    if mapped_domain != raw_domain:
        logger.info("域名映射: {} → {}", raw_domain, mapped_domain)
    domain = await confirm_domain_in_neo4j(mapped_domain, neo4j_driver)

    if not issues:
        logger.info("标准化结果 | L1 未提取到法律问题 domain={}", domain)
        return {
            "standard": [], "colloquial": [], "term_map": {}, "domain": domain,
            "collected_facts": extracted.facts,
            "case_updates": [item.model_dump() for item in extracted.case_updates],
            "evidence_details": extracted.evidence_details,
            "region": extracted.region,
            "time_info": extracted.time_info,
        }

    # ── 第二层：Neo4j LegalConcept 精确匹配（逐字相等）────────────────────────
    matched, unmatched_after_exact = await match_issues_in_neo4j(issues, neo4j_driver)

    # ── 第三层：Milvus 语义最近邻，把口语吸附到标准术语 ───────────────────────
    term_map, still_unmatched, mapped_domains = await semantic_fallback(
        unmatched_after_exact, embedding_model, milvus_client, domain=domain,
    )
    if (not domain or domain == "other") and len(mapped_domains) == 1:
        domain = await confirm_domain_in_neo4j(mapped_domains[0], neo4j_driver)
        logger.info("语义术语映射反推法律领域 | domain={}", domain)

    standard = _dedup(matched + list(term_map.values()))
    logger.info(
        "标准化结果 | L2命中={} L3映射={} 未映射={} standard={} domain={}",
        matched, term_map, still_unmatched, standard, domain,
    )
    return {
        "standard":   standard,
        "colloquial": still_unmatched,
        "term_map":   term_map,
        "domain":     domain,
        "collected_facts": extracted.facts,
        "case_updates": [item.model_dump() for item in extracted.case_updates],
        "evidence_details": extracted.evidence_details,
        "region": extracted.region,
        "time_info": extracted.time_info,
        "extraction_degraded": extracted.degraded,
    }
