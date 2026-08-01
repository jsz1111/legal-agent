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

_DATE_PATTERN = re.compile(
    r"(?:20\d{2}年)?(?:1[0-2]|0?[1-9])月(?:3[01]|[12]\d|0?[1-9])日"
    r"|20\d{2}[-/.](?:1[0-2]|0?[1-9])[-/.](?:3[01]|[12]\d|0?[1-9])"
)
_AMOUNT_PATTERN = re.compile(
    r"(?:人民币\s*)?\d+(?:\.\d{1,2})?\s*(?:元|万元|块钱|块)"
)
_EVIDENCE_PATTERN = re.compile(
    r"订单(?:截图|记录|详情)?|付款(?:记录|凭证|截图)|支付(?:记录|凭证|截图)|"
    r"聊天(?:记录|截图)|投诉(?:记录|工单)|平台工单|物流记录|银行流水|"
    r"交易记录|合同|协议|发票|收据|录音|录像|照片|视频|回执|通知"
)
_RELATION_PATTERN = re.compile(
    r"(?:对方是|对方为|双方是|我们是|我和对方是)"
    r"([^，。；;\n]{2,36})"
)
_CLAIM_PATTERN = re.compile(
    r"(?:我)?(?:希望|要求|请求|想要|诉求是)"
    r"([^，。；;\n]{2,80})"
)
_PROCEDURE_PATTERN = re.compile(
    r"[^，。；;\n]{0,24}(?:投诉|举报|报警|仲裁|起诉|联系|沟通|协商|"
    r"申请|反馈|工单|处理)[^，。；;\n]{0,48}"
)
_PLATFORM_PATTERN = re.compile(
    r"(?:在|通过)(闲鱼|淘宝|天猫|京东|拼多多|抖音|快手|微信|支付宝|"
    r"小红书|美团|饿了么|携程|滴滴)(?:平台)?"
)
_EVENT_MARKERS = (
    "未发", "没有发", "拒绝", "不退", "拖欠", "拉黑", "扣押", "扣款",
    "损坏", "泄露", "解除", "辞退", "受伤", "碰撞", "逾期", "违约",
    "约定", "支付", "付款", "转账", "签订", "交付", "维修", "拒收",
)
_UNKNOWN_MARKERS = ("不知道", "不清楚", "不能确定", "无法确认", "记不清")
_DENIAL_MARKERS = (
    "没有", "没留", "没保存", "未保存", "无此", "不存在", "找不到", "丢了",
)


def _stable_suffix(value: str, length: int = 10) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _update_identity(update: CaseFactUpdate) -> tuple[str, str]:
    return update.key, " ".join(update.source_text.split())


def _merge_case_updates(
    values: list[CaseFactUpdate],
) -> list[CaseFactUpdate]:
    result: list[CaseFactUpdate] = []
    seen: set[tuple[str, str]] = set()
    for item in values:
        identity = _update_identity(item)
        if not item.source_text or identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result[:40]


def _update_dimension(update: CaseFactUpdate) -> str:
    category = str(update.category or "").lower()
    key = str(update.key or "").lower()
    if category in {"actor", "relationship"}:
        return "relationship"
    for dimension in (
        "evidence",
        "procedure",
        "claim",
        "amount",
        "time",
        "location",
        "harm",
        "event",
    ):
        if category == dimension:
            return dimension
    if any(token in key for token in ("actor", "counterparty", "relationship")):
        return "relationship"
    if ".date" in key or "timeline" in key:
        return "time"
    return category or "event"


def _compact_compare(value: str) -> str:
    return re.sub(r"[\s，。；;、,:：()（）【】\[\]\"'“”‘’]+", "", value or "").lower()


def _same_grounded_atom(
    model_update: CaseFactUpdate,
    fallback_update: CaseFactUpdate,
) -> bool:
    if model_update.key == fallback_update.key:
        return True
    if _update_dimension(model_update) != _update_dimension(fallback_update):
        return False
    model_values = {
        _compact_compare(model_update.value),
        _compact_compare(model_update.source_text),
    } - {""}
    fallback_values = {
        _compact_compare(fallback_update.value),
        _compact_compare(fallback_update.source_text),
    } - {""}
    return any(
        left == right or left in right or right in left
        for left in model_values
        for right in fallback_values
    )


def _merge_model_and_fallback_updates(
    model_updates: list[CaseFactUpdate],
    fallback_updates: list[CaseFactUpdate],
) -> list[CaseFactUpdate]:
    """Let deterministic atoms fill extraction gaps without creating conflicts."""

    merged = _merge_case_updates(model_updates)
    for fallback in fallback_updates:
        if any(_same_grounded_atom(item, fallback) for item in merged):
            continue
        if (
            fallback.category == "event"
            and any(
                marker in fallback.source_text
                for marker in ("付款", "支付", "转账")
            )
            and not any(
                marker in fallback.source_text
                for marker in (
                    "未发", "没有发", "拒绝", "不退", "拖欠", "拉黑",
                    "扣押", "扣款", "损坏", "泄露", "解除", "辞退",
                    "受伤", "碰撞", "逾期", "违约", "拒收",
                )
            )
            and any(
                item.source_text
                and item.source_text in fallback.source_text
                and _update_dimension(item) in {"amount", "time", "location"}
                for item in merged
            )
        ):
            continue
        merged.append(fallback)
    return _merge_case_updates(merged)


def _certainty_for_context(value: str) -> str:
    if any(marker in value for marker in _UNKNOWN_MARKERS):
        return "unknown"
    if any(marker in value for marker in _DENIAL_MARKERS):
        return "denied"
    return "asserted"


def _context_window(text: str, start: int, end: int, radius: int = 28) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].strip("，。；;\n ").split())


def _evidence_context(text: str, start: int, end: int) -> str:
    """Return the smallest exact clause that carries evidence availability."""

    left = 0
    right = len(text)
    for marker in ("，", "。", "；", ";", "\n", "但是", "但", "不过", "然而"):
        before = text.rfind(marker, 0, start)
        if before >= 0:
            left = max(left, before + len(marker))
        after = text.find(marker, end)
        if after >= 0:
            right = min(right, after)
    return " ".join(text[left:right].strip("，。；;\n ").split())


def _deterministic_domain_and_issue(text: str) -> tuple[str, list[str]]:
    compact = "".join((text or "").split())
    profiles = (
        (
            "labor_social_security",
            ("工资", "劳动合同", "公司辞退", "社保", "工伤", "加班费"),
            "劳动关系及报酬争议",
        ),
        (
            "consumer_market",
            ("购买", "卖家", "商家", "退款", "订单", "商品", "消费者"),
            "消费或网络交易履行争议",
        ),
        (
            "contracts_property_housing",
            ("租房", "房东", "押金", "借款", "欠款", "合同", "租赁"),
            "合同、借贷或住房履行争议",
        ),
        (
            "traffic_personal_injury",
            ("交通事故", "撞车", "交警", "事故认定", "人身损害"),
            "交通事故或人身损害争议",
        ),
        (
            "family_vulnerable_groups",
            ("离婚", "抚养", "赡养", "家暴", "监护", "夫妻"),
            "婚姻家庭或弱势群体权益争议",
        ),
        (
            "cyber_data_fraud",
            ("诈骗", "盗号", "个人信息", "网络账户", "电信"),
            "网络、数据或欺诈风险争议",
        ),
    )
    for domain, markers, issue in profiles:
        if any(marker in compact for marker in markers):
            return domain, [issue]
    return "other", []


def _deterministic_case_updates(text: str) -> list[CaseFactUpdate]:
    """Extract only high-precision, source-grounded atoms when the model is slow.

    This is a reliability layer, not a second legal classifier. Every atom uses
    a literal substring from the current message and remains a user statement.
    """

    source = " ".join(str(text or "").split()).strip()
    if not source:
        return []
    updates: list[CaseFactUpdate] = []

    for index, match in enumerate(_DATE_PATTERN.finditer(source), 1):
        context = _context_window(source, match.start(), match.end())
        payment = any(marker in context for marker in ("付款", "支付", "转账"))
        key = (
            f"transaction.payment.pay_{index:02d}.date"
            if payment
            else f"event.timeline.time_{index:02d}"
        )
        updates.append(CaseFactUpdate(
            key=key,
            category="time",
            statement=f"关键时间为{match.group(0)}",
            value=match.group(0),
            certainty="asserted",
            source_text=match.group(0),
        ))

    for index, match in enumerate(_AMOUNT_PATTERN.finditer(source), 1):
        context = _context_window(source, match.start(), match.end())
        payment = any(marker in context for marker in ("付款", "支付", "转账", "购买"))
        loss = any(marker in context for marker in ("损失", "赔偿", "医疗费", "维修费"))
        key = (
            f"transaction.payment.pay_{index:02d}.amount"
            if payment
            else f"harm.loss.loss_{index:02d}.amount"
            if loss
            else f"transaction.amount.amount_{index:02d}"
        )
        updates.append(CaseFactUpdate(
            key=key,
            category="amount",
            statement=f"涉及金额为{match.group(0)}",
            value=match.group(0),
            certainty="asserted",
            source_text=match.group(0),
        ))

    for match in _PLATFORM_PATTERN.finditer(source):
        platform = match.group(1)
        updates.append(CaseFactUpdate(
            key="location.platform",
            category="location",
            statement=f"事项通过{platform}平台发生或办理",
            value=platform,
            certainty="asserted",
            source_text=match.group(0),
        ))

    for match in _RELATION_PATTERN.finditer(source):
        relation = match.group(1).strip()
        updates.append(CaseFactUpdate(
            key="relationship.counterparty",
            category="relationship",
            statement=f"用户称双方关系或对方身份为{relation}",
            value=relation,
            certainty=_certainty_for_context(match.group(0)),
            source_text=match.group(0),
        ))

    for match in _CLAIM_PATTERN.finditer(source):
        claim = match.group(1).strip()
        updates.append(CaseFactUpdate(
            key="claim.primary_request",
            category="claim",
            statement=f"用户希望{claim}",
            value=claim,
            certainty=_certainty_for_context(match.group(0)),
            source_text=match.group(0),
        ))

    for match in _PROCEDURE_PATTERN.finditer(source):
        value = match.group(0).strip()
        if not value:
            continue
        updates.append(CaseFactUpdate(
            key=f"procedure.history.{_stable_suffix(value)}",
            category="procedure",
            statement=value,
            value=value,
            certainty=_certainty_for_context(value),
            source_text=value,
        ))

    for match in _EVIDENCE_PATTERN.finditer(source):
        name = match.group(0)
        context = _evidence_context(source, match.start(), match.end())
        updates.append(CaseFactUpdate(
            key=f"evidence.{_stable_suffix(name)}",
            category="evidence",
            statement=f"用户提到{name}",
            value=name,
            certainty=_certainty_for_context(context),
            source_text=context,
        ))

    clauses = [
        value.strip()
        for value in re.split(r"[，。；;\n]+", source)
        if value.strip()
    ]
    for clause in clauses:
        if not any(marker in clause for marker in _EVENT_MARKERS):
            continue
        updates.append(CaseFactUpdate(
            key=f"event.core.{_stable_suffix(clause)}",
            category="event",
            statement=clause,
            value=clause,
            certainty=_certainty_for_context(clause),
            source_text=clause,
        ))

    return _merge_case_updates(updates)


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
    atoms = _deterministic_case_updates("\n".join(sections.values()))
    return _merge_case_updates([*updates, *atoms])


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
        domain, issues = _deterministic_domain_and_issue(case_summary)
        return IssuesOutput(
            issues=(
                list(fallback.issues)
                if fallback
                else issues or ([narrative] if narrative else [])
            ),
            domain=fallback.domain if fallback else domain,
            facts=facts,
            case_updates=updates,
            evidence_details=[],
            time_info=sections.get("时间、地点和金额", ""),
            degraded=True,
        )


def _deterministic_issue_fallback(user_input: str) -> IssuesOutput | None:
    """Recover only high-precision intents when structured LLM output is unavailable."""
    normalized = "".join((user_input or "").split())
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
    return None


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
        if not result.issues:
            result = _deterministic_issue_fallback(user_input) or result
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


async def extract_case_facts(
    user_input: str,
    llm: BaseChatModel,
    *,
    fallback_text: str = "",
) -> IssuesOutput:
    """Extract grounded case candidates without legal retrieval.

    ``update_facts`` must not call the legacy three-layer issue normalizer:
    Neo4j/Milvus matching belongs to later legal modelling. This helper keeps
    the same structured schema for compatibility, but returns only candidates
    grounded in the current user input.
    """

    structured_intake = await _extract_structured_intake(user_input, llm)
    if structured_intake is not None:
        return structured_intake
    try:
        prompt = _ISSUE_EXTRACT_FALLBACK_PROMPT.format(user_input=user_input)
        response = await ainvoke_bounded(
            llm_for_stage(llm, max_tokens=1400),
            [SystemMessage(content=prompt)],
            timeout=get_settings().GUIDE_LLM_TIMEOUT_EXTRACT,
            stage="case_fact_extraction",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        data = json.loads(content)
        deterministic = _deterministic_case_updates(
            str(fallback_text or user_input or "")
        )
        model_updates = [
            CaseFactUpdate.model_validate(item)
            for item in data.get("case_updates", [])
            if isinstance(item, dict)
        ]
        return IssuesOutput(
            issues=[str(item).strip() for item in data.get("issues", []) if str(item).strip()],
            domain=str(data.get("domain") or "other"),
            facts=[str(item).strip() for item in data.get("facts", []) if str(item).strip()],
            case_updates=_merge_model_and_fallback_updates(
                model_updates,
                deterministic,
            ),
            evidence_details=[
                item for item in data.get("evidence_details", [])
                if isinstance(item, dict)
            ],
            region=str(data.get("region") or "").strip(),
            time_info=str(data.get("time_info") or "").strip(),
        )
    except Exception as exc:
        logger.warning("事实候选提取失败，保留原文降级: {}", exc)
        semantic_seed = " ".join(str(fallback_text or user_input or "").split()).strip()
        domain, issues = _deterministic_domain_and_issue(semantic_seed)
        updates = _deterministic_case_updates(semantic_seed)
        return IssuesOutput(
            issues=issues,
            domain=domain,
            facts=[
                item.statement for item in updates
            ] or ([semantic_seed] if semantic_seed else []),
            case_updates=updates,
            evidence_details=[],
            degraded=True,
        )


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
