"""Explainable, conservative evidence-utility assessment for the legal guide.

The module evaluates preparation and proof coverage, not judicial admissibility.
It deliberately separates:

1. whether a material exists;
2. which proof target it may support;
3. whether basic source/integrity details are known;
4. what the material cannot establish on its own.

All legal proof targets come from the domain follow-up catalog.  This keeps the
assessment generic and avoids incident-specific keyword patches in graph nodes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Iterable

from langchain_core.messages import SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.legal_guide.case_model import active_case_facts
from src.agents.legal_guide.followup_catalog import (
    EvidenceFollowup,
    evidence_rule_resolved,
    get_domain_followups,
)
from src.agents.legal_guide.llm_runtime import ainvoke_bounded, llm_for_stage


ALLOWED_SOURCE_FORMS = {
    "paper_original",
    "native_electronic",
    "exported_file",
    "screenshot",
    "copy",
    "user_statement",
    "unknown",
}
ALLOWED_COMPLETENESS = {"complete", "partial", "unknown"}
ALLOWED_VISIBILITY = {"clear", "unclear", "not_applicable", "unknown"}
ALLOWED_ACQUISITION = {
    "user_created",
    "received_from_counterparty",
    "platform_or_institution_export",
    "third_party",
    "unknown",
}
ALLOWED_CASE_SPECIFICITY = {
    "case_specific",
    "blank_or_reference",
    "unclear",
    "unknown",
}
ALLOWED_PROOF_ROLES = {
    "relationship",
    "transaction",
    "agreement",
    "payment",
    "event",
    "problem",
    "identity",
    "time",
    "communication",
    "procedure",
    "harm",
    "loss",
    "liability",
    "ownership",
    "infringement",
}

# Stable proof semantics, not incident words.  Evidence keys are application
# catalog concepts; roles are the smaller cross-domain ontology used to link a
# user's description of what a material contains to the relevant proof target.
EVIDENCE_KEY_PROOF_ROLES: dict[str, set[str]] = {
    "employment_relation": {"relationship", "identity", "time"},
    "wage_payment": {"payment", "loss", "time"},
    "transaction": {"transaction", "payment", "identity", "time"},
    "defect": {"event", "problem", "communication"},
    "agreement": {"agreement", "relationship"},
    "performance": {"transaction", "payment", "event", "communication"},
    "original_clues": {"event", "identity", "communication"},
    "harm": {"harm", "loss", "time"},
    "liability": {"liability", "event", "identity"},
    "damage": {"harm", "loss", "payment"},
    "family_status": {"relationship", "identity"},
    "property_or_harm": {"harm", "loss", "event"},
    "service_records": {"relationship", "transaction", "procedure", "time"},
    "harm_and_payment": {"harm", "loss", "payment"},
    "decision_and_service": {"procedure", "time", "identity"},
    "supporting": {"event", "procedure", "problem"},
    "ownership": {"ownership", "identity", "time"},
    "infringement": {"infringement", "event", "identity", "time"},
    "scene": {"event", "problem", "time"},
    "monitoring_and_harm": {"problem", "harm", "loss", "procedure"},
    "transaction_account": {"transaction", "payment", "identity", "time"},
    "communication": {"communication", "identity", "time", "procedure"},
    "procedure_document": {"procedure", "agreement", "time"},
    "underlying": {"agreement", "transaction", "payment"},
    "harm_and_report": {"harm", "loss", "procedure"},
}


class EvidenceItem(BaseModel):
    id: str
    name: str
    availability: str = "unclear"
    source_form: str = "unknown"
    completeness: str = "unknown"
    identity_visibility: str = "unknown"
    time_visibility: str = "unknown"
    acquisition_method: str = "unknown"
    case_specificity: str = "unknown"
    proof_roles: list[str] = Field(default_factory=list)
    authenticity_status: str = "not_verified"
    inspection_basis: str = "user_statement"
    source_excerpt: str = ""
    content_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ProofTarget(BaseModel):
    id: str
    rule_id: str
    evidence_key: str
    label: str
    purpose: str
    required_for_planning: bool = True


class EvidenceLink(BaseModel):
    evidence_id: str
    target_id: str
    direction: str = "supports"
    relevance: str = "potentially_relevant"
    proof_scope: str
    basis: str
    limitations: list[str] = Field(default_factory=list)


class EvidenceCoverage(BaseModel):
    target_id: str
    label: str
    purpose: str
    status: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    quality_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_action: str = ""


class EvidenceEvaluationReport(BaseModel):
    schema_version: int = 1
    items: list[EvidenceItem] = Field(default_factory=list)
    targets: list[ProofTarget] = Field(default_factory=list)
    links: list[EvidenceLink] = Field(default_factory=list)
    coverage: list[EvidenceCoverage] = Field(default_factory=list)
    target_count: int = 0
    preliminarily_covered_count: int = 0
    partial_count: int = 0
    known_missing_count: int = 0
    unresolved_count: int = 0
    disclaimer: str = (
        "本评估只用于梳理材料用途和补强方向，不认定真实性、合法性、"
        "可采性或最终证明力。"
    )


_QUALITY_LABELS = {
    "source_form": "原件或原始电子载体情况",
    "completeness": "内容完整性",
    "identity_visibility": "相关主体身份",
    "time_visibility": "形成时间",
    "acquisition_method": "材料取得或导出方式",
    "case_specificity": "是否包含本案具体主体、时间或交易信息",
}

_STATUS_LABELS = {
    "preliminarily_covered": "初步覆盖",
    "partially_covered": "部分覆盖",
    "known_missing": "目前缺失",
    "conflicted": "材料状态或内容存在冲突",
    "unresolved": "尚未确认",
}

_STRUCTURED_CLAIM_FIELDS = (
    (
        re.compile(r"^(?:成交金额|付款金额|支付金额|订单金额|金额)$"),
        "transaction.amount",
        "amount",
    ),
    (
        re.compile(r"^(?:付款时间|支付时间|付款日期|支付日期)$"),
        "transaction.payment_date",
        "date",
    ),
    (
        re.compile(r"^(?:订单号|关联订单|订单备注)$"),
        "transaction.order_id",
        "text",
    ),
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [
        value for value in values
        if value and not (value in seen or seen.add(value))
    ]


def _normalize_claim_value(value: str, value_type: str) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if value_type == "amount":
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", compact)
        return match.group(0).replace(",", "") if match else ""
    if value_type == "date":
        match = re.search(
            r"(?:20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})",
            compact,
        )
        if not match:
            return ""
        year = re.search(r"20\d{2}", match.group(0)).group(0)
        return f"{year}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
    return compact[:120]


def _structured_material_claims(content: str) -> list[dict[str, str]]:
    """Extract a small, domain-stable set of comparable record fields."""

    candidates: dict[str, list[dict[str, str]]] = {}
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if "：" not in line:
            continue
        label, value = (part.strip() for part in line.split("：", 1))
        for pattern, key, value_type in _STRUCTURED_CLAIM_FIELDS:
            if not pattern.fullmatch(label):
                continue
            normalized = _normalize_claim_value(value, value_type)
            if normalized:
                candidates.setdefault(key, []).append({
                    "key": key,
                    "value": normalized,
                    "source_text": line[:240],
                })
            break
    claims: list[dict[str, str]] = []
    for key, rows in candidates.items():
        values = {row["value"] for row in rows}
        # A document containing several different values for one field is not
        # reduced to one claim; it needs material-specific interpretation.
        if len(values) == 1:
            claims.append(rows[0])
    return claims


def _grounded_excerpt_fragments(source_excerpt: str) -> list[str]:
    """Return independently checkable fragments from a compact model citation."""

    compact = " ".join((source_excerpt or "").split()).strip()
    if not compact:
        return []
    fragments = [
        item.strip(" ，,。.")
        for item in re.split(r"[；;]\s*", compact)
        if item.strip(" ，,。.")
    ]
    return fragments[:12]


def _excerpt_is_grounded(source_excerpt: str, source_text: str) -> bool:
    """Accept one verbatim quote or a semicolon-joined list of verbatim fields."""

    compact_source = " ".join((source_text or "").split())
    compact_excerpt = " ".join((source_excerpt or "").split()).strip()
    if not compact_excerpt or not compact_source:
        return False
    if compact_excerpt in compact_source:
        return True
    fragments = _grounded_excerpt_fragments(compact_excerpt)
    return bool(fragments) and all(
        len(fragment) >= 4 and fragment in compact_source
        for fragment in fragments
    )


def _comes_from_uploaded_block(source_excerpt: str, user_text: str) -> bool:
    """Return whether an excerpt is grounded in a fingerprinted attachment block."""

    for block in re.split(r"(?=【(?:图片|文档)证据补充)", user_text):
        if not block.startswith(("【图片证据补充", "【文档证据补充")):
            continue
        if "SHA-256：" not in block:
            continue
        if _excerpt_is_grounded(source_excerpt, block):
            return True
    return False


def _deterministic_uploaded_quality(
    *,
    file_name: str,
    source_form: str,
    content: str,
    truncated: bool,
) -> dict[str, Any]:
    """Conservatively retain visible attachment qualities when the LLM fails.

    These rules only describe observable document features.  They never decide
    authenticity, admissibility, liability or whether a stated event is true.
    """

    normalized = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    compact = "".join(normalized.split())
    lower_name = file_name.lower()
    copy_markers = (
        "转录", "重构", "节选", "复制件", "复印件", "不是原始", "不是银行",
        "不是平台", "不是微信原始", "不是合同原件", "不是事故认定书",
    )
    visible_source_form = source_form
    if any(marker in compact for marker in copy_markers):
        visible_source_form = "copy"
    elif "截图" in compact or lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        visible_source_form = "screenshot"

    is_partial = truncated or any(
        marker in compact
        for marker in (*copy_markers, "仅提取前部文字", "部分内容", "摘要")
    )
    date_visible = bool(re.search(
        r"(?:19|20)\d{2}(?:[-年/.]\d{1,2})?(?:[-月/.]\d{1,2})?",
        normalized,
    ))
    identity_visible = bool(re.search(
        r"(?:账号|账户|主体|卖家|买家|委托人|代购人|员工|主管|出租人|承租人|"
        r"驾驶人|收款方|付款方|用人单位)[^\n：:]{0,12}[：:]",
        normalized,
    )) or bool(re.search(r"\b(?:U|S|M|L|E)-\d{2}\b", normalized))

    if re.search(r"(?:公开原型|公开案例|公开裁判|公开典型案例|公开案件).{0,20}(?:来源|重构)", compact):
        acquisition = "third_party"
    elif re.search(r"(?:平台|银行|支付机构|医院|交管部门|鉴定机构).{0,12}(?:导出|出具)", compact):
        acquisition = "platform_or_institution_export"
    elif any(marker in compact for marker in ("本人整理", "用户整理", "自行制作")):
        acquisition = "user_created"
    else:
        acquisition = "unknown"

    reference_only = any(
        marker in compact
        for marker in ("空白模板", "空白表格", "纯参考资料", "办事指南", "示范文本")
    )
    specific_signals = sum((
        date_visible,
        bool(re.search(r"(?:人民币)?\d+(?:\.\d+)?元|金额[：:]|费用[：:]", normalized)),
        bool(re.search(r"(?:流水号|订单号|案件编号|票据编号|账号|账户)[：:]", normalized)),
        identity_visible,
    ))
    case_specificity = (
        "blank_or_reference"
        if reference_only and specific_signals < 2
        else "case_specific"
        if specific_signals >= 2
        else "unclear"
    )

    role_patterns = {
        "relationship": r"劳动关系|劳务关系|租赁关系|雇佣关系|双方关系|用人单位",
        "transaction": r"订单|购买|交易|商品|票款",
        "agreement": r"合同|协议|约定|委托|承诺",
        "payment": r"付款|支付|转账|金额|流水|工资|押金|票款|佣金",
        "event": r"事故|发货|交付|搬离|退租|到岗|解除|相撞",
        "problem": r"未发货|未交付|未支付|未退款|拒绝|拖欠|损坏|失联",
        "identity": r"账号|账户|主体|卖家|买家|委托人|代购人|员工|主管|出租人|承租人|驾驶人",
        "communication": r"聊天|微信|短信|沟通|回复|承诺",
        "procedure": r"报警|投诉|仲裁|事故认定|鉴定意见|调解|起诉",
        "harm": r"受伤|伤情|医疗|病历|护理|营养|误工",
        "loss": r"损失|费用|金额|工资|押金|医疗费|误工费|护理费|营养费",
    }
    roles = [role for role, pattern in role_patterns.items() if re.search(pattern, normalized)]
    if date_visible:
        roles.append("time")
    anchor = next((line for line in normalized.splitlines() if line), file_name)[:240]
    return {
        "source_form": visible_source_form,
        "completeness": "partial" if is_partial else "unknown",
        "identity_visibility": "clear" if identity_visible else "unknown",
        "time_visibility": "clear" if date_visible else "unknown",
        "acquisition_method": acquisition,
        "case_specificity": case_specificity,
        "proof_roles": _unique(roles)[:8],
        "source_text": anchor,
    }


def split_uploaded_evidence_blocks(
    user_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Separate case narrative from fingerprinted attachment payloads.

    Attachment text is untrusted evidence content, not the user's adopted case
    narrative.  Keeping it out of issue extraction prevents large batches from
    exhausting the extraction budget or turning an entire document into one
    colloquial legal issue.  A deterministic inventory is returned so an LLM
    timeout can never make uploaded files disappear from the evidence state.
    """

    marker = re.compile(r"(?=【(?:图片|文档)证据补充)")
    parts = marker.split(user_text or "")
    narrative_parts: list[str] = []
    observations: list[dict[str, Any]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped.startswith(("【图片证据补充", "【文档证据补充")):
            if stripped:
                narrative_parts.append(stripped)
            continue
        if "SHA-256：" not in stripped:
            # A visually similar user-authored heading is not proof that the
            # application actually received a file.
            narrative_parts.append(stripped)
            continue
        file_match = re.search(r"(?m)^文件：(.+?)\s*$", stripped)
        digest_match = re.search(r"(?m)^原(?:图|文件) SHA-256：([0-9a-fA-F]{16,})\s*$", stripped)
        if not file_match or not digest_match:
            narrative_parts.append(stripped)
            continue
        file_name = file_match.group(1).strip()[:180]
        source_match = re.search(r"(?m)^来源形式：([a-z_]+)\s*$", stripped)
        requirement_match = re.search(r"(?m)^清单项ID：([^\s]+)\s*$", stripped)
        requirement_label_match = re.search(r"(?m)^清单项：(.+?)\s*$", stripped)
        requirement_id = requirement_match.group(1).strip() if requirement_match else ""
        rule_id = requirement_id.split("proof_target:", 1)[-1] if requirement_id else ""
        source_form = (
            source_match.group(1)
            if source_match and source_match.group(1) in ALLOWED_SOURCE_FORMS
            else "screenshot"
            if stripped.startswith("【图片证据补充")
            else "unknown"
        )
        truncated = "仅提取前部文字" in stripped
        if "【提取文字】" in stripped:
            content_excerpt = stripped.split("【提取文字】", 1)[1].strip()
        else:
            content_excerpt = stripped[digest_match.end():].strip()
        deterministic_quality = _deterministic_uploaded_quality(
            file_name=file_name,
            source_form=source_form,
            content=content_excerpt,
            truncated=truncated,
        )
        observations.append({
            "name": file_name,
            **deterministic_quality,
            "uploaded_copy": True,
            "content_digest": digest_match.group(1).lower(),
            "content_excerpt": content_excerpt[:1_200],
            "material_claims": _structured_material_claims(content_excerpt),
            "requirement_id": requirement_id,
            "requirement_label": (
                requirement_label_match.group(1).strip()[:180]
                if requirement_label_match else ""
            ),
            "rule_id": rule_id,
        })
    narrative = "\n\n".join(narrative_parts).strip()
    return narrative, observations


async def inspect_uploaded_evidence_blocks(
    user_text: str,
    llm: Any,
) -> list[dict[str, Any]]:
    """Inspect observable material quality without deciding truth or liability."""

    _narrative, inventory = split_uploaded_evidence_blocks(user_text)
    inspectable = [
        item for item in inventory if item.get("content_excerpt")
    ][:8]
    if not inspectable:
        return []

    async def inspect_batch(
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        blocks: list[str] = []
        for item in batch:
            excerpt = str(item.get("content_excerpt") or "")[:1_200]
            blocks.append(
                f"文件名：{item['name']}\n"
                f"系统识别来源形式：{item['source_form']}\n"
                f"可见内容：\n{excerpt}"
            )
        prompt = """你只评估系统实际收到的材料副本中“肉眼可见的基础属性”，不判断真实性、违法、责任、可采性或胜诉。

材料：
{materials}

每个文件输出一条JSON记录，name必须逐字复制文件名。source_text只摘录一条最短、连续的可见原文，不得汇总或改写；无法由摘录支持的属性填unknown。proof_roles最多4项。

case_specificity：
- case_specific：可见内容含本案具体主体、账号、订单、日期、金额、诊疗或处理状态等实际记录；
- blank_or_reference：空白模板、票样、法规、办事指南、示范文本或纯参考资料，没有本案已填写的具体记录；
- unclear：无法判断是否为本案材料；
- unknown：没有足够可见内容。

只输出JSON：
{{"items":[{{"name":"原文件名","source_form":"paper_original|native_electronic|exported_file|screenshot|copy|user_statement|unknown","completeness":"complete|partial|unknown","identity_visibility":"clear|unclear|not_applicable|unknown","time_visibility":"clear|unclear|not_applicable|unknown","acquisition_method":"user_created|received_from_counterparty|platform_or_institution_export|third_party|unknown","case_specificity":"case_specific|blank_or_reference|unclear|unknown","proof_roles":["relationship|transaction|agreement|payment|event|problem|identity|time|communication|procedure|harm|loss|liability|ownership|infringement"],"source_text":"可见内容中的逐字短句"}}]}}
""".format(materials="\n\n---\n\n".join(blocks))
        try:
            response = await ainvoke_bounded(
                llm_for_stage(llm, max_tokens=700),
                [SystemMessage(content=prompt)],
                timeout=10.0,
                stage="evidence_quality_inspection",
            )
            content = str(response.content or "").strip()
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()
            data = json.loads(content)
            allowed_names = {item["name"] for item in batch}
            raw_items = [
                item for item in data.get("items", [])
                if isinstance(item, dict) and item.get("name") in allowed_names
            ]
            normalized = normalize_evidence_observations(
                raw_items,
                user_text=user_text,
            )
            if raw_items and not normalized:
                logger.warning(
                    "证据质量检查结果缺少可验证原文锚点，已丢弃: {}",
                    content[:800],
                )
                return None
            return normalized
        except Exception as exc:
            # The deterministic inventory is merged separately.  A quality
            # model failure may leave this batch unknown, but never drops it.
            logger.warning("证据质量检查批次降级为确定性库存: {}", exc)
            return None

    # One-file decisions keep the structured response short and isolate a
    # malformed or slow document from every other attachment.  Calls run in
    # parallel, so the number of files does not multiply visible latency.
    batches = [[item] for item in inspectable]
    results = await asyncio.gather(*(inspect_batch(batch) for batch in batches))
    successful = [
        item
        for batch_result in results
        if batch_result
        for item in batch_result
    ]
    retry_items = [
        batch[0]
        for batch, batch_result in zip(batches, results)
        if batch_result is None
    ]
    if retry_items:
        # Transient timeouts and malformed JSON are common enough to justify
        # one isolated retry.  Failed files still retain deterministic upload
        # inventory if the retry also fails.
        retry_results = await asyncio.gather(
            *(inspect_batch([item]) for item in retry_items)
        )
        successful.extend(
            item
            for retry_result in retry_results
            if retry_result
            for item in retry_result
        )
    return successful


def normalize_evidence_observations(
    raw_items: Any,
    *,
    user_text: str,
) -> list[dict[str, Any]]:
    """Accept only source-anchored metadata explicitly stated by the user.

    The language model may structure the statement, but it cannot manufacture
    quality metadata: every observation needs a verbatim source anchor.
    """

    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in raw_items[:12]:
        if not isinstance(raw, dict):
            continue
        excerpt = " ".join(str(raw.get("source_text") or "").split())[:240]
        name = " ".join(str(raw.get("name") or "").split())[:120]
        if not excerpt or not _excerpt_is_grounded(excerpt, user_text) or not name:
            continue
        source_form = str(raw.get("source_form") or "unknown")
        completeness = str(raw.get("completeness") or "unknown")
        identity_visibility = str(raw.get("identity_visibility") or "unknown")
        time_visibility = str(raw.get("time_visibility") or "unknown")
        acquisition_method = str(raw.get("acquisition_method") or "unknown")
        case_specificity = str(raw.get("case_specificity") or "unknown")
        proof_roles = [
            str(item) for item in (raw.get("proof_roles") or [])
            if str(item) in ALLOWED_PROOF_ROLES
        ]
        normalized.append({
            "name": name,
            "source_form": (
                source_form if source_form in ALLOWED_SOURCE_FORMS else "unknown"
            ),
            "completeness": (
                completeness if completeness in ALLOWED_COMPLETENESS else "unknown"
            ),
            "identity_visibility": (
                identity_visibility
                if identity_visibility in ALLOWED_VISIBILITY
                else "unknown"
            ),
            "time_visibility": (
                time_visibility if time_visibility in ALLOWED_VISIBILITY else "unknown"
            ),
            "acquisition_method": (
                acquisition_method
                if acquisition_method in ALLOWED_ACQUISITION
                else "unknown"
            ),
            "case_specificity": (
                case_specificity
                if case_specificity in ALLOWED_CASE_SPECIFICITY
                else "unknown"
            ),
            "proof_roles": _unique(proof_roles),
            "source_text": excerpt,
            "uploaded_copy": _comes_from_uploaded_block(excerpt, user_text),
        })
    return normalized


def _observation_matches(name: str, canonical: str, rule: EvidenceFollowup | None) -> bool:
    if not name or not canonical:
        return False
    if name in canonical or canonical in name:
        return True
    if not rule:
        return False
    return evidence_rule_resolved(rule, [name, canonical])


def _mark_evidence_content_conflicts(
    records: dict[str, dict],
) -> dict[str, dict]:
    result = {
        key: {**value, "content_conflicts": []}
        for key, value in records.items()
    }
    groups: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for record_id, record in result.items():
        for claim in record.get("material_claims") or []:
            key = str(claim.get("key") or "")
            value = str(claim.get("value") or "")
            source_text = str(claim.get("source_text") or "")
            if key and value and source_text:
                groups.setdefault(key, []).append((record_id, {
                    "value": value,
                    "source_text": source_text,
                }))
    for claim_key, rows in groups.items():
        values = {row["value"] for _record_id, row in rows}
        if len(values) <= 1:
            continue
        comparison = {
            "claim_key": claim_key,
            "values": sorted(values),
            "evidence_ids": [record_id for record_id, _row in rows],
        }
        for record_id, _row in rows:
            result[record_id]["content_conflicts"].append(comparison)
    return result


def merge_evidence_observations(
    assessments: dict[str, dict],
    observations: list[dict[str, Any]],
    *,
    domain: str,
) -> dict[str, dict]:
    """Merge explicit material-quality details into existing assessment rows."""

    result = {key: dict(value) for key, value in (assessments or {}).items()}
    rules = get_domain_followups(domain).evidence
    for observation in observations:
        matched_key = ""
        explicit_rule = next(
            (
                item for item in rules
                if item.id == str(observation.get("rule_id") or "")
            ),
            None,
        )
        if explicit_rule:
            matched_key = explicit_rule.id
            if matched_key not in result:
                result[matched_key] = {
                    "rule_id": explicit_rule.id,
                    "evidence_key": explicit_rule.evidence_key,
                    "canonical_item": explicit_rule.item,
                    "availability": "uploaded_copy",
                    "authenticity": "not_verified",
                    "relevance": "potentially_relevant",
                    "legal_admissibility": "not_determined",
                    "purpose": explicit_rule.purpose,
                    "alternatives": list(explicit_rule.alternatives),
                    "limitations": [],
                    "answer_excerpt": observation.get("source_text", ""),
                    "requirement_id": observation.get("requirement_id", ""),
                }
        for key, record in result.items():
            if matched_key:
                break
            rule = next(
                (
                    item for item in rules
                    if item.id == record.get("rule_id")
                    or item.evidence_key == record.get("evidence_key")
                ),
                None,
            )
            if _observation_matches(
                observation.get("name", ""),
                str(record.get("canonical_item") or ""),
                rule,
            ):
                matched_key = key
                break
        if not matched_key:
            name = observation["name"]
            matched_key = _stable_id("evidence", name)
            availability = (
                "uploaded_copy"
                if observation.get("uploaded_copy")
                else "user_claimed_present"
            )
            result[matched_key] = {
                "rule_id": matched_key,
                "evidence_key": matched_key,
                "canonical_item": name,
                "availability": availability,
                "authenticity": "not_verified",
                "relevance": "potentially_relevant",
                "legal_admissibility": "not_determined",
                "purpose": "需结合具体争议判断证明目的",
                "alternatives": [],
                "limitations": [],
                "answer_excerpt": observation["source_text"],
            }
        record = dict(result[matched_key])
        if observation.get("uploaded_copy"):
            record["availability"] = "uploaded_copy"
        for field in (
            "source_form",
            "completeness",
            "identity_visibility",
            "time_visibility",
            "acquisition_method",
            "case_specificity",
            "content_digest",
            "material_claims",
            "requirement_id",
        ):
            value = observation.get(field, "unknown")
            if value != "unknown":
                record[field] = value
        roles = _unique(
            list(record.get("proof_roles") or [])
            + list(observation.get("proof_roles") or [])
        )
        if roles:
            record["proof_roles"] = roles
        record["inspection_basis"] = (
            "uploaded_copy"
            if record.get("availability") == "uploaded_copy"
            else "user_statement"
        )
        record["quality_source_excerpt"] = observation["source_text"]
        result[matched_key] = record
    return _mark_evidence_content_conflicts(result)


def _record_to_item(key: str, record: dict[str, Any]) -> EvidenceItem:
    availability = str(record.get("availability") or "unclear")
    return EvidenceItem(
        id=key or _stable_id(
            "evidence", str(record.get("canonical_item") or "unknown")
        ),
        name=str(record.get("canonical_item") or "未命名材料"),
        availability=availability,
        source_form=str(record.get("source_form") or (
            "screenshot" if availability == "uploaded_copy" else "unknown"
        )),
        completeness=str(record.get("completeness") or "unknown"),
        identity_visibility=str(record.get("identity_visibility") or "unknown"),
        time_visibility=str(record.get("time_visibility") or "unknown"),
        acquisition_method=str(record.get("acquisition_method") or "unknown"),
        case_specificity=str(record.get("case_specificity") or "unknown"),
        proof_roles=[
            str(item) for item in (record.get("proof_roles") or [])
            if str(item) in ALLOWED_PROOF_ROLES
        ],
        authenticity_status=str(record.get("authenticity") or "not_verified"),
        inspection_basis=str(record.get("inspection_basis") or (
            "uploaded_copy" if availability == "uploaded_copy" else "user_statement"
        )),
        source_excerpt=str(
            record.get("quality_source_excerpt")
            or record.get("answer_excerpt")
            or ""
        )[:300],
        content_conflicts=list(record.get("content_conflicts") or []),
        limitations=_unique(record.get("limitations") or []),
    )


def _default_record(name: str, availability: str) -> dict[str, Any]:
    return {
        "canonical_item": name,
        "availability": availability,
        "authenticity": "not_verified",
        "source_form": "unknown",
        "completeness": "unknown",
        "identity_visibility": "unknown",
        "time_visibility": "unknown",
        "acquisition_method": "unknown",
        "case_specificity": "unknown",
        "proof_roles": [],
        "limitations": [],
        "answer_excerpt": name,
    }


def _material_quality_gaps(item: EvidenceItem) -> list[str]:
    if item.availability not in {"uploaded_copy", "user_claimed_present"}:
        return []
    if item.case_specificity == "blank_or_reference":
        return []
    gaps: list[str] = []
    if item.case_specificity not in {"case_specific"}:
        gaps.append(_QUALITY_LABELS["case_specificity"])
    if item.source_form not in {
        "paper_original",
        "native_electronic",
        "exported_file",
    }:
        gaps.append(_QUALITY_LABELS["source_form"])
    if item.completeness != "complete":
        gaps.append(_QUALITY_LABELS["completeness"])
    if item.identity_visibility not in {"clear", "not_applicable"}:
        gaps.append(_QUALITY_LABELS["identity_visibility"])
    if item.time_visibility not in {"clear", "not_applicable"}:
        gaps.append(_QUALITY_LABELS["time_visibility"])
    if item.acquisition_method == "unknown":
        gaps.append(_QUALITY_LABELS["acquisition_method"])
    return gaps


def _target_for_rule(rule: EvidenceFollowup) -> ProofTarget:
    return ProofTarget(
        id=f"proof_target:{rule.id}",
        rule_id=rule.id,
        evidence_key=rule.evidence_key,
        label=rule.item,
        purpose=rule.purpose,
    )


def evaluate_evidence(
    *,
    domain: str,
    assessments: dict[str, dict] | None,
    confirmed_items: list[str] | None,
    unavailable_items: list[str] | None,
) -> EvidenceEvaluationReport:
    """Build a deterministic proof-coverage report from current case state."""

    rows = {key: dict(value) for key, value in (assessments or {}).items()}
    known_names = {
        str(record.get("canonical_item") or "")
        for record in rows.values()
    }
    for name in confirmed_items or []:
        if name and name not in known_names:
            rows[_stable_id("evidence", name)] = _default_record(
                name, "user_claimed_present"
            )
            known_names.add(name)
    for name in unavailable_items or []:
        if name and name not in known_names:
            rows[_stable_id("evidence", f"missing:{name}")] = _default_record(
                name, "unavailable"
            )
            known_names.add(name)

    items = [_record_to_item(key, record) for key, record in rows.items()]
    records_by_id = {
        key: record for key, record in rows.items()
    }
    targets = [
        _target_for_rule(rule)
        for rule in get_domain_followups(domain).evidence
    ]
    links: list[EvidenceLink] = []
    coverage_rows: list[EvidenceCoverage] = []

    for rule, target in zip(get_domain_followups(domain).evidence, targets):
        linked_items: list[EvidenceItem] = []
        basis_by_item: dict[str, str] = {}
        for item in items:
            record = records_by_id.get(item.id, {})
            bound_target_id = str(record.get("requirement_id") or "")
            if bound_target_id and bound_target_id != target.id:
                # A user-selected upload channel is an explicit scope choice.
                # Do not broaden that material to other proof targets merely
                # because generic keywords or model-proposed roles overlap.
                continue
            direct_rule_match = (
                record.get("rule_id") == rule.id
                or record.get("evidence_key") == rule.evidence_key
                or bound_target_id == target.id
            )
            catalog_match = evidence_rule_resolved(rule, [item.name])
            target_roles = EVIDENCE_KEY_PROOF_ROLES.get(
                rule.evidence_key, set()
            )
            role_match = bool(set(item.proof_roles) & target_roles)
            if not (direct_rule_match or catalog_match or role_match):
                continue
            linked_items.append(item)
            basis_by_item[item.id] = (
                "answered_targeted_question"
                if direct_rule_match
                else "catalog_material_mapping"
                if catalog_match
                else "proof_role_mapping"
            )
            links.append(EvidenceLink(
                evidence_id=item.id,
                target_id=target.id,
                proof_scope=target.purpose,
                basis=basis_by_item[item.id],
                limitations=item.limitations,
            ))

        present = [
            item for item in linked_items
            if item.availability in {"uploaded_copy", "user_claimed_present"}
            and item.case_specificity != "blank_or_reference"
        ]
        reference_only = [
            item for item in linked_items
            if item.availability in {"uploaded_copy", "user_claimed_present"}
            and item.case_specificity == "blank_or_reference"
        ]
        conflicted = [
            item for item in linked_items if item.availability == "conflicted"
        ]
        content_conflicted = [
            item for item in present if item.content_conflicts
        ]
        missing = [
            item for item in linked_items if item.availability == "unavailable"
        ]
        limitations = _unique(
            limitation
            for item in present + conflicted
            for limitation in item.limitations
        )
        gaps_by_item = {
            item.id: _material_quality_gaps(item)
            for item in present
        }
        complete_present = [
            item for item in present if not gaps_by_item[item.id]
        ]
        quality_gaps = _unique(
            gap
            for item in present
            for gap in gaps_by_item[item.id]
        )
        if conflicted or content_conflicted:
            status = "conflicted"
            next_action = (
                "先核对不同材料中同一字段的数值或日期，并以可回查的"
                "原始平台、银行或机构记录为准。"
                if content_conflicted
                else "先确认该材料目前是否实际持有，并以最新陈述为准。"
            )
        elif complete_present:
            status = "preliminarily_covered"
            quality_gaps = []
            next_action = "保留原始载体；提交前仍需由受理机关核验。"
        elif present:
            status = "partially_covered"
            next_action = (
                "优先确认" + "、".join(quality_gaps[:3]) + "。"
            )
        elif missing:
            status = "known_missing"
            next_action = (
                "考虑使用替代材料：" + "、".join(rule.alternatives[:3])
                if rule.alternatives
                else "记录缺失原因，并寻找能够证明同一事实的其他材料。"
            )
        elif reference_only:
            status = "unresolved"
            next_action = (
                "当前上传的是空白模板或参考资料，需补充包含本案具体主体、"
                "时间、金额或处理结果的实际记录。"
            )
        else:
            status = "unresolved"
            next_action = (
                "先确认是否持有该类材料；没有时再寻找替代材料。"
            )
        coverage_rows.append(EvidenceCoverage(
            target_id=target.id,
            label=target.label,
            purpose=target.purpose,
            status=status,
            supporting_evidence_ids=[item.id for item in present + conflicted],
            quality_gaps=quality_gaps,
            limitations=limitations,
            next_action=next_action,
        ))

    counts = {
        status: sum(row.status == status for row in coverage_rows)
        for status in _STATUS_LABELS
    }
    return EvidenceEvaluationReport(
        items=items,
        targets=targets,
        links=links,
        coverage=coverage_rows,
        target_count=len(targets),
        preliminarily_covered_count=counts["preliminarily_covered"],
        partial_count=counts["partially_covered"],
        known_missing_count=counts["known_missing"],
        unresolved_count=counts["unresolved"] + counts["conflicted"],
    )


def evaluate_state_evidence(state: Any) -> EvidenceEvaluationReport:
    return evaluate_evidence(
        domain=str(getattr(state, "legal_domain", "") or "other"),
        assessments=getattr(state, "evidence_assessments", {}) or {},
        confirmed_items=getattr(state, "evidence_confirmed", []) or [],
        unavailable_items=getattr(state, "evidence_unavailable", []) or [],
    )


def merge_evidence_requirements(
    state: Any,
    report: EvidenceEvaluationReport,
    *,
    basis_refs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Incrementally version the user-facing evidence checklist.

    The coverage engine owns the current proof targets.  This function adds
    case-fact provenance, legal/official basis and stable version semantics so
    the checklist can evolve without recreating duplicate rows every turn.
    """

    previous = {
        str(item.get("id") or ""): dict(item)
        for item in (getattr(state, "evidence_requirements", []) or [])
        if isinstance(item, dict) and item.get("id")
    }
    current_round = int(getattr(state, "round", 0) or 0)
    active_facts = [
        item for item in active_case_facts(getattr(state, "case_facts", []) or [])
        if item.get("status") in {"asserted", "uncertain", "conflicted"}
    ]
    fact_keys = [str(item.get("key") or "") for item in active_facts if item.get("key")]
    coverage_by_target = {item.target_id: item for item in report.coverage}
    rules = {
        rule.id: rule
        for rule in get_domain_followups(str(getattr(state, "legal_domain", "") or "other")).evidence
    }
    normalized_basis = []
    for item in basis_refs or []:
        if not isinstance(item, dict):
            continue
        ref = {
            "title": str(item.get("title") or "").strip(),
            "article_no": str(item.get("article_no") or "").strip(),
            "source_type": str(item.get("source_type") or "statute").strip(),
            "text": str(item.get("text") or "").strip()[:1_200],
            "issuer": str(item.get("issuer") or "").strip()[:200],
            "url": str(item.get("url") or "").strip()[:1_000],
            "law_id": str(item.get("law_id") or "").strip()[:200],
        }
        if ref["title"] or ref["article_no"]:
            normalized_basis.append(ref)
    normalized_basis.sort(key=lambda ref: (
        0
        if ref["source_type"] in {"official_process", "official_evidence_guidance"}
        and ref["url"]
        else 1,
        0 if ref["text"] else 1,
    ))
    normalized_basis = normalized_basis[:3]

    rows: list[dict[str, Any]] = []
    changed = False
    active_ids: set[str] = set()
    for target in report.targets:
        requirement_id = target.id
        active_ids.add(requirement_id)
        coverage = coverage_by_target.get(requirement_id)
        rule = rules.get(target.rule_id)
        old = previous.get(requirement_id, {})
        row = {
            "id": requirement_id,
            "rule_id": target.rule_id,
            "label": target.label,
            "proof_target": target.purpose,
            "priority": "required" if target.required_for_planning else "recommended",
            "status": coverage.status if coverage else "unresolved",
            "supporting_evidence_ids": list(coverage.supporting_evidence_ids) if coverage else [],
            "quality_gaps": list(coverage.quality_gaps) if coverage else [],
            "next_action": coverage.next_action if coverage else "先确认是否持有该类材料。",
            "alternatives": list(rule.alternatives[:4]) if rule else [],
            "trigger_fact_keys": fact_keys[-12:],
            "basis_refs": normalized_basis,
            "active": True,
            "created_round": int(old.get("created_round") or current_round),
            "updated_round": current_round,
        }
        comparable = {key: value for key, value in row.items() if key not in {"updated_round"}}
        old_comparable = {key: value for key, value in old.items() if key not in {"updated_round", "version"}}
        if comparable != old_comparable:
            changed = True
        rows.append(row)

    for requirement_id, old in previous.items():
        if requirement_id in active_ids or not old.get("active", True):
            continue
        changed = True
        rows.append({**old, "active": False, "updated_round": current_round})

    version = int(getattr(state, "evidence_requirement_version", 0) or 0)
    if changed or not version:
        version += 1
    for row in rows:
        row["version"] = version
    return rows[-80:], version


def coverage_for_rule(
    report: EvidenceEvaluationReport | dict[str, Any] | None,
    rule_id: str,
) -> EvidenceCoverage | None:
    if not report:
        return None
    parsed = (
        report
        if isinstance(report, EvidenceEvaluationReport)
        else EvidenceEvaluationReport.model_validate(report)
    )
    target_id = f"proof_target:{rule_id}"
    return next(
        (item for item in parsed.coverage if item.target_id == target_id),
        None,
    )


def format_evidence_coverage(
    report: EvidenceEvaluationReport | dict[str, Any] | None,
    *,
    max_targets: int = 4,
) -> str:
    if not report:
        return "（暂未形成证据覆盖评估）"
    parsed = (
        report
        if isinstance(report, EvidenceEvaluationReport)
        else EvidenceEvaluationReport.model_validate(report)
    )
    item_names = {item.id: item.name for item in parsed.items}
    lines: list[str] = []
    for row in parsed.coverage[:max_targets]:
        status = _STATUS_LABELS.get(row.status, "状态待确认")
        materials = _unique(
            item_names.get(item_id, "")
            for item_id in row.supporting_evidence_ids
        )
        material_note = f"；关联材料：{'、'.join(materials)}" if materials else ""
        gap_note = (
            f"；待核验：{'、'.join(row.quality_gaps[:3])}"
            if row.quality_gaps else ""
        )
        lines.append(
            f"- {row.label}：{status}{material_note}；可能用途：{row.purpose}"
            f"{gap_note}；下一步：{row.next_action}"
        )
    if not lines:
        return "（当前领域尚未配置证明目标）"
    lines.append(f"- 说明：{parsed.disclaimer}")
    return "\n".join(lines)
