"""权威追问依据的可审计来源、版本和规则映射。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.agents.legal_guide.followup_catalog import load_followup_catalog
from src.modules.legal.model import AuthoritySource, AuthorityVersion, FollowupRuleCitation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BATCH04_REPORT = PROJECT_ROOT / "data/sources/formal/2026-07-23/batch-04/download_report.json"
BATCH02_REPORTS = (
    PROJECT_ROOT / "data/sources/formal/2026-07-23/batch-02/download_report.json",
    PROJECT_ROOT / "data/sources/formal/2026-07-23/batch-02/retry-01/download_report.json",
)
OFFICIAL_FORM_MANIFEST = PROJECT_ROOT / "resources/legal_document_templates/manifest.json"
OFFICIAL_FORM_PDF = (
    PROJECT_ROOT
    / "resources/legal_document_templates/sources/spc_2025/2025_67_types_pleading_models.pdf"
)


@dataclass(slots=True)
class SourceSnapshot:
    source_key: str
    title: str
    issuer: str
    source_type: str
    authority_level: str
    official_url: str
    domains: list[str]
    usage_note: str
    status: str
    version_key: str
    document_no: str = ""
    published_at: str = ""
    effective_from: str = ""
    effective_to: str = ""
    official_file_url: str = ""
    local_path: str = ""
    sha256: str = ""
    content_type: str = ""
    review_status: str = "pending_legal_review"
    verified_at: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CitationSnapshot:
    rule_id: str
    domain: str
    rule_type: str
    rule_text: str
    source_key: str
    locator: str
    source_excerpt: str
    derivation_note: str
    mapping_status: str


_BATCH04_SPECS = {
    "criminal_procedure_law": (["criminal_public_security"], "official_basis_derived"),
    "domestic_violence_law": (["family_vulnerable_groups"], "official_basis_derived"),
    "road_traffic_safety_law": (["traffic_personal_injury"], "official_basis_derived"),
    "basic_health_law": (["medical_education_tax"], "official_process_derived"),
    "education_law": (["medical_education_tax"], "official_process_derived"),
    "tax_collection_law": (["medical_education_tax"], "official_process_derived"),
    "environmental_protection_law": (["environment_pollution"], "official_basis_derived"),
    "telecom_fraud_law": (["cyber_data_fraud"], "official_process_derived"),
    "arbitration_law": (["mediation_notary_arbitration"], "official_basis_derived"),
    "peoples_mediation_law": (["mediation_notary_arbitration"], "official_basis_derived"),
    "notarization_law": (["mediation_notary_arbitration"], "official_basis_derived"),
}

_BATCH02_SPECS = {
    "labor_dispute_mediation_arbitration_law": {
        "domains": ["labor_social_security"],
        "issuer": "人力资源和社会保障部",
        "authority_level": "official_basis_derived",
        "published_at": "2007-12-29",
        "effective_from": "2008-05-01",
    },
    "consumer_rights_protection_law_current": {
        "domains": ["consumer_market"],
        "issuer": "全国人民代表大会常务委员会（国家市场监督管理总局转载）",
        "authority_level": "official_basis_derived",
        "published_at": "2013-10-25",
        "effective_from": "2014-03-15",
    },
    "administrative_reconsideration_law_2023": {
        "domains": ["administrative_remedies"],
        "issuer": "全国人民代表大会常务委员会（中国政府网公布）",
        "authority_level": "official_basis_derived",
        "published_at": "2023-09-01",
        "effective_from": "2024-01-01",
    },
    "personal_information_protection_law_current": {
        "domains": ["cyber_data_fraud"],
        "issuer": "全国人民代表大会常务委员会（国家统计局转载）",
        "authority_level": "official_process_derived",
        "published_at": "2021-08-20",
        "effective_from": "2021-11-01",
    },
}

DOMAIN_SOURCE_KEYS = {
    "labor_social_security": [
        "labor_dispute_mediation_arbitration_law",
        "beijing_labor_arbitration_guide",
        "spc_2025_pleading_models",
    ],
    "consumer_market": ["consumer_rights_protection_law_current", "spc_2025_pleading_models"],
    "contracts_property_housing": ["spc_2025_pleading_models"],
    "criminal_public_security": ["criminal_procedure_law"],
    "family_vulnerable_groups": ["spc_2025_pleading_models", "domestic_violence_law"],
    "traffic_personal_injury": ["spc_2025_pleading_models", "road_traffic_safety_law"],
    "medical_education_tax": ["basic_health_law", "education_law", "tax_collection_law"],
    "administrative_remedies": ["administrative_reconsideration_law_2023"],
    "intellectual_property": ["spc_2025_pleading_models"],
    "environment_pollution": ["environmental_protection_law"],
    "cyber_data_fraud": ["telecom_fraud_law", "personal_information_protection_law_current"],
    "mediation_notary_arbitration": ["arbitration_law", "peoples_mediation_law", "notarization_law"],
    "other": ["system_followup_guidance"],
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _integrity(path_value: str, expected_sha256: str) -> tuple[str, str]:
    if not path_value:
        return "", "source_located"
    path = PROJECT_ROOT / path_value
    if not path.is_file():
        return expected_sha256, "local_file_missing"
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if expected_sha256 and actual != expected_sha256.upper():
        return actual, "hash_mismatch"
    return actual, "integrity_verified_pending_legal_review"


def _batch04_sources() -> list[SourceSnapshot]:
    report = _json(BATCH04_REPORT)
    documents = {doc["id"]: doc for doc in report.get("documents", [])}
    sources = []
    for source_key, (domains, authority_level) in _BATCH04_SPECS.items():
        doc = documents[source_key]
        formats = doc.get("formats") or {}
        artifact = formats.get("pdf") or formats.get("docx") or {}
        local_path = artifact.get("saved_path") or ""
        expected_hash = artifact.get("sha256") or ""
        sha256, review_status = _integrity(local_path, expected_hash)
        published_at = doc.get("published_at") or ""
        version_key = f"{source_key}:{doc.get('bbbs') or published_at or sha256[:12]}"
        sources.append(SourceSnapshot(
            source_key=source_key,
            title=doc["title"],
            issuer=doc.get("authority") or "国家法律法规数据库",
            source_type="official_law",
            authority_level=authority_level,
            official_url=doc.get("detail_url") or doc.get("source_home_url") or "https://flk.npc.gov.cn/",
            domains=domains,
            usage_note="追问点由法律规定的权利、程序和材料要素提炼，不是发布机关的固定问卷。",
            status="active",
            version_key=version_key,
            published_at=published_at,
            effective_from=doc.get("effective_from") or "",
            official_file_url=artifact.get("download_request_url") or "",
            local_path=local_path,
            sha256=sha256,
            content_type=artifact.get("content_type") or "",
            review_status=review_status,
            verified_at=datetime.now(timezone.utc).isoformat(),
            source_metadata={
                "source_database": doc.get("source_database"),
                "bbbs": doc.get("bbbs"),
                "validity_code": doc.get("validity_code"),
                "historical_versions": doc.get("historical_versions") or [],
                "report_review_status": doc.get("review_status"),
                "report_production_status": doc.get("production_status"),
            },
        ))
    return sources


def _batch02_sources() -> list[SourceSnapshot]:
    records: dict[str, dict] = {}
    for report_path in BATCH02_REPORTS:
        for record in _json(report_path):
            if record.get("status") == "downloaded_full":
                records[record["id"]] = record
    sources = []
    for source_key, spec in _BATCH02_SPECS.items():
        record = records[source_key]
        local_path = record.get("saved_path") or ""
        sha256, review_status = _integrity(local_path, record.get("sha256") or "")
        version_seed = spec["published_at"] or sha256[:12]
        sources.append(SourceSnapshot(
            source_key=source_key,
            title=record["title"],
            issuer=spec["issuer"],
            source_type="official_law",
            authority_level=spec["authority_level"],
            official_url=record.get("url") or "",
            domains=spec["domains"],
            usage_note=record.get("note") or "追问由官方法律文本的程序和材料要素提炼。",
            status="active",
            version_key=f"{source_key}:{version_seed}",
            published_at=spec["published_at"],
            effective_from=spec["effective_from"],
            local_path=local_path,
            sha256=sha256,
            content_type="text/html",
            review_status=review_status,
            verified_at=datetime.now(timezone.utc).isoformat(),
            source_metadata={
                "captured_at": record.get("captured_at"),
                "permission_class": record.get("permission_class"),
                "access_constraints": record.get("access_constraints"),
            },
        ))
    return sources


def _form_source() -> SourceSnapshot:
    manifest = _json(OFFICIAL_FORM_MANIFEST)
    collection = manifest["collection"]
    local_path = _relative(OFFICIAL_FORM_PDF)
    sha256, review_status = _integrity(local_path, collection["source_pdf_sha256"])
    return SourceSnapshot(
        source_key="spc_2025_pleading_models",
        title=collection["title"],
        issuer="、".join(collection["issuers"]),
        source_type="official_form",
        authority_level="national_official_form_derived",
        official_url=collection["source_page_url"],
        domains=sorted({domain for item in manifest["templates"] for domain in item["domains"]}),
        usage_note="按官方示范文本的事实、请求和证据栏目提炼；不是发布机关对本案的审查结论。",
        status="active",
        version_key=f"spc_2025_pleading_models:{collection['document_no']}",
        document_no=collection["document_no"],
        published_at=collection["published_at"],
        effective_from=collection["effective_at"],
        official_file_url=collection["source_pdf_url"],
        local_path=local_path,
        sha256=sha256,
        content_type="application/pdf",
        review_status=review_status,
        verified_at=datetime.now(timezone.utc).isoformat(),
        source_metadata={"templates": manifest["templates"]},
    )


def _manual_sources() -> list[SourceSnapshot]:
    guide_path = "data/samples/parsed/beijing_arbitration_worker_manual/beijing_arbitration_worker_manual.md"
    guide_hash, guide_review = _integrity(guide_path, "")
    guide = SourceSnapshot(
        source_key="beijing_labor_arbitration_guide",
        title="劳动人事争议仲裁线上申请办事指南",
        issuer="北京市人力资源和社会保障局",
        source_type="official_guide",
        authority_level="official_basis_derived",
        official_url="https://rsj.beijing.gov.cn/ywsite/bjsrlsbfwgf/ldgx1/ldqyybz/202312/t20231229_3518839.html",
        domains=["labor_social_security"],
        usage_note="用于提炼仲裁请求、主体、事实理由及申请材料要素，不是固定问卷。",
        status="active",
        version_key=f"beijing_labor_arbitration_guide:{guide_hash[:12]}",
        local_path=guide_path,
        sha256=guide_hash,
        content_type="text/markdown",
        review_status=guide_review,
        verified_at=datetime.now(timezone.utc).isoformat(),
    )
    system = SourceSnapshot(
        source_key="system_followup_guidance",
        title="通用法律事项梳理规则",
        issuer="系统规则库",
        source_type="system_rule",
        authority_level="system_guidance",
        official_url="",
        domains=["other"],
        usage_note="无法稳定归类时使用，不冒充官方固定问卷。",
        status="system_only",
        version_key="system_followup_guidance:v1",
        review_status="system_only",
    )
    return [guide, system]


def build_source_snapshots() -> list[SourceSnapshot]:
    sources = [_form_source(), *_batch02_sources(), *_batch04_sources(), *_manual_sources()]
    by_key = {source.source_key: source for source in sources}
    missing = sorted({key for keys in DOMAIN_SOURCE_KEYS.values() for key in keys} - set(by_key))
    if missing:
        raise ValueError(f"权威来源映射缺少定义: {missing}")
    return sorted(sources, key=lambda item: item.source_key)


def _form_locator(domain: str, rule_type: str) -> str:
    manifest = _json(OFFICIAL_FORM_MANIFEST)
    pages = []
    case_types = []
    for template in manifest["templates"]:
        if domain not in template["domains"]:
            continue
        case_types.append(template["case_type"])
        source_pages = template.get("source_pages") or []
        if len(source_pages) == 2:
            pages.append(f"{source_pages[0]}-{source_pages[1]}")
    column = "事实栏目" if rule_type == "fact" else "证据和证明目的栏目"
    return f"官方PDF第{'、'.join(pages)}页；{'、'.join(case_types)}示范文本的{column}"


def build_citation_snapshots(sources: list[SourceSnapshot] | None = None) -> list[CitationSnapshot]:
    source_by_key = {source.source_key: source for source in (sources or build_source_snapshots())}
    catalog = load_followup_catalog()
    citations = []
    for domain, domain_rules in catalog.domains.items():
        source_keys = DOMAIN_SOURCE_KEYS[domain]
        for rule_type, rules in (("fact", domain_rules.facts), ("evidence", domain_rules.evidence)):
            for rule in rules:
                why = rule.why if rule_type == "fact" else rule.purpose
                rule_text = rule.question
                for source_key in source_keys:
                    source = source_by_key[source_key]
                    if source.source_type == "official_form":
                        locator = _form_locator(domain, rule_type)
                        mapping_status = "source_located"
                    elif source.source_type == "official_guide":
                        locator = "线上申请的受理条件、仲裁请求、事实理由和申请材料栏目"
                        mapping_status = "source_located"
                    elif source.source_type == "system_rule":
                        locator = "系统通用规则"
                        mapping_status = "system_only"
                    else:
                        locator = "尚需人工标注到具体条款；当前仅登记官方文件级来源"
                        mapping_status = "needs_pinpoint"
                    citations.append(CitationSnapshot(
                        rule_id=rule.id,
                        domain=domain,
                        rule_type=rule_type,
                        rule_text=rule_text,
                        source_key=source_key,
                        locator=locator,
                        source_excerpt="",
                        derivation_note=f"为{why}，将官方文件中的事实、程序或材料要素转化为单轮追问。",
                        mapping_status=mapping_status,
                    ))
    return citations


def format_domain_authority_summary(domain: str) -> str:
    source_by_key = {source.source_key: source for source in build_source_snapshots()}
    lines = []
    for key in DOMAIN_SOURCE_KEYS.get(domain, DOMAIN_SOURCE_KEYS["other"]):
        source = source_by_key[key]
        status = "官方来源已定位，具体条款仍待精标" if source.source_type == "official_law" else "来源及栏目已定位"
        if source.source_type == "system_rule":
            status = "系统通用规则，非官方固定问卷"
        line = f"- {source.issuer}《{source.title}》：{status}"
        if source.document_no:
            line += f"；文号{source.document_no}"
        if source.official_url:
            line += f"；{source.official_url}"
        lines.append(line)
    return "\n".join(lines)


async def sync_authority_registry(session) -> dict[str, int]:
    sources = build_source_snapshots()
    citations = build_citation_snapshots(sources)
    version_by_source: dict[str, AuthorityVersion] = {}

    for snapshot in sources:
        source = (await session.execute(
            select(AuthoritySource).where(AuthoritySource.source_key == snapshot.source_key)
        )).scalar_one_or_none()
        if source is None:
            source = AuthoritySource(source_key=snapshot.source_key, title=snapshot.title,
                                     issuer=snapshot.issuer, source_type=snapshot.source_type,
                                     authority_level=snapshot.authority_level)
            session.add(source)
        for field_name in (
            "title", "issuer", "source_type", "authority_level", "official_url",
            "domains", "usage_note", "status",
        ):
            setattr(source, field_name, getattr(snapshot, field_name))
        await session.flush()

        version = (await session.execute(
            select(AuthorityVersion).where(AuthorityVersion.version_key == snapshot.version_key)
        )).scalar_one_or_none()
        if version is None:
            version = AuthorityVersion(source_id=source.id, version_key=snapshot.version_key)
            session.add(version)
        version.source_id = source.id
        for field_name in (
            "document_no", "published_at", "effective_from", "effective_to", "official_file_url",
            "local_path", "sha256", "content_type", "review_status", "verified_at", "source_metadata",
        ):
            setattr(version, field_name, getattr(snapshot, field_name))
        await session.flush()
        version_by_source[snapshot.source_key] = version

    for snapshot in citations:
        version = version_by_source[snapshot.source_key]
        citation = (await session.execute(
            select(FollowupRuleCitation).where(
                FollowupRuleCitation.rule_id == snapshot.rule_id,
                FollowupRuleCitation.source_version_id == version.id,
            )
        )).scalar_one_or_none()
        if citation is None:
            citation = FollowupRuleCitation(
                rule_id=snapshot.rule_id,
                domain=snapshot.domain,
                rule_type=snapshot.rule_type,
                rule_text=snapshot.rule_text,
                source_version_id=version.id,
            )
            session.add(citation)
        for field_name in (
            "domain", "rule_type", "rule_text", "locator", "source_excerpt",
            "derivation_note", "mapping_status",
        ):
            setattr(citation, field_name, getattr(snapshot, field_name))

    await session.commit()
    return {"sources": len(sources), "versions": len(sources), "citations": len(citations)}


def build_authority_index_rows() -> list[dict[str, str]]:
    sources = build_source_snapshots()
    source_by_key = {source.source_key: source for source in sources}
    rows = []
    for citation in build_citation_snapshots(sources):
        source = source_by_key[citation.source_key]
        stable_id = hashlib.sha1(
            f"{citation.rule_id}:{source.version_key}".encode("utf-8")
        ).hexdigest()
        rows.append({
            "id": stable_id,
            "domain": citation.domain,
            "rule_id": citation.rule_id,
            "rule_type": citation.rule_type,
            "source_key": source.source_key,
            "title": source.title,
            "source_url": source.official_url,
            "locator": citation.locator,
            "mapping_status": citation.mapping_status,
            "text": (
                f"追问：{citation.rule_text}\n转化说明：{citation.derivation_note}\n"
                f"依据来源：{source.issuer}《{source.title}》\n定位：{citation.locator}\n"
                f"适用边界：{source.usage_note}"
            )[:60000],
        })
    return rows


def export_registry_payload() -> dict[str, Any]:
    sources = build_source_snapshots()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [asdict(source) for source in sources],
        "citations": [asdict(citation) for citation in build_citation_snapshots(sources)],
    }
