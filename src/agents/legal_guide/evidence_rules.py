"""Source-aware evidence checklists for the legal guide workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.legal_guide.document_templates import (
    OfficialDocumentTemplate,
    list_official_templates,
)
from src.agents.legal_guide.prompts import EVIDENCE_TEMPLATES, GENERIC_EVIDENCE


@dataclass(frozen=True, slots=True)
class EvidenceChecklist:
    items: tuple[str, ...]
    authority_level: str
    title: str
    source: dict[str, Any] | None = None
    usage_note: str = ""

    @property
    def is_officially_grounded(self) -> bool:
        return self.authority_level == "official_form_derived"


def _context_text(context_parts: tuple[str | list[str], ...]) -> str:
    flattened: list[str] = []
    for part in context_parts:
        if isinstance(part, list):
            flattened.extend(str(item) for item in part if item)
        elif part:
            flattened.append(str(part))
    return "".join("".join(flattened).split())


def _select_evidence_template(
    legal_domain: str,
    context: str,
) -> OfficialDocumentTemplate | None:
    """Select an official form for evidence fields, independent of filing stage.

    Evidence preservation usually happens before filing, so the labor-dispute
    form can ground the checklist even when the user has not yet sued. For a
    domain with several forms, a keyword hit is required to avoid attaching the
    wrong case-type checklist.
    """
    candidates: list[tuple[int, OfficialDocumentTemplate]] = []
    domain_templates = [
        template
        for template in list_official_templates()
        if legal_domain in template.domains and template.evidence_items
    ]
    for template in domain_templates:
        keyword_hits = sum(1 for keyword in template.keywords if keyword in context)
        if template.requires_keyword and keyword_hits == 0:
            continue
        candidates.append((keyword_hits, template))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].template_id), reverse=True)
    best_score, best = candidates[0]
    if len(domain_templates) > 1 and best_score == 0:
        return None
    return best


def resolve_evidence_checklist(
    legal_domain: str,
    *context_parts: str | list[str],
) -> EvidenceChecklist:
    """Return the most specific traceable checklist available for the case."""
    context = _context_text(context_parts)
    template = _select_evidence_template(legal_domain, context)
    if template:
        source = template.public_metadata()
        source["collection_title"] = template.collection.title
        return EvidenceChecklist(
            items=tuple(template.evidence_items),
            authority_level="official_form_derived",
            title=f"{template.case_type}证据准备要素",
            source=source,
            usage_note=(
                "本清单依据国家级示范文本的事实要素和证据栏目整理，"
                "不是对个案必须提交材料的穷尽列举；具体以受理机关要求为准。"
            ),
        )

    fallback = EVIDENCE_TEMPLATES.get(legal_domain) or GENERIC_EVIDENCE
    return EvidenceChecklist(
        items=tuple(fallback),
        authority_level="system_guidance",
        title="通用证据保存建议",
        usage_note=(
            "当前未匹配到对应案由的国家级官方证据指引，"
            "本清单为系统通用整理，不是官方固定材料目录。"
        ),
    )


def resolve_state_evidence_checklist(state: Any) -> EvidenceChecklist:
    """Resolve against the accumulated state blackboard, not only one turn."""
    recent_messages = [
        str(getattr(message, "content", ""))
        for message in getattr(state, "messages", [])[-6:]
        if getattr(message, "content", "")
    ]
    return resolve_evidence_checklist(
        str(getattr(state, "legal_domain", "") or ""),
        list(getattr(state, "confirmed_issues", []) or []),
        list(getattr(state, "unmatched_issues", []) or []),
        list(getattr(state, "collected_facts", []) or []),
        recent_messages,
    )


def format_evidence_source(checklist: EvidenceChecklist) -> str:
    """Format provenance for prompts and deterministic user-facing output."""
    if not checklist.is_officially_grounded or not checklist.source:
        return checklist.usage_note
    source = checklist.source
    issuers = "、".join(source.get("issuers") or [])
    pages = source.get("source_pages") or []
    page_text = f"，官方源 PDF 第 {pages[0]}-{pages[1]} 页" if len(pages) == 2 else ""
    source_url = source.get("source_page_url") or ""
    source_link = f" 官方发布页：{source_url}" if source_url else ""
    return (
        f"依据：{issuers}《{source.get('collection_title')}》"
        f"（{source.get('document_no')}）中的{source.get('case_type')}示范文本"
        f"{page_text}。{checklist.usage_note}{source_link}"
    )
