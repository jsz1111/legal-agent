"""Versioned authoritative legal-document template registry."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PROJECT_ROOT / "resources" / "legal_document_templates"
MANIFEST_PATH = TEMPLATE_ROOT / "manifest.json"

LITIGATION_STAGE_TERMS = (
    "仲裁裁决",
    "不服仲裁",
    "裁决书",
    "向法院",
    "法院起诉",
    "提起诉讼",
    "诉讼",
)


class TemplateCollection(BaseModel):
    title: str
    document_no: str
    issuers: list[str]
    published_at: str
    effective_at: str
    source_page_url: str
    source_pdf_url: str
    source_pdf_sha256: str


class OfficialDocumentTemplate(BaseModel):
    template_id: str
    title: str
    case_type: str
    domains: list[str]
    keywords: list[str]
    requires_keyword: bool = False
    requires_litigation_stage: bool = False
    blank_pdf: str
    blank_pdf_sha256: str
    source_pages: tuple[int, int]
    claim_items: list[str] = Field(default_factory=list)
    fact_items: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    collection: TemplateCollection

    @property
    def blank_pdf_path(self) -> Path:
        path = (TEMPLATE_ROOT / self.blank_pdf).resolve()
        if TEMPLATE_ROOT.resolve() not in path.parents:
            raise ValueError("模板文件路径越界")
        return path

    def public_metadata(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "title": self.title,
            "case_type": self.case_type,
            "authority_level": "national_official",
            "issuers": self.collection.issuers,
            "document_no": self.collection.document_no,
            "published_at": self.collection.published_at,
            "effective_at": self.collection.effective_at,
            "source_page_url": self.collection.source_page_url,
            "source_pdf_url": self.collection.source_pdf_url,
            "source_pages": list(self.source_pages),
            "blank_pdf_sha256": self.blank_pdf_sha256,
        }


@lru_cache(maxsize=1)
def load_official_templates() -> dict[str, OfficialDocumentTemplate]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    collection = TemplateCollection.model_validate(payload["collection"])
    result: dict[str, OfficialDocumentTemplate] = {}
    for raw in payload.get("templates", []):
        template = OfficialDocumentTemplate.model_validate(
            {**raw, "collection": collection}
        )
        result[template.template_id] = template
    return result


def get_official_template(template_id: str) -> OfficialDocumentTemplate | None:
    return load_official_templates().get(template_id)


def list_official_templates() -> list[OfficialDocumentTemplate]:
    return list(load_official_templates().values())


def select_official_template(
    legal_domain: str,
    *context_parts: str | list[str],
) -> OfficialDocumentTemplate | None:
    """Select an official form from domain, case facts and procedural stage."""
    flattened: list[str] = []
    for part in context_parts:
        if isinstance(part, list):
            flattened.extend(str(item) for item in part)
        elif part:
            flattened.append(str(part))
    context = "".join("".join(flattened).split())
    has_litigation_stage = any(term in context for term in LITIGATION_STAGE_TERMS)

    ranked: list[tuple[int, OfficialDocumentTemplate]] = []
    for template in list_official_templates():
        if legal_domain not in template.domains:
            continue
        keyword_hits = sum(1 for keyword in template.keywords if keyword in context)
        if template.requires_keyword and keyword_hits == 0:
            continue
        if template.requires_litigation_stage and not has_litigation_stage:
            continue
        ranked.append((keyword_hits, template))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].template_id), reverse=True)
    best_score, best = ranked[0]
    if len(ranked) > 1 and best_score == 0:
        return None
    return best


def format_official_framework(template: OfficialDocumentTemplate) -> str:
    claims = "\n".join(
        f"{index}. {item}：【请按案情填写或填写无】"
        for index, item in enumerate(template.claim_items, start=1)
    )
    facts = "\n".join(
        f"{index}. {item}：【请按案情填写】"
        for index, item in enumerate(template.fact_items, start=1)
    )
    return (
        f"{template.title}\n\n"
        "当事人信息：\n"
        "原告/申请人：【请填写姓名或名称、证件信息、地址、联系方式】\n"
        "被告/被申请人：【请填写姓名或名称、证件或统一社会信用代码、地址、联系方式】\n\n"
        f"诉讼请求：\n{claims}\n\n"
        f"事实与理由：\n{facts}\n\n"
        "证据清单：\n【请逐项填写证据名称、来源和证明目的】\n\n"
        "对纠纷解决方式的意愿：【同意调解/不同意调解/暂不确定】\n\n"
        "具状人（签字、盖章）：【请填写】\n"
        "日期：【请填写】"
    )
