"""权威模板选择、参考文书生成与 DOCX 输出。"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from src.agents.legal_guide.document_templates import (
    OfficialDocumentTemplate,
    format_official_framework,
    select_official_template,
    select_related_official_template,
)
from src.agents.legal_guide.prompts import (
    DOC_AUDIT_PROMPT, DOC_GEN_PROMPT, DOC_TYPE_MAP, DOC_TEMPLATES, DOMAIN_LABELS,
)


@dataclass(slots=True)
class GeneratedLegalDocument:
    doc_type: str
    text: str
    docx_bytes: bytes
    filename: str
    missing_fields: list[str]
    official_template: OfficialDocumentTemplate | None = None
    related_official_template: OfficialDocumentTemplate | None = None


def _extract_missing_fields(text: str) -> list[str]:
    fields = {
        value.strip() or "未填写内容"
        for value in re.findall(r"【请填写(?:或核实)?([^】]*)】", text)
    }
    return sorted(fields)


_UNSUPPORTED_FACT_PHRASES = (
    "提前一个月通知",
    "双方共同签署",
    "已按约履行全部义务",
    "已经按约履行全部义务",
    "无拖欠租金",
    "没有拖欠租金",
    "多次催告",
    "协商未果",
    "交付时状态良好",
    "维修已经发生",
    "已经完成维修",
    "证明损坏原因",
    "证明责任归属",
)


def _deterministic_fact_guard(text: str, allowed_facts: list[str]) -> str:
    """在 LLM 审校之外再做一次保守规则兜底，避免常见套话进入文书。"""
    allowed = "；".join(allowed_facts)
    guarded = text
    for phrase in _UNSUPPORTED_FACT_PHRASES:
        if phrase not in allowed:
            guarded = guarded.replace(phrase, "【请填写或核实相关事实】")
    return guarded


def _deterministic_legal_guard(text: str, legal_domain: str, doc_type: str) -> str:
    """Remove legally incompatible boilerplate that an LLM may reintroduce."""
    if legal_domain != "labor_social_security" or "劳动仲裁" not in doc_type:
        return text
    lines = []
    for line in text.splitlines():
        if re.search(r"仲裁费[^。；\n]*(?:承担|负担|缴纳|收取)", line):
            continue
        # 第八十五条加付赔偿属于劳动行政部门责令限期支付后仍逾期不付的
        # 行政执法后果，不应直接列为劳动仲裁申请事项。
        if re.search(r"(?:裁决|请求|申请)[^。；\n]{0,100}加付赔偿金", line):
            continue
        lines.append(line)
    guarded = "\n".join(lines).strip()
    guarded = re.sub(
        r"依据《中华人民共和国劳动合同法》第八十五条[^。\n]*加付赔偿金[。；]?",
        "",
        guarded,
    )
    guarded = guarded.replace("无正当理由无故拖欠", "未按约定支付")
    guarded = guarded.replace("无故拖欠申请人工资", "拖欠申请人工资")
    return guarded


def _clean_line(line: str) -> str:
    value = line.strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = value.replace("**", "")
    return value


def _set_run_font(run, name: str, size: int, bold: bool = False) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold


def render_legal_docx(
    doc_type: str,
    doc_text: str,
    official_template: OfficialDocumentTemplate | None = None,
) -> bytes:
    """Render an editable DOCX while keeping source and system status explicit."""
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)

    lines = [_clean_line(line) for line in doc_text.splitlines()]
    lines = [line for line in lines if line]
    title_written = False
    for line in lines:
        if not title_written:
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_run_font(paragraph.add_run(line), "黑体", 18, True)
            title_written = True
            continue
        is_heading = line.rstrip("：") in {
            "当事人信息",
            "诉讼请求",
            "申请事项",
            "投诉请求",
            "事实与理由",
            "事实经过",
            "证据清单",
            "随附证据",
            "对纠纷解决方式的意愿",
        }
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Cm(0 if is_heading else 0.74)
        paragraph.paragraph_format.line_spacing = 1.5
        _set_run_font(
            paragraph.add_run(line),
            "黑体" if is_heading else "宋体",
            12,
            is_heading,
        )

    document.add_paragraph()
    source_heading = document.add_paragraph()
    _set_run_font(source_heading.add_run("模板来源与文件性质"), "黑体", 11, True)
    if official_template:
        collection = official_template.collection
        source_text = (
            f"结构依据：{collection.title}（{collection.document_no}），"
            f"发布机关：{'、'.join(collection.issuers)}，"
            f"自 {collection.effective_at} 起推广使用。官方原文："
            f"{collection.source_page_url}"
        )
    else:
        source_text = "当前文书类型尚未匹配到全国统一官方空白模板，结构为系统通用参考格式。"
    source_paragraph = document.add_paragraph()
    source_run = source_paragraph.add_run(source_text)
    _set_run_font(source_run, "宋体", 9)
    source_run.font.color.rgb = RGBColor(89, 89, 89)

    disclaimer = document.add_paragraph()
    disclaimer_run = disclaimer.add_run(
        "本文件为系统根据用户提供的信息生成的可编辑参考稿，非人民法院、司法行政机关或其他发布机关出具。"
    )
    _set_run_font(disclaimer_run, "宋体", 9, True)
    disclaimer_run.font.color.rgb = RGBColor(192, 57, 43)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


async def generate_legal_document(
    *,
    legal_domain: str,
    confirmed_issues: list[str],
    collected_facts: list[str] | None = None,
    region: str,
    evidence_confirmed: list[str],
    law_context_str: str,
    llm: BaseChatModel,
    requested_doc_type: str = "",
) -> GeneratedLegalDocument:
    """Generate a source-grounded draft plus an editable DOCX artifact."""
    collected_facts = collected_facts or []
    desired_doc_type = requested_doc_type or DOC_TYPE_MAP.get(legal_domain, "投诉信")
    context = "".join([*confirmed_issues, *collected_facts])
    litigation_requested = (
        "起诉状" in desired_doc_type
        or any(term in context for term in ("向法院", "法院起诉", "提起诉讼", "准备起诉", "诉讼阶段"))
    )
    official_template = (
        select_official_template(
            legal_domain,
            confirmed_issues,
            collected_facts,
            desired_doc_type,
        )
        if litigation_requested
        else None
    )
    related_official_template = official_template or select_related_official_template(
        legal_domain,
        confirmed_issues,
        collected_facts,
    )
    doc_type = (
        official_template.title
        if official_template
        else desired_doc_type
    )

    template = (
        format_official_framework(official_template)
        if official_template
        else DOC_TEMPLATES.get(doc_type, "")
    )
    template_section = (
        f"## 参考格式框架（请严格按此结构填充）\n```\n{template}\n```\n"
        if template else ""
    )
    if official_template:
        collection = official_template.collection
        template_source = (
            f"国家级官方示范文本：{collection.title}（{collection.document_no}）；"
            f"发布机关：{'、'.join(collection.issuers)}；"
            f"适用类型：{official_template.title}；"
            f"官方原文：{collection.source_page_url}。"
        )
    else:
        template_source = "未匹配到全国统一官方空白模板，使用系统通用参考格式，不得表述为官方模板。"

    prompt = DOC_GEN_PROMPT.format(
        doc_type=doc_type,
        confirmed_issues="、".join(confirmed_issues) or "法律纠纷",
        legal_domain=DOMAIN_LABELS.get(legal_domain, legal_domain or "法律"),
        region=region or "【请填写所在地区】",
        evidence_confirmed="、".join(evidence_confirmed) or "相关证据",
        collected_facts="；".join(collected_facts) or "（用户尚未提供完整事实，请保留占位符）",
        law_context=law_context_str or "（暂无检索到的具体法条）",
        template_source=template_source,
        template_section=template_section,
    )

    response = await llm.ainvoke([SystemMessage(content=prompt)])
    draft_text = str(response.content).strip()
    audit_prompt = DOC_AUDIT_PROMPT.format(
        draftable_facts="\n".join(f"- {item}" for item in collected_facts) or "（无；所有个案事实均须保留占位符）",
        evidence_confirmed="、".join(evidence_confirmed) or "（无）",
        draft_text=draft_text,
    )
    try:
        audited = await llm.ainvoke([SystemMessage(content=audit_prompt)])
        doc_text = str(audited.content).strip() or draft_text
    except Exception:
        # 审校失败不能阻断已经生成的参考稿；下方仍会执行规则式事实兜底。
        doc_text = draft_text
    doc_text = _deterministic_fact_guard(doc_text, collected_facts)
    doc_text = _deterministic_legal_guard(doc_text, legal_domain, doc_type)
    filename_base = official_template.template_id if official_template else doc_type
    filename_base = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", filename_base)
    return GeneratedLegalDocument(
        doc_type=doc_type,
        text=doc_text,
        docx_bytes=render_legal_docx(doc_type, doc_text, official_template),
        filename=f"{filename_base}_智能填写参考稿.docx",
        missing_fields=_extract_missing_fields(doc_text),
        official_template=official_template,
        related_official_template=related_official_template,
    )


async def generate_legal_doc(
    *,
    legal_domain: str,
    confirmed_issues: list[str],
    region: str,
    evidence_confirmed: list[str],
    law_context_str: str,
    llm: BaseChatModel,
    collected_facts: list[str] | None = None,
) -> tuple[str, str]:
    """Backward-compatible text-only wrapper."""
    result = await generate_legal_document(
        legal_domain=legal_domain,
        confirmed_issues=confirmed_issues,
        collected_facts=collected_facts,
        region=region,
        evidence_confirmed=evidence_confirmed,
        law_context_str=law_context_str,
        llm=llm,
    )
    return result.doc_type, result.text
