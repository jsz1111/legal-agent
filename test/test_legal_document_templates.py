from __future__ import annotations

import asyncio
import hashlib
import io

from docx import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfReader

from src.agents.legal_guide.doc_generator import (
    generate_legal_document,
    render_legal_docx,
)
from src.agents.legal_guide.document_templates import (
    list_official_templates,
    select_official_template,
)


def test_official_manifest_files_and_hashes_are_valid():
    templates = list_official_templates()
    assert len(templates) == 8

    for template in templates:
        assert template.blank_pdf_path.is_file()
        assert template.evidence_items
        digest = hashlib.sha256(template.blank_pdf_path.read_bytes()).hexdigest().upper()
        assert digest == template.blank_pdf_sha256
        reader = PdfReader(template.blank_pdf_path)
        assert len(reader.pages) >= 4
        first_page = "".join((reader.pages[0].extract_text() or "").split())
        assert "民事起诉状" in first_page


def test_template_selector_uses_case_type_and_procedural_stage():
    lease = select_official_template(
        "contracts_property_housing",
        "房东拒不退还租房押金",
    )
    assert lease and lease.template_id == "spc_2025_house_lease_complaint"

    consumer = select_official_template(
        "consumer_market",
        "网购商品存在质量问题，商家拒绝退款",
    )
    assert consumer and consumer.template_id == "spc_2025_sales_contract_complaint"

    assert select_official_template(
        "labor_social_security",
        "公司拖欠工资，尚未申请劳动仲裁",
    ) is None

    labor_litigation = select_official_template(
        "labor_social_security",
        "不服劳动仲裁裁决，准备向法院起诉",
    )
    assert labor_litigation
    assert labor_litigation.template_id == "spc_2025_labor_dispute_complaint"


def test_rendered_docx_is_editable_and_labels_source():
    template = select_official_template(
        "traffic_personal_injury",
        "交通事故责任认定后准备起诉",
    )
    assert template
    payload = render_legal_docx(
        template.title,
        "民事起诉状（机动车交通事故责任纠纷）\n"
        "当事人信息：\n原告：【请填写姓名】\n"
        "诉讼请求：\n1. 赔偿医疗费【请填写金额】元。",
        template,
    )
    assert payload.startswith(b"PK")

    document = Document(io.BytesIO(payload))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "法〔2025〕82号" in text
    assert "系统根据用户提供的信息生成" in text
    assert "非人民法院" in text


def test_generate_document_returns_official_and_generated_artifacts():
    llm = FakeListChatModel(
        responses=[
            "民事起诉状（劳动争议纠纷）\n"
            "当事人信息：\n原告：【请填写姓名】\n被告：某公司\n"
            "诉讼请求：\n1. 支付拖欠工资【请填写金额】元。\n"
            "事实与理由：已完成劳动仲裁。"
        ]
    )
    result = asyncio.run(
        generate_legal_document(
            legal_domain="labor_social_security",
            confirmed_issues=["拖欠劳动报酬"],
            collected_facts=["已经收到劳动仲裁裁决书，准备向法院起诉"],
            region="北京",
            evidence_confirmed=["劳动合同", "仲裁裁决书"],
            law_context_str="《劳动合同法》第三十条",
            llm=llm,
        )
    )

    assert result.official_template
    assert result.official_template.template_id == "spc_2025_labor_dispute_complaint"
    assert result.docx_bytes.startswith(b"PK")
    assert "姓名" in result.missing_fields
    assert "金额" in result.missing_fields
