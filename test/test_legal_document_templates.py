from __future__ import annotations

import asyncio
import hashlib
import io

from docx import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfReader

from src.agents.legal_guide.doc_generator import (
    _deterministic_fact_guard,
    _deterministic_legal_guard,
    generate_legal_document,
    render_legal_docx,
)
from src.agents.legal_guide.formatters import requested_doc_type
from src.agents.legal_guide.document_templates import (
    list_official_templates,
    select_official_template,
    select_related_official_template,
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
    related_labor = select_related_official_template(
        "labor_social_security",
        "公司拖欠工资，尚未申请劳动仲裁",
    )
    assert related_labor
    assert related_labor.template_id == "spc_2025_labor_dispute_complaint"

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
    assert result.related_official_template == result.official_template
    assert result.docx_bytes.startswith(b"PK")
    assert "姓名" in result.missing_fields
    assert "金额" in result.missing_fields


def test_housing_generic_document_request_keeps_pre_litigation_stage():
    llm = FakeListChatModel(responses=[
        "催告函\n收件人：【请填写房东姓名】\n请返还押金500元。",
    ])

    result = asyncio.run(generate_legal_document(
        legal_domain="contracts_property_housing",
        confirmed_issues=["房屋租赁押金返还纠纷"],
        collected_facts=["押金500元", "一个月前退租"],
        region="北京",
        evidence_confirmed=["房屋租赁合同"],
        law_context_str="",
        llm=llm,
        requested_doc_type="催告函",
    ))

    assert result.doc_type == "催告函"
    assert result.official_template is None
    assert result.related_official_template
    assert result.related_official_template.template_id == "spc_2025_house_lease_complaint"


def test_explicit_housing_litigation_request_uses_official_complaint_template():
    llm = FakeListChatModel(responses=[
        "民事起诉状（房屋租赁合同纠纷）\n原告：【请填写姓名】\n诉讼请求：返还押金500元。",
    ])

    result = asyncio.run(generate_legal_document(
        legal_domain="contracts_property_housing",
        confirmed_issues=["房屋租赁押金返还纠纷"],
        collected_facts=["准备向法院起诉", "押金500元"],
        region="北京",
        evidence_confirmed=["房屋租赁合同"],
        law_context_str="",
        llm=llm,
        requested_doc_type="民事起诉状",
    ))

    assert result.official_template
    assert result.official_template.template_id == "spc_2025_house_lease_complaint"
    assert result.doc_type == "民事起诉状（房屋租赁合同纠纷）"


def test_document_audit_replaces_unsupported_boilerplate_facts():
    llm = FakeListChatModel(responses=[
        (
            "催告函\n本人已提前一个月通知退租，双方共同签署退房确认单，"
            "本人已按约履行全部义务且无拖欠租金。"
        ),
        (
            "催告函\n本人于一个月前退租。是否提前通知、是否由双方签署确认单、"
            "租金是否结清，均请【请填写或核实】。"
        ),
    ])

    result = asyncio.run(generate_legal_document(
        legal_domain="contracts_property_housing",
        confirmed_issues=["房屋租赁押金返还纠纷"],
        collected_facts=["一个月前退租", "押金500元"],
        region="",
        evidence_confirmed=["退房确认单"],
        law_context_str="",
        llm=llm,
        requested_doc_type="催告函",
    ))

    assert "提前一个月通知" not in result.text
    assert "共同签署" not in result.text
    assert "无拖欠租金" not in result.text
    assert "请填写或核实" in result.text


def test_document_fact_guard_still_works_when_llm_audit_fails():
    text = _deterministic_fact_guard(
        "催告函\n本人已提前一个月通知退租，双方共同签署确认单。",
        ["一个月前退租"],
    )

    assert "提前一个月通知" not in text
    assert "双方共同签署" not in text
    assert text.count("请填写或核实相关事实") == 2


def test_labor_arbitration_document_never_requests_arbitration_fee():
    text = _deterministic_legal_guard(
        "劳动仲裁申请书\n申请事项：\n1. 支付工资24000元；\n"
        "2. 裁决被申请人加付赔偿金12000元；\n"
        "3. 本案仲裁费由被申请人承担。\n"
        "依据《中华人民共和国劳动合同法》第八十五条，逾期不支付的加付赔偿金。",
        "labor_social_security",
        "劳动仲裁申请书",
    )

    assert "支付工资24000元" in text
    assert "仲裁费" not in text
    assert "加付赔偿金" not in text


def test_generic_arbitration_request_uses_domain_default_to_disambiguate():
    assert requested_doc_type("生成仲裁申请书", "劳动仲裁申请书") == "劳动仲裁申请书"
    assert requested_doc_type("生成仲裁申请书", "仲裁申请书") == "仲裁申请书"
