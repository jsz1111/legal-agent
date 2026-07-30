"""类案领域过滤、主题补充与条件式 HyDE 的回归测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.legal_knowledge.case_rag import (
    GENERIC_CIVIL_DOMAIN,
    _case_domains_for,
    _case_detail_matches_topic,
    _generic_case_matches_domain,
    _topic_case_terms,
    search_cases_raw,
    format_case_context,
)


def test_consumer_vector_filter_does_not_mix_all_generic_civil_cases():
    assert _case_domains_for("consumer_market") == ("consumer_market",)


def test_structured_topic_fallback_can_consider_generic_civil_cases():
    domains = _case_domains_for("consumer_market", include_generic=True)

    assert domains == ("consumer_market", GENERIC_CIVIL_DOMAIN)


def test_topic_terms_are_extracted_dynamically_from_any_case_language():
    terms = _topic_case_terms("饭馆饭菜里吃出玻璃渣，属于食品安全问题")

    assert any("食品" in item for item in terms)
    assert any("玻璃" in item for item in terms)
    assert "押金" in _topic_case_terms("房东不退租房押金")


def test_prepaid_topic_rejects_broad_food_or_product_case():
    terms = _topic_case_terms("理发店会员卡充值后关门，卡内还有300元余额")
    prepaid_case = {
        "title": "健身房预付会员卡退费纠纷",
        "cause": "服务合同纠纷",
        "gist": "消费者充值后经营者停止营业，诉请退还余额",
        "retrieval_text": "预付式消费会员卡余额退还",
    }
    wine_case = {
        "title": "酒水产品销售者责任纠纷",
        "cause": "产品销售者责任纠纷",
        "gist": "消费者购买酒水后主张惩罚性赔偿",
        "retrieval_text": "支付宝支付酒水货款8000元",
    }

    assert "会员卡" in terms
    assert _case_detail_matches_topic(prepaid_case, terms) is True
    assert _case_detail_matches_topic(wine_case, terms) is False


def test_generic_real_judgments_are_filtered_by_cause_without_rewriting_domain():
    labor = {
        "cause": "追索劳动报酬纠纷",
        "title": "某公司与王某劳动争议案",
        "retrieval_text": "公司拖欠工资",
    }
    unrelated = {
        "cause": "民间借贷纠纷",
        "title": "张某与李某借款案",
        "retrieval_text": "返还借款",
    }

    assert _generic_case_matches_domain("labor_social_security", labor) is True
    assert _generic_case_matches_domain("labor_social_security", unrelated) is False


def test_labor_filter_rejects_construction_or_service_contract_even_if_text_mentions_wages():
    service_contract = {
        "cause": "承揽合同纠纷",
        "title": "陈某与钟某劳务合同纠纷案",
        "retrieval_text": "请求支付拖欠的劳务工资",
    }

    assert _generic_case_matches_domain("labor_social_security", service_contract) is False


def test_case_context_limits_long_retrieval_text_before_prompt_injection():
    hits = [{"id": "1", "text": "备用文本", "score": 0.9}]
    details = {
        1: {
            "title": "真实裁判文书",
            "case_number": "（2021）测01民初1号",
            "cause": "劳动争议",
            "retrieval_text": "案情内容" * 1000,
            "gist": "裁判要旨" * 300,
            "original_url": "https://wenshu.court.gov.cn/example",
        }
    }

    context = format_case_context(hits, details)

    assert len(context) < 2200
    assert "（2021）测01民初1号" in context
    assert "原始链接：https://wenshu.court.gov.cn/example" in context


def test_hyde_is_used_for_dense_fallback_without_removing_domain_filter():
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2])
    milvus = MagicMock()
    milvus.search.return_value = [[]]
    llm = MagicMock()
    hyde = AsyncMock(return_value=[0.3, 0.4])

    with patch(
        "src.agents.legal_knowledge.hyde.generate_hyde_embedding",
        new=hyde,
    ):
        result = asyncio.run(search_cases_raw(
            question="老板一直不给钱",
            embedding_model=embeddings,
            milvus_client=milvus,
            domain="labor_social_security",
            use_rrf=False,
            llm=llm,
            use_hyde=True,
        ))

    assert result == []
    hyde.assert_awaited_once()
    assert hyde.await_args.kwargs["mode"] == "case"
    assert milvus.search.call_args.kwargs["filter"] == 'domain == "labor_social_security"'
