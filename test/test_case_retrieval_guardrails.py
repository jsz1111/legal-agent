"""类案领域过滤、主题补充与条件式 HyDE 的回归测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.legal_knowledge.case_rag import (
    GENERIC_CIVIL_DOMAIN,
    _case_domains_for,
    _generic_case_matches_topic,
    _topic_case_terms,
    search_cases_raw,
)


def test_consumer_vector_filter_does_not_mix_all_generic_civil_cases():
    assert _case_domains_for("consumer_market") == ("consumer_market",)


def test_structured_topic_fallback_can_consider_generic_civil_cases():
    domains = _case_domains_for("consumer_market", include_generic=True)

    assert domains == ("consumer_market", GENERIC_CIVIL_DOMAIN)


def test_food_safety_language_activates_food_case_terms_only():
    terms = _topic_case_terms("饭馆饭菜里吃出玻璃渣，属于食品安全问题")

    assert "食品" in terms
    assert "玻璃" in terms
    assert _topic_case_terms("房东不退租房押金") == ()


def test_food_topic_rejects_unrelated_generic_civil_causes():
    terms = _topic_case_terms("食品里有异物")

    assert _generic_case_matches_topic(
        GENERIC_CIVIL_DOMAIN,
        "网络购物合同纠纷",
        terms,
    ) is True
    assert _generic_case_matches_topic(
        GENERIC_CIVIL_DOMAIN,
        "劳动合同纠纷",
        terms,
    ) is False


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
