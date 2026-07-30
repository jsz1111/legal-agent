from src.agents.legal_guide.retrieval_query import build_case_retrieval_inputs
from src.agents.legal_knowledge.statute_rag import _expand_pg_keywords


def test_retrieval_query_uses_asserted_claim_event_and_time_facts_generically():
    facts = [
        {"category": "claim", "statement": "要求退还未消费余额", "status": "asserted"},
        {"category": "event", "statement": "经营者停止营业", "status": "asserted"},
        {"category": "time", "statement": "充值一周后停止服务", "status": "asserted"},
        {"category": "event", "statement": "听说经营者可能转移财产", "status": "uncertain"},
    ]

    query = build_case_retrieval_inputs(["预付款消费纠纷"], facts)

    assert "预付款" in query["lexical_terms"]
    assert "退还" in query["lexical_terms"]
    assert any("停止营业" in item for item in query["semantic_phrases"])
    assert "转移" not in query["sparse_query"]


def test_pg_keyword_expansion_splits_compound_legal_issues():
    keywords = _expand_pg_keywords(["预付款消费纠纷", "要求退还未消费余额"])

    assert "预付款" in keywords
    assert "退还" in keywords
    assert "余额" in keywords
