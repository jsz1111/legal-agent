"""验证法条 RAG 的自省/幻觉校验接入。

search_statutes 生成回答后，用检索到的法条原文校验回答是否有依据：
- is_grounded=True  → 原样返回
- is_grounded=False → 保留回答，追加免责提示 + 可疑陈述清单
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agents.legal_knowledge.statute_rag import (
    search_statutes, _apply_grounding_check,
)


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=text))
    return llm


def test_grounded_answer_unchanged():
    """校验通过时，回答原样返回，无追加内容。"""
    llm = _llm_returning("正常回答")
    with patch(
        "src.agents.legal_knowledge.hallucination_check.check_hallucination",
        new=AsyncMock(return_value={"is_grounded": True, "confidence": 0.9,
                                    "unsupported_claims": []}),
    ):
        out = asyncio.run(_apply_grounding_check("q", "法条证据", "正常回答", llm))
    assert out == "正常回答"


def test_ungrounded_answer_gets_disclaimer():
    """校验不通过时，保留原回答并追加免责提示 + 可疑陈述。"""
    with patch(
        "src.agents.legal_knowledge.hallucination_check.check_hallucination",
        new=AsyncMock(return_value={
            "is_grounded": False, "confidence": 0.3,
            "unsupported_claims": ["赔偿标准为月薪的10倍"],
        }),
    ):
        out = asyncio.run(_apply_grounding_check("q", "法条证据", "原始回答", MagicMock()))
    assert out.startswith("原始回答")
    assert "可信度提示" in out
    assert "12348" in out
    assert "赔偿标准为月薪的10倍" in out


def test_no_hits_short_circuits():
    """检索无结果时，不触发校验，直接返回提示语。"""
    with patch(
        "src.agents.legal_knowledge.statute_rag.search_statutes_raw",
        new=AsyncMock(return_value=[]),
    ):
        out = asyncio.run(search_statutes(
            "q", MagicMock(), MagicMock(), _llm_returning("x"),
            db_session=None, verify_grounding=True,
        ))
    assert "未找到" in out


def test_verify_grounding_off_skips_check():
    """verify_grounding=False 时不调用幻觉检测。"""
    hits = [{"law_id": "1", "article_no": "第1条", "domain": "labor", "text": "内容", "score": 0.9}]
    check_mock = AsyncMock(return_value={"is_grounded": False, "unsupported_claims": [], "confidence": 0.1})
    with patch("src.agents.legal_knowledge.statute_rag.search_statutes_raw",
               new=AsyncMock(return_value=hits)), \
         patch("src.agents.legal_knowledge.statute_rag.format_statute_context",
               return_value="ctx"), \
         patch("src.agents.legal_knowledge.hallucination_check.check_hallucination", new=check_mock):
        out = asyncio.run(search_statutes(
            "q", MagicMock(), MagicMock(), _llm_returning("干净回答"),
            db_session=None, verify_grounding=False,
        ))
    assert out == "干净回答"
    check_mock.assert_not_awaited()


if __name__ == "__main__":
    test_grounded_answer_unchanged()
    test_ungrounded_answer_gets_disclaimer()
    test_no_hits_short_circuits()
    test_verify_grounding_off_skips_check()
    print("ALL PASS")
