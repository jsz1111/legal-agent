"""Legal Q&A and case guidance share a retrieval display projection."""
from __future__ import annotations

import json

import pytest

from src.agents.tools.worker_tools import (
    _append_retrieved_statute_text,
    _legal_qa_retrieval_debug,
)
from src.api.routers.chat import _pop_legal_qa_debug_artifact


def _artifacts():
    return [{
        "source_type": "statute",
        "context": "## 核心法条\n《道路交通安全法实施条例》第四十条\n红色叉形灯亮时，禁止本车道车辆通行。",
        "hits": [{
            "law_id": "29",
            "title": "中华人民共和国道路交通安全法实施条例",
            "article_no": "第四十条",
            "text": "第四十条　车道信号灯表示：红色叉形灯亮时，禁止本车道车辆通行。",
            "score": 0.97,
        }],
    }]


def test_qa_retrieval_projection_contains_title_article_and_body():
    debug = _legal_qa_retrieval_debug(_artifacts())

    assert debug is not None
    assert "红色叉形灯亮时" in debug["statute_hits"]
    assert debug["followup_basis_refs"] == [{
        "law_id": "29",
        "title": "中华人民共和国道路交通安全法实施条例",
        "article_no": "第四十条",
        "source_type": "statute_index",
        "text": "第四十条　车道信号灯表示：红色叉形灯亮时，禁止本车道车辆通行。",
    }]


def test_qa_answer_appends_retrieved_body_instead_of_bare_citation():
    debug = _legal_qa_retrieval_debug(_artifacts())
    answer = _append_retrieved_statute_text(
        "## 核心结论\n\n不能通行。\n\n## 依据来源\n\n- 《道路交通安全法实施条例》第四十条",
        debug,
    )

    assert "## 检索法条原文" in answer
    assert "红色叉形灯亮时，禁止本车道车辆通行" in answer


def test_qa_answer_does_not_duplicate_body_already_in_source_section():
    debug = _legal_qa_retrieval_debug(_artifacts())
    original = (
        "## 核心结论\n\n不能通行。\n\n## 依据来源\n\n"
        "- 《道路交通安全法实施条例》第四十条\n\n"
        "  > 第四十条　车道信号灯表示：红色叉形灯亮时，禁止本车道车辆通行。"
    )

    answer = _append_retrieved_statute_text(original, debug)

    assert answer == original
    assert "## 检索法条原文" not in answer


def test_qa_answer_does_not_duplicate_body_quoted_before_source_section():
    debug = _legal_qa_retrieval_debug(_artifacts())
    original = (
        "## 核心结论\n\n"
        "> 车道信号灯表示：红色叉形灯亮时，禁止本车道车辆通行。\n\n"
        "## 依据来源\n\n- 《道路交通安全法实施条例》第四十条，正文见上。"
    )

    answer = _append_retrieved_statute_text(original, debug)

    assert answer == original
    assert "## 检索法条原文" not in answer


class _Redis:
    def __init__(self, payload):
        self.values = {"legal_qa_last_debug:u:s": json.dumps(payload, ensure_ascii=False)}

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)


@pytest.mark.asyncio
async def test_qa_retrieval_projection_is_popped_for_shared_inspector():
    payload = _legal_qa_retrieval_debug(_artifacts())
    redis = _Redis(payload)

    debug = await _pop_legal_qa_debug_artifact(redis, "u", "s")

    assert debug is not None
    assert "红色叉形灯亮时" in debug.statute_hits
    assert debug.followup_basis_refs[0]["article_no"] == "第四十条"
    assert redis.values == {}
