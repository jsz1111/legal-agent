"""Build retrieval inputs from provenance-aware case facts.

This module deliberately knows nothing about industries or example scenarios.
It turns the current legal issues and active fact atoms into two views of the
same case: a rich semantic query and a compact lexical query.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


_LEXICAL_FACT_CATEGORIES = {
    "relationship", "event", "claim", "time", "procedure", "harm",
}
_QUERY_STOPWORDS = {
    "用户", "本人", "对方", "目前", "现在", "已经", "仍然", "相关", "情况",
    "事情", "问题", "法律", "依据", "权利", "义务", "纠纷", "争议", "大概",
    "一个", "一种", "这个", "那个", "进行", "发生", "表示", "称", "认为", "属于",
}


def lexical_terms(values: Iterable[str], *, limit: int = 24) -> list[str]:
    """Extract reusable Chinese lexical terms without a scenario dictionary."""
    phrases = [" ".join(str(value or "").split()) for value in values]
    phrases = [value for value in phrases if value]
    if not phrases:
        return []

    tokens: list[str] = []
    try:
        import jieba

        for phrase in phrases:
            tokens.extend(jieba.lcut(phrase, cut_all=False))
    except ImportError:
        for phrase in phrases:
            tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,12}|[A-Za-z]{2,20}", phrase))

    result: list[str] = []
    for token in tokens:
        clean = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", token).strip()
        if (
            len(clean) < 2
            or clean.isdigit()
            or clean in _QUERY_STOPWORDS
            or clean in result
        ):
            continue
        result.append(clean[:24])
        if len(result) >= limit:
            break
    return result


def build_case_retrieval_inputs(
    issues: Iterable[str],
    case_facts: Iterable[dict[str, Any]],
) -> dict[str, list[str] | str]:
    """Return semantic phrases and lexical terms from the active case state.

    Only asserted facts influence lexical retrieval. Uncertain or conflicting
    statements remain available to the final reasoning layer, but cannot steer
    exact-match retrieval as if they had been established.
    """
    issue_phrases = [" ".join(str(item or "").split()) for item in issues]
    issue_phrases = [item for item in issue_phrases if item]
    semantic_facts: list[str] = []
    lexical_facts: list[str] = []
    for item in case_facts or []:
        if not isinstance(item, dict) or item.get("status") == "superseded":
            continue
        statement = " ".join(str(item.get("statement") or "").split())
        if not statement:
            continue
        semantic_facts.append(statement)
        if (
            item.get("status") == "asserted"
            and item.get("category") in _LEXICAL_FACT_CATEGORIES
        ):
            lexical_facts.append(statement)

    semantic_phrases = list(dict.fromkeys(issue_phrases + semantic_facts))
    lexical_phrases = list(dict.fromkeys(issue_phrases + lexical_facts))
    terms = lexical_terms(lexical_phrases)
    return {
        "semantic_phrases": semantic_phrases,
        "lexical_phrases": lexical_phrases,
        "sparse_query": " ".join(terms),
        "lexical_terms": terms,
    }
