"""法条语义检索：statute_index → rerank → PG 补充 law title → 生成回答。"""
from __future__ import annotations

import re

from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pathlib import Path
from src.agents.legal_knowledge.prompts import STATUTE_QA_PROMPT
from src.agents.legal_guide.retrieval_query import lexical_terms

COLLECTION_NAME = "statute_index"
CIVIL_CODE_LAW_ID = "75"  # 中华人民共和国民法典（基础私法，跨领域通用）


async def search_statutes_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 30,
    rerank_top_k: int = 8,
    domain: str = "",
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
    use_rrf: bool = True,
    sparse_query: str = "",  # BM25 专用查询（法律术语关键词），为空时退化为 question
    skip_rerank: bool = False,  # True 时跳过精排，由调用方统一 rerank（双路融合场景）
) -> list[dict]:
    """法条向量检索（支持 RRF 混合检索），返回原始结果列表。"""
    # 有领域时：返回领域内法条 + 民法典（基础私法，跨领域通用）
    # 无领域时：搜全库
    if domain:
        filter_expr = f'domain == "{domain}" || law_id == "{CIVIL_CODE_LAW_ID}"'
    else:
        filter_expr = None

    # RRF 混合检索（Dense + Sparse BM25）
    if use_rrf:
        try:
            from pymilvus import AnnSearchRequest, RRFRanker
            from milvus_model.sparse import BM25EmbeddingFunction

            # Dense 向量：HyDE 生成假设法条文本，再向量化（比直接向量化问句更贴近法条语义空间）
            if use_hyde and llm is not None:
                from src.agents.legal_knowledge.hyde import generate_hyde_embedding
                dense_vec = await generate_hyde_embedding(question, llm, embedding_model, mode="statute")
            else:
                dense_vec = await embedding_model.aembed_query(question)

            # Sparse 向量（BM25）—— 优先使用调用方提供的法律术语关键词
            import scipy.sparse as sp
            bm25_ef = BM25EmbeddingFunction()
            _bm25_model = Path(__file__).resolve().parent.parent.parent.parent / "models" / "bm25_statute.json"
            if _bm25_model.exists():
                bm25_ef.load(str(_bm25_model))
            _bm25_input = sparse_query or question  # sparse_query = confirmed_issues 关键词
            _sv = bm25_ef.encode_queries([_bm25_input])[0]
            _sv_csr = sp.csr_array(_sv)
            sparse_vec = {int(i): float(v) for i, v in zip(_sv_csr.indices, _sv_csr.data)}

            # 构建混合检索请求
            dense_req = AnnSearchRequest(
                data=[dense_vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=filter_expr,
            )
            sparse_req = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filter_expr,
            )

            # RRF 融合
            results = milvus_client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=top_k,
                output_fields=["law_id", "article_no", "domain", "text"],
            )

        except Exception as e:
            logger.warning("RRF 混合检索失败，降级为纯向量检索: {}", e, exc_info=True)
            use_rrf = False

    # 降级：纯 Dense 向量检索
    if not use_rrf:
        if use_hyde and llm is not None:
            from src.agents.legal_knowledge.hyde import generate_hyde_embedding
            query_vec = await generate_hyde_embedding(question, llm, embedding_model)
        else:
            query_vec = await embedding_model.aembed_query(question)

        try:
            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                limit=top_k,
                output_fields=["law_id", "article_no", "domain", "text"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning("法条检索失败: {}", e, exc_info=True)
            return []

    if not results or not results[0]:
        return []

    hits = [
        {
            "law_id": hit["entity"]["law_id"],
            "article_no": hit["entity"]["article_no"],
            "domain": hit["entity"]["domain"],
            "text": hit["entity"]["text"],
            "score": hit.get("distance", 0.0),
        }
        for hit in results[0]
    ]

    from src.agents.legal_knowledge.reranker import rerank_docs
    if skip_rerank:
        return hits  # 调用方负责统一精排
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    return reranked


async def _fetch_law_titles(
    hits: list[dict], db_session: AsyncSession
) -> dict[str, str]:
    """批量从 PG 取 law title，返回 {law_id_str: title}。"""
    from src.modules.legal.model import Law
    law_ids = list({int(h["law_id"]) for h in hits if h.get("law_id")})
    if not law_ids:
        return {}
    rows = (
        await db_session.execute(select(Law.id, Law.title).where(Law.id.in_(law_ids)))
    ).all()
    return {str(row.id): row.title for row in rows}


def format_statute_context(
    hits: list[dict],
    law_titles: dict[str, str],
    primary_count: int = 3,
) -> str:
    """将法条检索结果格式化为分层 LLM 上下文字符串。

    前 primary_count 条标注为「核心法条」，其余标注为「参考法条」，
    引导 LLM 优先引用高置信度结果，低置信度结果仅作补充。
    """
    if not hits:
        return ""

    def _display_article_no(value: object) -> str:
        article = str(value or "").strip()
        if not article:
            return "条号未标注"
        if article.startswith("第") and "条" in article:
            return article
        if re.fullmatch(r"[零〇一二三四五六七八九十百千万两\d]+(?:之[零〇一二三四五六七八九十百千万两\d]+)?", article):
            main, *sub = article.split("之", 1)
            return f"第{main}条" + (f"之{sub[0]}" if sub else "")
        return article

    def _fmt(hit: dict, idx: int) -> str:
        title = law_titles.get(str(hit["law_id"]), f"法律ID:{hit['law_id']}")
        return f"法条{idx}【{title} {_display_article_no(hit['article_no'])}】\n{hit['text']}"

    primary = hits[:primary_count]
    backup  = hits[primary_count:]

    sections: list[str] = []

    sections.append("## 核心法条（高度相关，优先引用）")
    sections.extend(_fmt(h, i + 1) for i, h in enumerate(primary))

    if backup:
        sections.append("## 参考法条（相关度次之，可作补充）")
        sections.extend(_fmt(h, primary_count + i + 1) for i, h in enumerate(backup))

    return "\n\n---\n\n".join(sections)


async def search_statutes(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    db_session: AsyncSession | None = None,
    domain: str = "",
    use_hyde: bool = True,
    verify_grounding: bool = True,
    retrieval_trace: dict | None = None,
) -> str:
    """法条 RAG 完整流程：检索 → 精排 → 补充标题 → 生成回答 → 自省校验。

    verify_grounding=True 时，生成回答后会用检索到的法条原文做幻觉校验，
    若回答含无法条支撑的陈述，追加免责提示并列出可疑内容——降低法条幻觉风险。
    """
    hits = await search_statutes_raw(
        question, embedding_model, milvus_client,
        domain=domain, llm=llm, use_hyde=use_hyde,
    )
    if not hits and db_session is not None:
        try:
            hits = await search_statutes_pg_fallback(
                domain=domain,
                issues=[question],
                db_session=db_session,
                limit=8,
            )
            if hits:
                logger.info(
                    "法条向量检索无结果，已使用 PostgreSQL 原文字面检索恢复 | hits={}",
                    len(hits),
                )
        except Exception as fallback_error:
            logger.warning("法条 PostgreSQL 降级检索失败: {}", fallback_error)
    if not hits:
        if retrieval_trace is not None:
            retrieval_trace.update({"hits": [], "context": ""})
        return "当前法条库中未找到与您问题相关的法律条文。"

    law_titles: dict[str, str] = {}
    if db_session is not None:
        law_titles = await _fetch_law_titles(hits, db_session)

    context = format_statute_context(hits, law_titles)
    if retrieval_trace is not None:
        retrieval_trace.update({
            "hits": [
                {
                    "law_id": str(hit.get("law_id") or ""),
                    "title": law_titles.get(
                        str(hit.get("law_id") or ""),
                        f"法律ID:{hit.get('law_id') or ''}",
                    ),
                    "article_no": str(hit.get("article_no") or ""),
                    "text": str(hit.get("text") or ""),
                    "score": hit.get("rerank_score", hit.get("score", 0.0)),
                }
                for hit in hits
            ],
            "context": context,
        })
    prompt = STATUTE_QA_PROMPT.format(question=question, context=context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    answer = response.content

    if verify_grounding:
        answer = await _apply_grounding_check(question, context, answer, llm)
    return answer


async def search_statutes_pg_fallback(
    domain: str,
    issues: list[str],
    db_session: AsyncSession,
    limit: int = 8,
) -> list[dict]:
    """PostgreSQL LIKE 兜底检索：Milvus 不可用或返回空时调用。

    按领域过滤 laws，对 articles.content 做关键词 LIKE 匹配，
    最多返回 limit 条，格式与 search_statutes_raw 返回值一致。
    """
    from sqlalchemy import or_
    from src.modules.legal.model import Article, Law

    keywords = _expand_pg_keywords(issues)
    query = (
        select(
            Article.content,
            Article.article_no,
            Law.title,
            Law.id.label("law_id"),
        )
        .join(Law, Article.law_id == Law.id)
    )
    if domain:
        query = query.where(Law.domain == domain)
    if keywords:
        conditions = [Article.content.ilike(f"%{kw}%") for kw in keywords]
        query = query.where(or_(*conditions))

    rows = (await db_session.execute(query.limit(max(limit * 4, 24)))).all()
    ranked_rows = sorted(
        rows,
        key=lambda row: sum(
            (3 if keyword in "".join(issues) else 1)
            for keyword in keywords
            if keyword in (row.content or "")
        ),
        reverse=True,
    )[:limit]
    logger.info("PG 法条补充检索 | domain={} keywords={} hits={}", domain, keywords, len(ranked_rows))
    return [
        {
            "law_id": str(r.law_id),
            "article_no": r.article_no,
            "domain": domain,
            "text": r.content,
            "score": 0.5,
        }
        for r in ranked_rows
    ]


def _expand_pg_keywords(issues: list[str]) -> list[str]:
    """把标准法律问题拆成更可能出现在法条原文中的字面词。"""
    action_prefixes = (
        "拖欠", "拒不", "拒绝", "未签订", "未支付", "未履行", "违法解除",
        "违法", "非法", "侵害", "侵犯", "销售", "造成",
    )
    generic_suffixes = ("纠纷", "争议")
    candidates: list[str] = []
    for raw in issues[:5]:
        issue = raw.strip()
        if not issue:
            continue
        candidates.append(issue)
        for prefix in action_prefixes:
            if issue.startswith(prefix) and len(issue) > len(prefix) + 1:
                candidates.extend([prefix, issue[len(prefix):]])
                break
        for suffix in generic_suffixes:
            if issue.endswith(suffix) and len(issue) > len(suffix) + 1:
                candidates.append(issue[:-len(suffix)])
        candidates.extend(part for part in re.split(r"[/、，,\s]+", issue) if len(part) >= 2)
    candidates.extend(lexical_terms(issues, limit=24))

    seen: set[str] = set()
    return [item for item in candidates if len(item) >= 2 and not (item in seen or seen.add(item))]


async def _apply_grounding_check(
    question: str, context: str, answer: str, llm: BaseChatModel,
) -> str:
    """对法条回答做幻觉校验，不可信时追加免责提示与可疑陈述清单。"""
    from src.agents.legal_knowledge.hallucination_check import check_hallucination

    result = await check_hallucination(question, context, answer, llm)
    if result.get("is_grounded", True):
        return answer

    unsupported = result.get("unsupported_claims", [])
    conf = result.get("confidence", 0.0)
    logger.warning(
        "法条回答幻觉校验未通过 | confidence={} unsupported={}", conf, unsupported,
    )
    note_lines = [
        "\n\n---",
        "⚠️ **可信度提示**：以下内容未能在检索到的法条中找到直接依据，请谨慎参考，",
        "建议拨打 **12348** 法律援助热线向专业律师核实：",
    ]
    if unsupported:
        note_lines += [f"  - {c}" for c in unsupported[:5]]
    return answer + "\n".join(note_lines)
