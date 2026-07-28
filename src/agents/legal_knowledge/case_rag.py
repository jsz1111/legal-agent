"""类案语义检索：case_index → PG 补充完整案情 → 生成回答。"""
from __future__ import annotations

import asyncio
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_, select

from pathlib import Path
from src.agents.legal_knowledge.prompts import CASE_QA_PROMPT

COLLECTION_NAME = "case_index"
GENERIC_CIVIL_DOMAIN = "civil_case"
NO_DATA_DOMAINS = set()   # 无类案数据时动态添加，此处不预设

_CASE_DOMAIN_MAP: dict[str, tuple[str, ...]] = {
    "contracts_property_housing": ("contract_commercial", "real_estate_construction"),
    "family_vulnerable_groups": ("family_marriage",),
}

_FOOD_TOPIC_TRIGGERS = ("食品", "玻璃渣", "玻璃", "异物", "饭菜", "饭馆", "吃出")
_FOOD_CASE_TERMS = ("食品", "玻璃", "异物", "饭菜", "食物", "菜品")
_FOOD_GENERIC_CAUSE_MARKERS = (
    "网络购物合同纠纷",
    "产品责任纠纷",
    "产品销售者责任纠纷",
    "消费者权益",
)


def _case_domains_for(domain: str, *, include_generic: bool = False) -> tuple[str, ...]:
    domains = _CASE_DOMAIN_MAP.get(domain, (domain,)) if domain else ()
    if include_generic and domain and GENERIC_CIVIL_DOMAIN not in domains:
        domains = (*domains, GENERIC_CIVIL_DOMAIN)
    return domains


def _topic_case_terms(question: str) -> tuple[str, ...]:
    if any(trigger in question for trigger in _FOOD_TOPIC_TRIGGERS):
        return _FOOD_CASE_TERMS
    return ()


def _generic_case_matches_topic(domain: str, cause: str, terms: tuple[str, ...]) -> bool:
    if domain != GENERIC_CIVIL_DOMAIN or terms != _FOOD_CASE_TERMS:
        return True
    return any(marker in (cause or "") for marker in _FOOD_GENERIC_CAUSE_MARKERS)


async def search_cases_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 15,
    rerank_top_k: int = 5,
    domain: str = "",
    use_rrf: bool = True,
    sparse_query: str = "",  # BM25 专用查询，为空时退化为 question
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
) -> list[dict]:
    """类案向量检索（含 rerank + RRF），返回精排后的结果列表。"""
    filter_expr = None
    if domain:
        case_domains = _case_domains_for(domain)
        encoded_domains = [json.dumps(item, ensure_ascii=False) for item in case_domains]
        if len(encoded_domains) == 1:
            filter_expr = f"domain == {encoded_domains[0]}"
        else:
            filter_expr = f"domain in [{', '.join(encoded_domains)}]"

    dense_vec: list[float] | None = None
    hyde_vec: list[float] | None = None

    # RRF 混合检索（原始 Dense + 可选 HyDE Dense + Sparse BM25）
    if use_rrf:
        try:
            from pymilvus import AnnSearchRequest, RRFRanker
            from milvus_model.sparse import BM25EmbeddingFunction
            import jieba

            if use_hyde and llm is not None:
                from src.agents.legal_knowledge.hyde import generate_hyde_embedding
                dense_vec, hyde_vec = await asyncio.gather(
                    embedding_model.aembed_query(question),
                    generate_hyde_embedding(
                        question,
                        llm,
                        embedding_model,
                        mode="case",
                    ),
                )
            else:
                dense_vec = await embedding_model.aembed_query(question)

            # Sparse 向量（BM25）—— 优先使用法律术语关键词
            import scipy.sparse as sp
            bm25_ef = BM25EmbeddingFunction(analyzer=jieba.lcut)
            _bm25_model = Path(__file__).resolve().parent.parent.parent.parent / "models" / "bm25_case.json"
            if _bm25_model.exists():
                bm25_ef.load(str(_bm25_model))
            _bm25_input = sparse_query or question
            _sv = bm25_ef.encode_queries([_bm25_input])[0]
            _sv_csr = sp.csr_array(_sv)
            sparse_vec = {int(i): float(v) for i, v in zip(_sv_csr.indices, _sv_csr.data)}

            # 构建混合检索请求。HyDE 只增加一路召回，不替代原始问句。
            requests = [AnnSearchRequest(
                data=[dense_vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=filter_expr,
            )]
            if hyde_vec is not None:
                requests.append(AnnSearchRequest(
                    data=[hyde_vec],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    limit=top_k,
                    expr=filter_expr,
                ))
            requests.append(AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filter_expr,
            ))

            # RRF 融合
            results = milvus_client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=requests,
                ranker=RRFRanker(k=60),
                limit=top_k,
                output_fields=["id", "domain", "source", "text"],
            )

        except Exception as e:
            logger.warning(f"RRF 混合检索失败，降级为纯向量检索: {e}")
            use_rrf = False

    # 降级：纯 Dense 向量检索
    if not use_rrf:
        if use_hyde and llm is not None:
            if hyde_vec is None:
                from src.agents.legal_knowledge.hyde import generate_hyde_embedding
                hyde_vec = await generate_hyde_embedding(
                    question,
                    llm,
                    embedding_model,
                    mode="case",
                )
            query_vec = hyde_vec
        else:
            query_vec = dense_vec or await embedding_model.aembed_query(question)

        try:
            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                limit=top_k,
                output_fields=["id", "domain", "source", "text"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning(f"类案检索失败: {e}")
            return []

    if not results or not results[0]:
        return []

    hits = [
        {
            "id": hit["entity"]["id"],
            "domain": hit["entity"]["domain"],
            "source": hit["entity"]["source"],
            "text": hit["entity"]["text"],
            "score": hit.get("distance", 0.0),
        }
        for hit in results[0]
    ]

    # 添加 rerank 机制，提高案例相关性
    from src.agents.legal_knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    return reranked


async def _fetch_case_details(
    hits: list[dict], db_session: AsyncSession
) -> dict[int, dict]:
    """从 PG 批量取案例元数据，返回 {数据库主键: {...}}。

    同时进行数据质量校验：过滤掉 title 和 gist 都为空的低质量案例。
    """
    from src.modules.legal.model import LegalCase
    case_ids = [int(h["id"]) for h in hits if h.get("id")]
    if not case_ids:
        return {}
    rows = (
        await db_session.execute(
            select(
                LegalCase.id,
                LegalCase.title,
                LegalCase.gist,
                LegalCase.case_id,
                LegalCase.case_number,
                LegalCase.cause,
                LegalCase.court,
                LegalCase.region,
                LegalCase.procedure,
                LegalCase.judgment_date,
                LegalCase.legal_basis,
                LegalCase.original_url,
                LegalCase.retrieval_text,
            )
            .where(LegalCase.id.in_(case_ids))
        )
    ).all()

    # 数据质量校验：过滤低质量案例
    valid_cases = {}
    for row in rows:
        # 至少有标题或要旨其一，否则认为是低质量数据
        if row.title or row.gist:
            valid_cases[row.id] = {
                "title": row.title,
                "gist": row.gist,
                "case_id": row.case_id,
                "case_number": row.case_number,
                "cause": row.cause,
                "court": row.court,
                "region": row.region,
                "procedure": row.procedure,
                "judgment_date": row.judgment_date,
                "legal_basis": row.legal_basis,
                "original_url": row.original_url,
                "retrieval_text": row.retrieval_text,
            }
        else:
            logger.debug(f"跳过低质量案例 case_id={row.id}（title 和 gist 均为空）")

    return valid_cases


async def _search_topic_cases_pg(
    question: str,
    domain: str,
    db_session: AsyncSession,
    limit: int = 8,
) -> list[dict]:
    """Use structured text search for explicit topics that generic vectors often dilute."""
    terms = _topic_case_terms(question)
    if not terms:
        return []

    from src.modules.legal.model import LegalCase

    domains = _case_domains_for(domain, include_generic=True)
    conditions = []
    for term in terms:
        pattern = f"%{term}%"
        conditions.extend((
            LegalCase.title.ilike(pattern),
            LegalCase.cause.ilike(pattern),
            LegalCase.gist.ilike(pattern),
            LegalCase.retrieval_text.ilike(pattern),
        ))

    statement = (
        select(
            LegalCase.id,
            LegalCase.domain,
            LegalCase.retrieval_text,
            LegalCase.gist,
            LegalCase.title,
            LegalCase.cause,
        )
        .where(LegalCase.domain.in_(domains), or_(*conditions))
        .limit(limit * 20)
    )
    rows = (await db_session.execute(statement)).all()
    ranked: list[tuple[int, dict]] = []
    for row in rows:
        if not _generic_case_matches_topic(row.domain, row.cause or "", terms):
            continue
        text = " ".join(
            str(value or "")
            for value in (row.title, row.cause, row.gist, row.retrieval_text)
        )
        keyword_score = sum(text.count(term) for term in terms)
        if keyword_score <= 0:
            continue
        ranked.append((keyword_score, {
            "id": row.id,
            "domain": row.domain,
            "source": "postgres_topic_fallback",
            "text": row.retrieval_text or row.gist or row.title or "",
            "score": float(keyword_score),
        }))
    ranked.sort(key=lambda item: (-item[0], int(item[1]["id"])))
    return [hit for _, hit in ranked[:limit]]


def format_case_context(hits: list[dict], details: dict[int, dict]) -> str:
    """格式化类案上下文字符串。"""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        detail = details.get(int(hit["id"]), {})
        title = detail.get("title") or f"案例{i}"
        gist = detail.get("gist") or ""
        facts_snippet = detail.get("retrieval_text") or hit["text"]

        lines = [f"案例{i}【{title}】"]
        metadata = [
            detail.get("case_number"),
            detail.get("cause"),
            detail.get("court"),
            detail.get("region"),
            detail.get("procedure"),
            detail.get("judgment_date"),
        ]
        metadata = [str(item) for item in metadata if item]
        if metadata:
            lines.append("基本信息：" + "｜".join(metadata))
        lines.append(f"案情摘要：{facts_snippet}")
        if gist:
            lines.append(f"裁判要旨：{gist}")
        if detail.get("legal_basis"):
            lines.append(f"法律依据：{detail['legal_basis']}")
        if detail.get("original_url"):
            lines.append(f"原始链接：{detail['original_url']}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


async def search_cases_context(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session: AsyncSession | None = None,
    domain: str = "",
    sparse_query: str = "",
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
) -> dict:
    """为维权图返回原始类案上下文，不额外调用 LLM 生成二次回答。"""
    if db_session is not None:
        topic_hits = await _search_topic_cases_pg(question, domain, db_session)
        if topic_hits:
            topic_details = await _fetch_case_details(topic_hits, db_session)
            valid_topic_hits = [
                hit for hit in topic_hits if int(hit["id"]) in topic_details
            ]
            if valid_topic_hits:
                logger.info(
                    "类案主题字面补充命中 | domain={} hits={}",
                    domain,
                    len(valid_topic_hits),
                )
                cases = []
                for hit in valid_topic_hits:
                    detail = topic_details[int(hit["id"])]
                    cases.append({
                        "id": hit.get("id"),
                        "title": detail.get("title") or "",
                        "gist": detail.get("gist") or "",
                        "text": hit.get("text") or "",
                        "score": hit.get("score", 0.0),
                        "case_number": detail.get("case_number") or "",
                        "cause": detail.get("cause") or "",
                        "court": detail.get("court") or "",
                        "judgment_date": detail.get("judgment_date") or "",
                        "original_url": detail.get("original_url") or "",
                    })
                return {
                    "context": format_case_context(valid_topic_hits, topic_details),
                    "cases": cases,
                    "fallback_guide": None,
                }

    hits = await search_cases_raw(
        question=question,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        domain=domain,
        sparse_query=sparse_query,
        llm=llm,
        use_hyde=use_hyde,
    )
    details: dict[int, dict] = {}
    if hits and db_session is not None:
        details = await _fetch_case_details(hits, db_session)

    valid_hits = hits if db_session is None else [
        hit for hit in hits if int(hit["id"]) in details
    ]
    if valid_hits:
        cases = []
        for hit in valid_hits:
            detail = details.get(int(hit["id"]), {})
            cases.append({
                "id": hit.get("id"),
                "title": detail.get("title") or "",
                "gist": detail.get("gist") or "",
                "text": hit.get("text") or "",
                "score": hit.get("score", 0.0),
                "case_number": detail.get("case_number") or "",
                "cause": detail.get("cause") or "",
                "court": detail.get("court") or "",
                "judgment_date": detail.get("judgment_date") or "",
                "original_url": detail.get("original_url") or "",
            })
        return {
            "context": format_case_context(valid_hits, details),
            "cases": cases,
            "fallback_guide": None,
        }

    keywords = sparse_query or question[:80]
    return {
        "context": "",
        "cases": [],
        "fallback_guide": {
            "platform": "中国裁判文书网",
            "url": "https://wenshu.court.gov.cn",
            "search_tips": f"可使用“{keywords}”检索，并按案由、地区和审理程序筛选。",
        },
    }


async def _generate_search_tips(question: str, llm: BaseChatModel) -> str:
    """根据用户问题生成个性化的裁判文书网搜索提示。"""
    prompt = f"""用户咨询：{question}

请为用户生成在中国裁判文书网（https://wenshu.court.gov.cn）查询相关案例的具体建议。要求：
1. 提取用户问题中的核心法律概念、案由类型、争议焦点
2. 推荐2-3个具体搜索关键词（如案由名称、法条编号、核心争议点）
3. 建议筛选条件（如案件类型、审理程序）
4. 控制在80字以内，语言简洁实用

直接输出搜索建议，不要额外解释。"""

    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content.strip()


async def search_cases(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    db_session: AsyncSession | None = None,
    domain: str = "",
) -> str:
    """类案 RAG 完整流程：检索 → 补充案情 → 数据质量校验 → 生成回答。"""
    hits = await search_cases_raw(
        question,
        embedding_model,
        milvus_client,
        domain=domain,
        llm=llm,
        use_hyde=True,
    )

    if not hits:
        search_tips = await _generate_search_tips(question, llm)
        fallback_message = (
            f"暂无类案数据，建议通过法条检索了解相关法律规定。\n\n"
            f"💡 您也可以访问**中国裁判文书网**（https://wenshu.court.gov.cn）查找更多案例：\n{search_tips}"
        )
        return json.dumps(
            {
                "cases": [],
                "message": fallback_message,
                "fallback_guide": {
                    "platform": "中国裁判文书网",
                    "url": "https://wenshu.court.gov.cn",
                    "search_tips": search_tips,
                },
            },
            ensure_ascii=False,
        )

    details: dict[int, dict] = {}
    if db_session is not None:
        details = await _fetch_case_details(hits, db_session)

    # 数据质量二次过滤：只保留有有效 detail 的 hits
    valid_hits = [h for h in hits if int(h["id"]) in details]

    if not valid_hits:
        logger.warning(f"案例检索命中 {len(hits)} 条，但数据质量校验后全部过滤（title/gist 均为空）")
        search_tips = await _generate_search_tips(question, llm)
        fallback_message = (
            f"检索到的案例数据质量较差，建议通过法条检索了解相关法律规定。\n\n"
            f"💡 您也可以访问**中国裁判文书网**（https://wenshu.court.gov.cn）查找更多案例：\n{search_tips}"
        )
        return json.dumps(
            {
                "cases": [],
                "message": fallback_message,
                "fallback_guide": {
                    "platform": "中国裁判文书网",
                    "url": "https://wenshu.court.gov.cn",
                    "search_tips": search_tips,
                },
            },
            ensure_ascii=False,
        )

    context = format_case_context(valid_hits, details)
    prompt = CASE_QA_PROMPT.format(question=question, context=context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
