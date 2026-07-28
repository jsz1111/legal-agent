"""LegalKnowledgeDeps + build_legal_knowledge_tools()：法律知识与统计工具。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.legal_knowledge.audit import QueryAuditLog, Timer


@asynccontextmanager
async def _db_session_or_new(db_session: AsyncSession | None):
    """复用调用方会话；独立 Worker 调用时按工具生命周期创建只读会话。"""
    if db_session is not None:
        yield db_session
        return

    from src.infra.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        yield session


class LegalKnowledgeDeps:
    """法律知识 Agent 的依赖注入容器。"""

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        neo4j_driver: AsyncDriver,
        db_session: AsyncSession | None = None,
        user_id: str = "anonymous",
        statistics_previous_sql: str = "",
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.user_id = user_id
        self.statistics_previous_sql = statistics_previous_sql


async def _rewrite(question: str, llm: BaseChatModel) -> str:
    """将口语化问题改写为标准法律术语表达，失败时回退原始问题。"""
    from langchain_core.messages import SystemMessage
    from src.agents.legal_knowledge.prompts import LEGAL_QUERY_REWRITE_PROMPT

    try:
        prompt = LEGAL_QUERY_REWRITE_PROMPT.format(question=question)
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        rewritten = response.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def build_legal_knowledge_tools(deps: LegalKnowledgeDeps) -> list:
    """构建法律知识 Agent 的工具集（含 Query 改写预处理）。"""

    @tool
    async def search_statute(question: str, domain: str = "") -> str:
        """从法条库中语义检索相关法律条文并生成回答。
        适用：查询具体法律规定、了解某行为的法律后果、确认权利和义务。
        question: 用户的法律问题
        domain: 可选，法律领域过滤（如 labor_social_security / consumer_market / contract_commercial）"""
        from src.agents.legal_knowledge.statute_rag import search_statutes

        rewritten = await _rewrite(question, deps.llm)
        with Timer() as t:
            async with _db_session_or_new(deps.db_session) as db_session:
                result = await search_statutes(
                    question=rewritten,
                    embedding_model=deps.embedding_model,
                    milvus_client=deps.milvus_client,
                    llm=deps.llm,
                    db_session=db_session,
                    domain=domain,
                )
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "statute_rag",
            ["statute_index"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_similar_cases(question: str, domain: str = "") -> str:
        """检索与用户情况相似的法律案例（类案）并生成参考意见。
        适用：了解同类纠纷的处理结果、参考裁判要旨、评估维权可行性。
        question: 描述用户的案情或法律问题
        domain: 可选，法律领域过滤（如 labor_social_security / consumer_market / criminal_public_security）"""
        from src.agents.legal_knowledge.case_rag import search_cases

        with Timer() as t:
            async with _db_session_or_new(deps.db_session) as db_session:
                result = await search_cases(
                    question=question,
                    embedding_model=deps.embedding_model,
                    milvus_client=deps.milvus_client,
                    llm=deps.llm,
                    db_session=db_session,
                    domain=domain,
                )
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "case_rag",
            ["case_index"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_legal_graph(question: str) -> str:
        """从法律知识图谱中查询法律关系和维权渠道信息。
        适用：查询某领域适用哪些法律、对口的投诉渠道是什么、法律关系推理。
        question: 用户的法律问题（可包含法律领域、渠道名称等关键词）"""
        from src.agents.legal_knowledge.graph_rag import search_graph

        rewritten = await _rewrite(question, deps.llm)
        with Timer() as t:
            result = await search_graph(
                question=rewritten,
                neo4j_driver=deps.neo4j_driver,
                llm=deps.llm,
            )
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "graph_rag",
            ["graph_rag"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_channels(domain: str, region_code: str = "CN") -> str:
        """确定性查询指定法律领域和地区的权威维权渠道及办理细节。
        适用：需要投诉热线、法律援助、政务转办的联系方式和材料要求。
        domain: 法律领域（如 labor_social_security / consumer_market）
        region_code: 当前试点支持 CN/110000/310000，也兼容北京/上海/BJ/SH"""
        from src.agents.legal_guide.db_queries import query_recommended_channels
        from src.agents.legal_guide.formatters import fmt_channels

        question = f"查询{domain}领域在{region_code}地区的维权渠道，包括名称、电话和网址"
        with Timer() as t:
            async with _db_session_or_new(deps.db_session) as db_session:
                channels = await query_recommended_channels(
                    domain=domain,
                    region=region_code,
                    db=db_session,
                )
                result = fmt_channels(channels)
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "channel_repository",
            ["channels"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_legal_docs(question: str) -> str:
        """从律师上传的法律文书/裁判文书知识库中检索信息。
        适用：查询具体合同条款、裁判文书内容、律师上传的专业文档。
        question: 用户的问题"""
        from src.agents.legal_knowledge.doc_rag import search_docs

        rewritten = await _rewrite(question, deps.llm)
        with Timer() as t:
            result = await search_docs(
                question=rewritten,
                embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client,
                llm=deps.llm,
            )
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "doc_rag",
            ["knowledge_docs"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_legal_statistics(question: str) -> str:
        """查询《中国法律年鉴》统计数据并返回回答、原始数据和推荐图表。
        适用：查询案件数量、年度趋势、比例、机构统计等数值问题。
        question: 必须是可独立理解的问题；连续追问时要继承上轮的年份和类别；
        用户要求新增指标或同图对比时，必须同时包含原指标和新增指标"""
        import json

        from src.agents.legal_knowledge.legal_statistics_chatbi import (
            run_legal_statistics_chatbi,
        )

        with Timer() as t:
            result = await run_legal_statistics_chatbi(
                question,
                deps.llm,
                previous_sql=deps.statistics_previous_sql,
            )
        payload = {
            "answer": result.answer,
            "statistics": result.model_dump(mode="json"),
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        QueryAuditLog.log(
            deps.user_id,
            "citizen",
            question,
            "legal_statistics_nl2sql",
            ["legal_statistics_db"],
            result.answer[:80],
            t.elapsed_ms,
        )
        return serialized

    return [
        search_statute,
        search_similar_cases,
        search_legal_graph,
        search_channels,
        search_legal_docs,
        search_legal_statistics,
    ]
