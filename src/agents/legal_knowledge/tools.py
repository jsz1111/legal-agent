"""LegalKnowledgeDeps + build_legal_knowledge_tools()：5 个法律知识工具。"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.knowledge.audit import QueryAuditLog, Timer


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
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.user_id = user_id


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
            result = await search_statutes(
                question=rewritten,
                embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client,
                llm=deps.llm,
                db_session=deps.db_session,
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
            result = await search_cases(
                question=question,
                embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client,
                llm=deps.llm,
                db_session=deps.db_session,
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
        """查询指定法律领域和地区的维权渠道（含电话/网址）。
        适用：需要具体联系方式时，如投诉热线、仲裁委、消协等。
        domain: 法律领域（如 labor_social_security / consumer_market）
        region_code: 地区代码（CN=全国，BJ=北京，SH=上海，GD=广东等）"""
        if deps.db_session is None:
            return "数据库连接不可用，无法查询渠道信息。"
        from src.agents.legal_knowledge.nl2sql import search_sql

        question = f"查询{domain}领域在{region_code}地区的维权渠道，包括名称、电话和网址"
        with Timer() as t:
            result = await search_sql(
                question=question,
                llm=deps.llm,
                db=deps.db_session,
            )
        QueryAuditLog.log(
            deps.user_id, "citizen", question, "nl2sql",
            ["channels"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_legal_docs(question: str) -> str:
        """从律师上传的法律文书/裁判文书知识库中检索信息。
        适用：查询具体合同条款、裁判文书内容、律师上传的专业文档。
        question: 用户的问题"""
        from src.agents.knowledge.doc_rag import search_docs

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

    return [
        search_statute,
        search_similar_cases,
        search_legal_graph,
        search_channels,
        search_legal_docs,
    ]
