from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from pymilvus import MilvusClient

from src.agents.knowledge.tools import KnowledgeDeps
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver


@lru_cache
def get_shared_knowledge_runtime() -> tuple[ChatDeepSeek, DashScopeEmbeddings, MilvusClient, object]:
    settings = get_settings()

    llm = ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return llm, embedding_model, milvus_client, neo4j_driver


def build_knowledge_deps(
    db_session=None,
    user_id: str = "anonymous",
    role: str = "patient",
) -> KnowledgeDeps:
    llm, embedding_model, milvus_client, neo4j_driver = get_shared_knowledge_runtime()
    return KnowledgeDeps(
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
        db_session=db_session,
        user_id=user_id,
        role=role,
    )
