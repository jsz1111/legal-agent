from langchain_core.tools import tool
from src.rag.engine import RAGEngine

KNOWLEDGE_BASES = {
    "medical_docs": {"collection": "knowledge_docs", "description": "医学指南、药品说明书、SOP 文档"},
    "medical_dialogue": {"collection": "medical_dialogue", "description": "医患对话语料库 (6 科室)"},
    "drug_knowledge": {"collection": "drug_knowledge", "description": "药物专项知识库"},
    "exam_questions": {"collection": "exam_questions", "description": "执业医师考试题库"},
}


def create_rag_tools(engine: RAGEngine) -> list:

    @tool
    async def search_knowledge(
        query: str, knowledge_base: str = "medical_docs", doc_type: str | None = None,
    ) -> str:
        """在指定知识库中检索并回答问题。
        knowledge_base 可选: medical_docs/medical_dialogue/drug_knowledge/exam_questions"""
        kb = KNOWLEDGE_BASES.get(knowledge_base)
        if not kb:
            return f"未知知识库: {knowledge_base}，可选: {list(KNOWLEDGE_BASES.keys())}"
        filters = {"doc_type": doc_type} if doc_type else None
        result = await engine.query(query=query, collection_name=kb["collection"], filters=filters)
        return result["answer"]

    @tool
    async def search_knowledge_with_sources(
        query: str, knowledge_base: str = "medical_docs",
    ) -> dict:
        """检索并返回答案及来源信息"""
        kb = KNOWLEDGE_BASES.get(knowledge_base)
        if not kb:
            return {"answer": f"未知知识库: {knowledge_base}", "sources": []}
        result = await engine.query(query=query, collection_name=kb["collection"])
        sources = [{"doc_name": ctx["doc_name"], "score": ctx.get("rerank_score", ctx.get("score", 0))}
                   for ctx in result["contexts"]]
        return {"answer": result["answer"], "sources": sources}

    @tool
    async def list_knowledge_bases() -> str:
        """列出所有可用的知识库"""
        return "\n".join(f"- {name}: {cfg['description']}" for name, cfg in KNOWLEDGE_BASES.items())

    return [search_knowledge, search_knowledge_with_sources, list_knowledge_bases]
