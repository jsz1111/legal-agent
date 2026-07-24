from __future__ import annotations

from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection, ingest_file
from src.agents.knowledge.runtime import build_knowledge_deps
from src.rag.config import IngestionConfig


class IngestionPipeline:
    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()

    def ensure_collection(self) -> None:
        deps = build_knowledge_deps()
        ensure_knowledge_collection(deps.milvus_client)

    async def ingest_file(
        self,
        *,
        file_path: str,
        doc_name: str,
        doc_type: str,
        category: str,
    ) -> int:
        deps = build_knowledge_deps()
        return await ingest_file(
            file_path=file_path,
            doc_name=doc_name,
            doc_type=doc_type,
            category=category,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
        )
