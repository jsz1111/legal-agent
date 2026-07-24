from __future__ import annotations

from src.agents.knowledge.runtime import build_knowledge_deps


def build_embedding_model():
    deps = build_knowledge_deps()
    return deps.embedding_model


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await build_embedding_model().aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    return await build_embedding_model().aembed_query(text)
