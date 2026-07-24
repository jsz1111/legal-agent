from __future__ import annotations

from src.agents.knowledge.doc_ingestion import _parse_with_llamaindex, _parse_with_mineru


async def parse_with_mineru(file_path: str, file_name: str) -> str | None:
    return await _parse_with_mineru(file_path, file_name)


async def parse_with_fallback(file_path: str) -> list:
    return await _parse_with_llamaindex(file_path)


async def parse_document(file_path: str, file_name: str) -> str | list:
    parsed = await parse_with_mineru(file_path, file_name)
    if parsed:
        return parsed
    return await parse_with_fallback(file_path)
