from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.gradio_rag_demo import run_rag_ui, upload_document_ui
from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME
from src.agents.knowledge.runtime import build_knowledge_deps


FIXTURE_DIR = Path(r"D:\learn\tiangong-agent\test\fixtures\drug_instructions")


def print_json(title: str, data) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    file_path = next(FIXTURE_DIR.glob("*.txt"))
    print(f"FILE={file_path}")

    msg, docs_md = upload_document_ui(str(file_path), "drug_instruction", "测试")
    print(f"UPLOAD_MSG={msg}")
    print(f"DOCS_MD_HEAD={docs_md[:800]}")

    deps = build_knowledge_deps(role="patient")
    ensure_knowledge_collection(deps.milvus_client)

    listed = deps.milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="chunk_index >= 0",
        output_fields=["id", "doc_id", "doc_name", "doc_type", "category", "chunk_index"],
        limit=50,
    )
    print_json("MILVUS_LIST", listed)

    exact_rows = deps.milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter=f'doc_name == "{file_path.name}"',
        output_fields=["id", "doc_id", "doc_name", "doc_type", "category", "chunk_index"],
        limit=50,
    )
    print_json("MILVUS_DOC_FILTER", exact_rows)

    doc_id = exact_rows[0]["doc_id"] if exact_rows else None
    if doc_id:
        id_rows = deps.milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["id", "doc_id", "doc_name", "doc_type", "category", "chunk_index"],
            limit=50,
        )
        print_json("MILVUS_DOC_ID_FILTER", id_rows)

    cases = [
        ("文档RAG", "5-氨基水杨酸肠溶片的适应症是什么？", "patient", "drug_instruction", True),
        ("图谱RAG", "糖尿病常用药有哪些？", "patient", "", True),
        ("融合RAG", "高血压合并肾功能不全时，降压药怎么选？", "patient", "", True),
    ]
    for mode, question, role, doc_type, debug in cases:
        print(f"\n=== {mode} ===")
        answer, dbg = run_rag_ui(question, mode, role, doc_type, debug)
        print(f"ANSWER={answer[:1200]}")
        print(f"DEBUG_HEAD={dbg[:1200]}")


if __name__ == "__main__":
    main()
