from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import sys
import traceback
from pathlib import Path

import gradio as gr
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection, ingest_file
from src.agents.knowledge.doc_rag import COLLECTION_NAME, search_docs, search_docs_raw
from src.agents.knowledge.fusion import multi_channel_search
from src.agents.knowledge.graph_rag import search_graph, search_graph_raw
from src.agents.knowledge.prescription_review import review_prescription
from src.agents.knowledge.runtime import build_knowledge_deps
from src.infra.minio_client import ensure_bucket_exists, upload_file as minio_upload


DOC_TYPE_CHOICES = ["", "guideline", "drug_instruction", "sop", "literature"]
ROLE_CHOICES = ["patient", "doctor", "pharmacist"]
MODE_CHOICES = ["文档RAG", "图谱RAG", "融合RAG", "处方审核"]
FUSION_CHANNEL_CHOICES = [
    ("文档检索", "doc_rag"),
    ("图谱检索", "graph_rag"),
]
DEFAULT_FUSION_CHANNELS = ["doc_rag", "graph_rag"]


def _json_dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _render_documents_markdown(docs: list[list[str]]) -> str:
    if not docs:
        return "当前知识库中还没有文档。"

    lines = [
        "| doc_name | doc_type | category |",
        "| --- | --- | --- |",
    ]
    for doc_name, doc_type, category in docs:
        lines.append(f"| {doc_name} | {doc_type} | {category} |")
    return "\n".join(lines)


def _format_doc_hits_markdown(hits: list[dict]) -> str:
    if not hits:
        return "未检索到文档片段。"

    sections: list[str] = []
    for index, hit in enumerate(hits, 1):
        doc_name = hit.get("doc_name", "未知文档")
        page_number = hit.get("page_number", "?")
        chunk_index = hit.get("chunk_index", "?")
        score = hit.get("score", 0.0)
        text = (hit.get("text", "") or "").strip()
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        sections.append(
            "\n".join(
                [
                    f"### 文档片段 {index}",
                    f"- 来源：`{doc_name}`",
                    f"- 页码：`{page_number}`",
                    f"- 分块：`{chunk_index}`",
                    f"- 相似度分数：`{score_text}`",
                    "",
                    text or "_片段内容为空_",
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def _format_graph_records_markdown(records: list[dict]) -> str:
    if not records:
        return "未检索到图谱结果。"
    return "```json\n" + _json_dumps(records) + "\n```"


def _format_retrieval_markdown(
    mode: str,
    doc_hits: list[dict] | None = None,
    graph_records: list[dict] | None = None,
    note: str | None = None,
) -> str:
    parts: list[str] = []

    if doc_hits is not None:
        parts.append("## 文档检索片段\n" + _format_doc_hits_markdown(doc_hits))
    if graph_records is not None:
        parts.append("## 图谱检索结果\n" + _format_graph_records_markdown(graph_records))
    if note:
        parts.append("## 说明\n" + note)

    if parts:
        return "\n\n".join(parts)
    return f"当前模式 `{mode}` 没有可展示的检索片段。"


def _normalize_fusion_channels(channels: list[str] | None) -> list[str]:
    channels = channels or []
    valid_channels = {value for _, value in FUSION_CHANNEL_CHOICES}
    return [channel for channel in channels if channel in valid_channels]


async def _list_documents_async() -> list[list[str]]:
    deps = build_knowledge_deps()
    ensure_knowledge_collection(deps.milvus_client)
    results = deps.milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="chunk_index == 0",
        output_fields=["doc_name", "doc_type", "category"],
        limit=500,
    )
    docs = [[row["doc_name"], row["doc_type"], row["category"]] for row in results]
    docs.sort(key=lambda row: (row[1], row[0]))
    return docs


def list_documents_ui():
    docs = asyncio.run(_list_documents_async())
    return _render_documents_markdown(docs)


async def _upload_document_async(file_path: str, doc_type: str, category: str):
    if not file_path:
        docs = await _list_documents_async()
        return "请选择要导入的文件。", _render_documents_markdown(docs)

    path = Path(file_path)
    if not path.exists():
        docs = await _list_documents_async()
        return f"文件不存在：{file_path}", _render_documents_markdown(docs)

    ext = path.suffix.lower()
    allowed_ext = {".pdf", ".docx", ".doc", ".txt", ".md"}
    if ext not in allowed_ext:
        docs = await _list_documents_async()
        return f"暂不支持该文件格式：{ext}", _render_documents_markdown(docs)

    deps = build_knowledge_deps()

    try:
        ensure_bucket_exists()
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        minio_upload(f"knowledge/{doc_type}/{path.name}", content, content_type)
    except Exception as exc:
        logger.warning(f"上传原始文件到 MinIO 失败: {exc}")

    chunk_count = await ingest_file(
        file_path=str(path),
        doc_name=path.name,
        doc_type=doc_type,
        category=category or "通用",
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )
    docs = await _list_documents_async()
    return (
        f"导入完成：{path.name}，共写入 {chunk_count} 个分块。",
        _render_documents_markdown(docs),
    )


def upload_document_ui(file_path: str, doc_type: str, category: str):
    return asyncio.run(_upload_document_async(file_path, doc_type, category))


async def _delete_document_async(doc_name: str):
    doc_name = (doc_name or "").strip()
    if not doc_name:
        docs = await _list_documents_async()
        return "请输入要删除的文档名。", _render_documents_markdown(docs)

    deps = build_knowledge_deps()
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]
    deps.milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )
    docs = await _list_documents_async()
    return f"已从 Milvus 删除文档：{doc_name}", _render_documents_markdown(docs)


def delete_document_ui(doc_name: str):
    return asyncio.run(_delete_document_async(doc_name))


async def _run_rag_async(
    question: str,
    mode: str,
    role: str,
    doc_type: str,
    show_debug: bool,
    use_hyde: bool,
    fusion_channels: list[str] | None,
):
    question = (question or "").strip()
    if not question:
        return "请输入问题。", "", ""

    normalized_channels = _normalize_fusion_channels(fusion_channels)
    if mode == "融合RAG" and not normalized_channels:
        return "请至少选择一个融合通道。", "", ""

    deps = build_knowledge_deps(role=role)
    debug: dict[str, object] = {
        "mode": mode,
        "role": role,
        "use_hyde": use_hyde,
        "fusion_channels": normalized_channels,
    }
    doc_hits: list[dict] | None = None
    graph_records: list[dict] | None = None
    retrieval_note: str | None = None

    if mode == "文档RAG":
        answer = await search_docs(
            question=question,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
            llm=deps.llm,
            doc_type=doc_type or None,
            role=role,
            use_hyde=use_hyde,
        )
        doc_hits = await search_docs_raw(
            question=question,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
            doc_type=doc_type or None,
            llm=deps.llm,
            use_hyde=use_hyde,
        )
        if show_debug:
            debug["doc_hits"] = doc_hits

    elif mode == "图谱RAG":
        answer = await search_graph(
            question=question,
            neo4j_driver=deps.neo4j_driver,
            llm=deps.llm,
            role=role,
        )
        graph_records = await search_graph_raw(
            question=question,
            neo4j_driver=deps.neo4j_driver,
            llm=deps.llm,
        )
        retrieval_note = "图谱RAG 不使用 HyDE，开关仅对文档检索通道生效。"
        if show_debug:
            debug["graph_records"] = graph_records

    elif mode == "融合RAG":
        answer = await multi_channel_search(
            question=question,
            llm=deps.llm,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
            neo4j_driver=deps.neo4j_driver,
            channels=normalized_channels,
            role=role,
            doc_use_hyde=use_hyde,
        )

        tasks = []
        if "doc_rag" in normalized_channels:
            tasks.append(
                search_docs_raw(
                    question=question,
                    embedding_model=deps.embedding_model,
                    milvus_client=deps.milvus_client,
                    llm=deps.llm,
                    use_hyde=use_hyde,
                )
            )
        if "graph_rag" in normalized_channels:
            tasks.append(
                search_graph_raw(
                    question=question,
                    neo4j_driver=deps.neo4j_driver,
                    llm=deps.llm,
                )
            )

        results = await asyncio.gather(*tasks) if tasks else []
        idx = 0
        if "doc_rag" in normalized_channels:
            doc_hits = results[idx]
            idx += 1
        if "graph_rag" in normalized_channels:
            graph_records = results[idx]

        if show_debug:
            if doc_hits is not None:
                debug["doc_hits"] = doc_hits
            if graph_records is not None:
                debug["graph_records"] = graph_records

    else:
        answer = await review_prescription(
            question=question,
            llm=deps.llm,
            embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client,
            neo4j_driver=deps.neo4j_driver,
        )
        retrieval_note = "处方审核当前走内部综合检索流程，测试页暂不拆出单独片段列表。"
        if show_debug:
            debug["note"] = retrieval_note

    debug_text = _json_dumps(debug) if show_debug else ""
    retrieval_text = _format_retrieval_markdown(
        mode=mode,
        doc_hits=doc_hits,
        graph_records=graph_records,
        note=retrieval_note,
    )
    return answer, retrieval_text, debug_text


def run_rag_ui(
    question: str,
    mode: str,
    role: str,
    doc_type: str,
    show_debug: bool,
    use_hyde: bool,
    fusion_channels: list[str] | None,
):
    try:
        return asyncio.run(
            _run_rag_async(
                question,
                mode,
                role,
                doc_type,
                show_debug,
                use_hyde,
                fusion_channels,
            )
        )
    except Exception as exc:
        logger.exception("RAG demo execution failed")
        debug_text = traceback.format_exc() if show_debug else ""
        return f"运行错误：{exc}", "", debug_text


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="法律平台 RAG 调试台") as demo:
        gr.Markdown(
            """
            # 法律平台 RAG 调试台
            用于测试知识库导入、文档 RAG、图谱 RAG、融合 RAG 与法条检索能力。
            """
        )

        with gr.Tab("RAG测试"):
            with gr.Row():
                mode = gr.Radio(
                    MODE_CHOICES,
                    value="融合RAG",
                    label="测试模式",
                )
                role = gr.Radio(
                    ROLE_CHOICES,
                    value="patient",
                    label="回答角色",
                )
            with gr.Row():
                doc_type = gr.Dropdown(
                    DOC_TYPE_CHOICES,
                    value="",
                    label="文档类型过滤（仅文档RAG有效）",
                )
                show_debug = gr.Checkbox(value=True, label="显示调试信息")
                use_hyde = gr.Checkbox(value=True, label="启用 HyDE（仅文档通道）")

            fusion_channels = gr.CheckboxGroup(
                choices=FUSION_CHANNEL_CHOICES,
                value=DEFAULT_FUSION_CHANNELS,
                label="融合RAG通道",
                info="仅在融合RAG模式下生效，至少选择一个通道。",
            )

            question = gr.Textbox(
                label="问题",
                lines=5,
                placeholder="例如：高血压合并肾功能不全时，降压药怎么选？",
            )
            with gr.Row():
                run_button = gr.Button("开始测试", variant="primary")
                clear_button = gr.Button("清空")

            answer = gr.Markdown(label="回答")
            retrieval_output = gr.Markdown(label="检索片段")
            debug_output = gr.Code(label="调试信息", language="json")

            gr.Examples(
                examples=[
                    ["高血压合并肾功能不全时，降压药怎么选？", "融合RAG", "patient", "", True, True, ["doc_rag", "graph_rag"]],
                    ["高血压合并肾功能不全时，降压药怎么选？", "融合RAG", "patient", "", True, True, ["doc_rag"]],
                    ["糖尿病常用药有哪些？", "融合RAG", "patient", "", True, True, ["graph_rag"]],
                    ["阿莫西林的禁忌症有哪些？", "文档RAG", "patient", "drug_instruction", True, True, ["doc_rag", "graph_rag"]],
                ],
                inputs=[question, mode, role, doc_type, show_debug, use_hyde, fusion_channels],
            )

            run_button.click(
                run_rag_ui,
                inputs=[question, mode, role, doc_type, show_debug, use_hyde, fusion_channels],
                outputs=[answer, retrieval_output, debug_output],
            )
            clear_button.click(
                lambda: ("", "", "", ""),
                outputs=[question, answer, retrieval_output, debug_output],
            )

        with gr.Tab("知识库管理"):
            with gr.Row():
                upload_file = gr.File(label="选择文档", type="filepath")
                with gr.Column():
                    upload_doc_type = gr.Dropdown(
                        DOC_TYPE_CHOICES[1:],
                        value="guideline",
                        label="文档类型",
                    )
                    upload_category = gr.Textbox(
                        value="通用",
                        label="分类",
                    )
            with gr.Row():
                upload_button = gr.Button("导入知识库", variant="primary")
                refresh_button = gr.Button("刷新文档列表")

            upload_status = gr.Textbox(label="导入结果", interactive=False)
            docs_table = gr.Markdown(label="当前知识库文档")

            gr.Markdown("### 删除文档")
            with gr.Row():
                delete_doc_name = gr.Textbox(
                    label="文档名",
                    placeholder="输入与列表中一致的 doc_name",
                )
                delete_button = gr.Button("从 Milvus 删除")

            delete_status = gr.Textbox(label="删除结果", interactive=False)

            upload_button.click(
                upload_document_ui,
                inputs=[upload_file, upload_doc_type, upload_category],
                outputs=[upload_status, docs_table],
            )
            refresh_button.click(
                list_documents_ui,
                outputs=[docs_table],
            )
            delete_button.click(
                delete_document_ui,
                inputs=[delete_doc_name],
                outputs=[delete_status, docs_table],
            )

            demo.load(list_documents_ui, outputs=[docs_table])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio RAG demo for tiangong-agent")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
