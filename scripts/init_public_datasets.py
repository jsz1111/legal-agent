"""
公共医学数据集初始化脚本。
从 HuggingFace 下载公共医学数据集，清洗后导入 Milvus 或保存到本地。

用法:
    python scripts/init_public_datasets.py
    python scripts/init_public_datasets.py --dataset cmirb
    python scripts/init_public_datasets.py --dataset dialogue --limit 100
    python scripts/init_public_datasets.py --dataset medqa --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from zipfile import ZipFile

from huggingface_hub import HfApi, hf_hub_download
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias


settings = get_settings()
BATCH_SIZE = 50


def _get_deps() -> tuple[MilvusClient, DashScopeEmbeddings]:
    get_milvus_client_alias()
    milvus_client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    ensure_knowledge_collection(milvus_client)
    return milvus_client, embedding_model


def _download_repo_zip(repo_id: str) -> str:
    api = HfApi()
    zip_files = [
        name
        for name in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if name.endswith(".zip")
    ]
    if not zip_files:
        raise RuntimeError(f"No zip asset found in dataset repo: {repo_id}")
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=zip_files[0])


def _is_real_zip_entry(name: str) -> bool:
    return not name.endswith("/") and not name.startswith("__MACOSX/") and "/." not in name


async def _insert_texts(
    milvus_client: MilvusClient,
    embedding_model: DashScopeEmbeddings,
    texts: list[str],
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
) -> int:
    """批量向量化并写入 Milvus。"""
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )

    all_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = await embedding_model.aembed_documents(batch)
        for j, (text_content, emb) in enumerate(zip(batch, embeddings)):
            idx = i + j
            all_data.append(
                {
                    "id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "category": category,
                    "page_number": 0,
                    "chunk_index": idx,
                    "text": text_content[:65000],
                    "embedding": emb,
                }
            )

    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i : i + 1000],
            )
    return len(all_data)


async def download_cmirb(
    milvus_client: MilvusClient,
    embedding_model: DashScopeEmbeddings,
    limit: int | None = None,
) -> int:
    """下载 CMIRB/MedicalRetrieval 并导入 Milvus。"""
    from datasets import load_dataset

    print("[INFO] 下载 CMIRB/MedicalRetrieval 数据集...")
    ds = load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus")

    texts = []
    for row in ds:
        text = row.get("text", "") or row.get("content", "")
        if text and len(text.strip()) > 20:
            texts.append(text.strip()[:2000])
            if limit and len(texts) >= limit:
                break

    print(f"[INFO] CMIRB 共 {len(texts)} 条有效文本，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        doc_id = hashlib.md5(f"cmirb_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client,
            embedding_model,
            chunk,
            doc_id=doc_id,
            doc_name=f"CMIRB医学检索语料_{i // chunk_size + 1}",
            doc_type="literature",
            category="医学文献",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] CMIRB 导入完成，共 {total} 条")
    return total


async def download_med_dialogue(
    milvus_client: MilvusClient,
    embedding_model: DashScopeEmbeddings,
    limit: int | None = None,
) -> int:
    """
    下载中文医患对话数据集并导入 Milvus。
    仓库中为 zip + json 文件，直接读取原始文件，避免依赖 datasets 脚本。
    """
    print("[INFO] 下载 Chinese-medical-dialogue-data 数据集...")
    zip_path = _download_repo_zip("BillGPT/Chinese-medical-dialogue-data")

    texts = []
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not (_is_real_zip_entry(name) and name.endswith(".json")):
                continue
            row = json.loads(zf.read(name).decode("utf-8"))
            text = (row.get("input") or "").strip()
            if text and len(text) > 20:
                texts.append(text[:2000])
                if limit and len(texts) >= limit:
                    break

    print(f"[INFO] 医患对话共 {len(texts)} 条有效记录，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        doc_id = hashlib.md5(f"med_dialogue_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client,
            embedding_model,
            chunk,
            doc_id=doc_id,
            doc_name=f"中文医患对话_{i // chunk_size + 1}",
            doc_type="literature",
            category="医患对话",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] 医患对话导入完成，共 {total} 条")
    return total


async def download_medqa(limit: int | None = None) -> int:
    """
    下载 MedQA 中文执业医师考试题，保存到 data/eval/medqa_zh.json。
    使用压缩包中的 Mainland/4_options 题库，不导入 Milvus。
    """
    print("[INFO] 下载 MedQA 数据集...")
    zip_path = hf_hub_download(
        repo_id="bigbio/med_qa",
        repo_type="dataset",
        filename="data_clean.zip",
    )

    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    split_files = {
        "train": "data_clean/questions/Mainland/4_options/train.jsonl",
        "dev": "data_clean/questions/Mainland/4_options/dev.jsonl",
        "test": "data_clean/questions/Mainland/4_options/test.jsonl",
    }

    records = []
    with ZipFile(zip_path) as zf:
        for split_name, inner_path in split_files.items():
            with zf.open(inner_path) as fp:
                for raw_line in fp:
                    raw_line = raw_line.decode("utf-8").strip()
                    if not raw_line:
                        continue
                    row = json.loads(raw_line)
                    question = (row.get("question") or "").strip()
                    if not question:
                        continue
                    records.append(
                        {
                            "question": question,
                            "choices": row.get("options") or {},
                            "answer": row.get("answer"),
                            "answer_idx": row.get("answer_idx"),
                            "meta_info": row.get("meta_info"),
                            "split": split_name,
                        }
                    )
                    if limit and len(records) >= limit:
                        break
            if limit and len(records) >= limit:
                break

    output_path = os.path.join(eval_dir, "medqa_zh.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] MedQA 保存完成：{output_path}，共 {len(records)} 道题")
    return len(records)


async def main() -> None:
    parser = argparse.ArgumentParser(description="公共医学数据集初始化")
    parser.add_argument(
        "--dataset",
        choices=["cmirb", "dialogue", "medqa", "all"],
        default="all",
        help="要导入的数据集，默认 all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条记录，便于小批量验证",
    )
    args = parser.parse_args()

    milvus_client = None
    embedding_model = None
    if args.dataset in ("cmirb", "dialogue", "all"):
        milvus_client, embedding_model = _get_deps()

    if args.dataset in ("cmirb", "all"):
        await download_cmirb(milvus_client, embedding_model, limit=args.limit)

    if args.dataset in ("dialogue", "all"):
        await download_med_dialogue(milvus_client, embedding_model, limit=args.limit)

    if args.dataset in ("medqa", "all"):
        await download_medqa(limit=args.limit)

    print("\n[DONE] 数据集初始化完成")


if __name__ == "__main__":
    asyncio.run(main())
