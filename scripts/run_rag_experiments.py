"""
实验矩阵 - 每组记录为不同 app_version，TruLens Dashboard 对比。

用法:
  python scripts/run_rag_experiments.py
  python scripts/run_rag_experiments.py --only v2_hyde
  python scripts/run_rag_experiments.py --dashboard  # 仅启动 Dashboard
"""
import asyncio
import json
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trulens.apps.app import TruApp

from src.rag.evaluation.trulens_config import get_trulens_session, get_llm_provider, launch_dashboard
from src.rag.evaluation.feedbacks import build_rag_triad_metrics
from src.rag.evaluation.tracked_rag import TrackedRAG
from src.rag.engine import RAGEngine, RAGDeps
from src.rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from src.core.config import get_settings

settings = get_settings()

EXPERIMENTS = {
    "baseline": RAGConfig(
        chunking=ChunkingConfig(strategy="fixed", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(use_hyde=False, use_rerank=False, use_hybrid=False),
    ),
    "v1_rerank": RAGConfig(
        chunking=ChunkingConfig(strategy="fixed", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(use_hyde=False, use_rerank=True, use_hybrid=False),
    ),
    "v2_hyde": RAGConfig(
        chunking=ChunkingConfig(strategy="fixed", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(use_hyde=True, use_rerank=True, use_hybrid=False),
    ),
    "v3_semantic": RAGConfig(
        chunking=ChunkingConfig(strategy="semantic"),
        retrieval=RetrievalConfig(use_hyde=True, use_rerank=True, use_hybrid=False),
    ),
    "v4_parent": RAGConfig(
        chunking=ChunkingConfig(strategy="parent_child", chunk_size=256, parent_chunk_size=2048),
        retrieval=RetrievalConfig(use_hyde=True, use_rerank=True, use_hybrid=False),
    ),
    "v5_hybrid": RAGConfig(
        chunking=ChunkingConfig(strategy="parent_child", chunk_size=256, parent_chunk_size=2048),
        retrieval=RetrievalConfig(use_hyde=True, use_rerank=True, use_hybrid=True),
    ),
}


def _build_deps(config: RAGConfig) -> RAGDeps:
    from pymilvus import MilvusClient
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_openai import ChatOpenAI

    return RAGDeps(
        milvus=MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"),
        embedding_model=DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL, dashscope_api_key=settings.DASHSCOPE_API_KEY
        ),
        llm=ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.BASE_URL_CHAT,
            temperature=config.generation.temperature,
        ),
        config=config,
    )


async def run_single_experiment(version: str, config: RAGConfig, dataset: list[dict]):
    session = get_trulens_session()
    provider = get_llm_provider()
    metrics = build_rag_triad_metrics(provider)

    deps = _build_deps(config)
    engine = RAGEngine(deps)
    tracked = TrackedRAG(engine)

    tru_app = TruApp(
        tracked,
        app_name="medical_rag",
        app_version=version,
        feedbacks=metrics,
        session=session,
    )

    print(f"\n{'='*60}")
    print(f"实验: {version}")
    print(f"  分块: {config.chunking.strategy} ({config.chunking.chunk_size})")
    print(f"  HyDE: {config.retrieval.use_hyde} | Rerank: {config.retrieval.use_rerank} | Hybrid: {config.retrieval.use_hybrid}")
    print(f"{'='*60}")

    # 仅测试 10条数据
    for i, item in enumerate(dataset[:10]):
        print(f"  [{i+1}/{len(dataset)}] {item['question'][:40]}...")
        with tru_app as recording:
            await tracked.query(item["question"])

    # 1. 先把所有 trace 数据写入数据库
    session.force_flush()

    # 2. 停止后台评估线程（防止和主线程评估竞争）
    tru_app.stop_evaluator()

    # 3. 在主线程同步执行所有未完成的评估
    print(f"  >>> {version} 记录完成，开始同步评估...")
    tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)

    # 4. 最终确保数据落库
    session.force_flush()

    leaderboard = session.get_leaderboard()
    print(f"  >>> {version} 评估完成，指标如下:\n{leaderboard}")


async def main(only: str | None = None, dashboard: bool = False):
    if dashboard:
        print("启动 TruLens Dashboard...")
        launch_dashboard(port=8501)
        return

    with open("data/eval/rag_eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    experiments = {only: EXPERIMENTS[only]} if only else EXPERIMENTS
    for version, config in experiments.items():
        await run_single_experiment(version, config, dataset)

    print("\n全部实验完成!")
    print("启动 Dashboard 查看结果:")
    print("  python scripts/run_rag_experiments.py --dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(EXPERIMENTS.keys()), help="只运行指定实验")
    parser.add_argument("--dashboard", action="store_true", help="启动 TruLens Dashboard")
    args = parser.parse_args()
    asyncio.run(main(args.only, args.dashboard))
