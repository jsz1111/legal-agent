"""Import the reviewed pilot case package into PostgreSQL and Milvus.

The import is idempotent. PostgreSQL stores the complete cleaned judgment and
all metadata, while Milvus embeds only ``retrieval_text``. The source package's
commercial ``来源`` field is neither read nor imported.

Examples:
    python scripts/import_case_pilot.py --limit 5 --skip-index
    python scripts/import_case_pilot.py
    python scripts/import_case_pilot.py --index-only
"""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import (  # noqa: E402
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.infra.database import AsyncSessionLocal, engine  # noqa: E402
from src.infra.embedding import get_embedding_model  # noqa: E402
from src.modules.legal.model import LegalCase  # noqa: E402


DEFAULT_INPUT = ROOT / "案例数据包" / "cases_pilot.sqlite3"
SCHEMA_FILE = ROOT / "database" / "case_library" / "schema.sql"
MODEL_FILE = ROOT / "models" / "bm25_case.json"
COLLECTION_NAME = "case_index"
GENERIC_DOMAIN = "civil_case"
EMBEDDING_DIM = 1024
REQUIRED_COLUMNS = {
    "case_id",
    "original_url",
    "case_number",
    "case_name",
    "court",
    "region",
    "case_type",
    "case_type_code",
    "procedure",
    "judgment_date",
    "publication_date",
    "parties",
    "cause",
    "legal_basis",
    "full_text",
    "full_text_length",
    "retrieval_text",
    "selection_tags",
}


def load_sqlite_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"案例数据包不存在: {path}")

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            item[1]
            for item in connection.execute("PRAGMA table_info(legal_cases)").fetchall()
        }
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"案例数据包缺少字段: {sorted(missing)}")

        sql = "SELECT * FROM legal_cases ORDER BY case_id"
        params: tuple[Any, ...] = ()
        if limit > 0:
            sql += " LIMIT ?"
            params = (limit,)
        rows = [dict(row) for row in connection.execute(sql, params).fetchall()]
    finally:
        connection.close()

    if not rows:
        raise ValueError("案例数据包没有可导入记录")
    for row in rows:
        if not row["case_id"] or not row["case_number"] or not row["retrieval_text"]:
            raise ValueError(f"案例关键字段为空: {row.get('case_id') or '<unknown>'}")
        if not row["full_text"] or int(row["full_text_length"] or 0) <= 0:
            raise ValueError(f"案例全文为空: {row['case_id']}")
    return rows


def _schema_statements() -> list[str]:
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    without_comments = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return [statement.strip() for statement in without_comments.split(";") if statement.strip()]


async def ensure_postgres_schema() -> None:
    async with engine.begin() as connection:
        for statement in _schema_statements():
            await connection.execute(text(statement))


def _postgres_payload(row: dict[str, Any]) -> dict[str, Any]:
    retrieval_text = str(row["retrieval_text"]).strip()
    return {
        "case_id": row["case_id"],
        "original_url": row["original_url"],
        "case_number": row["case_number"],
        "title": row["case_name"],
        "court": row["court"],
        "region": row["region"],
        "case_type": row["case_type"],
        "case_type_code": row["case_type_code"],
        "procedure": row["procedure"],
        "judgment_date": row["judgment_date"],
        "publication_date": row["publication_date"],
        "parties": row["parties"],
        "cause": row["cause"],
        "legal_basis": row["legal_basis"],
        "full_text": row["full_text"],
        "full_text_length": int(row["full_text_length"]),
        "retrieval_text": retrieval_text,
        "selection_tags": row["selection_tags"],
        "facts": retrieval_text,
        "gist": None,
        "domain": GENERIC_DOMAIN,
        "source": "",
    }


async def upsert_postgres(rows: list[dict[str, Any]], batch_size: int = 100) -> int:
    mutable_columns = [
        "original_url",
        "case_number",
        "title",
        "court",
        "region",
        "case_type",
        "case_type_code",
        "procedure",
        "judgment_date",
        "publication_date",
        "parties",
        "cause",
        "legal_basis",
        "full_text",
        "full_text_length",
        "retrieval_text",
        "selection_tags",
        "facts",
        "gist",
        "domain",
        "source",
    ]

    async with AsyncSessionLocal() as session:
        for start in range(0, len(rows), batch_size):
            payloads = [_postgres_payload(row) for row in rows[start : start + batch_size]]
            statement = pg_insert(LegalCase).values(payloads)
            statement = statement.on_conflict_do_update(
                index_elements=[LegalCase.case_id],
                set_={column: getattr(statement.excluded, column) for column in mutable_columns},
            )
            await session.execute(statement)
        await session.commit()

        imported = (
            await session.execute(
                select(LegalCase.case_id).where(
                    LegalCase.case_id.in_([row["case_id"] for row in rows])
                )
            )
        ).scalars().all()
    return len(imported)


async def fetch_index_rows() -> list[dict[str, str]]:
    async with AsyncSessionLocal() as session:
        cases = (
            await session.execute(
                select(LegalCase)
                .where(LegalCase.case_id.is_not(None))
                .order_by(LegalCase.id)
            )
        ).scalars().all()

    rows = []
    for case in cases:
        retrieval_text = (case.retrieval_text or "").strip()
        if not retrieval_text or not case.title:
            continue
        rows.append(
            {
                "id": str(case.id),
                "domain": case.domain or "",
                "source": case.source or "",
                "text": retrieval_text,
            }
        )
    return rows


def _create_staging_collection(name: str, alias: str) -> Collection:
    fields = [
        FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("domain", DataType.VARCHAR, max_length=100),
        FieldSchema("source", DataType.VARCHAR, max_length=50),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
    ]
    collection = Collection(
        name,
        CollectionSchema(fields, description="High-quality legal case retrieval index"),
        using=alias,
    )
    collection.create_index(
        "embedding",
        {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.create_index(
        "sparse_embedding",
        {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
    )
    return collection


async def _embed_with_retry(embed_model, texts: list[str]) -> list[list[float]]:
    for attempt in range(3):
        try:
            return await embed_model.aembed_documents(texts)
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError("embedding retry loop exited unexpectedly")


async def rebuild_milvus_index(rows: list[dict[str, str]], batch_size: int = 10) -> str | None:
    if not rows:
        raise ValueError("PostgreSQL 中没有可索引的高质量案例")

    from milvus_model.sparse import BM25EmbeddingFunction
    import jieba
    import scipy.sparse as sp

    settings = get_settings()
    alias = "case_pilot_import"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_name = f"case_index_build_{timestamp}"
    backup_name = f"case_index_backup_{timestamp}"

    connections.connect(alias=alias, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    collection = _create_staging_collection(staging_name, alias)
    texts = [row["text"] for row in rows]

    bm25 = BM25EmbeddingFunction(analyzer=jieba.lcut)
    bm25.fit(texts)
    sparse_vectors = bm25.encode_documents(texts)
    sparse_dicts = []
    for vector in sparse_vectors:
        csr = sp.csr_array(vector)
        sparse_dicts.append(
            {int(index): float(value) for index, value in zip(csr.indices, csr.data)}
        )

    embed_model = get_embedding_model()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            dense_vectors = await _embed_with_retry(embed_model, batch_texts)
            records = []
            for row, dense, sparse in zip(
                batch,
                dense_vectors,
                sparse_dicts[start : start + batch_size],
            ):
                records.append({**row, "embedding": dense, "sparse_embedding": sparse})
            collection.insert(records)
            print(f"[Milvus] {min(start + batch_size, len(rows))}/{len(rows)}")
        collection.flush()
    except Exception:
        collection.release()
        utility.drop_collection(staging_name, using=alias)
        raise
    finally:
        client = getattr(embed_model, "client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    entity_count = collection.num_entities
    if entity_count != len(rows):
        collection.release()
        raise RuntimeError(f"Milvus staging 数量异常: expected={len(rows)}, actual={entity_count}")
    collection.load()

    model_next = MODEL_FILE.with_suffix(".next.json")
    if model_next.exists():
        model_next.unlink()
    bm25.save(str(model_next))

    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    previous_name: str | None = None
    collection.release()
    try:
        if client.has_collection(COLLECTION_NAME):
            client.release_collection(COLLECTION_NAME)
            client.rename_collection(COLLECTION_NAME, backup_name)
            previous_name = backup_name
        client.rename_collection(staging_name, COLLECTION_NAME)
        client.load_collection(COLLECTION_NAME)
    except Exception:
        if previous_name and not client.has_collection(COLLECTION_NAME):
            client.rename_collection(previous_name, COLLECTION_NAME)
        raise
    finally:
        connections.disconnect(alias=alias)

    model_next.replace(MODEL_FILE)
    return previous_name


async def run(args: argparse.Namespace) -> None:
    await ensure_postgres_schema()

    if not args.index_only:
        rows = load_sqlite_rows(args.input, args.limit)
        imported = await upsert_postgres(rows)
        if imported != len(rows):
            raise RuntimeError(f"PostgreSQL 校验失败: expected={len(rows)}, actual={imported}")
        print(f"[PostgreSQL] 已校验 {imported} 条案例")

    if not args.skip_index:
        index_rows = await fetch_index_rows()
        print(f"[Milvus] 准备索引 {len(index_rows)} 条可用案例")
        backup = await rebuild_milvus_index(index_rows, args.batch_size)
        print(f"[Milvus] case_index 已切换，旧索引备份: {backup or '无'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入高质量案例数据包")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=0, help="仅导入前 N 条；0 表示全部")
    parser.add_argument("--batch-size", type=int, default=10, help="单批向量化数量")
    parser.add_argument("--skip-index", action="store_true", help="只写 PostgreSQL")
    parser.add_argument("--index-only", action="store_true", help="只根据 PostgreSQL 重建索引")
    args = parser.parse_args()
    if args.skip_index and args.index_only:
        parser.error("--skip-index 与 --index-only 不能同时使用")
    if args.limit < 0 or args.batch_size <= 0:
        parser.error("--limit 不能为负数，--batch-size 必须大于 0")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
