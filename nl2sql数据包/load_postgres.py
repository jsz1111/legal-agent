"""Load an offline legal-statistics package into PostgreSQL.

This script is intentionally separate from the package builder. Building the
package never connects to PostgreSQL; loading requires an explicit database URL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COPY_TABLES = {
    "datasets": [
        "dataset_id", "yearbook_year", "statistical_year_start",
        "statistical_year_end", "title", "institution", "topic", "unit",
        "source_file", "source_sheet", "source_url", "source_sha256",
        "source_rows", "source_columns", "data_start_row", "year_quality",
        "extraction_status", "notes",
    ],
    "records": [
        "record_id", "dataset_id", "source_row", "row_label", "row_data",
        "search_text",
    ],
    "facts": [
        "fact_id", "dataset_id", "record_id", "yearbook_year",
        "statistical_year_start", "statistical_year_end", "institution",
        "topic", "dimension_label", "metric", "metric_path",
        "numeric_value", "text_value", "value_kind", "unit", "source_row",
        "source_column", "quality_flag",
    ],
    "cells": [
        "dataset_id", "source_row", "source_column", "cell_text",
        "resolved_text", "cell_type", "is_merged", "merged_range",
    ],
}


def load_package(package_dir: Path, database_url: str, replace: bool) -> None:
    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit(
            "PostgreSQL loading requires psycopg2. Install the project "
            "requirements or run: pip install psycopg2-binary"
        ) from exc

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    schema_sql = (package_dir / "schema.sql").read_text(encoding="utf-8")

    with psycopg2.connect(database_url) as connection:
        with connection.cursor() as cursor:
            if replace:
                cursor.execute("DROP SCHEMA IF EXISTS legal_statistics CASCADE")
            cursor.execute(schema_sql)

            for table_name, columns in COPY_TABLES.items():
                csv_path = package_dir / "csv" / f"{table_name}.csv"
                expected = int(manifest["counts"].get(table_name, 0))
                if not csv_path.exists():
                    raise FileNotFoundError(csv_path)

                column_sql = ", ".join(columns)
                copy_sql = (
                    f"COPY legal_statistics.{table_name} ({column_sql}) "
                    "FROM STDIN WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')"
                )
                with csv_path.open("r", encoding="utf-8", newline="") as source:
                    cursor.copy_expert(copy_sql, source)

                cursor.execute(f"SELECT count(*) FROM legal_statistics.{table_name}")
                actual = cursor.fetchone()[0]
                if actual != expected:
                    raise RuntimeError(
                        f"{table_name}: expected {expected} rows, loaded {actual}"
                    )

    print(
        "Loaded legal statistics package into PostgreSQL: "
        f"{manifest['counts']['datasets']} datasets, "
        f"{manifest['counts']['facts']} facts"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop and recreate the legal_statistics schema before loading.",
    )
    args = parser.parse_args()
    load_package(args.package_dir.resolve(), args.database_url, args.replace)


if __name__ == "__main__":
    main()
