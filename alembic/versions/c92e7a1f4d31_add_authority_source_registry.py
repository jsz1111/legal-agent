"""add authority source registry

Revision ID: c92e7a1f4d31
Revises: 98789af4c4dc
Create Date: 2026-07-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c92e7a1f4d31"
down_revision: Union[str, Sequence[str], None] = "98789af4c4dc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "authority_sources",
        sa.Column("source_key", sa.String(length=120), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("issuer", sa.String(length=500), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("authority_level", sa.String(length=50), nullable=False),
        sa.Column("official_url", sa.String(length=1200), nullable=True),
        sa.Column("domains", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("usage_note", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_authority_sources_source_key", "authority_sources", ["source_key"], unique=True)
    op.create_index("ix_authority_sources_status", "authority_sources", ["status"])

    op.create_table(
        "authority_versions",
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("version_key", sa.String(length=180), nullable=False),
        sa.Column("document_no", sa.String(length=150), nullable=True),
        sa.Column("published_at", sa.String(length=30), nullable=True),
        sa.Column("effective_from", sa.String(length=30), nullable=True),
        sa.Column("effective_to", sa.String(length=30), nullable=True),
        sa.Column("official_file_url", sa.String(length=1500), nullable=True),
        sa.Column("local_path", sa.String(length=1000), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("review_status", sa.String(length=40), nullable=False),
        sa.Column("verified_at", sa.String(length=40), nullable=True),
        sa.Column("source_metadata", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["authority_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_authority_versions_version_key", "authority_versions", ["version_key"], unique=True)
    op.create_index("ix_authority_versions_source", "authority_versions", ["source_id"])
    op.create_index("ix_authority_versions_review", "authority_versions", ["review_status"])

    op.create_table(
        "followup_rule_citations",
        sa.Column("rule_id", sa.String(length=120), nullable=False),
        sa.Column("domain", sa.String(length=100), nullable=False),
        sa.Column("rule_type", sa.String(length=20), nullable=False),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column("source_version_id", sa.BigInteger(), nullable=False),
        sa.Column("locator", sa.Text(), nullable=True),
        sa.Column("source_excerpt", sa.Text(), nullable=True),
        sa.Column("derivation_note", sa.Text(), nullable=True),
        sa.Column("mapping_status", sa.String(length=40), nullable=False),
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["source_version_id"], ["authority_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rule_id", "source_version_id", name="uq_followup_rule_source_version"),
    )
    op.create_index("ix_followup_citations_domain", "followup_rule_citations", ["domain"])
    op.create_index("ix_followup_citations_rule", "followup_rule_citations", ["rule_id"])
    op.create_index("ix_followup_citations_status", "followup_rule_citations", ["mapping_status"])


def downgrade() -> None:
    op.drop_table("followup_rule_citations")
    op.drop_table("authority_versions")
    op.drop_table("authority_sources")
