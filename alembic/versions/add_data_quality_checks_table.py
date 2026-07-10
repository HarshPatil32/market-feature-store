"""add data_quality_checks table

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-09 00:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_quality_checks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("symbol_id", sa.Integer(), nullable=True),
        sa.Column("check_name", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("affected_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["symbol_id"],
            ["symbols.id"],
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="ck_data_quality_checks_severity",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_data_quality_checks_run_id",
        "data_quality_checks",
        ["run_id"],
    )
    op.create_index(
        "ix_data_quality_checks_symbol_id",
        "data_quality_checks",
        ["symbol_id"],
    )
    op.create_index(
        "ix_data_quality_checks_created_at",
        "data_quality_checks",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_data_quality_checks_created_at",
        table_name="data_quality_checks",
    )
    op.drop_index(
        "ix_data_quality_checks_symbol_id",
        table_name="data_quality_checks",
    )
    op.drop_index(
        "ix_data_quality_checks_run_id",
        table_name="data_quality_checks",
    )
    op.drop_table("data_quality_checks")
