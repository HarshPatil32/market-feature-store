"""add raw_market_data table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-08 23:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_market_data",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("symbol_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=True),
        sa.Column("request_params", postgresql.JSONB(), nullable=True),
        sa.Column("response_payload", postgresql.JSONB(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_raw_market_data_run_id",
        "raw_market_data",
        ["run_id"],
    )
    op.create_index(
        "ix_raw_market_data_symbol_id",
        "raw_market_data",
        ["symbol_id"],
    )
    op.create_index(
        "ix_raw_market_data_created_at",
        "raw_market_data",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_raw_market_data_created_at", table_name="raw_market_data")
    op.drop_index("ix_raw_market_data_symbol_id", table_name="raw_market_data")
    op.drop_index("ix_raw_market_data_run_id", table_name="raw_market_data")
    op.drop_table("raw_market_data")
