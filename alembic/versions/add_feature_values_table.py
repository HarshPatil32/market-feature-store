"""add feature_values table

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-09 01:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("feature_definition_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Numeric(18, 8), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["symbol_id"],
            ["symbols.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["feature_definition_id"],
            ["feature_definitions.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "symbol_id",
            "timestamp",
            "feature_definition_id",
            name="uq_feature_values_symbol_id_timestamp_feature_definition_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("feature_values")
