"""add raw payload object storage columns

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-07-23 01:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d0e1f2a3b4"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "raw_market_data",
        sa.Column("payload_object_key", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "raw_market_data",
        sa.Column("payload_size_bytes", sa.Integer(), nullable=True),
    )
    op.alter_column(
        "raw_market_data",
        "response_payload",
        existing_type=postgresql.JSONB(),
        nullable=True,
    )
    op.create_check_constraint(
        "ck_raw_market_data_payload_location",
        "raw_market_data",
        "(response_payload IS NULL) != (payload_object_key IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_raw_market_data_payload_location",
        "raw_market_data",
        type_="check",
    )
    op.alter_column(
        "raw_market_data",
        "response_payload",
        existing_type=postgresql.JSONB(),
        nullable=False,
    )
    op.drop_column("raw_market_data", "payload_size_bytes")
    op.drop_column("raw_market_data", "payload_object_key")
