"""add description and requirements to feature_definitions

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-09-24 01:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d0e1f2a3b4c5"
down_revision: str | Sequence[str] | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "feature_definitions",
        sa.Column("description", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "feature_definitions",
        sa.Column(
            "requirements", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("feature_definitions", "requirements")
    op.drop_column("feature_definitions", "description")
