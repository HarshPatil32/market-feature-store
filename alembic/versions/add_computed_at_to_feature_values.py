"""add computed_at to feature_values and widen value column

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-10-19 04:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "feature_values",
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.alter_column(
        "feature_values",
        "value",
        existing_type=sa.Numeric(precision=18, scale=8),
        type_=sa.Numeric(precision=20, scale=8),
        existing_nullable=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    too_large = bind.execute(
        sa.text(
            "SELECT 1 FROM feature_values " "WHERE abs(value) >= 10000000000 " "LIMIT 1"
        )
    ).first()
    if too_large is not None:
        raise RuntimeError(
            "Cannot downgrade feature_values.value to Numeric(18,8): "
            "one or more rows have abs(value) >= 1e10. "
            "Delete or adjust those rows before downgrading."
        )
    op.alter_column(
        "feature_values",
        "value",
        existing_type=sa.Numeric(precision=20, scale=8),
        type_=sa.Numeric(precision=18, scale=8),
        existing_nullable=False,
    )
    op.drop_column("feature_values", "computed_at")
