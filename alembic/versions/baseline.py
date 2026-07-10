"""baseline

Revision ID: c89e3409f16f
Revises:
Create Date: 2026-07-08 15:49:08.845412

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "c89e3409f16f"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
