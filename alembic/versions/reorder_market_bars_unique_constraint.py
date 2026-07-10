"""reorder market_bars unique constraint for symbol+timeframe+timestamp lookups

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-07-09 14:30:00.000000

Reorders the market_bars unique constraint from (symbol_id, timestamp, timeframe)
to (symbol_id, timeframe, timestamp) so the btree index matches the hot-path
list_by_symbol query (WHERE symbol_id = ? AND timeframe = ? ... ORDER BY timestamp).

At large table sizes, use CREATE UNIQUE INDEX CONCURRENTLY + ADD CONSTRAINT
USING INDEX instead of drop/recreate inside a transaction.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_market_bars_symbol_id_timestamp_timeframe",
        "market_bars",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_market_bars_symbol_id_timeframe_timestamp",
        "market_bars",
        ["symbol_id", "timeframe", "timestamp"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_market_bars_symbol_id_timeframe_timestamp",
        "market_bars",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_market_bars_symbol_id_timestamp_timeframe",
        "market_bars",
        ["symbol_id", "timestamp", "timeframe"],
    )
