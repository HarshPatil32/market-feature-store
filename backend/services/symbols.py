"""Symbol registry business logic."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Symbol
from backend.storage.repository import MarketBarRepository, SymbolRepository
from backend.storage.schemas import SymbolCreate


class DuplicateSymbolError(Exception):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(f"Symbol already exists: {symbol}")


class SymbolNotFoundError(Exception):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(f"Symbol not found: {symbol}")


async def add_symbol(session: AsyncSession, data: SymbolCreate) -> Symbol:
    repo = SymbolRepository(session)
    try:
        async with session.begin_nested():
            return await repo.create(symbol=data.symbol, asset_type=data.asset_type)
    except IntegrityError as exc:
        raise DuplicateSymbolError(data.symbol) from exc


async def list_symbols(
    session: AsyncSession,
    *,
    active_only: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[Symbol]:
    repo = SymbolRepository(session)
    return await repo.list(active_only=active_only, limit=limit, offset=offset)


async def detect_stale_symbols(
    session: AsyncSession,
    *,
    threshold: timedelta,
) -> Sequence[Symbol]:
    """Return active symbols whose coverage_end is missing or older than threshold."""
    repo = SymbolRepository(session)
    return await repo.list_stale(threshold)


async def get_symbol(session: AsyncSession, symbol: str) -> Symbol:
    ticker = symbol.strip().upper()
    repo = SymbolRepository(session)
    row = await repo.get_by_symbol(ticker)
    if row is None:
        raise SymbolNotFoundError(ticker)
    return row


async def sync_symbol_coverage(session: AsyncSession, symbol_id: int) -> Symbol | None:
    """Recompute symbol coverage from stored market bars."""
    bar_repo = MarketBarRepository(session)
    coverage_start, coverage_end = await bar_repo.get_timestamp_bounds(symbol_id)
    repo = SymbolRepository(session)
    return await repo.update_coverage(
        symbol_id,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        last_ingested_at=datetime.now(tz=UTC),
    )


async def deactivate_symbol(session: AsyncSession, symbol: str) -> Symbol:
    ticker = symbol.strip().upper()
    repo = SymbolRepository(session)
    row = await repo.get_by_symbol(ticker)
    if row is None:
        raise SymbolNotFoundError(ticker)
    updated = await repo.set_active(row, active=False)
    assert updated is not None
    return updated
