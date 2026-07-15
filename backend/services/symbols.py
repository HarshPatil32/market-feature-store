"""Symbol registry business logic."""

from collections.abc import Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Symbol
from backend.storage.repository import SymbolRepository
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


async def get_symbol(session: AsyncSession, symbol: str) -> Symbol:
    ticker = symbol.strip().upper()
    repo = SymbolRepository(session)
    row = await repo.get_by_symbol(ticker)
    if row is None:
        raise SymbolNotFoundError(ticker)
    return row


async def deactivate_symbol(session: AsyncSession, symbol: str) -> Symbol:
    ticker = symbol.strip().upper()
    repo = SymbolRepository(session)
    row = await repo.get_by_symbol(ticker)
    if row is None:
        raise SymbolNotFoundError(ticker)
    updated = await repo.set_active(row, active=False)
    assert updated is not None
    return updated
