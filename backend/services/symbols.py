"""Symbol registry business logic."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Symbol
from backend.storage.repository import SymbolRepository
from backend.storage.schemas import SymbolCreate


class DuplicateSymbolError(Exception):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(f"Symbol already exists: {symbol}")


async def add_symbol(session: AsyncSession, data: SymbolCreate) -> Symbol:
    repo = SymbolRepository(session)
    try:
        async with session.begin_nested():
            return await repo.create(symbol=data.symbol, asset_type=data.asset_type)
    except IntegrityError as exc:
        raise DuplicateSymbolError(data.symbol) from exc
