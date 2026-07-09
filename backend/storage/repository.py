"""Repository layer for persisted market data and features."""

from collections.abc import Sequence
from datetime import datetime
from typing import TypeGuard

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Symbol


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _Unset()


def _is_provided(value: datetime | None | _Unset) -> TypeGuard[datetime | None]:
    return value is not _UNSET


class SymbolRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, symbol: str, asset_type: str = "equity") -> Symbol:
        row = Symbol(symbol=symbol, asset_type=asset_type)
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, symbol_id: int) -> Symbol | None:
        return await self._session.get(Symbol, symbol_id)

    async def get_by_symbol(self, symbol: str) -> Symbol | None:
        result = await self._session.execute(
            select(Symbol).where(Symbol.symbol == symbol)
        )
        return result.scalar_one_or_none()

    async def list(self, *, active_only: bool = False) -> Sequence[Symbol]:
        stmt = select(Symbol).order_by(Symbol.symbol)
        if active_only:
            stmt = stmt.where(Symbol.active.is_(True))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_coverage(
        self,
        symbol_id: int,
        *,
        coverage_start: datetime | None | _Unset = _UNSET,
        coverage_end: datetime | None | _Unset = _UNSET,
        last_ingested_at: datetime | None | _Unset = _UNSET,
    ) -> Symbol | None:
        row = await self.get_by_id(symbol_id)
        if row is None:
            return None
        if _is_provided(coverage_start):
            row.coverage_start = coverage_start
        if _is_provided(coverage_end):
            row.coverage_end = coverage_end
        if _is_provided(last_ingested_at):
            row.last_ingested_at = last_ingested_at
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def set_active(self, symbol_id: int, active: bool) -> Symbol | None:
        row = await self.get_by_id(symbol_id)
        if row is None:
            return None
        row.active = active
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def delete(self, symbol_id: int) -> None:
        row = await self.get_by_id(symbol_id)
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()
