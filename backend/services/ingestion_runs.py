"""Ingestion run trigger logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.symbols import get_symbol
from backend.storage.models import IngestionRun
from backend.storage.repository import IngestionRunRepository


async def trigger_backfill(session: AsyncSession, symbol: str) -> IngestionRun:
    row = await get_symbol(session, symbol)
    repo = IngestionRunRepository(session)
    return await repo.create(run_type="backfill", symbol_id=row.id)


async def trigger_incremental(session: AsyncSession, symbol: str) -> IngestionRun:
    row = await get_symbol(session, symbol)
    repo = IngestionRunRepository(session)
    return await repo.create(run_type="incremental", symbol_id=row.id)
