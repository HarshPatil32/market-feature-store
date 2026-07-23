"""Raw market data persistence service."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import RawMarketData
from backend.storage.repository import RawMarketDataRepository


async def persist_raw_fetch(
    session: AsyncSession,
    *,
    run_id: int,
    response_payload: dict[str, Any],
    symbol_id: int | None = None,
    source: str | None = None,
    request_params: dict[str, Any] | None = None,
) -> RawMarketData:
    repo = RawMarketDataRepository(session)
    return await repo.create(
        run_id=run_id,
        symbol_id=symbol_id,
        source=source,
        request_params=request_params,
        response_payload=response_payload,
    )
