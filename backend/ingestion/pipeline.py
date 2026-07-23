"""Market data ingestion pipeline."""

import inspect
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.raw_market_data import persist_raw_fetch
from backend.storage.models import RawMarketData


async def ingest_raw_data(
    session: AsyncSession,
    *,
    run_id: int,
    response_payload: dict[str, Any],
    normalize: Callable[[dict[str, Any]], object],
    symbol_id: int | None = None,
    source: str | None = None,
    request_params: dict[str, Any] | None = None,
) -> RawMarketData:
    """Persist the raw provider response, then attempt normalization."""
    if inspect.iscoroutinefunction(normalize):
        raise TypeError("normalize must be a synchronous callable")

    raw_row = await persist_raw_fetch(
        session,
        run_id=run_id,
        symbol_id=symbol_id,
        source=source,
        request_params=request_params,
        response_payload=response_payload,
    )

    async with session.begin_nested():
        normalize(response_payload)

    return raw_row
