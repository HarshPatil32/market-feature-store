"""Market data ingestion pipeline."""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar, bars_from_normalize_result
from backend.services.data_quality import persist_check_results
from backend.services.raw_market_data import persist_raw_fetch
from backend.storage.models import RawMarketData
from backend.validation.validator import QualityCheckResult, run_checks


async def ingest_raw_data(
    session: AsyncSession,
    *,
    run_id: int,
    response_payload: dict[str, Any],
    normalize: Callable[[dict[str, Any]], object],
    symbol_id: int | None = None,
    source: str | None = None,
    request_params: dict[str, Any] | None = None,
    validate: (
        Callable[[Sequence[Bar]], Sequence[QualityCheckResult]] | None
    ) = run_checks,
) -> RawMarketData:
    """Persist the raw provider response, normalize it, and run validation."""
    if inspect.iscoroutinefunction(normalize):
        raise TypeError("normalize must be a synchronous callable")
    if validate is not None and inspect.iscoroutinefunction(validate):
        raise TypeError("validate must be a synchronous callable")

    raw_row = await persist_raw_fetch(
        session,
        run_id=run_id,
        symbol_id=symbol_id,
        source=source,
        request_params=request_params,
        response_payload=response_payload,
    )

    async with session.begin_nested():
        bars = bars_from_normalize_result(normalize(response_payload))
        if validate is not None and symbol_id is not None:
            results = validate(bars)
            if results:
                await persist_check_results(
                    session,
                    results,
                    symbol_id=symbol_id,
                    run_id=run_id,
                )

    return raw_row
