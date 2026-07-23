"""Reprocess stored raw market data through normalize/validate."""

import inspect
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.services.raw_market_data import load_response_payload
from backend.storage.models import IngestionRun, RunStatus
from backend.storage.repository import (
    DataQualityCheckRepository,
    IngestionRunRepository,
    MarketBarRepository,
    RawMarketDataRepository,
)

logger = structlog.get_logger(__name__)


def _ensure_sync_callable(name: str, fn: Callable[..., object]) -> None:
    if inspect.iscoroutinefunction(fn):
        raise TypeError(f"{name} must be a synchronous callable")


def _bars_from_normalize(result: object) -> list[Bar]:
    if isinstance(result, Bar):
        return [result]
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        bars = list(result)
        if any(not isinstance(bar, Bar) for bar in bars):
            raise TypeError("normalize must return Bar instances")
        return bars
    raise TypeError("normalize must return a Bar or iterable of Bar")


async def reprocess_from_raw(
    session: AsyncSession,
    *,
    symbol_id: int,
    normalize: Callable[[dict[str, Any]], object],
    validate: Callable[[Sequence[Bar]], Sequence[dict[str, Any]]] | None = None,
    run_id: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> IngestionRun:
    """Rebuild market bars from stored raw rows without refetching from a provider."""
    _ensure_sync_callable("normalize", normalize)
    if validate is not None:
        _ensure_sync_callable("validate", validate)

    raw_repo = RawMarketDataRepository(session)
    raw_rows = await raw_repo.list_by_symbol(
        symbol_id,
        run_id=run_id,
        start=start,
        end=end,
    )
    if not raw_rows:
        raise ValueError("no raw market data rows match the given filters")

    run_repo = IngestionRunRepository(session)
    bar_repo = MarketBarRepository(session)
    check_repo = DataQualityCheckRepository(session)

    run = await run_repo.create(run_type="reprocess", symbol_id=symbol_id)
    started_at = datetime.now(tz=UTC)
    await run_repo.update(run.id, status=RunStatus.running, started_at=started_at)

    inserted = 0
    failed = 0
    staged_checks: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        try:
            payload = await load_response_payload(raw_row)
        except Exception:
            logger.exception("reprocess_load_payload_failed", raw_id=raw_row.id)
            failed += 1
            continue

        try:
            async with session.begin_nested():
                bars = _bars_from_normalize(normalize(payload))
                for bar in bars:
                    await bar_repo.upsert(
                        symbol_id=symbol_id,
                        timestamp=bar.ts,
                        timeframe=bar.timeframe,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                    )

                if validate is not None:
                    for check in validate(bars):
                        staged_checks.append(
                            {
                                **check,
                                "run_id": run.id,
                                "symbol_id": symbol_id,
                            }
                        )
            inserted += len(bars)
        except Exception:
            logger.exception("reprocess_row_failed", raw_id=raw_row.id)
            failed += 1

    if staged_checks:
        await check_repo.bulk_create(staged_checks)

    if inserted > 0:
        status = RunStatus.succeeded
        error_message = (
            f"{failed} raw row(s) failed to reprocess" if failed > 0 else None
        )
    elif failed > 0:
        status = RunStatus.failed
        error_message = f"all {failed} raw row(s) failed to reprocess"
    else:
        status = RunStatus.succeeded
        error_message = None

    updated = await run_repo.update(
        run.id,
        status=status,
        fetched=len(raw_rows),
        inserted=inserted,
        failed=failed,
        error_message=error_message,
        finished_at=datetime.now(tz=UTC),
    )
    if updated is None:
        raise RuntimeError(f"reprocess run {run.id} disappeared during update")
    return updated
