"""Incremental ingestion orchestration: fetch missing bars and upsert."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.pipeline import ingest_raw_data
from backend.logging import bind_ingestion_run_id, clear_ingestion_run_id
from backend.providers.base import MarketDataProvider
from backend.services.symbols import get_symbol, sync_symbol_coverage
from backend.storage.models import IngestionRun, RunStatus
from backend.storage.repository import IngestionRunRepository, MarketBarRepository
from backend.validation.validator import run_checks

logger = structlog.get_logger(__name__)


def _bars_payload(bars: Sequence[Bar]) -> dict[str, Any]:
    return {"bars": [bar.model_dump(mode="json") for bar in bars]}


def _normalize_bars_payload(payload: dict[str, Any]) -> list[Bar]:
    return [Bar.model_validate(entry) for entry in payload["bars"]]


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _incremental_start(
    coverage_end: datetime | None,
    *,
    lookback_days: int,
    end: AwareDatetime,
) -> AwareDatetime:
    if coverage_end is not None:
        return _ensure_utc(coverage_end)
    return end - timedelta(days=lookback_days)


async def incremental_symbol(
    session: AsyncSession,
    *,
    symbol: str,
    timeframe: str,
    provider: MarketDataProvider,
    source: str,
    lookback_days: int = 7,
) -> IngestionRun:
    """Fetch and insert bars from the last known coverage point up to now."""
    ticker = symbol.strip().upper()
    symbol_row = await get_symbol(session, ticker)
    symbol_id = symbol_row.id

    end = datetime.now(tz=UTC)
    start = _incremental_start(
        symbol_row.coverage_end,
        lookback_days=lookback_days,
        end=end,
    )

    run_repo = IngestionRunRepository(session)
    bar_repo = MarketBarRepository(session)
    run = await run_repo.create(run_type="incremental", symbol_id=symbol_id)

    if start >= end:
        updated = await run_repo.update(
            run.id,
            status=RunStatus.succeeded,
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            fetched=0,
            inserted=0,
            failed=0,
            error_message=None,
        )
        if updated is None:
            raise RuntimeError(f"incremental run {run.id} disappeared during update")
        await session.commit()
        logger.info(
            "incremental_symbol_already_current",
            symbol=ticker,
            run_id=run.id,
        )
        return updated

    await run_repo.update(
        run.id,
        status=RunStatus.running,
        started_at=datetime.now(tz=UTC),
        error_message=None,
        finished_at=None,
    )
    await session.commit()

    bind_ingestion_run_id(str(run.id))
    logger.info(
        "incremental_symbol_started",
        symbol=ticker,
        run_id=run.id,
        timeframe=timeframe,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    total_fetched = 0
    total_inserted = 0

    try:
        try:
            bars = await provider.fetch_historical_bars(ticker, timeframe, start, end)
            total_fetched = len(bars)
            await ingest_raw_data(
                session,
                run_id=run.id,
                symbol_id=symbol_id,
                source=source,
                response_payload=_bars_payload(bars),
                request_params={
                    "symbol": ticker,
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                normalize=_normalize_bars_payload,
                validate=run_checks,
            )
            for bar in bars:
                inserted_bar = await bar_repo.insert_if_not_exists(
                    symbol_id=symbol_id,
                    timestamp=bar.ts,
                    timeframe=bar.timeframe,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                if inserted_bar is not None:
                    total_inserted += 1
        except Exception as exc:
            await session.rollback()
            updated = await run_repo.update(
                run.id,
                status=RunStatus.failed,
                fetched=total_fetched,
                inserted=total_inserted,
                failed=1,
                error_message=str(exc),
                finished_at=datetime.now(tz=UTC),
            )
            if updated is None:
                raise RuntimeError(
                    f"incremental run {run.id} disappeared during update"
                ) from exc
            await session.commit()
            logger.error(
                "incremental_symbol_failed",
                symbol=ticker,
                run_id=run.id,
                error=str(exc),
            )
            raise

        if total_inserted > 0:
            await sync_symbol_coverage(session, symbol_id)
        updated = await run_repo.update(
            run.id,
            status=RunStatus.succeeded,
            fetched=total_fetched,
            inserted=total_inserted,
            failed=0,
            finished_at=datetime.now(tz=UTC),
        )
        if updated is None:
            raise RuntimeError(f"incremental run {run.id} disappeared during update")
        await session.commit()
        logger.info(
            "incremental_symbol_succeeded",
            symbol=ticker,
            run_id=updated.id,
            fetched=total_fetched,
            inserted=total_inserted,
        )
        return updated
    finally:
        clear_ingestion_run_id()


async def incremental_symbols(
    session: AsyncSession,
    *,
    symbols: Sequence[str],
    timeframe: str,
    provider: MarketDataProvider,
    source: str,
    lookback_days: int = 7,
) -> dict[str, IngestionRun | BaseException]:
    """Run incremental ingestion for each symbol, continuing after failures."""
    results: dict[str, IngestionRun | BaseException] = {}
    for symbol in symbols:
        ticker = symbol.strip().upper()
        try:
            results[ticker] = await incremental_symbol(
                session,
                symbol=ticker,
                timeframe=timeframe,
                provider=provider,
                source=source,
                lookback_days=lookback_days,
            )
        except Exception as exc:
            results[ticker] = exc
    succeeded = sum(1 for value in results.values() if isinstance(value, IngestionRun))
    logger.info(
        "incremental_symbols_done",
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )
    return results
