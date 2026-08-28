"""Backfill orchestration: fetch → raw → normalize → validate → upsert."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.chunking import chunk_date_range
from backend.ingestion.pipeline import ingest_raw_data
from backend.logging import bind_ingestion_run_id, clear_ingestion_run_id
from backend.providers.base import MarketDataProvider
from backend.services.symbols import get_symbol, sync_symbol_coverage
from backend.storage.models import IngestionRun, RunStatus
from backend.storage.repository import (
    IngestionRunRepository,
    MarketBarRepository,
    RawMarketDataRepository,
)
from backend.validation.validator import run_checks

logger = structlog.get_logger(__name__)


def _bars_payload(bars: Sequence[Bar]) -> dict[str, Any]:
    return {"bars": [bar.model_dump(mode="json") for bar in bars]}


def _normalize_bars_payload(payload: dict[str, Any]) -> list[Bar]:
    return [Bar.model_validate(entry) for entry in payload["bars"]]


def _chunk_key(
    ticker: str,
    timeframe: str,
    chunk_start: AwareDatetime,
    chunk_end: AwareDatetime,
) -> tuple[str, str, str, str]:
    return (ticker, timeframe, chunk_start.isoformat(), chunk_end.isoformat())


def _chunk_key_from_params(params: dict[str, Any]) -> tuple[str, str, str, str]:
    return (params["symbol"], params["timeframe"], params["start"], params["end"])


async def _load_resume_run(
    session: AsyncSession,
    *,
    resume_run_id: int,
    symbol_id: int,
) -> IngestionRun:
    run_repo = IngestionRunRepository(session)
    run = await run_repo.get_by_id(resume_run_id)
    if run is None:
        raise ValueError(f"backfill run {resume_run_id} not found")
    if run.run_type != "backfill":
        raise ValueError(f"backfill run {resume_run_id} is not a backfill run")
    if run.symbol_id != symbol_id:
        raise ValueError(f"backfill run {resume_run_id} belongs to a different symbol")
    if run.status == RunStatus.succeeded:
        raise ValueError(f"backfill run {resume_run_id} already succeeded")
    return run


async def _completed_chunk_keys(
    session: AsyncSession,
    *,
    symbol_id: int,
    run_id: int,
) -> set[tuple[str, str, str, str]]:
    raw_repo = RawMarketDataRepository(session)
    raw_rows = await raw_repo.list_by_symbol(symbol_id, run_id=run_id)
    return {
        _chunk_key_from_params(row.request_params)
        for row in raw_rows
        if row.request_params is not None
    }


async def backfill_symbol(
    session: AsyncSession,
    *,
    symbol: str,
    timeframe: str,
    start: AwareDatetime,
    end: AwareDatetime,
    provider: MarketDataProvider,
    source: str,
    resume_run_id: int | None = None,
) -> IngestionRun:
    """Run a full backfill for one symbol and date range."""
    if start > end:
        raise ValueError("start must be <= end")

    ticker = symbol.strip().upper()
    symbol_row = await get_symbol(session, ticker)
    symbol_id = symbol_row.id

    run_repo = IngestionRunRepository(session)
    bar_repo = MarketBarRepository(session)
    existing_start, existing_end = await bar_repo.get_timeframe_coverage(
        symbol_id, timeframe=timeframe
    )

    if resume_run_id is None:
        run = await run_repo.create(run_type="backfill", symbol_id=symbol_id)
        already_processed: set[tuple[str, str, str, str]] = set()
        total_fetched = 0
        total_inserted = 0
        total_failed = 0
    else:
        run = await _load_resume_run(
            session,
            resume_run_id=resume_run_id,
            symbol_id=symbol_id,
        )
        already_processed = await _completed_chunk_keys(
            session,
            symbol_id=symbol_id,
            run_id=run.id,
        )
        total_fetched = run.fetched
        total_inserted = run.inserted
        total_failed = run.failed

    chunks = chunk_date_range(start, end, timeframe)
    multi_chunk = len(chunks) > 1
    chunk_start, chunk_end = start, end

    await run_repo.update(
        run.id,
        status=RunStatus.running,
        started_at=run.started_at or datetime.now(tz=UTC),
        error_message=None,
        finished_at=None,
    )
    await session.commit()

    bind_ingestion_run_id(str(run.id))
    logger.info(
        "backfill_symbol_started",
        symbol=ticker,
        run_id=run.id,
        timeframe=timeframe,
    )

    try:
        try:
            for chunk_start, chunk_end in chunks:
                key = _chunk_key(ticker, timeframe, chunk_start, chunk_end)
                if key in already_processed:
                    continue
                # Skip provider fetch for chunks fully within existing bar coverage.
                if (
                    existing_start is not None
                    and existing_end is not None
                    and chunk_start >= existing_start
                    and chunk_end <= existing_end
                ):
                    continue

                bars = await provider.fetch_historical_bars(
                    ticker, timeframe, chunk_start, chunk_end
                )
                await ingest_raw_data(
                    session,
                    run_id=run.id,
                    symbol_id=symbol_id,
                    source=source,
                    response_payload=_bars_payload(bars),
                    request_params={
                        "symbol": ticker,
                        "timeframe": timeframe,
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                    },
                    normalize=_normalize_bars_payload,
                    validate=run_checks,
                )
                chunk_inserted = 0
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
                        chunk_inserted += 1
                if chunk_inserted > 0:
                    await sync_symbol_coverage(session, symbol_id)
                total_fetched += len(bars)
                total_inserted += chunk_inserted
                await run_repo.update(
                    run.id,
                    fetched=total_fetched,
                    inserted=total_inserted,
                )
                await session.commit()
        except Exception as exc:
            await session.rollback()
            total_failed += 1
            error_message = (
                f"chunk {chunk_start.isoformat()}..{chunk_end.isoformat()} failed: {exc}"
                if multi_chunk
                else str(exc)
            )
            updated = await run_repo.update(
                run.id,
                status=RunStatus.failed,
                fetched=total_fetched,
                inserted=total_inserted,
                failed=total_failed,
                error_message=error_message,
                finished_at=datetime.now(tz=UTC),
            )
            if updated is None:
                raise RuntimeError(
                    f"backfill run {run.id} disappeared during update"
                ) from exc
            await session.commit()
            logger.error(
                "backfill_symbol_failed",
                symbol=ticker,
                run_id=run.id,
                error=error_message,
            )
            raise

        updated = await run_repo.update(
            run.id,
            status=RunStatus.succeeded,
            fetched=total_fetched,
            inserted=total_inserted,
            failed=total_failed,
            finished_at=datetime.now(tz=UTC),
        )
        if updated is None:
            raise RuntimeError(f"backfill run {run.id} disappeared during update")
        await session.commit()
        logger.info(
            "backfill_symbol_succeeded",
            symbol=ticker,
            run_id=updated.id,
            fetched=total_fetched,
            inserted=total_inserted,
        )
        return updated
    finally:
        clear_ingestion_run_id()


async def backfill_symbols(
    session: AsyncSession,
    *,
    symbols: Sequence[str],
    timeframe: str,
    start: AwareDatetime,
    end: AwareDatetime,
    provider: MarketDataProvider,
    source: str,
) -> dict[str, IngestionRun | BaseException]:
    """Run backfill for each symbol, continuing after individual failures."""
    results: dict[str, IngestionRun | BaseException] = {}
    for symbol in symbols:
        try:
            results[symbol] = await backfill_symbol(
                session,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                provider=provider,
                source=source,
            )
        except Exception as exc:
            results[symbol] = exc
    succeeded = sum(1 for value in results.values() if isinstance(value, IngestionRun))
    logger.info(
        "backfill_symbols_done",
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )
    return results
