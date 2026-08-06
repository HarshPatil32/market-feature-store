"""Backfill orchestration: fetch → raw → normalize → validate → upsert."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.chunking import chunk_date_range
from backend.ingestion.pipeline import ingest_raw_data
from backend.providers.base import MarketDataProvider
from backend.services.ingestion_runs import trigger_backfill
from backend.storage.models import IngestionRun, RunStatus
from backend.storage.repository import IngestionRunRepository, MarketBarRepository
from backend.validation.validator import run_checks


def _bars_payload(bars: Sequence[Bar]) -> dict[str, Any]:
    return {"bars": [bar.model_dump(mode="json") for bar in bars]}


def _normalize_bars_payload(payload: dict[str, Any]) -> list[Bar]:
    return [Bar.model_validate(entry) for entry in payload["bars"]]


async def backfill_symbol(
    session: AsyncSession,
    *,
    symbol: str,
    timeframe: str,
    start: AwareDatetime,
    end: AwareDatetime,
    provider: MarketDataProvider,
    source: str,
) -> IngestionRun:
    """Run a full backfill for one symbol and date range."""
    if start > end:
        raise ValueError("start must be <= end")

    ticker = symbol.strip().upper()
    run = await trigger_backfill(session, ticker)
    if run.symbol_id is None:
        raise RuntimeError(f"backfill run {run.id} has no symbol_id")
    symbol_id = run.symbol_id

    run_repo = IngestionRunRepository(session)
    bar_repo = MarketBarRepository(session)
    chunks = chunk_date_range(start, end, timeframe)
    multi_chunk = len(chunks) > 1
    chunk_start, chunk_end = start, end

    await run_repo.update(
        run.id,
        status=RunStatus.running,
        started_at=datetime.now(tz=UTC),
    )

    total_fetched = 0
    total_inserted = 0

    try:
        for chunk_start, chunk_end in chunks:
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
            total_fetched += len(bars)
            total_inserted += len(bars)
    except Exception as exc:
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
            error_message=error_message,
            finished_at=datetime.now(tz=UTC),
        )
        if updated is None:
            raise RuntimeError(
                f"backfill run {run.id} disappeared during update"
            ) from exc
        raise

    # Any upsert failure aborts the run, so inserted always equals fetched on success.
    updated = await run_repo.update(
        run.id,
        status=RunStatus.succeeded,
        fetched=total_fetched,
        inserted=total_inserted,
        finished_at=datetime.now(tz=UTC),
    )
    if updated is None:
        raise RuntimeError(f"backfill run {run.id} disappeared during update")
    return updated
