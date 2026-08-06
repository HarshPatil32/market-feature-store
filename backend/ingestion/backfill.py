"""Backfill orchestration: fetch → raw → normalize → validate → upsert."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
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
    ticker = symbol.strip().upper()
    run = await trigger_backfill(session, ticker)
    if run.symbol_id is None:
        raise RuntimeError(f"backfill run {run.id} has no symbol_id")
    symbol_id = run.symbol_id

    run_repo = IngestionRunRepository(session)
    bar_repo = MarketBarRepository(session)

    await run_repo.update(
        run.id,
        status=RunStatus.running,
        started_at=datetime.now(tz=UTC),
    )

    try:
        bars = await provider.fetch_historical_bars(ticker, timeframe, start, end)
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
    except Exception as exc:
        updated = await run_repo.update(
            run.id,
            status=RunStatus.failed,
            error_message=str(exc),
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
        fetched=len(bars),
        inserted=len(bars),
        finished_at=datetime.now(tz=UTC),
    )
    if updated is None:
        raise RuntimeError(f"backfill run {run.id} disappeared during update")
    return updated
