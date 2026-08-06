"""Tests for backfill orchestration."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
from pydantic import AwareDatetime
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.backfill import backfill_symbol
from backend.ingestion.chunking import chunk_date_range
from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.fake import FakeProvider
from backend.services.symbols import SymbolNotFoundError, add_symbol
from backend.storage.models import CheckSeverity, IngestionRun, RunStatus
from backend.storage.repository import (
    DataQualityCheckRepository,
    MarketBarRepository,
    RawMarketDataRepository,
)
from backend.storage.schemas import SymbolCreate, Ticker


class _ZeroVolumeProvider(MarketDataProvider):
    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        return [
            Bar(
                symbol=symbol,
                ts=start,
                timeframe=timeframe,
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("103"),
                volume=Decimal("0"),
                source="fake",
            )
        ]

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return []


class _FailingProvider(MarketDataProvider):
    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        raise ProviderError("provider unavailable")

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return []


class _PartialFailProvider(MarketDataProvider):
    def __init__(self, inner: FakeProvider) -> None:
        self._inner = inner
        self._calls = 0

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        self._calls += 1
        if self._calls == 1:
            return await self._inner.fetch_historical_bars(
                symbol, timeframe, start, end
            )
        raise ProviderError("chunk failed")

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return []


@pytest.mark.asyncio
async def test_backfill_happy_path(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    result = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
        start=start,
        end=end,
    )
    raw_rows = await RawMarketDataRepository(db_session).list_by_symbol(
        symbol.id,
        run_id=result.id,
    )

    assert result.run_type == "backfill"
    assert result.status == RunStatus.succeeded
    assert result.fetched == 3
    assert result.inserted == 3
    assert result.error_message is None
    assert result.started_at is not None
    assert result.finished_at is not None
    assert len(bars) == 3
    assert len(raw_rows) == 1
    assert raw_rows[0].run_id == result.id
    assert raw_rows[0].source == "fake"


@pytest.mark.asyncio
async def test_backfill_is_idempotent(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)
    bar_repo = MarketBarRepository(db_session)

    await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )
    first_bars = await bar_repo.list_by_symbol(
        symbol.id,
        timeframe="1d",
        start=start,
        end=end,
    )
    first_ids = {bar.id for bar in first_bars}

    await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )
    second_bars = await bar_repo.list_by_symbol(
        symbol.id,
        timeframe="1d",
        start=start,
        end=end,
    )

    assert len(first_bars) == 3
    assert len(second_bars) == 3
    assert {bar.id for bar in second_bars} == first_ids


@pytest.mark.asyncio
async def test_backfill_validation_persists_quality_checks(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)

    result = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=_ZeroVolumeProvider(),
        source="fake",
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(result.id)
    zero_volume_checks = [c for c in checks if c.check_name == "zero_volume"]

    assert result.status == RunStatus.succeeded
    assert len(zero_volume_checks) == 1
    assert zero_volume_checks[0].severity == CheckSeverity.warning
    assert zero_volume_checks[0].symbol_id == symbol.id


@pytest.mark.asyncio
async def test_backfill_provider_failure_marks_run_failed(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    with pytest.raises(ProviderError, match="provider unavailable"):
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
            provider=_FailingProvider(),
            source="fake",
        )

    run = await db_session.scalar(
        select(IngestionRun)
        .where(IngestionRun.symbol_id == symbol.id)
        .order_by(IngestionRun.id.desc())
        .limit(1)
    )
    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert run is not None
    assert run.status == RunStatus.failed
    assert run.error_message == "provider unavailable"
    assert run.finished_at is not None
    assert bars == []


@pytest.mark.asyncio
async def test_backfill_pipeline_failure_marks_run_failed_without_upserting(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    with patch(
        "backend.ingestion.backfill.ingest_raw_data",
        side_effect=ValueError("validate boom"),
    ):
        with pytest.raises(ValueError, match="validate boom"):
            await backfill_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1d",
                start=start,
                end=end,
                provider=fake_provider,
                source="fake",
            )

    run = await db_session.scalar(
        select(IngestionRun)
        .where(IngestionRun.symbol_id == symbol.id)
        .order_by(IngestionRun.id.desc())
        .limit(1)
    )
    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert run is not None
    assert run.status == RunStatus.failed
    assert run.error_message == "validate boom"
    assert run.finished_at is not None
    assert bars == []


@pytest.mark.asyncio
async def test_backfill_multi_chunk_fetches_and_persists_all_bars(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 15, tzinfo=UTC)
    expected_bars = await fake_provider.fetch_historical_bars(
        "AAPL",
        "1m",
        start,
        end,
    )
    chunks = chunk_date_range(start, end, "1m")

    result = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1m",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1m",
        start=start,
        end=end,
    )
    raw_rows = await RawMarketDataRepository(db_session).list_by_symbol(
        symbol.id,
        run_id=result.id,
    )
    timestamps = [bar.timestamp for bar in bars]

    assert result.status == RunStatus.succeeded
    assert result.fetched == len(expected_bars)
    assert result.inserted == len(expected_bars)
    assert len(bars) == len(expected_bars)
    assert len(set(timestamps)) == len(timestamps)
    assert len(raw_rows) == len(chunks)


@pytest.mark.asyncio
async def test_backfill_partial_chunk_failure_keeps_prior_bars(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 15, tzinfo=UTC)
    chunks = chunk_date_range(start, end, "1m")
    first_chunk_bars = await fake_provider.fetch_historical_bars(
        "AAPL",
        "1m",
        chunks[0][0],
        chunks[0][1],
    )

    with pytest.raises(ProviderError, match="chunk failed"):
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1m",
            start=start,
            end=end,
            provider=_PartialFailProvider(fake_provider),
            source="fake",
        )

    run = await db_session.scalar(
        select(IngestionRun)
        .where(IngestionRun.symbol_id == symbol.id)
        .order_by(IngestionRun.id.desc())
        .limit(1)
    )
    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1m",
        start=chunks[0][0],
        end=chunks[0][1],
    )

    assert run is not None
    assert run.status == RunStatus.failed
    assert run.fetched == len(first_chunk_bars)
    assert run.inserted == len(first_chunk_bars)
    assert len(bars) == len(first_chunk_bars)
    assert run.error_message is not None
    assert "chunk" in run.error_message
    assert "chunk failed" in run.error_message


@pytest.mark.asyncio
async def test_backfill_invalid_range_raises_without_creating_run(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    before = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "backfill")
    )

    with pytest.raises(ValueError, match="start must be <= end"):
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            start=datetime(2024, 1, 3, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
            provider=fake_provider,
            source="fake",
        )

    after = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "backfill")
    )

    assert before == after


@pytest.mark.asyncio
async def test_backfill_unknown_symbol_raises_without_creating_run(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    before = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "backfill")
    )

    with pytest.raises(SymbolNotFoundError) as exc_info:
        await backfill_symbol(
            db_session,
            symbol="UNKNOWN",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 3, tzinfo=UTC),
            provider=fake_provider,
            source="fake",
        )

    after = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "backfill")
    )

    assert exc_info.value.symbol == "UNKNOWN"
    assert before == after
