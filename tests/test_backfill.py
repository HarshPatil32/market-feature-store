"""Tests for backfill orchestration."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from pydantic import AwareDatetime
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

import backend.ingestion.chunking as chunking_module
from backend.bar import Bar
from backend.ingestion.backfill import backfill_symbol
from backend.ingestion.chunking import chunk_date_range
from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.fake import FakeProvider
from backend.services.ingestion_runs import trigger_incremental
from backend.services.symbols import SymbolNotFoundError, add_symbol
from backend.storage.models import (
    CheckSeverity,
    DataQualityCheck,
    IngestionRun,
    MarketBar,
    RawMarketData,
    RunStatus,
    Symbol,
)
from backend.storage.repository import (
    DataQualityCheckRepository,
    IngestionRunRepository,
    MarketBarRepository,
    RawMarketDataRepository,
    SymbolRepository,
)
from backend.storage.schemas import SymbolCreate, Ticker

_FAST_MULTI_CHUNK_START = datetime(2024, 1, 1, tzinfo=UTC)
_FAST_MULTI_CHUNK_END = datetime(2024, 1, 3, tzinfo=UTC)
_FAST_BARS_PER_CHUNK = 3


@contextmanager
def _fast_multi_chunk_window() -> Iterator[None]:
    """Use a 1-day chunk window so short ranges still produce multiple chunks."""
    patched = {**chunking_module._TIMEFRAME_WINDOWS, "1m": timedelta(days=1)}
    with patch.object(chunking_module, "_TIMEFRAME_WINDOWS", patched):
        yield


class _LimitedBarsProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider, *, max_bars: int) -> None:
        self._inner = inner
        self._max_bars = max_bars

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        bars = await self._inner.fetch_historical_bars(symbol, timeframe, start, end)
        return bars[: self._max_bars]

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return await self._inner.fetch_latest_bars(symbol, timeframe, limit)


def _fast_multi_chunk_provider(fake_provider: FakeProvider) -> _LimitedBarsProvider:
    return _LimitedBarsProvider(fake_provider, max_bars=_FAST_BARS_PER_CHUNK)


async def _cleanup_committed_symbol(engine: AsyncEngine, symbol_id: int) -> None:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await session.execute(delete(MarketBar).where(MarketBar.symbol_id == symbol_id))
        await session.execute(
            delete(DataQualityCheck).where(DataQualityCheck.symbol_id == symbol_id)
        )
        await session.execute(
            delete(RawMarketData).where(RawMarketData.symbol_id == symbol_id)
        )
        await session.execute(
            delete(IngestionRun).where(IngestionRun.symbol_id == symbol_id)
        )
        await session.execute(delete(Symbol).where(Symbol.id == symbol_id))
        await session.commit()


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
    def __init__(self, inner: MarketDataProvider) -> None:
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


class _DifferentOpenProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        bars = await self._inner.fetch_historical_bars(symbol, timeframe, start, end)
        return [
            Bar(
                symbol=bar.symbol,
                ts=bar.ts,
                timeframe=bar.timeframe,
                open=Decimal("999.00"),
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                source=bar.source,
            )
            for bar in bars
        ]

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return []


class _CountingProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self.calls = 0

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        self.calls += 1
        return await self._inner.fetch_historical_bars(symbol, timeframe, start, end)

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
    assert result.failed == 0
    assert result.error_message is None
    assert result.started_at is not None
    assert result.finished_at is not None
    assert len(bars) == 3
    assert len(raw_rows) == 1
    assert raw_rows[0].run_id == result.id
    assert raw_rows[0].source == "fake"

    refreshed = await SymbolRepository(db_session).get_by_id(symbol.id)
    assert refreshed is not None
    assert refreshed.coverage_start == start
    assert refreshed.coverage_end == end
    assert refreshed.last_ingested_at is not None


@pytest.mark.asyncio
async def test_backfill_does_not_overwrite_existing_bars(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)
    middle = datetime(2024, 1, 2, tzinfo=UTC)
    bar_repo = MarketBarRepository(db_session)

    bars = await fake_provider.fetch_historical_bars(Ticker("AAPL"), "1d", start, end)
    middle_bar = next(bar for bar in bars if bar.ts == middle)
    await bar_repo.upsert(
        symbol_id=symbol.id,
        timestamp=middle_bar.ts,
        timeframe=middle_bar.timeframe,
        open=middle_bar.open,
        high=middle_bar.high,
        low=middle_bar.low,
        close=middle_bar.close,
        volume=middle_bar.volume,
    )
    await db_session.commit()

    result = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=_DifferentOpenProvider(fake_provider),
        source="fake",
    )
    stored = {
        bar.timestamp: bar.open
        for bar in await bar_repo.list_by_symbol(symbol.id, timeframe="1d")
    }

    assert stored[middle] == middle_bar.open
    assert stored[start] == Decimal("999.00")
    assert stored[end] == Decimal("999.00")
    assert result.fetched == 3
    assert result.inserted == 2


@pytest.mark.asyncio
async def test_backfill_skips_coverage_sync_when_no_bars_inserted(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)
    bar_repo = MarketBarRepository(db_session)
    symbol_repo = SymbolRepository(db_session)
    last_ingested_at = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    bars = await fake_provider.fetch_historical_bars(Ticker("AAPL"), "1d", start, end)
    for bar in bars:
        await bar_repo.upsert(
            symbol_id=symbol.id,
            timestamp=bar.ts,
            timeframe=bar.timeframe,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
    await symbol_repo.update_coverage(
        symbol.id,
        coverage_start=start,
        coverage_end=end,
        last_ingested_at=last_ingested_at,
    )
    await db_session.commit()

    async def _no_coverage(*_args: object, **_kwargs: object) -> tuple[None, None]:
        return (None, None)

    with patch.object(
        MarketBarRepository,
        "get_timeframe_coverage",
        side_effect=_no_coverage,
    ):
        result = await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
            provider=fake_provider,
            source="fake",
        )

    refreshed = await symbol_repo.get_by_id(symbol.id)
    assert refreshed is not None
    assert result.fetched == 3
    assert result.inserted == 0
    assert refreshed.last_ingested_at == last_ingested_at


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
async def test_backfill_skip_covered_chunks_makes_no_provider_calls(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )

    counting_provider = _CountingProvider(fake_provider)
    await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=counting_provider,
        source="fake",
    )

    assert counting_provider.calls == 0


@pytest.mark.asyncio
async def test_backfill_skip_covered_chunks_preserves_bar_count(
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
    first_count = len(
        await bar_repo.list_by_symbol(
            symbol.id,
            timeframe="1d",
            start=start,
            end=end,
        )
    )

    await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )
    second_count = len(
        await bar_repo.list_by_symbol(
            symbol.id,
            timeframe="1d",
            start=start,
            end=end,
        )
    )

    assert first_count == 3
    assert second_count == first_count


@pytest.mark.asyncio
async def test_backfill_extended_range_fetches_only_new_chunks(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = _FAST_MULTI_CHUNK_START
    mid = datetime(2024, 1, 2, tzinfo=UTC)
    end = datetime(2024, 1, 4, tzinfo=UTC)

    with _fast_multi_chunk_window():
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1m",
            start=start,
            end=mid,
            provider=fake_provider,
            source="fake",
        )

        bar_repo = MarketBarRepository(db_session)
        existing_start, existing_end = await bar_repo.get_timeframe_coverage(
            symbol.id,
            timeframe="1m",
        )
        assert existing_start is not None
        assert existing_end is not None

        extended_chunks = chunk_date_range(start, end, "1m")
        expected_calls = sum(
            1
            for chunk_start, chunk_end in extended_chunks
            if not (chunk_start >= existing_start and chunk_end <= existing_end)
        )

        counting_provider = _CountingProvider(fake_provider)
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1m",
            start=start,
            end=end,
            provider=counting_provider,
            source="fake",
        )

        assert expected_calls > 0
        assert counting_provider.calls == expected_calls
        assert counting_provider.calls < len(extended_chunks)


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
    assert run.failed == 1
    assert run.error_message == "provider unavailable"
    assert run.finished_at is not None
    assert bars == []

    refreshed = await SymbolRepository(db_session).get_by_id(symbol.id)
    assert refreshed is not None
    assert refreshed.coverage_start is None
    assert refreshed.coverage_end is None
    assert refreshed.last_ingested_at is None


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
    assert run.failed == 1
    assert run.error_message == "validate boom"
    assert run.finished_at is not None
    assert bars == []


@pytest.mark.asyncio
async def test_backfill_multi_chunk_fetches_and_persists_all_bars(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = _FAST_MULTI_CHUNK_START
    end = _FAST_MULTI_CHUNK_END
    provider = _fast_multi_chunk_provider(fake_provider)

    with _fast_multi_chunk_window():
        chunks = chunk_date_range(start, end, "1m")
        expected_count = len(chunks) * _FAST_BARS_PER_CHUNK

        result = await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1m",
            start=start,
            end=end,
            provider=provider,
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
        assert result.fetched == expected_count
        assert result.inserted == expected_count
        assert result.failed == 0
        assert len(bars) == expected_count
        assert len(set(timestamps)) == len(timestamps)
        assert len(raw_rows) == len(chunks)


@pytest.mark.asyncio
async def test_backfill_partial_chunk_failure_keeps_prior_bars(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = _FAST_MULTI_CHUNK_START
    end = _FAST_MULTI_CHUNK_END
    provider = _PartialFailProvider(_fast_multi_chunk_provider(fake_provider))

    with _fast_multi_chunk_window():
        chunks = chunk_date_range(start, end, "1m")

        with pytest.raises(ProviderError, match="chunk failed"):
            await backfill_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1m",
                start=start,
                end=end,
                provider=provider,
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
        assert run.fetched == _FAST_BARS_PER_CHUNK
        assert run.inserted == _FAST_BARS_PER_CHUNK
        assert run.failed == 1
        assert len(bars) == _FAST_BARS_PER_CHUNK
        assert run.error_message is not None
        assert "chunk" in run.error_message
        assert "chunk failed" in run.error_message

        refreshed = await SymbolRepository(db_session).get_by_id(symbol.id)
        assert refreshed is not None
        assert refreshed.last_ingested_at is not None
        bar_timestamps = {bar.timestamp for bar in bars}
        assert refreshed.coverage_start == min(bar_timestamps)
        assert refreshed.coverage_end == max(bar_timestamps)


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


@pytest.mark.asyncio
async def test_backfill_failure_survives_caller_rollback(
    engine: AsyncEngine,
    fake_provider: FakeProvider,
) -> None:
    ticker = "DURBL"
    start = _FAST_MULTI_CHUNK_START
    end = _FAST_MULTI_CHUNK_END
    provider = _PartialFailProvider(_fast_multi_chunk_provider(fake_provider))
    symbol_id: int | None = None

    try:
        with _fast_multi_chunk_window():
            chunks = chunk_date_range(start, end, "1m")

            async with AsyncSession(engine, expire_on_commit=False) as session:
                symbol = await add_symbol(session, SymbolCreate(symbol=ticker))
                symbol_id = symbol.id
                await session.commit()

                with pytest.raises(ProviderError, match="chunk failed"):
                    await backfill_symbol(
                        session,
                        symbol=ticker,
                        timeframe="1m",
                        start=start,
                        end=end,
                        provider=provider,
                        source="fake",
                    )
                await session.rollback()

            async with AsyncSession(engine, expire_on_commit=False) as verify_session:
                run = await verify_session.scalar(
                    select(IngestionRun)
                    .where(IngestionRun.symbol_id == symbol_id)
                    .order_by(IngestionRun.id.desc())
                    .limit(1)
                )
                bars = await MarketBarRepository(verify_session).list_by_symbol(
                    symbol_id,
                    timeframe="1m",
                    start=chunks[0][0],
                    end=chunks[0][1],
                )

                assert run is not None
                assert run.status == RunStatus.failed
                assert run.fetched == _FAST_BARS_PER_CHUNK
                assert run.failed == 1
                assert len(bars) == _FAST_BARS_PER_CHUNK
    finally:
        if symbol_id is not None:
            await _cleanup_committed_symbol(engine, symbol_id)


@pytest.mark.asyncio
async def test_backfill_resume_skips_completed_chunks(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = _FAST_MULTI_CHUNK_START
    end = _FAST_MULTI_CHUNK_END
    limited = _fast_multi_chunk_provider(fake_provider)
    partial_fail = _PartialFailProvider(limited)

    with _fast_multi_chunk_window():
        chunks = chunk_date_range(start, end, "1m")
        expected_count = len(chunks) * _FAST_BARS_PER_CHUNK

        with pytest.raises(ProviderError, match="chunk failed"):
            await backfill_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1m",
                start=start,
                end=end,
                provider=partial_fail,
                source="fake",
            )

        failed_run = await db_session.scalar(
            select(IngestionRun)
            .where(IngestionRun.symbol_id == symbol.id)
            .order_by(IngestionRun.id.desc())
            .limit(1)
        )
        assert failed_run is not None
        assert failed_run.status == RunStatus.failed
        assert failed_run.failed == 1

        raw_before = await RawMarketDataRepository(db_session).list_by_symbol(
            symbol.id,
            run_id=failed_run.id,
        )
        counting_provider = _CountingProvider(limited)

        result = await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1m",
            start=start,
            end=end,
            provider=counting_provider,
            source="fake",
            resume_run_id=failed_run.id,
        )
        raw_after = await RawMarketDataRepository(db_session).list_by_symbol(
            symbol.id,
            run_id=failed_run.id,
        )
        bars = await MarketBarRepository(db_session).list_by_symbol(
            symbol.id,
            timeframe="1m",
            start=start,
            end=end,
        )

        assert result.id == failed_run.id
        assert result.status == RunStatus.succeeded
        assert result.fetched == expected_count
        assert result.failed == 1
        assert len(bars) == expected_count
        assert len(raw_after) == len(chunks)
        assert len(raw_before) == 1
        assert counting_provider.calls == len(chunks) - 1


@pytest.mark.asyncio
async def test_backfill_resume_failure_increments_failed_count(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = _FAST_MULTI_CHUNK_START
    end = _FAST_MULTI_CHUNK_END
    limited = _fast_multi_chunk_provider(fake_provider)

    with _fast_multi_chunk_window():
        with pytest.raises(ProviderError, match="chunk failed"):
            await backfill_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1m",
                start=start,
                end=end,
                provider=_PartialFailProvider(limited),
                source="fake",
            )

        failed_run = await db_session.scalar(
            select(IngestionRun)
            .where(IngestionRun.symbol_id == symbol.id)
            .order_by(IngestionRun.id.desc())
            .limit(1)
        )
        assert failed_run is not None
        assert failed_run.status == RunStatus.failed
        assert failed_run.failed == 1

        with pytest.raises(ProviderError, match="provider unavailable"):
            await backfill_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1m",
                start=start,
                end=end,
                provider=_FailingProvider(),
                source="fake",
                resume_run_id=failed_run.id,
            )

        run = await db_session.scalar(
            select(IngestionRun).where(IngestionRun.id == failed_run.id)
        )
        assert run is not None
        assert run.status == RunStatus.failed
        assert run.failed == 2


@pytest.mark.asyncio
async def test_backfill_resume_rejects_succeeded_run(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    succeeded_run = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
    )
    assert succeeded_run.status == RunStatus.succeeded

    with pytest.raises(ValueError, match="already succeeded"):
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
            provider=fake_provider,
            source="fake",
            resume_run_id=succeeded_run.id,
        )


@pytest.mark.asyncio
async def test_backfill_resume_accepts_running_run(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)
    run_repo = IngestionRunRepository(db_session)

    running_run = await run_repo.create(run_type="backfill", symbol_id=symbol.id)
    await run_repo.update(
        running_run.id,
        status=RunStatus.running,
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    await db_session.commit()

    result = await backfill_symbol(
        db_session,
        symbol="AAPL",
        timeframe="1d",
        start=start,
        end=end,
        provider=fake_provider,
        source="fake",
        resume_run_id=running_run.id,
    )

    assert result.id == running_run.id
    assert result.status == RunStatus.succeeded
    assert result.fetched == 3
    assert result.inserted == 3


@pytest.mark.asyncio
async def test_backfill_resume_rejects_non_backfill_run(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    incremental_run = await trigger_incremental(db_session, "AAPL")

    with pytest.raises(ValueError, match="not a backfill run"):
        await backfill_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
            provider=fake_provider,
            source="fake",
            resume_run_id=incremental_run.id,
        )
