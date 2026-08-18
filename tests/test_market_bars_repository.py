"""Tests for MarketBarRepository CRUD and upsert operations."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TypedDict

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import MarketBar
from backend.storage.repository import MarketBarRepository, SymbolRepository


class BarKwargs(TypedDict):
    symbol_id: int
    timestamp: datetime
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def _bar_kwargs(
    *,
    symbol_id: int,
    timestamp: datetime,
    timeframe: str = "1d",
    open_: Decimal = Decimal("100.00"),
    high: Decimal = Decimal("105.00"),
    low: Decimal = Decimal("99.00"),
    close: Decimal = Decimal("103.50"),
    volume: Decimal = Decimal("1000000"),
) -> BarKwargs:
    return {
        "symbol_id": symbol_id,
        "timestamp": timestamp,
        "timeframe": timeframe,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


@pytest.mark.asyncio
async def test_upsert_inserts_new_row(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")
    ts = datetime(2024, 1, 2, tzinfo=UTC)

    created = await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts))
    fetched = await bar_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.symbol_id == symbol.id
    assert fetched.timestamp == ts
    assert fetched.timeframe == "1d"
    assert fetched.open == Decimal("100.00")
    assert fetched.high == Decimal("105.00")
    assert fetched.low == Decimal("99.00")
    assert fetched.close == Decimal("103.50")
    assert fetched.volume == Decimal("1000000")


@pytest.mark.asyncio
async def test_upsert_on_conflict_updates_existing_row(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")
    ts = datetime(2024, 1, 2, tzinfo=UTC)

    first = await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts))
    await db_session.execute(
        text("UPDATE market_bars SET updated_at = :old WHERE id = :id"),
        {"old": datetime(2020, 1, 1, tzinfo=UTC), "id": first.id},
    )
    await db_session.flush()
    second = await bar_repo.upsert(
        **_bar_kwargs(
            symbol_id=symbol.id,
            timestamp=ts,
            open_=Decimal("110.00"),
            high=Decimal("115.00"),
            low=Decimal("108.00"),
            close=Decimal("112.00"),
            volume=Decimal("2000000"),
        )
    )
    bars = await bar_repo.list_by_symbol(symbol.id, timeframe="1d")

    assert len(bars) == 1
    assert second.id == first.id
    assert bars[0].open == Decimal("110.00")
    assert bars[0].high == Decimal("115.00")
    assert bars[0].low == Decimal("108.00")
    assert bars[0].close == Decimal("112.00")
    assert bars[0].volume == Decimal("2000000")
    assert second.updated_at > datetime(2020, 1, 1, tzinfo=UTC)
    assert bars[0].updated_at > datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_duplicate_direct_insert_violates_unique_constraint(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")
    ts = datetime(2024, 1, 2, tzinfo=UTC)

    db_session.add(
        MarketBar(
            symbol_id=symbol.id,
            timestamp=ts,
            timeframe="1d",
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.50"),
            volume=Decimal("1000000"),
        )
    )
    await db_session.flush()

    db_session.add(
        MarketBar(
            symbol_id=symbol.id,
            timestamp=ts,
            timeframe="1d",
            open=Decimal("110.00"),
            high=Decimal("115.00"),
            low=Decimal("108.00"),
            close=Decimal("112.00"),
            volume=Decimal("2000000"),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_upsert_with_invalid_symbol_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    bar_repo = MarketBarRepository(db_session)

    with pytest.raises(IntegrityError):
        await bar_repo.upsert(
            **_bar_kwargs(
                symbol_id=999_999,
                timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            )
        )


@pytest.mark.asyncio
async def test_symbol_deletion_restricted_when_bars_exist(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")
    await bar_repo.upsert(
        **_bar_kwargs(
            symbol_id=symbol.id,
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        )
    )

    with pytest.raises(IntegrityError):
        await symbol_repo.delete(symbol.id)


@pytest.mark.asyncio
async def test_list_by_symbol_returns_bars_ordered_by_timestamp(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")
    other_symbol = await symbol_repo.create(symbol="MSFT")

    ts1 = datetime(2024, 1, 3, tzinfo=UTC)
    ts2 = datetime(2024, 1, 1, tzinfo=UTC)
    ts3 = datetime(2024, 1, 2, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts1))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts2))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts3))
    await bar_repo.upsert(
        **_bar_kwargs(
            symbol_id=other_symbol.id,
            timestamp=ts1,
        )
    )
    await bar_repo.upsert(
        **_bar_kwargs(
            symbol_id=symbol.id,
            timestamp=ts3,
            timeframe="1h",
        )
    )

    bars = await bar_repo.list_by_symbol(symbol.id, timeframe="1d")

    assert [bar.timestamp for bar in bars] == [ts2, ts3, ts1]


@pytest.mark.asyncio
async def test_list_by_symbol_filters_by_start_and_end(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    ts3 = datetime(2024, 1, 3, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts1))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts2))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts3))

    bars = await bar_repo.list_by_symbol(
        symbol.id,
        timeframe="1d",
        start=ts2,
        end=ts2,
    )

    assert [bar.timestamp for bar in bars] == [ts2]


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing_row(db_session: AsyncSession) -> None:
    bar_repo = MarketBarRepository(db_session)

    assert await bar_repo.get_by_id(999_999) is None


@pytest.mark.asyncio
async def test_get_timestamp_bounds_returns_none_when_no_bars(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    bounds = await bar_repo.get_timestamp_bounds(symbol.id)

    assert bounds == (None, None)


@pytest.mark.asyncio
async def test_get_timestamp_bounds_returns_min_max_across_timeframes(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 3, tzinfo=UTC)
    ts3 = datetime(2024, 1, 2, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts1))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts2))
    await bar_repo.upsert(
        **_bar_kwargs(symbol_id=symbol.id, timestamp=ts3, timeframe="1h")
    )

    bounds = await bar_repo.get_timestamp_bounds(symbol.id)

    assert bounds == (ts1, ts2)


@pytest.mark.asyncio
async def test_get_timestamp_bounds_scoped_by_symbol_id(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    aapl = await symbol_repo.create(symbol="AAPL")
    msft = await symbol_repo.create(symbol="MSFT")

    aapl_ts = datetime(2024, 1, 1, tzinfo=UTC)
    msft_ts = datetime(2024, 6, 1, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=aapl.id, timestamp=aapl_ts))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=msft.id, timestamp=msft_ts))

    assert await bar_repo.get_timestamp_bounds(aapl.id) == (aapl_ts, aapl_ts)
    assert await bar_repo.get_timestamp_bounds(msft.id) == (msft_ts, msft_ts)


@pytest.mark.asyncio
async def test_get_existing_timestamps_returns_matching_timestamps(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    ts3 = datetime(2024, 1, 3, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts1))
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts2))

    existing = await bar_repo.get_existing_timestamps(
        symbol.id,
        timeframe="1d",
        timestamps=[ts1, ts2, ts3],
    )

    assert existing == {ts1, ts2}


@pytest.mark.asyncio
async def test_get_existing_timestamps_scoped_by_symbol_and_timeframe(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    aapl = await symbol_repo.create(symbol="AAPL")
    msft = await symbol_repo.create(symbol="MSFT")

    ts = datetime(2024, 1, 2, tzinfo=UTC)

    await bar_repo.upsert(**_bar_kwargs(symbol_id=aapl.id, timestamp=ts))
    await bar_repo.upsert(
        **_bar_kwargs(symbol_id=aapl.id, timestamp=ts, timeframe="1h")
    )

    assert (
        await bar_repo.get_existing_timestamps(
            msft.id,
            timeframe="1d",
            timestamps=[ts],
        )
        == set()
    )

    assert await bar_repo.get_existing_timestamps(
        aapl.id,
        timeframe="1h",
        timestamps=[ts],
    ) == {ts}

    assert await bar_repo.get_existing_timestamps(
        aapl.id,
        timeframe="1d",
        timestamps=[ts],
    ) == {ts}


@pytest.mark.asyncio
async def test_get_existing_timestamps_returns_empty_when_none_match(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    existing = await bar_repo.get_existing_timestamps(
        symbol.id,
        timeframe="1d",
        timestamps=[datetime(2024, 1, 2, tzinfo=UTC)],
    )

    assert existing == set()


@pytest.mark.asyncio
async def test_get_existing_timestamps_returns_empty_for_empty_input(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    assert (
        await bar_repo.get_existing_timestamps(
            symbol.id,
            timeframe="1d",
            timestamps=[],
        )
        == set()
    )


@pytest.mark.asyncio
async def test_get_existing_timestamps_chunks_large_input(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    bar_repo = MarketBarRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    await bar_repo.upsert(**_bar_kwargs(symbol_id=symbol.id, timestamp=ts1))

    timestamps = [
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day) for day in range(1500)
    ]

    existing = await bar_repo.get_existing_timestamps(
        symbol.id,
        timeframe="1d",
        timestamps=timestamps,
    )

    assert existing == {ts1}
