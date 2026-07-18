"""Tests for the deterministic fake market data provider."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.fake import FakeProvider

FIXED_NOW = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
START = datetime(2024, 6, 10, tzinfo=UTC)
END = datetime(2024, 6, 14, tzinfo=UTC)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider(now=FIXED_NOW)


async def test_fake_provider_satisfies_interface(fake_provider: FakeProvider) -> None:
    assert isinstance(fake_provider, MarketDataProvider)


async def test_fetch_historical_bars_returns_ascending_bars_in_range(
    fake_provider: FakeProvider,
) -> None:
    bars = await fake_provider.fetch_historical_bars("AAPL", "1d", START, END)

    assert len(bars) == 5
    assert bars[0].ts == START
    assert all(START <= bar.ts <= END for bar in bars)
    assert [bar.ts for bar in bars] == sorted(bar.ts for bar in bars)
    assert all(bar.symbol == "AAPL" for bar in bars)
    assert all(bar.timeframe == "1d" for bar in bars)
    assert all(bar.source == "fake" for bar in bars)
    assert all(
        bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high
        for bar in bars
    )


async def test_fetch_historical_bars_is_deterministic(
    fake_provider: FakeProvider,
) -> None:
    first = await fake_provider.fetch_historical_bars("AAPL", "1d", START, END)
    second = await FakeProvider(now=FIXED_NOW).fetch_historical_bars(
        "AAPL", "1d", START, END
    )

    assert first == second


async def test_fetch_historical_bars_varies_by_symbol_and_timestamp(
    fake_provider: FakeProvider,
) -> None:
    aapl_bars = await fake_provider.fetch_historical_bars("AAPL", "1d", START, END)
    msft_bars = await fake_provider.fetch_historical_bars("MSFT", "1d", START, END)

    assert aapl_bars[0] != msft_bars[0]

    single_day_end = START + timedelta(days=1)
    day_one = await fake_provider.fetch_historical_bars(
        "AAPL", "1d", START, single_day_end
    )
    day_two_start = START + timedelta(days=1)
    day_two_end = START + timedelta(days=2)
    day_two = await fake_provider.fetch_historical_bars(
        "AAPL", "1d", day_two_start, day_two_end
    )

    assert day_one[0] != day_two[0]


async def test_fetch_historical_bars_rejects_unsupported_timeframe(
    fake_provider: FakeProvider,
) -> None:
    with pytest.raises(ProviderError, match="unsupported timeframe"):
        await fake_provider.fetch_historical_bars("AAPL", "2h", START, END)


async def test_fetch_historical_bars_rejects_start_after_end(
    fake_provider: FakeProvider,
) -> None:
    with pytest.raises(ProviderError, match="start must be <= end"):
        await fake_provider.fetch_historical_bars("AAPL", "1d", END, START)


async def test_fetch_latest_bars_returns_limit_bars_ending_at_now(
    fake_provider: FakeProvider,
) -> None:
    bars = await fake_provider.fetch_latest_bars("AAPL", "1d", limit=3)

    assert len(bars) == 3
    assert bars[-1].ts == FIXED_NOW
    assert [bar.ts for bar in bars] == sorted(bar.ts for bar in bars)


async def test_fetch_latest_bars_rejects_invalid_limit(
    fake_provider: FakeProvider,
) -> None:
    with pytest.raises(ProviderError, match="limit must be >= 1"):
        await fake_provider.fetch_latest_bars("AAPL", "1d", limit=0)


async def test_fetch_latest_bars_rejects_unsupported_timeframe(
    fake_provider: FakeProvider,
) -> None:
    with pytest.raises(ProviderError, match="unsupported timeframe"):
        await fake_provider.fetch_latest_bars("AAPL", "2h", limit=1)
