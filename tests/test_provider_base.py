"""Tests for MarketDataProvider ABC and shared provider types."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from backend.providers.base import Bar, MarketDataProvider
from backend.storage.schemas import Ticker


class _FakeProvider(MarketDataProvider):
    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        return [
            Bar(
                symbol=symbol,
                ts=start,
                timeframe=timeframe,
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("103"),
                volume=Decimal("1000"),
                source="alpaca",
            )
        ]

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> list[Bar]:
        return []


class _PartialProvider(MarketDataProvider):
    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        return []


def test_market_data_provider_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        MarketDataProvider()  # type: ignore[abstract]


def test_partial_implementation_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        _PartialProvider()  # type: ignore[abstract]


async def test_fake_provider_satisfies_interface() -> None:
    provider = _FakeProvider()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    historical = await provider.fetch_historical_bars("AAPL", "1d", start, end)
    latest = await provider.fetch_latest_bars("AAPL", "1d", limit=5)

    assert isinstance(provider, MarketDataProvider)
    assert len(historical) == 1
    assert historical[0].symbol == "AAPL"
    assert historical[0].source == "alpaca"
    assert latest == []
