"""Deterministic in-memory market data provider for tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pydantic import AwareDatetime

from backend.bar import Bar
from backend.providers.base import MarketDataProvider, ProviderError
from backend.storage.schemas import Ticker

SOURCE = "fake"

_TIMEFRAME_STEPS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "1wk": timedelta(weeks=1),
    "1mo": timedelta(days=30),
}


def _timeframe_step(timeframe: str) -> timedelta:
    step = _TIMEFRAME_STEPS.get(timeframe)
    if step is None:
        supported = ", ".join(sorted(_TIMEFRAME_STEPS))
        raise ProviderError(
            f"unsupported timeframe {timeframe!r}; supported: {supported}"
        )
    return step


def _deterministic_bar(symbol: Ticker, timeframe: str, ts: AwareDatetime) -> Bar:
    digest = hashlib.sha256(f"{symbol}:{timeframe}:{ts.isoformat()}".encode()).digest()
    base = Decimal(int.from_bytes(digest[:4], "big") % 900_000 + 100_000) / Decimal(
        10_000
    )
    spread_units = int.from_bytes(digest[4:6], "big") % 500 + 10
    spread = Decimal(spread_units) / Decimal(1_000)
    close_units = int.from_bytes(digest[6:8], "big") % (spread_units + 1)
    close_offset = (Decimal(close_units) / Decimal(1_000)) - (spread / 2)
    open_price = base
    close = base + close_offset
    upper_wick = Decimal(int.from_bytes(digest[8:10], "big") % 100 + 1) / Decimal(1_000)
    lower_wick = Decimal(int.from_bytes(digest[10:12], "big") % 100 + 1) / Decimal(
        1_000
    )
    high = max(open_price, close) + upper_wick
    low = min(open_price, close) - lower_wick
    volume = Decimal(int.from_bytes(digest[12:16], "big") % 100_000_000)

    return Bar(
        symbol=symbol,
        ts=ts,
        timeframe=timeframe,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source=SOURCE,
    )


def _iter_timestamps(
    start: AwareDatetime,
    end: AwareDatetime,
    step: timedelta,
) -> list[AwareDatetime]:
    timestamps: list[AwareDatetime] = []
    current = start
    while current <= end:
        timestamps.append(current)
        current = current + step
    return timestamps


def _bars_for_range(
    symbol: Ticker,
    timeframe: str,
    start: AwareDatetime,
    end: AwareDatetime,
    step: timedelta,
) -> list[Bar]:
    return [
        _deterministic_bar(symbol, timeframe, ts)
        for ts in _iter_timestamps(start, end, step)
    ]


class FakeProvider(MarketDataProvider):
    """Generates deterministic OHLCV bars without network or database access."""

    def __init__(self, *, now: AwareDatetime | None = None) -> None:
        self._now = now

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        if start > end:
            raise ProviderError("start must be <= end")

        step = _timeframe_step(timeframe)
        return _bars_for_range(symbol, timeframe, start, end, step)

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        if limit < 1:
            raise ProviderError("limit must be >= 1")

        step = _timeframe_step(timeframe)
        end = self._now if self._now is not None else datetime.now(tz=UTC)
        start = end - step * (limit - 1)
        return _bars_for_range(symbol, timeframe, start, end, step)
