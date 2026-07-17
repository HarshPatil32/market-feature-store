"""MarketDataProvider ABC and shared provider types."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from backend.storage.schemas import Ticker


class ProviderError(Exception):
    """Base exception for provider adapter failures."""


class Bar(BaseModel):
    """Minimal OHLCV bar returned by a provider."""

    model_config = ConfigDict(frozen=True)

    symbol: Ticker
    ts: AwareDatetime
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class MarketDataProvider(ABC):
    """Interface for fetching OHLCV bars from an external market data source."""

    @abstractmethod
    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        """Return bars for `symbol` within [start, end], ascending by ts.

        Args:
            start: Inclusive range start; must be timezone-aware.
            end: Inclusive range end; must be timezone-aware.

        Raises:
            ProviderError: if the provider cannot fulfill the request.
        """
        ...

    @abstractmethod
    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        """Return up to `limit` most recent bars for `symbol`, ascending by ts.

        Args:
            limit: Number of bars to return; must be >= 1.

        Raises:
            ProviderError: if the provider cannot fulfill the request.
        """
        ...
