"""Alpaca market data provider adapter."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import AwareDatetime, ValidationError

from backend.bar import Bar
from backend.config import get_settings
from backend.providers.base import MarketDataProvider, ProviderError
from backend.storage.schemas import Ticker

DATA_BASE_URL = "https://data.alpaca.markets"
BARS_PATH = "/v2/stocks/bars"
SOURCE = "alpaca"
DEFAULT_MAX_PAGES = 1000
MAX_RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_MAX_DELAY_SECONDS = 8.0
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

TIMEFRAME_TO_ALPACA: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
}


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _to_alpaca_timeframe(timeframe: str) -> str:
    mapped = TIMEFRAME_TO_ALPACA.get(timeframe)
    if mapped is None:
        supported = ", ".join(sorted(TIMEFRAME_TO_ALPACA))
        raise ProviderError(
            f"unsupported timeframe {timeframe!r}; supported: {supported}"
        )
    return mapped


def _format_datetime(value: AwareDatetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _parse_retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        seconds = float(header.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, RETRY_MAX_DELAY_SECONDS)


def _compute_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after
    delay: float = min(
        RETRY_BASE_DELAY_SECONDS * (2**attempt),
        RETRY_MAX_DELAY_SECONDS,
    )
    return delay + float(random.uniform(0, 0.1 * delay))


class AlpacaProvider(MarketDataProvider):
    """Fetches OHLCV bars from Alpaca's Market Data API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = DATA_BASE_URL,
        feed: str = "iex",
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._feed = feed
        self._max_pages = max_pages
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> AlpacaProvider:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        alpaca_timeframe = _to_alpaca_timeframe(timeframe)
        params: dict[str, str | int] = {
            "symbols": symbol,
            "timeframe": alpaca_timeframe,
            "start": _format_datetime(start),
            "end": _format_datetime(end),
            "sort": "asc",
            "feed": self._feed,
            "limit": 10000,
        }

        raw_bars: list[dict[str, Any]] = []
        for _ in range(self._max_pages):
            payload = await self._request_bars(params)
            raw_bars.extend(self._extract_symbol_bars(payload, symbol))
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            params["page_token"] = str(next_token)
        else:
            raise ProviderError(
                f"Alpaca pagination exceeded {self._max_pages} pages for {symbol!r}"
            )

        bars = self._parse_bars(raw_bars, symbol, timeframe)
        return [bar for bar in bars if start <= bar.ts <= end]

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        if limit < 1:
            raise ProviderError("limit must be >= 1")

        alpaca_timeframe = _to_alpaca_timeframe(timeframe)
        params: dict[str, str | int] = {
            "symbols": symbol,
            "timeframe": alpaca_timeframe,
            "sort": "desc",
            "feed": self._feed,
            "limit": limit,
        }

        payload = await self._request_bars(params)
        raw_bars = self._extract_symbol_bars(payload, symbol)
        bars = self._parse_bars(raw_bars, symbol, timeframe)
        return list(reversed(bars))

    async def _request_bars(self, params: Mapping[str, str | int]) -> dict[str, Any]:
        client = await self._get_client()

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await client.get(
                    BARS_PATH,
                    params=params,
                    headers={
                        "APCA-API-KEY-ID": self._api_key,
                        "APCA-API-SECRET-KEY": self._api_secret,
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt + 1 < MAX_RETRY_ATTEMPTS:
                    await asyncio.sleep(_compute_delay(attempt, None))
                    continue
                raise ProviderError("Alpaca request failed") from exc
            except httpx.HTTPError as exc:
                raise ProviderError("Alpaca request failed") from exc

            if _is_retryable_status(response.status_code):
                if attempt + 1 < MAX_RETRY_ATTEMPTS:
                    retry_after = (
                        _parse_retry_after(response)
                        if response.status_code == 429
                        else None
                    )
                    await asyncio.sleep(_compute_delay(attempt, retry_after))
                    continue

                message = self._error_message(response)
                raise ProviderError(
                    f"Alpaca returned HTTP {response.status_code}: {message}"
                )

            if response.status_code >= 400:
                message = self._error_message(response)
                raise ProviderError(
                    f"Alpaca returned HTTP {response.status_code}: {message}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("Alpaca returned invalid JSON") from exc

            if not isinstance(payload, dict):
                raise ProviderError("Alpaca response must be a JSON object")

            return payload

        raise AssertionError("_request_bars retry loop exhausted without result")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT,
            )
        return self._client

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text.strip() or "unknown error"

        if isinstance(body, dict):
            message = body.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        return response.text.strip() or "unknown error"

    @staticmethod
    def _extract_symbol_bars(
        payload: dict[str, Any], symbol: Ticker
    ) -> list[dict[str, Any]]:
        bars_by_symbol = payload.get("bars")
        if bars_by_symbol is None:
            return []
        if not isinstance(bars_by_symbol, dict):
            raise ProviderError("Alpaca response bars must be an object")

        symbol_bars = bars_by_symbol.get(symbol)
        if symbol_bars is None:
            return []
        if not isinstance(symbol_bars, list):
            raise ProviderError(f"Alpaca bars for {symbol!r} must be a list")

        extracted: list[dict[str, Any]] = []
        for item in symbol_bars:
            if not isinstance(item, dict):
                raise ProviderError(
                    f"Alpaca bar entry for {symbol!r} must be an object"
                )
            extracted.append(item)
        return extracted

    def _parse_bars(
        self,
        raw_bars: Sequence[Mapping[str, Any]],
        symbol: Ticker,
        timeframe: str,
    ) -> list[Bar]:
        parsed: list[Bar] = []
        for raw in raw_bars:
            try:
                parsed.append(self._parse_bar(raw, symbol, timeframe))
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("Alpaca bar payload is malformed") from exc
            except ValidationError as exc:
                raise ProviderError("Alpaca bar failed validation") from exc
        return parsed

    @staticmethod
    def _parse_bar(
        raw: Mapping[str, Any],
        symbol: Ticker,
        timeframe: str,
    ) -> Bar:
        timestamp = raw["t"]
        if not isinstance(timestamp, str):
            raise TypeError("bar timestamp must be a string")

        return Bar(
            symbol=symbol,
            ts=_parse_timestamp(timestamp),
            timeframe=timeframe,
            open=Decimal(str(raw["o"])),
            high=Decimal(str(raw["h"])),
            low=Decimal(str(raw["l"])),
            close=Decimal(str(raw["c"])),
            volume=Decimal(str(raw["v"])),
            source=SOURCE,
        )


def get_market_data_provider() -> AlpacaProvider:
    """Build an Alpaca provider using API credentials from settings/env."""
    settings = get_settings()
    return AlpacaProvider(
        api_key=settings.provider_api_key.get_secret_value(),
        api_secret=settings.provider_api_secret.get_secret_value(),
    )
