"""Tests for the Alpaca provider adapter."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.provider_adapter import (
    MAX_RETRY_ATTEMPTS,
    RETRY_MAX_DELAY_SECONDS,
    AlpacaProvider,
    _compute_delay,
    _parse_retry_after,
)

API_KEY = "test-key-id"
API_SECRET = "test-secret"


def _sample_bar_payload(
    *,
    timestamp: str = "2024-01-02T05:00:00Z",
    next_page_token: str | None = None,
) -> dict[str, object]:
    return {
        "bars": {
            "AAPL": [
                {
                    "t": timestamp,
                    "o": 187.15,
                    "h": 188.44,
                    "l": 186.89,
                    "c": 188.01,
                    "v": 45678900,
                    "n": 12345,
                    "vw": 187.5,
                }
            ]
        },
        "next_page_token": next_page_token,
    }


def _provider(transport: httpx.BaseTransport) -> AlpacaProvider:
    client = httpx.AsyncClient(
        transport=transport,
        base_url="https://data.alpaca.markets",
    )
    return AlpacaProvider(API_KEY, API_SECRET, client=client)


@pytest.fixture
def mock_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sleep = AsyncMock()
    monkeypatch.setattr(
        "backend.providers.provider_adapter.asyncio.sleep",
        sleep,
    )
    return sleep


async def test_fetch_historical_bars_returns_standard_bars() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/stocks/bars"
        assert request.headers["APCA-API-KEY-ID"] == API_KEY
        assert request.headers["APCA-API-SECRET-KEY"] == API_SECRET
        assert request.url.params["symbols"] == "AAPL"
        assert request.url.params["timeframe"] == "1Day"
        assert request.url.params["sort"] == "asc"
        assert request.url.params["feed"] == "iex"
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    bars = await provider.fetch_historical_bars("AAPL", "1d", start, end)

    assert isinstance(provider, MarketDataProvider)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "AAPL"
    assert bar.timeframe == "1d"
    assert bar.source == "alpaca"
    assert bar.open == Decimal("187.15")
    assert bar.high == Decimal("188.44")
    assert bar.low == Decimal("186.89")
    assert bar.close == Decimal("188.01")
    assert bar.volume == Decimal("45678900")
    assert bar.ts == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


async def test_fetch_historical_bars_paginates() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json=_sample_bar_payload(
                    timestamp="2024-01-02T05:00:00Z",
                    next_page_token="page-2",
                ),
            )
        assert request.url.params.get("page_token") == "page-2"
        return httpx.Response(
            200,
            json=_sample_bar_payload(timestamp="2024-01-03T05:00:00Z"),
        )

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    bars = await provider.fetch_historical_bars("AAPL", "1d", start, end)

    assert calls == 2
    assert len(bars) == 2
    assert bars[0].ts < bars[1].ts


async def test_fetch_latest_bars_returns_ascending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["sort"] == "desc"
        assert request.url.params["limit"] == "2"
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2024-01-03T05:00:00Z",
                            "o": 190.0,
                            "h": 191.0,
                            "l": 189.0,
                            "c": 190.5,
                            "v": 1000,
                            "n": 10,
                            "vw": 190.2,
                        },
                        {
                            "t": "2024-01-02T05:00:00Z",
                            "o": 187.15,
                            "h": 188.44,
                            "l": 186.89,
                            "c": 188.01,
                            "v": 45678900,
                            "n": 12345,
                            "vw": 187.5,
                        },
                    ]
                },
                "next_page_token": None,
            },
        )

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d", limit=2)

    assert len(bars) == 2
    assert bars[0].ts == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert bars[1].ts == datetime(2024, 1, 3, 5, 0, tzinfo=UTC)


async def test_unsupported_timeframe_raises_provider_error() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    with pytest.raises(ProviderError, match="unsupported timeframe"):
        await provider.fetch_historical_bars("AAPL", "2d", start, end)


async def test_fetch_latest_bars_rejects_invalid_limit() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json={})))

    with pytest.raises(ProviderError, match="limit must be >= 1"):
        await provider.fetch_latest_bars("AAPL", "1d", limit=0)


async def test_http_error_raises_provider_error(
    mock_sleep: AsyncMock,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"message": "internal error"})

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    with pytest.raises(ProviderError, match="HTTP 500"):
        await provider.fetch_historical_bars("AAPL", "1d", start, end)

    assert calls == MAX_RETRY_ATTEMPTS
    assert mock_sleep.await_count == MAX_RETRY_ATTEMPTS - 1


async def test_unauthorized_raises_provider_error(
    mock_sleep: AsyncMock,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"message": "invalid credentials"})

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(ProviderError, match="HTTP 401"):
        await provider.fetch_latest_bars("AAPL", "1d")

    assert calls == 1
    mock_sleep.assert_not_awaited()


async def test_malformed_payload_raises_provider_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"bars": {"AAPL": [{"t": "2024-01-02T05:00:00Z", "o": 1.0}]}},
        )

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    with pytest.raises(ProviderError, match="malformed"):
        await provider.fetch_historical_bars("AAPL", "1d", start, end)


async def test_unknown_symbol_returns_empty_list() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    bars = await provider.fetch_historical_bars("ZZZZ", "1d", start, end)

    assert bars == []


async def test_connection_error_raises_provider_error(
    mock_sleep: AsyncMock,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(ProviderError, match="Alpaca request failed"):
        await provider.fetch_latest_bars("AAPL", "1d")

    assert calls == MAX_RETRY_ATTEMPTS
    assert mock_sleep.await_count == MAX_RETRY_ATTEMPTS - 1


async def test_async_context_manager_closes_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[httpx.AsyncClient] = []
    original_client = httpx.AsyncClient

    def capturing_client(**kwargs: object) -> httpx.AsyncClient:
        client = original_client(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", capturing_client)

    async with AlpacaProvider(API_KEY, API_SECRET) as provider:
        await provider._get_client()
        assert len(created_clients) == 1
        assert provider._client is created_clients[0]

    assert provider._client is None


async def test_pagination_exceeds_max_pages_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_sample_bar_payload(next_page_token="always-more"),
        )

    provider = AlpacaProvider(
        API_KEY,
        API_SECRET,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://data.alpaca.markets",
        ),
        max_pages=2,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    with pytest.raises(ProviderError, match="exceeded 2 pages"):
        await provider.fetch_historical_bars("AAPL", "1d", start, end)


async def test_non_dict_bar_entry_raises_provider_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"bars": {"AAPL": ["not-a-dict"]}, "next_page_token": None},
        )

    provider = _provider(httpx.MockTransport(handler))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    with pytest.raises(ProviderError, match="must be an object"):
        await provider.fetch_historical_bars("AAPL", "1d", start, end)


async def test_429_retries_then_succeeds(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"message": "rate limited"})
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert calls == 3
    assert len(bars) == 1
    assert mock_sleep.await_count == 2


async def test_429_respects_retry_after_header(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                json={"message": "rate limited"},
                headers={"Retry-After": "2"},
            )
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert len(bars) == 1
    mock_sleep.assert_awaited_once_with(2.0)


async def test_429_caps_retry_after_header(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                json={"message": "rate limited"},
                headers={"Retry-After": "999"},
            )
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    await provider.fetch_latest_bars("AAPL", "1d")

    mock_sleep.assert_awaited_once_with(RETRY_MAX_DELAY_SECONDS)


async def test_5xx_retries_then_succeeds(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"message": "unavailable"})
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert len(bars) == 1
    assert calls == 2
    mock_sleep.assert_awaited_once()


async def test_timeout_retries_then_succeeds(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("read timed out")
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert len(bars) == 1
    assert calls == 2
    mock_sleep.assert_awaited_once()


async def test_connect_timeout_retries_then_succeeds(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("connect timed out")
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert len(bars) == 1
    assert calls == 2
    mock_sleep.assert_awaited_once()


async def test_429_with_invalid_retry_after_uses_exponential_backoff(
    mock_sleep: AsyncMock,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                json={"message": "rate limited"},
                headers={"Retry-After": "bad"},
            )
        return httpx.Response(200, json=_sample_bar_payload())

    provider = _provider(httpx.MockTransport(handler))

    bars = await provider.fetch_latest_bars("AAPL", "1d")

    assert len(bars) == 1
    mock_sleep.assert_awaited_once()
    await_args = mock_sleep.await_args
    assert await_args is not None
    delay = await_args.args[0]
    assert delay != 2.0
    assert delay <= RETRY_MAX_DELAY_SECONDS * 1.1


async def test_non_retryable_4xx_fails_immediately(mock_sleep: AsyncMock) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"message": "not found"})

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(ProviderError, match="HTTP 404"):
        await provider.fetch_latest_bars("AAPL", "1d")

    assert calls == 1
    mock_sleep.assert_not_awaited()


def test_compute_delay_uses_retry_after() -> None:
    assert _compute_delay(0, 2.0) == 2.0


def test_compute_delay_caps_exponential_backoff() -> None:
    delay = _compute_delay(10, None)
    assert delay <= RETRY_MAX_DELAY_SECONDS * 1.1


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        ("2", 2.0),
        ("999", RETRY_MAX_DELAY_SECONDS),
        ("bad", None),
        ("-1", None),
    ],
)
def test_parse_retry_after(retry_after: str, expected: float | None) -> None:
    response = httpx.Response(429, headers={"Retry-After": retry_after})

    assert _parse_retry_after(response) == expected


def test_parse_retry_after_missing_header() -> None:
    response = httpx.Response(429)

    assert _parse_retry_after(response) is None
