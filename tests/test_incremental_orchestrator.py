"""Tests for incremental ingestion orchestration and bulk trigger endpoint."""

from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import AwareDatetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.db import get_db_session
from backend.ingestion.incremental import incremental_symbol, incremental_symbols
from backend.main import app
from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.fake import FakeProvider
from backend.services.symbols import add_symbol
from backend.storage.models import IngestionRun, RunStatus
from backend.storage.repository import MarketBarRepository, SymbolRepository
from backend.storage.schemas import SymbolCreate, Ticker


class _RecordingProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self.last_start: AwareDatetime | None = None
        self.last_end: AwareDatetime | None = None

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        self.last_start = start
        self.last_end = end
        return await self._inner.fetch_historical_bars(symbol, timeframe, start, end)

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return await self._inner.fetch_latest_bars(symbol, timeframe, limit)


class _FixedBarsProvider(MarketDataProvider):
    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = list(bars)

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        return self._bars

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return self._bars[:limit]


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


class _FailingForSymbolProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider, *, fail_symbol: str) -> None:
        self._inner = inner
        self._fail_symbol = fail_symbol

    async def fetch_historical_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        start: AwareDatetime,
        end: AwareDatetime,
    ) -> Sequence[Bar]:
        if symbol == self._fail_symbol:
            raise ProviderError("provider unavailable")
        return await self._inner.fetch_historical_bars(symbol, timeframe, start, end)

    async def fetch_latest_bars(
        self,
        symbol: Ticker,
        timeframe: str,
        limit: int = 1,
    ) -> Sequence[Bar]:
        return await self._inner.fetch_latest_bars(symbol, timeframe, limit)


def _bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
) -> Bar:
    return Bar(
        symbol=symbol,
        ts=ts or datetime(2024, 1, 2, tzinfo=UTC),
        timeframe="1d",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=Decimal("1000"),
        source="fake",
    )


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db_session] = _override_session
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        yield http_client
    del app.dependency_overrides[get_db_session]


@pytest.mark.asyncio
async def test_incremental_symbol_inserts_bars_and_succeeds(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    fixed_now = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        run = await incremental_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            provider=fake_provider,
            source="fake",
            lookback_days=2,
        )

    assert run.run_type == "incremental"
    assert run.status == RunStatus.succeeded
    assert run.symbol_id == symbol.id
    assert run.fetched > 0
    assert run.inserted > 0

    updated_symbol = await SymbolRepository(db_session).get_by_id(symbol.id)
    assert updated_symbol is not None
    assert updated_symbol.coverage_end is not None
    assert updated_symbol.last_ingested_at is not None


@pytest.mark.asyncio
async def test_incremental_symbol_uses_coverage_end_as_start(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    coverage_end = datetime(2024, 1, 5, tzinfo=UTC)
    await SymbolRepository(db_session).update_coverage(
        symbol.id,
        coverage_end=coverage_end,
    )
    provider = _RecordingProvider(fake_provider)
    fixed_now = datetime(2024, 1, 10, tzinfo=UTC)

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        await incremental_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            provider=provider,
            source="fake",
        )

    assert provider.last_start == coverage_end
    assert provider.last_end == fixed_now


@pytest.mark.asyncio
async def test_incremental_symbol_already_current(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    fixed_now = datetime(2024, 1, 10, tzinfo=UTC)
    await SymbolRepository(db_session).update_coverage(
        symbol.id,
        coverage_end=fixed_now + timedelta(hours=1),
    )

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        run = await incremental_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            provider=fake_provider,
            source="fake",
        )

    assert run.status == RunStatus.succeeded
    assert run.fetched == 0
    assert run.inserted == 0
    assert run.failed == 0


@pytest.mark.asyncio
async def test_incremental_symbol_deduplicates_bars(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    existing_ts = datetime(2024, 1, 2, tzinfo=UTC)
    bar_repo = MarketBarRepository(db_session)
    await bar_repo.insert_if_not_exists(
        symbol_id=symbol.id,
        timestamp=existing_ts,
        timeframe="1d",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=Decimal("1000"),
    )
    await SymbolRepository(db_session).update_coverage(
        symbol.id,
        coverage_end=datetime(2024, 1, 1, tzinfo=UTC),
    )

    new_ts = datetime(2024, 1, 3, tzinfo=UTC)
    provider = _FixedBarsProvider(
        [_bar(symbol="AAPL", ts=existing_ts), _bar(symbol="AAPL", ts=new_ts)]
    )
    fixed_now = datetime(2024, 1, 4, tzinfo=UTC)

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        run = await incremental_symbol(
            db_session,
            symbol="AAPL",
            timeframe="1d",
            provider=provider,
            source="fake",
        )

    assert run.status == RunStatus.succeeded
    assert run.fetched == 2
    assert run.inserted == 1


@pytest.mark.asyncio
async def test_incremental_symbol_fails_on_provider_error(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    fixed_now = datetime(2024, 1, 10, tzinfo=UTC)

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        with pytest.raises(ProviderError, match="provider unavailable"):
            await incremental_symbol(
                db_session,
                symbol="AAPL",
                timeframe="1d",
                provider=_FailingProvider(),
                source="fake",
                lookback_days=1,
            )

    result = await db_session.execute(
        select(IngestionRun).where(IngestionRun.symbol_id == symbol.id)
    )
    failed_run = result.scalar_one()
    assert failed_run.status == RunStatus.failed
    assert failed_run.error_message == "provider unavailable"


@pytest.mark.asyncio
async def test_incremental_symbols_continues_after_failure(
    db_session: AsyncSession,
    fake_provider: FakeProvider,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    provider = _FailingForSymbolProvider(fake_provider, fail_symbol="AAPL")
    fixed_now = datetime(2024, 1, 10, tzinfo=UTC)

    with patch(
        "backend.ingestion.incremental.datetime",
        wraps=datetime,
    ) as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        results = await incremental_symbols(
            db_session,
            symbols=["AAPL", "MSFT"],
            timeframe="1d",
            provider=provider,
            source="fake",
            lookback_days=1,
        )

    assert isinstance(results["AAPL"], ProviderError)
    assert isinstance(results["MSFT"], IngestionRun)
    assert results["MSFT"].status == RunStatus.succeeded


@pytest.mark.asyncio
async def test_trigger_bulk_incremental_endpoint(
    client: AsyncClient,
    fake_provider: FakeProvider,
) -> None:
    fixed_now = datetime(2024, 1, 10, tzinfo=UTC)
    with (
        patch(
            "backend.api.router.get_market_data_provider",
            return_value=fake_provider,
        ),
        patch(
            "backend.ingestion.incremental.datetime",
            wraps=datetime,
        ) as mocked_datetime,
    ):
        mocked_datetime.now.return_value = fixed_now
        create_response = await client.post("/symbols", json={"symbol": "AAPL"})
        assert create_response.status_code == 201

        response = await client.post("/ingestion/incremental?timeframe=1d")

    assert response.status_code == 201
    payload = response.json()
    assert "AAPL" in payload["succeeded"]
    assert payload["failed"] == {}
    assert payload["succeeded"]["AAPL"]["run_type"] == "incremental"
    assert payload["succeeded"]["AAPL"]["status"] == RunStatus.succeeded.value


@pytest.mark.asyncio
async def test_trigger_bulk_incremental_no_stale_symbols(
    client: AsyncClient,
) -> None:
    response = await client.post("/ingestion/incremental")
    assert response.status_code == 201
    assert response.json() == {"succeeded": {}, "failed": {}}


@pytest.mark.asyncio
async def test_trigger_bulk_incremental_rejects_invalid_timeframe(
    client: AsyncClient,
) -> None:
    response = await client.post("/ingestion/incremental?timeframe=bad!timeframe")
    assert response.status_code == 422
