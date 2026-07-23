"""Tests for raw market data persistence service."""

from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.services.ingestion_runs import trigger_backfill
from backend.services.raw_market_data import load_response_payload, persist_raw_fetch
from backend.services.symbols import add_symbol
from backend.storage.object_store import ObjectStoreError, clear_object_store_cache
from backend.storage.repository import RawMarketDataRepository
from backend.storage.schemas import SymbolCreate

VALID_DB_URL = (
    "postgresql+asyncpg://postgres:changeme@localhost:5433/market_feature_store"
)


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.fail_put = False

    async def put_json(self, key: str, payload: dict[str, Any]) -> None:
        if self.fail_put:
            raise ObjectStoreError("upload failed")
        self.objects[key] = payload

    async def get_json(self, key: str) -> dict[str, Any]:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise ObjectStoreError("object not found") from exc


@pytest.fixture(autouse=True)
def clear_settings_and_object_store_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    clear_object_store_cache()
    yield
    get_settings.cache_clear()
    clear_object_store_cache()


def _enable_s3_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold_bytes: int = 10,
) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.setenv("S3_BUCKET", "raw-market-data")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("RAW_PAYLOAD_S3_THRESHOLD_BYTES", str(threshold_bytes))
    get_settings.cache_clear()
    clear_object_store_cache()


@pytest.mark.asyncio
async def test_persist_raw_fetch_links_row_to_ingestion_run(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    request_params = {
        "symbol": "AAPL",
        "interval": "1d",
        "start": "2024-01-01",
        "end": "2024-01-31",
    }
    response_payload = {
        "meta": {"symbol": "AAPL", "count": 2},
        "bars": [
            {"timestamp": "2024-01-02", "open": 100.0, "close": 101.5},
            {"timestamp": "2024-01-03", "open": 101.5, "close": 99.0},
        ],
    }

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        source="alpha_vantage",
        request_params=request_params,
        response_payload=response_payload,
    )
    fetched = await RawMarketDataRepository(db_session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id == run.id
    assert fetched.symbol_id == symbol.id
    assert fetched.source == "alpha_vantage"
    assert fetched.request_params == request_params
    assert fetched.response_payload == response_payload
    assert fetched.payload_object_key is None


@pytest.mark.asyncio
async def test_persist_raw_fetch_requires_valid_run_id(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(IntegrityError):
        await persist_raw_fetch(
            db_session,
            run_id=999_999,
            response_payload={"bars": []},
        )


@pytest.mark.asyncio
async def test_persist_raw_fetch_allows_optional_metadata(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    run = await trigger_backfill(db_session, symbol.symbol)

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload={"bars": []},
    )
    fetched = await RawMarketDataRepository(db_session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id == run.id
    assert fetched.symbol_id is None
    assert fetched.source is None
    assert fetched.request_params is None
    assert fetched.response_payload == {"bars": []}


@pytest.mark.asyncio
async def test_persist_raw_fetch_keeps_small_payload_in_jsonb_when_s3_enabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_s3_settings(monkeypatch, threshold_bytes=1024)
    fake_store = FakeObjectStore()
    monkeypatch.setattr(
        "backend.services.raw_market_data.get_object_store",
        lambda: fake_store,
    )

    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02"}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload=response_payload,
    )

    assert created.response_payload == response_payload
    assert created.payload_object_key is None
    assert fake_store.objects == {}


@pytest.mark.asyncio
async def test_persist_raw_fetch_offloads_large_payload_to_object_store(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_s3_settings(monkeypatch, threshold_bytes=10)
    fake_store = FakeObjectStore()
    monkeypatch.setattr(
        "backend.services.raw_market_data.get_object_store",
        lambda: fake_store,
    )

    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02", "data": "x" * 100}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload=response_payload,
    )

    assert created.response_payload is None
    assert created.payload_object_key is not None
    assert created.payload_object_key.startswith(f"raw/{run.id}/")
    assert fake_store.objects[created.payload_object_key] == response_payload


@pytest.mark.asyncio
async def test_persist_raw_fetch_large_payload_stays_in_jsonb_when_s3_disabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    get_settings.cache_clear()
    clear_object_store_cache()

    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02", "data": "x" * 100}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload=response_payload,
    )

    assert created.response_payload == response_payload
    assert created.payload_object_key is None


@pytest.mark.asyncio
async def test_persist_raw_fetch_falls_back_to_jsonb_on_upload_failure(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_s3_settings(monkeypatch, threshold_bytes=10)
    fake_store = FakeObjectStore()
    fake_store.fail_put = True
    monkeypatch.setattr(
        "backend.services.raw_market_data.get_object_store",
        lambda: fake_store,
    )

    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02", "data": "x" * 100}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload=response_payload,
    )

    assert created.response_payload == response_payload
    assert created.payload_object_key is None
    assert fake_store.objects == {}


@pytest.mark.asyncio
async def test_load_response_payload_reads_inline_jsonb(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02"}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload=response_payload,
    )

    assert await load_response_payload(created) == response_payload


@pytest.mark.asyncio
async def test_load_response_payload_reads_from_object_store(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_s3_settings(monkeypatch, threshold_bytes=10)
    fake_store = FakeObjectStore()
    monkeypatch.setattr(
        "backend.services.raw_market_data.get_object_store",
        lambda: fake_store,
    )

    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02", "data": "x" * 100}]}

    created = await persist_raw_fetch(
        db_session,
        run_id=run.id,
        response_payload=response_payload,
    )

    assert await load_response_payload(created) == response_payload


@pytest.mark.asyncio
async def test_load_response_payload_raises_when_object_store_unconfigured(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.services.raw_market_data.get_object_store", lambda: None
    )
    repo = RawMarketDataRepository(db_session)
    row = await repo.create(
        response_payload=None,
        payload_object_key="raw/1/abc.json",
        payload_size_bytes=1024,
    )

    with pytest.raises(RuntimeError, match="no object store configured"):
        await load_response_payload(row)


def test_s3_settings_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    settings = Settings(_env_file=None)

    assert settings.s3_enabled is False
    assert settings.raw_payload_s3_threshold_bytes == 262_144
