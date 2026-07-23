"""Tests for RawMarketDataRepository CRUD operations."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import (
    IngestionRunRepository,
    RawMarketDataRepository,
    SymbolRepository,
)


@pytest.mark.asyncio
async def test_create_stores_jsonb_payloads(db_session: AsyncSession) -> None:
    repo = RawMarketDataRepository(db_session)
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

    created = await repo.create(
        request_params=request_params,
        response_payload=response_payload,
        source="alpha_vantage",
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.request_params == request_params
    assert fetched.response_payload == response_payload
    assert fetched.source == "alpha_vantage"
    assert fetched.run_id is None
    assert fetched.symbol_id is None


@pytest.mark.asyncio
async def test_create_with_run_and_symbol_ids(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    run_repo = IngestionRunRepository(db_session)
    raw_repo = RawMarketDataRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    run = await run_repo.create(run_type="backfill", symbol_id=symbol.id)
    created = await raw_repo.create(
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
    )
    fetched = await raw_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id == run.id
    assert fetched.symbol_id == symbol.id


@pytest.mark.asyncio
async def test_create_with_invalid_run_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    repo = RawMarketDataRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.create(run_id=999_999, response_payload={"bars": []})


@pytest.mark.asyncio
async def test_create_with_invalid_symbol_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    repo = RawMarketDataRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.create(symbol_id=999_999, response_payload={"bars": []})


@pytest.mark.asyncio
async def test_null_response_payload_without_object_key_rejected(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(IntegrityError):
        await db_session.execute(
            text("INSERT INTO raw_market_data (response_payload) VALUES (:payload)"),
            {"payload": None},
        )
        await db_session.flush()


@pytest.mark.asyncio
async def test_create_with_object_key_and_no_inline_payload(
    db_session: AsyncSession,
) -> None:
    repo = RawMarketDataRepository(db_session)
    created = await repo.create(
        response_payload=None,
        payload_object_key="raw/1/abc.json",
        payload_size_bytes=1024,
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.response_payload is None
    assert fetched.payload_object_key == "raw/1/abc.json"
    assert fetched.payload_size_bytes == 1024


@pytest.mark.asyncio
async def test_create_rejects_both_payload_locations(
    db_session: AsyncSession,
) -> None:
    repo = RawMarketDataRepository(db_session)

    with pytest.raises(ValueError, match="exactly one of"):
        await repo.create(
            response_payload={"bars": []},
            payload_object_key="raw/1/abc.json",
        )


@pytest.mark.asyncio
async def test_create_rejects_neither_payload_location(
    db_session: AsyncSession,
) -> None:
    repo = RawMarketDataRepository(db_session)

    with pytest.raises(ValueError, match="exactly one of"):
        await repo.create(response_payload=None)


@pytest.mark.asyncio
async def test_get_by_id(db_session: AsyncSession) -> None:
    repo = RawMarketDataRepository(db_session)
    created = await repo.create(response_payload={"bars": []})

    fetched = await repo.get_by_id(created.id)
    missing = await repo.get_by_id(999_999)

    assert fetched is not None
    assert fetched.id == created.id
    assert missing is None


@pytest.mark.asyncio
async def test_run_id_set_null_when_ingestion_run_deleted(
    db_session: AsyncSession,
) -> None:
    run_repo = IngestionRunRepository(db_session)
    raw_repo = RawMarketDataRepository(db_session)

    run = await run_repo.create(run_type="backfill")
    raw_row = await raw_repo.create(run_id=run.id, response_payload={"bars": []})
    raw_row_id = raw_row.id  # save before expire_all evicts the identity-map entry

    await run_repo.delete(run.id)
    db_session.expire_all()
    fetched = await raw_repo.get_by_id(raw_row_id)

    assert fetched is not None
    assert fetched.run_id is None


@pytest.mark.asyncio
async def test_symbol_id_set_null_when_symbol_deleted(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    raw_repo = RawMarketDataRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    raw_row = await raw_repo.create(
        symbol_id=symbol.id,
        response_payload={"bars": []},
    )
    raw_row_id = raw_row.id  # save before expire_all evicts the identity-map entry

    await symbol_repo.delete(symbol.id)
    db_session.expire_all()
    fetched = await raw_repo.get_by_id(raw_row_id)

    assert fetched is not None
    assert fetched.symbol_id is None
