"""Tests for raw market data persistence service."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.ingestion_runs import trigger_backfill
from backend.services.raw_market_data import persist_raw_fetch
from backend.services.symbols import add_symbol
from backend.storage.repository import RawMarketDataRepository
from backend.storage.schemas import SymbolCreate


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
