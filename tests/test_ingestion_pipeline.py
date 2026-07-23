"""Tests for ingestion pipeline orchestration."""

from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ingestion.pipeline import ingest_raw_data
from backend.services.ingestion_runs import trigger_backfill
from backend.services.raw_market_data import persist_raw_fetch
from backend.services.symbols import add_symbol
from backend.storage.models import RawMarketData
from backend.storage.repository import RawMarketDataRepository
from backend.storage.schemas import SymbolCreate


@pytest.mark.asyncio
async def test_ingest_raw_data_persists_raw_row_before_calling_normalize(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {
        "meta": {"symbol": "AAPL", "count": 1},
        "bars": [{"timestamp": "2024-01-02", "open": 100.0, "close": 101.5}],
    }
    normalize_calls: list[dict[str, object]] = []

    def normalize(payload: dict[str, object]) -> None:
        normalize_calls.append(payload)

    created = await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        source="fake",
        response_payload=response_payload,
        normalize=normalize,
    )
    fetched = await RawMarketDataRepository(db_session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id == run.id
    assert fetched.response_payload == response_payload
    assert normalize_calls == [response_payload]


@pytest.mark.asyncio
async def test_ingest_raw_data_raw_row_survives_normalization_failure(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    run = await trigger_backfill(db_session, "MSFT")
    response_payload: dict[str, Any] = {"bars": []}

    def normalize(_payload: dict[str, object]) -> None:
        raise ValueError("boom")

    raw_id: int | None = None

    async def capture_persist(
        session: AsyncSession,
        *,
        run_id: int,
        response_payload: dict[str, Any],
        symbol_id: int | None = None,
        source: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RawMarketData:
        nonlocal raw_id
        row = await persist_raw_fetch(
            session,
            run_id=run_id,
            symbol_id=symbol_id,
            source=source,
            request_params=request_params,
            response_payload=response_payload,
        )
        raw_id = row.id
        return row

    with patch(
        "backend.ingestion.pipeline.persist_raw_fetch",
        side_effect=capture_persist,
    ):
        with pytest.raises(ValueError, match="boom"):
            await ingest_raw_data(
                db_session,
                run_id=run.id,
                symbol_id=symbol.id,
                response_payload=response_payload,
                normalize=normalize,
            )

    assert raw_id is not None
    fetched = await RawMarketDataRepository(db_session).get_by_id(raw_id)
    assert fetched is not None
    assert fetched.response_payload == response_payload


@pytest.mark.asyncio
async def test_ingest_raw_data_normalization_failure_does_not_poison_session(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            response_payload={"bars": []},
            normalize=normalize,
        )

    created = await add_symbol(db_session, SymbolCreate(symbol="GOOG"))

    assert created.symbol == "GOOG"


@pytest.mark.asyncio
async def test_ingest_raw_data_rejects_async_normalize(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    async def normalize(_payload: dict[str, object]) -> None:
        pass

    with pytest.raises(TypeError, match="synchronous callable"):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            response_payload={"bars": []},
            normalize=normalize,
        )
