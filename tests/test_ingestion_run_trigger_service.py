"""Tests for ingestion run trigger service logic."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.ingestion_runs import trigger_backfill, trigger_incremental
from backend.services.symbols import SymbolNotFoundError, add_symbol
from backend.storage.models import RunStatus
from backend.storage.schemas import SymbolCreate


@pytest.mark.asyncio
async def test_trigger_backfill_creates_pending_run_for_symbol(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    run = await trigger_backfill(db_session, "AAPL")

    assert run.run_type == "backfill"
    assert run.status == RunStatus.pending
    assert run.symbol_id == created.id
    assert run.fetched == 0
    assert run.inserted == 0
    assert run.failed == 0
    assert run.error_message is None


@pytest.mark.asyncio
async def test_trigger_incremental_creates_pending_run_for_symbol(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="MSFT"))

    run = await trigger_incremental(db_session, "MSFT")

    assert run.run_type == "incremental"
    assert run.status == RunStatus.pending
    assert run.symbol_id == created.id
    assert run.fetched == 0
    assert run.inserted == 0
    assert run.failed == 0
    assert run.error_message is None


@pytest.mark.asyncio
async def test_trigger_backfill_is_case_insensitive(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    run = await trigger_backfill(db_session, "aapl")

    assert run.symbol_id == created.id
    assert run.run_type == "backfill"


@pytest.mark.asyncio
async def test_trigger_incremental_is_case_insensitive(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    run = await trigger_incremental(db_session, "aapl")

    assert run.symbol_id == created.id
    assert run.run_type == "incremental"


@pytest.mark.asyncio
async def test_trigger_backfill_raises_not_found_for_unknown_ticker(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(SymbolNotFoundError) as exc_info:
        await trigger_backfill(db_session, "UNKNOWN")

    assert exc_info.value.symbol == "UNKNOWN"


@pytest.mark.asyncio
async def test_trigger_incremental_raises_not_found_for_unknown_ticker(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(SymbolNotFoundError) as exc_info:
        await trigger_incremental(db_session, "UNKNOWN")

    assert exc_info.value.symbol == "UNKNOWN"
