"""Tests for symbol registry service logic."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.services.symbols import (
    DuplicateSymbolError,
    SymbolNotFoundError,
    add_symbol,
    deactivate_symbol,
    list_symbols,
)
from backend.storage.repository import SymbolRepository
from backend.storage.schemas import SymbolCreate


@pytest.mark.asyncio
async def test_add_symbol_uses_defaults(db_session: AsyncSession) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    assert created.symbol == "AAPL"
    assert created.asset_type == "equity"
    assert created.active is True


@pytest.mark.asyncio
async def test_add_symbol_respects_explicit_asset_type(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(
        db_session,
        SymbolCreate(symbol="BTC-USD", asset_type="crypto"),
    )

    assert created.symbol == "BTC-USD"
    assert created.asset_type == "crypto"
    assert created.active is True


@pytest.mark.asyncio
async def test_add_symbol_rejects_duplicate_ticker(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    with pytest.raises(DuplicateSymbolError) as exc_info:
        await add_symbol(db_session, SymbolCreate(symbol="aapl"))

    assert exc_info.value.symbol == "AAPL"


@pytest.mark.asyncio
async def test_add_symbol_duplicate_does_not_poison_session(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))

    with pytest.raises(DuplicateSymbolError):
        await add_symbol(db_session, SymbolCreate(symbol="MSFT"))

    created = await add_symbol(db_session, SymbolCreate(symbol="GOOG"))
    fetched = await SymbolRepository(db_session).get_by_symbol("GOOG")

    assert created.symbol == "GOOG"
    assert fetched is not None
    assert fetched.symbol == "GOOG"


@pytest.mark.asyncio
async def test_add_symbol_rejects_duplicate_from_committed_row(
    engine: AsyncEngine,
    db_session: AsyncSession,
) -> None:
    symbol = "COMMITDUP"
    async with engine.connect() as connection:
        async with connection.begin():
            seed_session = AsyncSession(connection, expire_on_commit=False)
            await add_symbol(seed_session, SymbolCreate(symbol=symbol))
            await seed_session.close()

    with pytest.raises(DuplicateSymbolError) as exc_info:
        await add_symbol(db_session, SymbolCreate(symbol=symbol.lower()))

    assert exc_info.value.symbol == symbol

    async with engine.connect() as connection:
        async with connection.begin():
            cleanup_session = AsyncSession(connection, expire_on_commit=False)
            row = await SymbolRepository(cleanup_session).get_by_symbol(symbol)
            if row is not None:
                await SymbolRepository(cleanup_session).delete(row.id)
            await cleanup_session.close()


@pytest.mark.asyncio
async def test_list_symbols_returns_empty_when_no_symbols(
    db_session: AsyncSession,
) -> None:
    symbols = await list_symbols(db_session)

    assert symbols == []


@pytest.mark.asyncio
async def test_list_symbols_returns_all_symbols_by_default(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    await deactivate_symbol(db_session, "MSFT")

    symbols = await list_symbols(db_session)

    assert [row.symbol for row in symbols] == ["AAPL", "MSFT"]


@pytest.mark.asyncio
async def test_list_symbols_active_only_excludes_inactive(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    await deactivate_symbol(db_session, "MSFT")

    symbols = await list_symbols(db_session, active_only=True)

    assert [row.symbol for row in symbols] == ["AAPL"]


@pytest.mark.asyncio
async def test_list_symbols_includes_coverage_and_freshness_metadata(
    db_session: AsyncSession,
) -> None:
    created = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    assert created.coverage_start is None
    assert created.coverage_end is None
    assert created.last_ingested_at is None

    coverage_start = datetime(2024, 1, 1, tzinfo=UTC)
    coverage_end = datetime(2024, 6, 1, tzinfo=UTC)
    last_ingested_at = datetime(2024, 6, 2, tzinfo=UTC)
    await SymbolRepository(db_session).update_coverage(
        created.id,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        last_ingested_at=last_ingested_at,
    )

    symbols = await list_symbols(db_session)

    assert len(symbols) == 1
    assert symbols[0].coverage_start == coverage_start
    assert symbols[0].coverage_end == coverage_end
    assert symbols[0].last_ingested_at == last_ingested_at


@pytest.mark.asyncio
async def test_deactivate_symbol_sets_active_false(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    deactivated = await deactivate_symbol(db_session, "AAPL")

    assert deactivated.symbol == "AAPL"
    assert deactivated.active is False


@pytest.mark.asyncio
async def test_deactivate_symbol_is_case_insensitive(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    deactivated = await deactivate_symbol(db_session, "aapl")

    assert deactivated.symbol == "AAPL"
    assert deactivated.active is False


@pytest.mark.asyncio
async def test_deactivate_symbol_raises_not_found_for_unknown_ticker(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(SymbolNotFoundError) as exc_info:
        await deactivate_symbol(db_session, "UNKNOWN")

    assert exc_info.value.symbol == "UNKNOWN"


@pytest.mark.asyncio
async def test_deactivate_symbol_is_idempotent(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    first = await deactivate_symbol(db_session, "AAPL")
    second = await deactivate_symbol(db_session, "AAPL")

    assert first.active is False
    assert second.active is False


@pytest.mark.asyncio
async def test_deactivate_symbol_preserves_other_fields(
    db_session: AsyncSession,
) -> None:
    await add_symbol(
        db_session,
        SymbolCreate(symbol="BTC-USD", asset_type="crypto"),
    )

    deactivated = await deactivate_symbol(db_session, "BTC-USD")

    assert deactivated.symbol == "BTC-USD"
    assert deactivated.asset_type == "crypto"
    assert deactivated.active is False
