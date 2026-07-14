"""Tests for symbol registry service logic."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.services.symbols import DuplicateSymbolError, add_symbol
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
