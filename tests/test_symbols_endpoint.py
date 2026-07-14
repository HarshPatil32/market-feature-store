"""Tests for GET /symbols endpoint."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.main import app
from backend.services.symbols import add_symbol, deactivate_symbol
from backend.storage.schemas import SymbolCreate


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
async def test_get_symbols_returns_empty_list(client: AsyncClient) -> None:
    response = await client.get("/symbols")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_symbols_returns_created_symbols(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    response = await client.get("/symbols")

    assert response.status_code == 200
    payload = response.json()
    assert [row["symbol"] for row in payload] == ["AAPL", "MSFT"]
    assert payload[0]["asset_type"] == "equity"
    assert payload[0]["active"] is True


@pytest.mark.asyncio
async def test_get_symbols_active_filter_excludes_inactive(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    await deactivate_symbol(db_session, "MSFT")

    response = await client.get("/symbols", params={"active": True})

    assert response.status_code == 200
    assert [row["symbol"] for row in response.json()] == ["AAPL"]


@pytest.mark.asyncio
async def test_get_symbols_respects_limit_and_offset(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    await add_symbol(db_session, SymbolCreate(symbol="NVDA"))

    response = await client.get("/symbols", params={"limit": 1, "offset": 1})

    assert response.status_code == 200
    assert [row["symbol"] for row in response.json()] == ["MSFT"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params",),
    [
        ({"limit": 0},),
        ({"limit": 201},),
        ({"offset": -1},),
        ({"active": "not-a-bool"},),
    ],
)
async def test_get_symbols_rejects_invalid_query_params(
    client: AsyncClient,
    params: dict[str, object],
) -> None:
    response = await client.get("/symbols", params=params)

    assert response.status_code == 422
