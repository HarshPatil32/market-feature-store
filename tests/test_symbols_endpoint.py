"""Tests for symbol registry endpoints."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.main import app
from backend.services.symbols import add_symbol, deactivate_symbol
from backend.storage.schemas import SymbolCreate

INVALID_SYMBOL_PATH_SEGMENTS = [
    "A" * 21,
    "AA%20PL",
    "AA%24PL",
    "AAPL%21",
    "%20%20%20",
]


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


@pytest.mark.asyncio
async def test_post_symbols_creates_and_returns_201(client: AsyncClient) -> None:
    response = await client.post("/symbols", json={"symbol": "AAPL"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["asset_type"] == "equity"
    assert payload["active"] is True
    assert isinstance(payload["id"], int)
    assert payload["created_at"] is not None
    assert payload["updated_at"] is not None


@pytest.mark.asyncio
async def test_post_symbols_normalizes_and_respects_asset_type(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/symbols",
        json={"symbol": "btc-usd", "asset_type": "crypto"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["symbol"] == "BTC-USD"
    assert payload["asset_type"] == "crypto"


@pytest.mark.asyncio
async def test_post_symbols_duplicate_returns_409(client: AsyncClient) -> None:
    first = await client.post("/symbols", json={"symbol": "AAPL"})
    assert first.status_code == 201

    second = await client.post("/symbols", json={"symbol": "AAPL"})

    assert second.status_code == 409
    assert "detail" in second.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ticker",
    [
        "",
        "A" * 21,
        "AA PL",
        "AA$PL",
        "AAPL!",
        "   ",
    ],
)
async def test_post_symbols_rejects_invalid_payload(
    client: AsyncClient,
    ticker: str,
) -> None:
    response = await client.post("/symbols", json={"symbol": ticker})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_symbol_by_ticker_returns_200_with_data(
    client: AsyncClient,
) -> None:
    create = await client.post("/symbols", json={"symbol": "AAPL"})
    assert create.status_code == 201

    response = await client.get("/symbols/AAPL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["asset_type"] == "equity"
    assert payload["active"] is True
    assert isinstance(payload["id"], int)
    assert payload["created_at"] is not None
    assert payload["updated_at"] is not None


@pytest.mark.asyncio
async def test_get_symbol_by_ticker_is_case_insensitive(
    client: AsyncClient,
) -> None:
    create = await client.post("/symbols", json={"symbol": "AAPL"})
    assert create.status_code == 201

    response = await client.get("/symbols/aapl")

    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_symbol_by_ticker_unknown_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.get("/symbols/UNKNOWN")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("path_segment", INVALID_SYMBOL_PATH_SEGMENTS)
async def test_get_symbol_by_ticker_rejects_invalid_path(
    client: AsyncClient,
    path_segment: str,
) -> None:
    response = await client.get(f"/symbols/{path_segment}")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_symbol_by_ticker_returns_inactive_symbol(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    await deactivate_symbol(db_session, "AAPL")

    response = await client.get("/symbols/AAPL")

    assert response.status_code == 200
    assert response.json()["active"] is False


@pytest.mark.asyncio
async def test_post_symbols_persists_and_is_listable(client: AsyncClient) -> None:
    create = await client.post("/symbols", json={"symbol": "NVDA"})
    assert create.status_code == 201

    response = await client.get("/symbols")

    assert response.status_code == 200
    assert [row["symbol"] for row in response.json()] == ["NVDA"]


@pytest.mark.asyncio
async def test_post_symbol_backfill_returns_201_with_run_id(
    client: AsyncClient,
) -> None:
    create = await client.post("/symbols", json={"symbol": "AAPL"})
    assert create.status_code == 201
    symbol_id = create.json()["id"]

    response = await client.post("/symbols/AAPL/backfill")

    assert response.status_code == 201
    payload = response.json()
    assert isinstance(payload["id"], int)
    assert payload["run_type"] == "backfill"
    assert payload["status"] == "pending"
    assert payload["symbol_id"] == symbol_id
    assert payload["fetched"] == 0
    assert payload["inserted"] == 0
    assert payload["failed"] == 0
    assert payload["error_message"] is None
    assert payload["created_at"] is not None


@pytest.mark.asyncio
async def test_post_symbol_incremental_returns_201_with_run_id(
    client: AsyncClient,
) -> None:
    create = await client.post("/symbols", json={"symbol": "MSFT"})
    assert create.status_code == 201
    symbol_id = create.json()["id"]

    response = await client.post("/symbols/MSFT/incremental")

    assert response.status_code == 201
    payload = response.json()
    assert isinstance(payload["id"], int)
    assert payload["run_type"] == "incremental"
    assert payload["status"] == "pending"
    assert payload["symbol_id"] == symbol_id
    assert payload["created_at"] is not None


@pytest.mark.asyncio
async def test_post_symbol_backfill_is_case_insensitive(
    client: AsyncClient,
) -> None:
    create = await client.post("/symbols", json={"symbol": "AAPL"})
    assert create.status_code == 201

    response = await client.post("/symbols/aapl/backfill")

    assert response.status_code == 201
    payload = response.json()
    assert payload["run_type"] == "backfill"
    assert payload["symbol_id"] == create.json()["id"]


@pytest.mark.asyncio
async def test_post_symbol_backfill_unknown_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.post("/symbols/UNKNOWN/backfill")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_post_symbol_incremental_unknown_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.post("/symbols/UNKNOWN/incremental")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("path_segment", INVALID_SYMBOL_PATH_SEGMENTS)
async def test_post_symbol_backfill_rejects_invalid_path(
    client: AsyncClient,
    path_segment: str,
) -> None:
    response = await client.post(f"/symbols/{path_segment}/backfill")

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("path_segment", INVALID_SYMBOL_PATH_SEGMENTS)
async def test_post_symbol_incremental_rejects_invalid_path(
    client: AsyncClient,
    path_segment: str,
) -> None:
    response = await client.post(f"/symbols/{path_segment}/incremental")

    assert response.status_code == 422
