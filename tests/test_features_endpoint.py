"""Tests for feature definition endpoints."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.main import app
from backend.storage.repository import FeatureDefinitionRepository


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
async def test_get_features_returns_empty_list(client: AsyncClient) -> None:
    response = await client.get("/features")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_features_returns_seeded_features(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(
        name="rsi_14",
        version=1,
        lookback_window=14,
        description="RSI",
        requirements=["lookback_window"],
    )
    await repo.create(name="sma_20", version=1, lookback_window=20)

    response = await client.get("/features")

    assert response.status_code == 200
    payload = response.json()
    assert [row["name"] for row in payload] == ["rsi_14", "sma_20"]
    assert payload[0]["description"] == "RSI"
    assert payload[0]["requirements"] == ["lookback_window"]
    assert payload[0]["version"] == 1
    assert payload[0]["active"] is True


@pytest.mark.asyncio
async def test_get_features_active_filter_excludes_inactive(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    await repo.create(
        name="rsi_14",
        version=1,
        lookback_window=14,
        active=False,
    )

    response = await client.get("/features", params={"active": True})

    assert response.status_code == 200
    assert [row["name"] for row in response.json()] == ["sma_20"]


@pytest.mark.asyncio
async def test_get_features_respects_limit_and_offset(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="a_feat", version=1, lookback_window=1)
    await repo.create(name="b_feat", version=1, lookback_window=1)
    await repo.create(name="c_feat", version=1, lookback_window=1)

    response = await client.get("/features", params={"limit": 1, "offset": 1})

    assert response.status_code == 200
    assert [row["name"] for row in response.json()] == ["b_feat"]


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
async def test_get_features_rejects_invalid_query_params(
    client: AsyncClient,
    params: dict[str, object],
) -> None:
    response = await client.get("/features", params=params)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_feature_by_name_returns_all_versions(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    await repo.create(name="sma_20", version=2, lookback_window=20)

    response = await client.get("/features/sma_20")

    assert response.status_code == 200
    payload = response.json()
    assert [row["version"] for row in payload] == [1, 2]
    assert all(row["name"] == "sma_20" for row in payload)


@pytest.mark.asyncio
async def test_get_feature_by_name_includes_description_and_requirements(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(
        name="sma_20",
        version=1,
        lookback_window=20,
        description="Simple moving average",
        requirements=["lookback_window"],
    )

    response = await client.get("/features/sma_20")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["description"] == "Simple moving average"
    assert payload[0]["requirements"] == ["lookback_window"]


@pytest.mark.asyncio
async def test_get_feature_by_name_unknown_returns_404(client: AsyncClient) -> None:
    response = await client.get("/features/unknown_feat")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_get_feature_by_name_rejects_invalid_path(client: AsyncClient) -> None:
    response = await client.get(f"/features/{'x' * 101}")

    assert response.status_code == 422
