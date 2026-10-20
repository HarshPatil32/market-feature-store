"""Tests for feature definition endpoints."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.features import calculator
from backend.features.seed import seed_feature_definitions
from backend.main import app
from backend.storage.repository import (
    FeatureDefinitionRepository,
    FeatureValueRepository,
    MarketBarRepository,
    SymbolRepository,
)
from backend.storage.schemas import MAX_COMPUTE_SYMBOLS

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DEFINITIONS_PATH = REPO_ROOT / "config" / "feature_definitions.json"


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
    assert payload[0]["lookback_window"] == 14
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
    assert payload[0]["lookback_window"] == 20


@pytest.mark.asyncio
async def test_get_feature_by_name_unknown_returns_404(client: AsyncClient) -> None:
    response = await client.get("/features/unknown_feat")

    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_get_feature_by_name_rejects_invalid_path(client: AsyncClient) -> None:
    response = await client.get(f"/features/{'x' * 101}")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_active_feature_version_returns_single_version(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)

    response = await client.get("/features/sma_20/active")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "sma_20"
    assert payload["version"] == 1
    assert payload["lookback_window"] == 20
    assert payload["active"] is True


@pytest.mark.asyncio
async def test_get_active_feature_version_returns_highest_active(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    await repo.create(name="sma_20", version=2, lookback_window=20)

    response = await client.get("/features/sma_20/active")

    assert response.status_code == 200
    assert response.json()["version"] == 2


@pytest.mark.asyncio
async def test_get_active_feature_version_skips_inactive_higher_version(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    await repo.create(name="sma_20", version=2, lookback_window=20, active=False)

    response = await client.get("/features/sma_20/active")

    assert response.status_code == 200
    assert response.json()["version"] == 1


@pytest.mark.asyncio
async def test_get_active_feature_version_all_inactive_returns_404(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20, active=False)

    response = await client.get("/features/sma_20/active")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_active_feature_version_unknown_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.get("/features/unknown_feat/active")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_active_feature_version_rejects_invalid_path(
    client: AsyncClient,
) -> None:
    response = await client.get(f"/features/{'x' * 101}/active")

    assert response.status_code == 422


async def _seed_definitions(db_session: AsyncSession) -> None:
    await seed_feature_definitions(db_session, FEATURE_DEFINITIONS_PATH)


async def _insert_daily_bars(
    db_session: AsyncSession,
    *,
    symbol_id: int,
    closes: list[float],
    start: datetime | None = None,
) -> None:
    bar_repo = MarketBarRepository(db_session)
    base = start or datetime(2024, 1, 1, tzinfo=UTC)
    for offset, close in enumerate(closes):
        ts = base + timedelta(days=offset)
        price = Decimal(str(close))
        await bar_repo.upsert(
            symbol_id=symbol_id,
            timestamp=ts,
            timeframe="1d",
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("1000000"),
        )


@pytest.mark.asyncio
async def test_compute_features_single_symbol(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["failed"] == {}
    assert payload["computed"]["AAPL"] == len(closes) - 4
    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert len(rows) == payload["computed"]["AAPL"]


@pytest.mark.asyncio
async def test_compute_features_multiple_symbols(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    aapl = await SymbolRepository(db_session).create(symbol="AAPL")
    msft = await SymbolRepository(db_session).create(symbol="MSFT")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=aapl.id, closes=closes)
    await _insert_daily_bars(db_session, symbol_id=msft.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL", "MSFT"], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["failed"] == {}
    assert payload["computed"]["AAPL"] == len(closes) - 4
    assert payload["computed"]["MSFT"] == len(closes) - 4


@pytest.mark.asyncio
async def test_compute_features_respects_start_end_range(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)
    start = datetime(2024, 1, 6, tzinfo=UTC)
    end = datetime(2024, 1, 8, tzinfo=UTC)

    response = await client.post(
        "/features/compute",
        json={
            "symbols": ["AAPL"],
            "features": ["sma_5"],
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )

    assert response.status_code == 200
    assert response.json()["computed"]["AAPL"] == 3
    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert all(start <= row.timestamp <= end for row in rows)


@pytest.mark.asyncio
async def test_compute_features_all_registered_when_features_omitted(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 251)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"]},
    )

    assert response.status_code == 200
    assert response.json()["failed"] == {}
    assert response.json()["computed"]["AAPL"] > 0


@pytest.mark.asyncio
async def test_compute_features_symbol_with_no_bars_returns_zero(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    await SymbolRepository(db_session).create(symbol="AAPL")

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    assert response.json() == {"computed": {"AAPL": 0}, "failed": {}}


@pytest.mark.asyncio
async def test_compute_features_unknown_symbol_returns_404(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL", "NOPE"], "features": ["sma_5"]},
    )

    assert response.status_code == 404
    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert rows == []


@pytest.mark.asyncio
async def test_compute_features_unknown_feature_returns_404(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    await SymbolRepository(db_session).create(symbol="AAPL")

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"], "features": ["not_a_feature"]},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"symbols": [], "features": ["sma_5"]},
        {"symbols": ["AAPL"] * (MAX_COMPUTE_SYMBOLS + 1), "features": ["sma_5"]},
        {"symbols": ["bad ticker!"], "features": ["sma_5"]},
        {
            "symbols": ["AAPL"],
            "features": ["sma_5"],
            "start": "2024-01-01T00:00:00",
        },
        {
            "symbols": ["AAPL"],
            "features": ["sma_5"],
            "start": "2024-01-10T00:00:00+00:00",
            "end": "2024-01-01T00:00:00+00:00",
        },
        {"symbols": ["AAPL"], "features": ["sma_5"], "timeframe": "1d!"},
        {"symbols": ["AAPL"], "features": ["x" * 101]},
        {"symbols": ["AAPL"], "features": []},
    ],
)
async def test_compute_features_rejects_invalid_body(
    client: AsyncClient,
    body: dict[str, object],
) -> None:
    response = await client.post("/features/compute", json=body)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_compute_features_partial_symbol_failure(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    aapl = await SymbolRepository(db_session).create(symbol="AAPL")
    msft = await SymbolRepository(db_session).create(symbol="MSFT")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=aapl.id, closes=closes)
    await _insert_daily_bars(db_session, symbol_id=msft.id, closes=closes)

    real_compute = calculator.compute_and_store_features

    async def _side_effect(
        session: AsyncSession,
        symbol_id: int,
        timeframe: str,
        feature_names: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        computed_at: datetime | None = None,
    ) -> int:
        result = await real_compute(
            session,
            symbol_id,
            timeframe,
            feature_names=feature_names,
            start=start,
            end=end,
            computed_at=computed_at,
        )
        if symbol_id == msft.id:
            raise RuntimeError("boom")
        return result

    with patch(
        "backend.services.feature_compute.compute_and_store_features",
        new=AsyncMock(side_effect=_side_effect),
    ):
        response = await client.post(
            "/features/compute",
            json={"symbols": ["MSFT", "AAPL"], "features": ["sma_5"]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["computed"]["AAPL"] == len(closes) - 4
    assert payload["failed"]["MSFT"] == "Feature computation failed"
    msft_rows = await FeatureValueRepository(db_session).list_by_symbol(msft.id)
    aapl_rows = await FeatureValueRepository(db_session).list_by_symbol(aapl.id)
    assert msft_rows == []
    assert len(aapl_rows) == len(closes) - 4


@pytest.mark.asyncio
async def test_compute_features_dedupes_symbols(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL", "AAPL"], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["computed"] == {"AAPL": len(closes) - 4}
    assert payload["failed"] == {}


@pytest.mark.asyncio
async def test_compute_features_dedupes_feature_names(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"], "features": ["sma_5", "sma_5"]},
    )

    assert response.status_code == 200
    assert response.json()["computed"]["AAPL"] == len(closes) - 4


@pytest.mark.asyncio
async def test_compute_features_normalizes_symbol_ticker(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["  aapl  "], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    assert "AAPL" in response.json()["computed"]


@pytest.mark.asyncio
async def test_compute_features_missing_definition_in_failed(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    response = await client.post(
        "/features/compute",
        json={"symbols": ["AAPL"], "features": ["sma_5"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["computed"] == {}
    assert "AAPL" in payload["failed"]
    assert "Feature definition not found" in payload["failed"]["AAPL"]


@pytest.mark.asyncio
async def test_compute_features_value_error_returns_generic_message(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    with patch(
        "backend.services.feature_compute.compute_and_store_features",
        new=AsyncMock(side_effect=ValueError("sensitive internals")),
    ):
        response = await client.post(
            "/features/compute",
            json={"symbols": ["AAPL"], "features": ["sma_5"]},
        )

    assert response.status_code == 200
    assert response.json()["failed"]["AAPL"] == "Feature computation failed"
