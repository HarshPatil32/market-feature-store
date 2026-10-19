"""Tests for compute_and_store_features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.features.calculator import (
    FeatureDefinitionLookbackMismatchError,
    FeatureDefinitionNotFoundError,
    UnknownSymbolError,
    compute_and_store_features,
)
from backend.features.moving_averages import Sma5
from backend.features.seed import seed_feature_definitions
from backend.services.features import FeatureNotFoundError
from backend.storage.repository import (
    FeatureDefinitionRepository,
    FeatureValueRepository,
    MarketBarRepository,
    SymbolRepository,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DEFINITIONS_PATH = REPO_ROOT / "config" / "feature_definitions.json"


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
async def test_compute_sma_5_persists_non_warmup_rows(db_session: AsyncSession) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="AAPL")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)
    computed_at = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    count = await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
        computed_at=computed_at,
    )

    assert count == len(closes) - 4
    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert len(rows) == count
    assert all(row.computed_at == computed_at for row in rows)


@pytest.mark.asyncio
async def test_stored_values_match_feature_compute(db_session: AsyncSession) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="MSFT")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )
    index = pd.DatetimeIndex([bar.timestamp for bar in bars])
    frame = pd.DataFrame(
        {"close": [float(bar.close) for bar in bars]},
        index=index,
    )
    expected = Sma5().compute(frame)
    definition = await FeatureDefinitionRepository(db_session).get_by_name_and_version(
        "sma_5",
        1,
    )
    assert definition is not None
    rows = await FeatureValueRepository(db_session).list_by_symbol(
        symbol.id,
        feature_definition_id=definition.id,
    )
    by_ts = {row.timestamp: row.value for row in rows}
    for ts, val in expected.items():
        if pd.isna(val):
            assert ts not in by_ts
        else:
            assert by_ts[ts] == Decimal(str(float(val))).quantize(Decimal("0.00000001"))


@pytest.mark.asyncio
async def test_recompute_updates_computed_at_without_duplicates(
    db_session: AsyncSession,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="GOOG")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)
    first_at = datetime(2024, 6, 1, tzinfo=UTC)
    second_at = datetime(2024, 6, 2, tzinfo=UTC)

    await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
        computed_at=first_at,
    )
    first_rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)

    await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
        computed_at=second_at,
    )
    second_rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)

    assert len(second_rows) == len(first_rows)
    assert {row.id for row in second_rows} == {row.id for row in first_rows}
    assert all(row.computed_at == second_at for row in second_rows)
    first_by_id = {row.id: row for row in first_rows}
    for row in second_rows:
        assert row.created_at == first_by_id[row.id].created_at


@pytest.mark.asyncio
async def test_missing_definition_raises(db_session: AsyncSession) -> None:
    symbol = await SymbolRepository(db_session).create(symbol="NODEF")
    await _insert_daily_bars(
        db_session, symbol_id=symbol.id, closes=[1.0, 2.0, 3.0, 4.0, 5.0]
    )

    with pytest.raises(FeatureDefinitionNotFoundError, match="sma_5"):
        await compute_and_store_features(
            db_session,
            symbol.id,
            "1d",
            feature_names=["sma_5"],
        )


@pytest.mark.asyncio
async def test_lookback_mismatch_raises(db_session: AsyncSession) -> None:
    await FeatureDefinitionRepository(db_session).create(
        name="sma_5",
        version=1,
        lookback_window=99,
    )
    symbol = await SymbolRepository(db_session).create(symbol="MISMATCH")
    await _insert_daily_bars(
        db_session, symbol_id=symbol.id, closes=[1.0, 2.0, 3.0, 4.0, 5.0]
    )

    with pytest.raises(FeatureDefinitionLookbackMismatchError, match="lookback_window"):
        await compute_and_store_features(
            db_session,
            symbol.id,
            "1d",
            feature_names=["sma_5"],
        )


@pytest.mark.asyncio
async def test_start_end_range_filters_persisted_rows(
    db_session: AsyncSession,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="RANGE")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)
    start = datetime(2024, 1, 6, tzinfo=UTC)
    end = datetime(2024, 1, 8, tzinfo=UTC)

    count = await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
        start=start,
        end=end,
    )

    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert count == len(rows)
    assert all(start <= row.timestamp <= end for row in rows)
    by_ts = {row.timestamp: row.value for row in rows}
    # sma_5 at start uses closes from the five days before start (full history loaded).
    assert by_ts[start] == Decimal("4.00000000")


@pytest.mark.asyncio
async def test_duplicate_feature_names_compute_once(db_session: AsyncSession) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="DUPDEF")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    count = await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5", "sma_5"],
    )

    assert count == len(closes) - 4
    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    assert len(rows) == count


@pytest.mark.asyncio
async def test_unknown_feature_name_raises(db_session: AsyncSession) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="UNKFEAT")
    await _insert_daily_bars(
        db_session, symbol_id=symbol.id, closes=[1.0, 2.0, 3.0, 4.0, 5.0]
    )

    with pytest.raises(FeatureNotFoundError, match="not_a_feature"):
        await compute_and_store_features(
            db_session,
            symbol.id,
            "1d",
            feature_names=["not_a_feature"],
        )


@pytest.mark.asyncio
async def test_unknown_symbol_id_raises(db_session: AsyncSession) -> None:
    await _seed_definitions(db_session)

    with pytest.raises(UnknownSymbolError):
        await compute_and_store_features(
            db_session,
            999_999,
            "1d",
            feature_names=["sma_5"],
        )


@pytest.mark.asyncio
async def test_subset_of_features_only_persists_requested(
    db_session: AsyncSession,
) -> None:
    await _seed_definitions(db_session)
    symbol = await SymbolRepository(db_session).create(symbol="SUBSET")
    closes = [float(i) for i in range(1, 11)]
    await _insert_daily_bars(db_session, symbol_id=symbol.id, closes=closes)

    await compute_and_store_features(
        db_session,
        symbol.id,
        "1d",
        feature_names=["sma_5"],
    )

    rows = await FeatureValueRepository(db_session).list_by_symbol(symbol.id)
    definition_ids = {row.feature_definition_id for row in rows}
    sma5 = await FeatureDefinitionRepository(db_session).get_by_name_and_version(
        "sma_5",
        1,
    )
    sma20 = await FeatureDefinitionRepository(db_session).get_by_name_and_version(
        "sma_20",
        1,
    )
    assert sma5 is not None
    assert sma20 is not None
    assert definition_ids == {sma5.id}
    assert sma20.id not in definition_ids
