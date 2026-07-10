"""Tests for FeatureValueRepository operations."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import (
    FeatureDefinitionRepository,
    FeatureValueRepository,
    SymbolRepository,
)


async def _create_symbol_and_feature(
    db_session: AsyncSession,
    *,
    symbol: str = "AAPL",
) -> tuple[int, int]:
    symbol_row = await SymbolRepository(db_session).create(symbol=symbol)
    feature = await FeatureDefinitionRepository(db_session).create(
        name="sma_20",
        version=1,
        lookback_window=20,
    )
    return symbol_row.id, feature.id


@pytest.mark.asyncio
async def test_upsert_inserts_row_with_all_fields(db_session: AsyncSession) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)

    created = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.symbol_id == symbol_id
    assert fetched.feature_definition_id == feature_definition_id
    assert fetched.timestamp == ts
    assert fetched.value == Decimal("150.25")
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


@pytest.mark.asyncio
async def test_upsert_on_same_key_updates_value_not_duplicate(
    db_session: AsyncSession,
) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)

    first = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
    )
    second = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("151.00"),
    )
    listed = await repo.list_by_symbol(symbol_id)

    assert second.id == first.id
    assert second.created_at == first.created_at
    assert len(listed) == 1
    assert listed[0].value == Decimal("151.00")
    assert listed[0].updated_at >= first.updated_at


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing_row(db_session: AsyncSession) -> None:
    repo = FeatureValueRepository(db_session)

    assert await repo.get_by_id(999_999) is None


@pytest.mark.asyncio
async def test_list_by_symbol_returns_ordered_by_timestamp(
    db_session: AsyncSession,
) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts_early = datetime(2024, 1, 10, tzinfo=UTC)
    ts_late = datetime(2024, 1, 20, tzinfo=UTC)

    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_late,
        value=Decimal("152.00"),
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_early,
        value=Decimal("148.00"),
    )

    listed = await repo.list_by_symbol(symbol_id)

    assert len(listed) == 2
    assert [row.timestamp for row in listed] == [ts_early, ts_late]


@pytest.mark.asyncio
async def test_list_by_symbol_filters_by_feature_definition_id(
    db_session: AsyncSession,
) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    other_feature = await FeatureDefinitionRepository(db_session).create(
        name="rsi_14",
        version=1,
        lookback_window=14,
    )
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)

    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=other_feature.id,
        timestamp=ts,
        value=Decimal("65.50"),
    )

    filtered = await repo.list_by_symbol(
        symbol_id,
        feature_definition_id=feature_definition_id,
    )

    assert len(filtered) == 1
    assert filtered[0].feature_definition_id == feature_definition_id


@pytest.mark.asyncio
async def test_list_by_symbol_filters_by_date_range(db_session: AsyncSession) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts_before = datetime(2024, 1, 5, tzinfo=UTC)
    ts_in_range = datetime(2024, 1, 15, tzinfo=UTC)
    ts_after = datetime(2024, 1, 25, tzinfo=UTC)

    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_before,
        value=Decimal("140.00"),
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_in_range,
        value=Decimal("150.25"),
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_after,
        value=Decimal("160.00"),
    )

    listed = await repo.list_by_symbol(
        symbol_id,
        start=datetime(2024, 1, 10, tzinfo=UTC),
        end=datetime(2024, 1, 20, tzinfo=UTC),
    )

    assert len(listed) == 1
    assert listed[0].timestamp == ts_in_range


@pytest.mark.asyncio
async def test_foreign_key_violation_on_unknown_symbol_id(
    db_session: AsyncSession,
) -> None:
    _, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.upsert(
            symbol_id=999_999,
            feature_definition_id=feature_definition_id,
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            value=Decimal("150.25"),
        )


@pytest.mark.asyncio
async def test_foreign_key_violation_on_unknown_feature_definition_id(
    db_session: AsyncSession,
) -> None:
    symbol_id, _ = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.upsert(
            symbol_id=symbol_id,
            feature_definition_id=999_999,
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            value=Decimal("150.25"),
        )
