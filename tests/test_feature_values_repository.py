"""Tests for FeatureValueRepository operations."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import (
    FeatureDefinitionRepository,
    FeatureValueRepository,
    SymbolRepository,
)


def _computed_at() -> datetime:
    return datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


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

    computed_at = _computed_at()
    created = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
        computed_at=computed_at,
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.symbol_id == symbol_id
    assert fetched.feature_definition_id == feature_definition_id
    assert fetched.timestamp == ts
    assert fetched.value == Decimal("150.25")
    assert fetched.computed_at == computed_at
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


@pytest.mark.asyncio
async def test_upsert_on_same_key_updates_value_not_duplicate(
    db_session: AsyncSession,
) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)

    first_at = _computed_at()
    second_at = datetime(2024, 6, 2, tzinfo=UTC)
    first = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
        computed_at=first_at,
    )
    second = await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("151.00"),
        computed_at=second_at,
    )
    listed = await repo.list_by_symbol(symbol_id)

    assert second.id == first.id
    assert second.created_at == first.created_at
    assert len(listed) == 1
    assert listed[0].value == Decimal("151.00")
    assert listed[0].computed_at == second_at
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

    computed_at = _computed_at()
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_late,
        value=Decimal("152.00"),
        computed_at=computed_at,
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_early,
        value=Decimal("148.00"),
        computed_at=computed_at,
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

    computed_at = _computed_at()
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts,
        value=Decimal("150.25"),
        computed_at=computed_at,
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=other_feature.id,
        timestamp=ts,
        value=Decimal("65.50"),
        computed_at=computed_at,
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

    computed_at = _computed_at()
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_before,
        value=Decimal("140.00"),
        computed_at=computed_at,
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_in_range,
        value=Decimal("150.25"),
        computed_at=computed_at,
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=feature_definition_id,
        timestamp=ts_after,
        value=Decimal("160.00"),
        computed_at=computed_at,
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
            computed_at=_computed_at(),
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
            computed_at=_computed_at(),
        )


@pytest.mark.asyncio
async def test_same_timestamp_different_versions_are_distinct_rows(
    db_session: AsyncSession,
) -> None:
    symbol_id, _ = await _create_symbol_and_feature(db_session)
    def_repo = FeatureDefinitionRepository(db_session)
    v1 = await def_repo.get_by_name_and_version("sma_20", 1)
    assert v1 is not None
    v2 = await def_repo.create(name="sma_20", version=2, lookback_window=20)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)
    computed_at = _computed_at()

    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=v1.id,
        timestamp=ts,
        value=Decimal("150.00"),
        computed_at=computed_at,
    )
    await repo.upsert(
        symbol_id=symbol_id,
        feature_definition_id=v2.id,
        timestamp=ts,
        value=Decimal("151.00"),
        computed_at=computed_at,
    )

    listed = await repo.list_by_symbol(symbol_id)
    assert len(listed) == 2


@pytest.mark.asyncio
async def test_upsert_many_is_idempotent(db_session: AsyncSession) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)
    computed_at = _computed_at()
    row = {
        "symbol_id": symbol_id,
        "feature_definition_id": feature_definition_id,
        "timestamp": ts,
        "value": Decimal("150.25"),
        "computed_at": computed_at,
    }

    assert await repo.upsert_many([row]) == 1
    first_listed = await repo.list_by_symbol(symbol_id)
    assert len(first_listed) == 1
    first_id = first_listed[0].id

    assert (
        await repo.upsert_many(
            [
                {
                    **row,
                    "value": Decimal("160.00"),
                    "computed_at": datetime(2024, 6, 3, tzinfo=UTC),
                }
            ]
        )
        == 1
    )
    second_listed = await repo.list_by_symbol(symbol_id)
    assert len(second_listed) == 1
    assert second_listed[0].id == first_id
    assert second_listed[0].value == Decimal("160.00")


@pytest.mark.asyncio
async def test_upsert_many_empty_list_is_no_op(db_session: AsyncSession) -> None:
    repo = FeatureValueRepository(db_session)
    assert await repo.upsert_many([]) == 0


@pytest.mark.asyncio
async def test_upsert_many_dedupes_duplicate_keys_in_one_batch(
    db_session: AsyncSession,
) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    ts = datetime(2024, 1, 15, tzinfo=UTC)
    base = {
        "symbol_id": symbol_id,
        "feature_definition_id": feature_definition_id,
        "timestamp": ts,
        "computed_at": _computed_at(),
    }

    count = await repo.upsert_many(
        [
            {**base, "value": Decimal("150.00")},
            {**base, "value": Decimal("160.00")},
        ]
    )

    assert count == 1
    listed = await repo.list_by_symbol(symbol_id)
    assert len(listed) == 1
    assert listed[0].value == Decimal("160.00")


@pytest.mark.asyncio
async def test_upsert_many_chunks_large_batches(db_session: AsyncSession) -> None:
    symbol_id, feature_definition_id = await _create_symbol_and_feature(db_session)
    repo = FeatureValueRepository(db_session)
    computed_at = _computed_at()
    rows = [
        {
            "symbol_id": symbol_id,
            "feature_definition_id": feature_definition_id,
            "timestamp": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=offset),
            "value": Decimal("100.00") + Decimal(offset),
            "computed_at": computed_at,
        }
        for offset in range(1500)
    ]

    assert await repo.upsert_many(rows) == 1500
    assert len(await repo.list_by_symbol(symbol_id)) == 1500
