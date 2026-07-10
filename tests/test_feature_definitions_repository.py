"""Tests for FeatureDefinitionRepository CRUD operations."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import FeatureDefinitionRepository


@pytest.mark.asyncio
async def test_create_inserts_row_with_all_fields(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)

    created = await repo.create(
        name="sma_20",
        version=1,
        lookback_window=20,
        active=False,
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.name == "sma_20"
    assert fetched.version == 1
    assert fetched.lookback_window == 20
    assert fetched.active is False
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


@pytest.mark.asyncio
async def test_create_defaults_active_to_true(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)

    created = await repo.create(
        name="rsi_14",
        version=1,
        lookback_window=14,
    )
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.active is True


@pytest.mark.asyncio
async def test_unique_constraint_violation_on_duplicate_name_version(
    db_session: AsyncSession,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma", version=1, lookback_window=20)

    with pytest.raises(IntegrityError):
        await repo.create(name="sma", version=1, lookback_window=30)


@pytest.mark.asyncio
async def test_same_name_with_different_version_succeeds(
    db_session: AsyncSession,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma", version=1, lookback_window=20)

    different_version = await repo.create(name="sma", version=2, lookback_window=20)

    assert different_version.version == 2


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing_row(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)

    assert await repo.get_by_id(999_999) is None


@pytest.mark.asyncio
async def test_get_by_name_and_version_returns_matching_row(
    db_session: AsyncSession,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)

    fetched = await repo.get_by_name_and_version("sma_20", 1)
    missing = await repo.get_by_name_and_version("sma_20", 2)

    assert fetched is not None
    assert fetched.name == "sma_20"
    assert fetched.version == 1
    assert missing is None


@pytest.mark.asyncio
async def test_list_returns_all_ordered_by_name_and_version(
    db_session: AsyncSession,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="rsi_14", version=1, lookback_window=14)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    await repo.create(name="sma_20", version=2, lookback_window=20)

    listed = await repo.list()

    assert len(listed) == 3
    assert [(row.name, row.version) for row in listed] == [
        ("rsi_14", 1),
        ("sma_20", 1),
        ("sma_20", 2),
    ]


@pytest.mark.asyncio
async def test_list_active_only_filters_inactive(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=20)
    inactive = await repo.create(
        name="rsi_14",
        version=1,
        lookback_window=14,
        active=False,
    )

    all_features = await repo.list()
    active_features = await repo.list(active_only=True)

    assert len(all_features) == 2
    assert len(active_features) == 1
    assert active_features[0].name == "sma_20"
    assert inactive.active is False


@pytest.mark.asyncio
async def test_set_active_toggles_flag(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)
    created = await repo.create(name="sma_20", version=1, lookback_window=20)

    deactivated = await repo.set_active(created.id, active=False)

    assert deactivated is not None
    assert deactivated.active is False

    reactivated = await repo.set_active(created.id, active=True)

    assert reactivated is not None
    assert reactivated.active is True


@pytest.mark.asyncio
async def test_set_active_returns_none_for_missing_row(
    db_session: AsyncSession,
) -> None:
    repo = FeatureDefinitionRepository(db_session)

    updated = await repo.set_active(999_999, active=False)

    assert updated is None


@pytest.mark.asyncio
async def test_delete_removes_row(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)
    created = await repo.create(name="sma_20", version=1, lookback_window=20)

    await repo.delete(created.id)
    fetched = await repo.get_by_id(created.id)

    assert fetched is None


@pytest.mark.asyncio
async def test_delete_missing_row_is_idempotent(db_session: AsyncSession) -> None:
    repo = FeatureDefinitionRepository(db_session)
    created = await repo.create(name="sma_20", version=1, lookback_window=20)

    await repo.delete(created.id)
    await repo.delete(created.id)
    await repo.delete(999_999)

    assert await repo.get_by_id(created.id) is None


@pytest.mark.parametrize("lookback_window", [0, -1])
@pytest.mark.asyncio
async def test_create_with_non_positive_lookback_window_raises_integrity_error(
    db_session: AsyncSession,
    lookback_window: int,
) -> None:
    repo = FeatureDefinitionRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.create(
            name="sma_20",
            version=1,
            lookback_window=lookback_window,
        )
