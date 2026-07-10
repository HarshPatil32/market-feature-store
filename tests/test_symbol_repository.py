"""Tests for SymbolRepository CRUD operations."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import SymbolRepository


@pytest.mark.asyncio
async def test_create_and_get_by_id(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="AAPL")

    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.symbol == "AAPL"
    assert fetched.asset_type == "equity"
    assert fetched.active is True
    assert fetched.coverage_start is None
    assert fetched.coverage_end is None
    assert fetched.last_ingested_at is None


@pytest.mark.asyncio
async def test_get_by_symbol(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    await repo.create(symbol="MSFT")

    fetched = await repo.get_by_symbol("MSFT")
    missing = await repo.get_by_symbol("UNKNOWN")

    assert fetched is not None
    assert fetched.symbol == "MSFT"
    assert missing is None


@pytest.mark.asyncio
async def test_duplicate_symbol_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)
    await repo.create(symbol="GOOG")

    with pytest.raises(IntegrityError):
        await repo.create(symbol="GOOG")


@pytest.mark.asyncio
async def test_list_active_only(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    await repo.create(symbol="AAPL")
    inactive = await repo.create(symbol="MSFT")
    await repo.set_active(inactive.id, active=False)

    all_symbols = await repo.list()
    active_symbols = await repo.list(active_only=True)

    assert {row.symbol for row in all_symbols} == {"AAPL", "MSFT"}
    assert {row.symbol for row in active_symbols} == {"AAPL"}


@pytest.mark.asyncio
async def test_update_coverage_partial_update_preserves_other_fields(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="META")
    coverage_start = datetime(2024, 1, 1, tzinfo=UTC)
    coverage_end = datetime(2024, 12, 31, tzinfo=UTC)
    await repo.update_coverage(
        created.id,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
    )

    last_ingested_at = datetime(2024, 12, 31, 12, 0, tzinfo=UTC)
    updated = await repo.update_coverage(
        created.id,
        last_ingested_at=last_ingested_at,
    )

    assert updated is not None
    assert updated.coverage_start == coverage_start
    assert updated.coverage_end == coverage_end
    assert updated.last_ingested_at == last_ingested_at


@pytest.mark.asyncio
async def test_update_coverage_can_clear_field_with_none(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="NFLX")
    coverage_start = datetime(2024, 1, 1, tzinfo=UTC)
    await repo.update_coverage(created.id, coverage_start=coverage_start)

    updated = await repo.update_coverage(created.id, coverage_start=None)

    assert updated is not None
    assert updated.coverage_start is None


@pytest.mark.asyncio
async def test_update_coverage(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="TSLA")
    coverage_start = datetime(2024, 1, 1, tzinfo=UTC)
    coverage_end = datetime(2024, 12, 31, tzinfo=UTC)
    last_ingested_at = datetime(2024, 12, 31, 12, 0, tzinfo=UTC)

    updated = await repo.update_coverage(
        created.id,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        last_ingested_at=last_ingested_at,
    )

    assert updated is not None
    assert updated.coverage_start == coverage_start
    assert updated.coverage_end == coverage_end
    assert updated.last_ingested_at == last_ingested_at


@pytest.mark.asyncio
async def test_update_coverage_missing_symbol_returns_none(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)

    updated = await repo.update_coverage(
        999_999,
        last_ingested_at=datetime(2024, 12, 31, tzinfo=UTC),
    )

    assert updated is None


@pytest.mark.asyncio
async def test_set_active_missing_symbol_returns_none(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)

    updated = await repo.set_active(999_999, active=False)

    assert updated is None


@pytest.mark.asyncio
async def test_set_active_round_trip(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="NVDA")

    deactivated = await repo.set_active(created.id, active=False)

    assert deactivated is not None
    assert deactivated.active is False

    reactivated = await repo.set_active(created.id, active=True)

    assert reactivated is not None
    assert reactivated.active is True


@pytest.mark.asyncio
async def test_delete(db_session: AsyncSession) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="AMD")

    await repo.delete(created.id)
    fetched = await repo.get_by_id(created.id)

    assert fetched is None


@pytest.mark.asyncio
async def test_delete_missing_symbol_is_idempotent(
    db_session: AsyncSession,
) -> None:
    repo = SymbolRepository(db_session)
    created = await repo.create(symbol="INTC")

    await repo.delete(created.id)
    await repo.delete(created.id)
    await repo.delete(999_999)

    assert await repo.get_by_id(created.id) is None
