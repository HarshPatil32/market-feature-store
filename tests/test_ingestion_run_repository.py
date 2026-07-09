"""Tests for IngestionRunRepository CRUD operations."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import RunStatus
from backend.storage.repository import IngestionRunRepository, SymbolRepository


@pytest.mark.asyncio
async def test_create_defaults_to_pending_status_and_zero_counts(
    db_session: AsyncSession,
) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")

    assert created.run_type == "backfill"
    assert created.status == RunStatus.pending
    assert created.symbol_id is None
    assert created.fetched == 0
    assert created.inserted == 0
    assert created.failed == 0
    assert created.error_message is None
    assert created.started_at is None
    assert created.finished_at is None


@pytest.mark.asyncio
async def test_create_with_symbol_id(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    run_repo = IngestionRunRepository(db_session)
    symbol = await symbol_repo.create(symbol="AAPL")

    created = await run_repo.create(run_type="incremental", symbol_id=symbol.id)
    fetched = await run_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.symbol_id == symbol.id
    assert fetched.run_type == "incremental"


@pytest.mark.asyncio
async def test_create_with_invalid_symbol_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    repo = IngestionRunRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.create(run_type="backfill", symbol_id=999_999)


@pytest.mark.asyncio
async def test_get_by_id(db_session: AsyncSession) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")

    fetched = await repo.get_by_id(created.id)
    missing = await repo.get_by_id(999_999)

    assert fetched is not None
    assert fetched.id == created.id
    assert missing is None


@pytest.mark.asyncio
async def test_update_partial_update_preserves_other_fields(
    db_session: AsyncSession,
) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")
    started_at = datetime(2024, 1, 1, tzinfo=UTC)
    await repo.update(
        created.id,
        status=RunStatus.running,
        started_at=started_at,
    )

    updated = await repo.update(created.id, fetched=10, inserted=8, failed=2)

    assert updated is not None
    assert updated.status == RunStatus.running
    assert updated.started_at == started_at
    assert updated.fetched == 10
    assert updated.inserted == 8
    assert updated.failed == 2


@pytest.mark.asyncio
async def test_update_can_clear_error_message_with_none(
    db_session: AsyncSession,
) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")
    await repo.update(created.id, error_message="provider timeout")

    updated = await repo.update(created.id, error_message=None)

    assert updated is not None
    assert updated.error_message is None


@pytest.mark.asyncio
async def test_update_missing_run_returns_none(db_session: AsyncSession) -> None:
    repo = IngestionRunRepository(db_session)

    updated = await repo.update(999_999, status=RunStatus.failed)

    assert updated is None


@pytest.mark.asyncio
async def test_delete(db_session: AsyncSession) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")

    await repo.delete(created.id)
    fetched = await repo.get_by_id(created.id)

    assert fetched is None


@pytest.mark.asyncio
async def test_delete_missing_run_is_idempotent(db_session: AsyncSession) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")

    await repo.delete(created.id)
    await repo.delete(created.id)
    await repo.delete(999_999)

    assert await repo.get_by_id(created.id) is None


@pytest.mark.asyncio
async def test_invalid_status_value_rejected_by_check_constraint(
    db_session: AsyncSession,
) -> None:
    repo = IngestionRunRepository(db_session)
    created = await repo.create(run_type="backfill")

    with pytest.raises(DBAPIError):
        await db_session.execute(
            text("UPDATE ingestion_runs SET status = :status WHERE id = :id"),
            {"status": "bogus", "id": created.id},
        )
