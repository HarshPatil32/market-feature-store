"""Tests for DataQualityCheckRepository CRUD and listing operations."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import CheckSeverity
from backend.storage.repository import (
    DataQualityCheckRepository,
    IngestionRunRepository,
    SymbolRepository,
)


@pytest.mark.asyncio
async def test_create_inserts_row_with_all_fields(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    run_repo = IngestionRunRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    run = await run_repo.create(run_type="backfill", symbol_id=symbol.id)
    affected_ts = datetime(2024, 1, 2, tzinfo=UTC)

    created = await check_repo.create(
        check_name="negative_prices",
        severity=CheckSeverity.error,
        run_id=run.id,
        symbol_id=symbol.id,
        message="Open price is negative",
        affected_timestamp=affected_ts,
    )
    fetched = await check_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.check_name == "negative_prices"
    assert fetched.severity == CheckSeverity.error
    assert fetched.run_id == run.id
    assert fetched.symbol_id == symbol.id
    assert fetched.message == "Open price is negative"
    assert fetched.affected_timestamp == affected_ts
    assert fetched.created_at is not None


@pytest.mark.asyncio
async def test_create_with_minimal_fields_leaves_optional_columns_none(
    db_session: AsyncSession,
) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    created = await check_repo.create(
        check_name="empty_response",
        severity=CheckSeverity.warning,
    )
    fetched = await check_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id is None
    assert fetched.symbol_id is None
    assert fetched.message is None
    assert fetched.affected_timestamp is None


@pytest.mark.parametrize(
    "severity",
    [CheckSeverity.info, CheckSeverity.warning, CheckSeverity.error],
)
@pytest.mark.asyncio
async def test_severity_persists_for_each_enum_value(
    db_session: AsyncSession,
    severity: CheckSeverity,
) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    created = await check_repo.create(
        check_name="test_check",
        severity=severity,
    )
    fetched = await check_repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.severity == severity


@pytest.mark.asyncio
async def test_bulk_create_inserts_multiple_rows_for_run(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    run_repo = IngestionRunRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    run = await run_repo.create(run_type="backfill", symbol_id=symbol.id)
    rows = [
        {
            "check_name": "negative_prices",
            "severity": CheckSeverity.error,
            "run_id": run.id,
            "symbol_id": symbol.id,
            "message": "Negative open",
        },
        {
            "check_name": "high_lt_low",
            "severity": CheckSeverity.error,
            "run_id": run.id,
            "symbol_id": symbol.id,
            "message": "High less than low",
        },
        {
            "check_name": "stale_symbol",
            "severity": CheckSeverity.warning,
            "run_id": run.id,
            "symbol_id": symbol.id,
        },
    ]

    created = await check_repo.bulk_create(rows)
    listed = await check_repo.list_by_run(run.id)

    assert len(created) == 3
    assert len(listed) == 3
    assert {row.check_name for row in listed} == {
        "negative_prices",
        "high_lt_low",
        "stale_symbol",
    }
    assert [row.check_name for row in listed] == [
        "negative_prices",
        "high_lt_low",
        "stale_symbol",
    ]


@pytest.mark.asyncio
async def test_bulk_create_with_invalid_run_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    with pytest.raises(IntegrityError):
        await check_repo.bulk_create(
            [
                {
                    "check_name": "test_check",
                    "severity": CheckSeverity.error,
                    "run_id": 999_999,
                }
            ]
        )


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing_row(db_session: AsyncSession) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    assert await check_repo.get_by_id(999_999) is None


@pytest.mark.asyncio
async def test_list_by_run_returns_only_matching_run(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    run_repo = IngestionRunRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    run_a = await run_repo.create(run_type="backfill", symbol_id=symbol.id)
    run_b = await run_repo.create(run_type="incremental", symbol_id=symbol.id)

    await check_repo.create(
        check_name="check_a",
        severity=CheckSeverity.error,
        run_id=run_a.id,
        symbol_id=symbol.id,
    )
    await check_repo.create(
        check_name="check_b",
        severity=CheckSeverity.warning,
        run_id=run_b.id,
        symbol_id=symbol.id,
    )

    listed = await check_repo.list_by_run(run_a.id)

    assert len(listed) == 1
    assert listed[0].check_name == "check_a"
    assert listed[0].run_id == run_a.id


@pytest.mark.asyncio
async def test_list_by_symbol_filters_by_severity(db_session: AsyncSession) -> None:
    symbol_repo = SymbolRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    symbol_a = await symbol_repo.create(symbol="AAPL")
    symbol_b = await symbol_repo.create(symbol="MSFT")

    await check_repo.create(
        check_name="error_check",
        severity=CheckSeverity.error,
        symbol_id=symbol_a.id,
    )
    await check_repo.create(
        check_name="warning_check",
        severity=CheckSeverity.warning,
        symbol_id=symbol_a.id,
    )
    await check_repo.create(
        check_name="other_symbol_check",
        severity=CheckSeverity.error,
        symbol_id=symbol_b.id,
    )

    all_checks = await check_repo.list_by_symbol(symbol_a.id)
    error_checks = await check_repo.list_by_symbol(
        symbol_a.id,
        severity=CheckSeverity.error,
    )

    assert len(all_checks) == 2
    assert {row.check_name for row in all_checks} == {"error_check", "warning_check"}
    assert len(error_checks) == 1
    assert error_checks[0].check_name == "error_check"


@pytest.mark.asyncio
async def test_create_with_invalid_symbol_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    with pytest.raises(IntegrityError):
        await check_repo.create(
            check_name="test_check",
            severity=CheckSeverity.error,
            symbol_id=999_999,
        )


@pytest.mark.asyncio
async def test_create_with_invalid_run_id_raises_integrity_error(
    db_session: AsyncSession,
) -> None:
    check_repo = DataQualityCheckRepository(db_session)

    with pytest.raises(IntegrityError):
        await check_repo.create(
            check_name="test_check",
            severity=CheckSeverity.error,
            run_id=999_999,
        )


@pytest.mark.asyncio
async def test_symbol_deletion_sets_symbol_id_null_when_checks_exist(
    db_session: AsyncSession,
) -> None:
    symbol_repo = SymbolRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    symbol = await symbol_repo.create(symbol="AAPL")
    check_row = await check_repo.create(
        check_name="test_check",
        severity=CheckSeverity.warning,
        symbol_id=symbol.id,
    )
    check_row_id = check_row.id

    await symbol_repo.delete(symbol.id)
    db_session.expire_all()
    fetched = await check_repo.get_by_id(check_row_id)

    assert fetched is not None
    assert fetched.symbol_id is None


@pytest.mark.asyncio
async def test_run_deletion_sets_run_id_null_when_checks_exist(
    db_session: AsyncSession,
) -> None:
    run_repo = IngestionRunRepository(db_session)
    check_repo = DataQualityCheckRepository(db_session)

    run = await run_repo.create(run_type="backfill")
    check_row = await check_repo.create(
        check_name="test_check",
        severity=CheckSeverity.error,
        run_id=run.id,
    )
    check_row_id = check_row.id

    await run_repo.delete(run.id)
    db_session.expire_all()
    fetched = await check_repo.get_by_id(check_row_id)

    assert fetched is not None
    assert fetched.run_id is None
