"""Tests for reprocess-from-raw orchestration."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.reprocess import reprocess_from_raw
from backend.services.ingestion_runs import trigger_backfill
from backend.services.raw_market_data import persist_raw_fetch
from backend.services.symbols import add_symbol
from backend.storage.models import CheckSeverity, IngestionRun, RunStatus
from backend.storage.repository import (
    DataQualityCheckRepository,
    MarketBarRepository,
)
from backend.storage.schemas import SymbolCreate


def _make_bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
    close: Decimal = Decimal("103"),
) -> Bar:
    return Bar(
        symbol=symbol,
        ts=ts or datetime(2024, 1, 2, tzinfo=UTC),
        timeframe="1d",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=close,
        volume=Decimal("1000"),
        source="fake",
    )


async def _seed_raw_row(
    session: AsyncSession,
    *,
    symbol_id: int,
    run_id: int,
    payload: dict[str, Any] | None = None,
) -> int:
    row = await persist_raw_fetch(
        session,
        run_id=run_id,
        symbol_id=symbol_id,
        source="fake",
        response_payload=payload or {"bars": []},
    )
    return row.id


@pytest.mark.asyncio
async def test_reprocess_upserts_bars_from_stored_raw(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    source_run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(
        db_session,
        symbol_id=symbol.id,
        run_id=source_run.id,
        payload={"bars": [{"timestamp": "2024-01-02"}]},
    )

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert result.run_type == "reprocess"
    assert result.status == RunStatus.succeeded
    assert result.fetched == 1
    assert result.inserted == 1
    assert result.failed == 0
    assert len(bars) == 1
    assert bars[0].close == Decimal("103")


@pytest.mark.asyncio
async def test_reprocess_is_idempotent(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    source_run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=source_run.id)

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL", close=Decimal("103"))]

    await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
    )

    def normalize_updated(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL", close=Decimal("110"))]

    await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize_updated,
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert len(bars) == 1
    assert bars[0].close == Decimal("110")


@pytest.mark.asyncio
async def test_reprocess_filters_by_run_id(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run_a = await trigger_backfill(db_session, "AAPL")
    run_b = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run_a.id)
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run_b.id)
    normalize_calls: list[dict[str, object]] = []

    def normalize(payload: dict[str, object]) -> list[Bar]:
        normalize_calls.append(payload)
        return [_make_bar(symbol="AAPL")]

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
        run_id=run_a.id,
    )

    assert result.fetched == 1
    assert result.inserted == 1
    assert len(normalize_calls) == 1


@pytest.mark.asyncio
async def test_reprocess_filters_by_created_at_range(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    early_id = await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run.id)
    late_id = await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run.id)

    await db_session.execute(
        text("UPDATE raw_market_data SET created_at = :ts WHERE id = :id"),
        {"ts": datetime(2024, 1, 1, tzinfo=UTC), "id": early_id},
    )
    await db_session.execute(
        text("UPDATE raw_market_data SET created_at = :ts WHERE id = :id"),
        {"ts": datetime(2024, 6, 1, tzinfo=UTC), "id": late_id},
    )
    await db_session.flush()

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
        start=datetime(2024, 2, 1, tzinfo=UTC),
        end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    assert result.fetched == 1
    assert result.inserted == 1


@pytest.mark.asyncio
async def test_reprocess_validate_creates_new_quality_checks(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    source_run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=source_run.id)

    await DataQualityCheckRepository(db_session).create(
        check_name="old_check",
        severity=CheckSeverity.warning,
        run_id=source_run.id,
        symbol_id=symbol.id,
        message="from original run",
    )

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    def validate(_bars: Sequence[Bar]) -> list[dict[str, Any]]:
        return [
            {
                "check_name": "negative_prices",
                "severity": CheckSeverity.error,
                "message": "found during reprocess",
            }
        ]

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
        validate=validate,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_symbol(symbol.id)
    reprocess_checks = [c for c in checks if c.run_id == result.id]
    original_checks = [c for c in checks if c.run_id == source_run.id]

    assert len(reprocess_checks) == 1
    assert reprocess_checks[0].check_name == "negative_prices"
    assert reprocess_checks[0].message == "found during reprocess"
    assert len(original_checks) == 1
    assert original_checks[0].check_name == "old_check"


@pytest.mark.asyncio
async def test_reprocess_row_failure_does_not_abort_batch(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(
        db_session,
        symbol_id=symbol.id,
        run_id=run.id,
        payload={"label": "bad"},
    )
    await _seed_raw_row(
        db_session,
        symbol_id=symbol.id,
        run_id=run.id,
        payload={"label": "good"},
    )
    call_count = 0

    def normalize(payload: dict[str, object]) -> list[Bar]:
        nonlocal call_count
        call_count += 1
        if payload.get("label") == "bad":
            raise ValueError("boom")
        return [_make_bar(symbol="AAPL")]

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert call_count == 2
    assert result.status == RunStatus.succeeded
    assert result.fetched == 2
    assert result.inserted == 1
    assert result.failed == 1
    assert result.error_message == "1 raw row(s) failed to reprocess"
    assert len(bars) == 1


@pytest.mark.asyncio
async def test_reprocess_all_rows_failing_marks_run_failed(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run.id)
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run.id)

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        raise ValueError("boom")

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
    )

    assert result.status == RunStatus.failed
    assert result.fetched == 2
    assert result.inserted == 0
    assert result.failed == 2
    assert result.error_message == "all 2 raw row(s) failed to reprocess"


@pytest.mark.asyncio
async def test_reprocess_no_matching_rows_raises_without_creating_run(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return []

    before = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "reprocess")
    )

    with pytest.raises(ValueError, match="no raw market data rows"):
        await reprocess_from_raw(
            db_session,
            symbol_id=symbol.id,
            normalize=normalize,
        )

    after = await db_session.scalar(
        select(func.count())
        .select_from(IngestionRun)
        .where(IngestionRun.run_type == "reprocess")
    )
    assert before == after


@pytest.mark.asyncio
async def test_reprocess_validate_failure_does_not_count_inserted_bars(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    await _seed_raw_row(db_session, symbol_id=symbol.id, run_id=run.id)

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    def validate(_bars: Sequence[Bar]) -> list[dict[str, Any]]:
        raise ValueError("validation failed")

    result = await reprocess_from_raw(
        db_session,
        symbol_id=symbol.id,
        normalize=normalize,
        validate=validate,
    )

    bars = await MarketBarRepository(db_session).list_by_symbol(
        symbol.id,
        timeframe="1d",
    )

    assert result.status == RunStatus.failed
    assert result.inserted == 0
    assert result.failed == 1
    assert len(bars) == 0


@pytest.mark.asyncio
async def test_reprocess_rejects_async_normalize(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    async def normalize(_payload: dict[str, object]) -> list[Bar]:
        return []

    with pytest.raises(TypeError, match="synchronous callable"):
        await reprocess_from_raw(
            db_session,
            symbol_id=symbol.id,
            normalize=normalize,
        )


@pytest.mark.asyncio
async def test_reprocess_rejects_async_validate(db_session: AsyncSession) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return []

    async def validate(_bars: Sequence[Bar]) -> list[dict[str, Any]]:
        return []

    with pytest.raises(TypeError, match="synchronous callable"):
        await reprocess_from_raw(
            db_session,
            symbol_id=symbol.id,
            normalize=normalize,
            validate=validate,  # type: ignore[arg-type]
        )
