"""Tests for data quality check persistence service."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.services.data_quality import persist_check_results
from backend.services.ingestion_runs import trigger_backfill
from backend.services.symbols import add_symbol
from backend.storage.models import CheckSeverity
from backend.storage.repository import DataQualityCheckRepository
from backend.storage.schemas import SymbolCreate
from backend.validation.validator import (
    QualityCheckResult,
    check_high_lt_low,
    check_negative_prices,
    check_zero_volume,
)


def _make_bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("99"),
    open_: Decimal = Decimal("100"),
    close: Decimal = Decimal("103"),
    volume: Decimal = Decimal("1000"),
) -> Bar:
    return Bar.model_construct(
        symbol=symbol,
        ts=ts or datetime(2024, 1, 2, tzinfo=UTC),
        timeframe="1d",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="test",
    )


@pytest.mark.asyncio
async def test_persist_check_results_returns_empty_list_for_empty_results(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))

    created = await persist_check_results(
        db_session,
        [],
        symbol_id=symbol.id,
    )
    listed = await DataQualityCheckRepository(db_session).list_by_symbol(symbol.id)

    assert created == []
    assert listed == []


@pytest.mark.asyncio
async def test_persist_check_results_inserts_multiple_results_for_run(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, symbol.symbol)

    bars = [
        _make_bar(high=Decimal("98"), low=Decimal("99")),
        Bar.model_construct(
            symbol="AAPL",
            ts=datetime(2024, 1, 3, tzinfo=UTC),
            timeframe="1d",
            open=Decimal("-1"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=Decimal("1000"),
            source="test",
        ),
        _make_bar(ts=datetime(2024, 1, 4, tzinfo=UTC), volume=Decimal("0")),
    ]
    high_lt_low = check_high_lt_low(bars[0])
    negative_prices = check_negative_prices(bars[1])
    zero_volume = check_zero_volume(bars[2])
    assert high_lt_low is not None
    assert negative_prices is not None
    assert zero_volume is not None
    results = [high_lt_low, negative_prices, zero_volume]

    created = await persist_check_results(
        db_session,
        results,
        symbol_id=symbol.id,
        run_id=run.id,
    )
    listed = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert len(created) == 3
    assert len(listed) == 3
    assert {row.check_name for row in listed} == {
        "high_lt_low",
        "negative_prices",
        "zero_volume",
    }


@pytest.mark.asyncio
async def test_persist_check_results_covers_mixed_severities(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    results = [
        QualityCheckResult(
            symbol="AAPL",
            check="info_check",
            severity=CheckSeverity.info,
            message="Informational",
        ),
        QualityCheckResult(
            symbol="AAPL",
            check="warning_check",
            severity=CheckSeverity.warning,
            message="Warning",
        ),
        QualityCheckResult(
            symbol="AAPL",
            check="error_check",
            severity=CheckSeverity.error,
            message="Error",
        ),
    ]

    created = await persist_check_results(
        db_session,
        results,
        symbol_id=symbol.id,
    )

    assert {row.severity for row in created} == {
        CheckSeverity.info,
        CheckSeverity.warning,
        CheckSeverity.error,
    }


@pytest.mark.asyncio
async def test_persist_check_results_run_id_override_takes_precedence(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run_a = await trigger_backfill(db_session, symbol.symbol)
    run_b = await trigger_backfill(db_session, symbol.symbol)
    result = QualityCheckResult(
        symbol="AAPL",
        check="test_check",
        severity=CheckSeverity.error,
        run_id=run_a.id,
    )

    created = await persist_check_results(
        db_session,
        [result],
        symbol_id=symbol.id,
        run_id=run_b.id,
    )

    assert len(created) == 1
    assert created[0].run_id == run_b.id


@pytest.mark.asyncio
async def test_persist_check_results_falls_back_to_result_run_id(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, symbol.symbol)
    result = QualityCheckResult(
        symbol="AAPL",
        check="test_check",
        severity=CheckSeverity.warning,
        run_id=run.id,
    )

    created = await persist_check_results(
        db_session,
        [result],
        symbol_id=symbol.id,
    )

    assert len(created) == 1
    assert created[0].run_id == run.id


@pytest.mark.asyncio
async def test_persist_check_results_stamps_all_results_with_given_symbol_id(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    results = [
        QualityCheckResult(
            symbol="MSFT",
            check="check_one",
            severity=CheckSeverity.error,
        ),
        QualityCheckResult(
            symbol="GOOG",
            check="check_two",
            severity=CheckSeverity.warning,
        ),
        QualityCheckResult(
            symbol="TSLA",
            check="check_three",
            severity=CheckSeverity.info,
        ),
    ]

    created = await persist_check_results(
        db_session,
        results,
        symbol_id=symbol.id,
    )

    assert len(created) == 3
    assert all(row.symbol_id == symbol.id for row in created)
