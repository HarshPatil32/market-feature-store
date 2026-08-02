"""Tests for data quality validation checks."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.bar import Bar
from backend.storage.models import CheckSeverity
from backend.validation.validator import (
    QualityCheckResult,
    check_high_lt_low,
    check_zero_volume,
    run_checks,
)


def _make_bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
    open_: Decimal = Decimal("100"),
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("99"),
    close: Decimal = Decimal("103"),
    volume: Decimal = Decimal("1000"),
) -> Bar:
    return Bar(
        symbol=symbol,
        ts=ts or datetime(2024, 1, 2, tzinfo=UTC),
        timeframe="1d",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="fake",
    )


def test_check_high_lt_low_returns_none_for_valid_bar() -> None:
    assert check_high_lt_low(_make_bar()) is None


def test_check_high_lt_low_returns_none_when_high_equals_low() -> None:
    assert check_high_lt_low(_make_bar(high=Decimal("100"), low=Decimal("100"))) is None


def test_check_high_lt_low_returns_result_when_high_below_low() -> None:
    bar = _make_bar(high=Decimal("98"), low=Decimal("99"))
    result = check_high_lt_low(bar)

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.check == "high_lt_low"
    assert result.severity == CheckSeverity.error
    assert result.affected_ts == bar.ts
    assert "98" in (result.message or "")


def test_check_zero_volume_returns_none_for_nonzero_volume() -> None:
    assert check_zero_volume(_make_bar()) is None


def test_check_zero_volume_returns_warning_for_zero_volume() -> None:
    bar = _make_bar(volume=Decimal("0"))
    result = check_zero_volume(bar)

    assert result is not None
    assert result.check == "zero_volume"
    assert result.severity == CheckSeverity.warning
    assert result.affected_ts == bar.ts


def test_run_checks_collects_results_from_multiple_bars() -> None:
    bars = [
        _make_bar(),
        _make_bar(ts=datetime(2024, 1, 3, tzinfo=UTC), volume=Decimal("0")),
        _make_bar(
            ts=datetime(2024, 1, 4, tzinfo=UTC),
            high=Decimal("98"),
            low=Decimal("99"),
        ),
    ]

    results = run_checks(bars)

    assert len(results) == 2
    assert {result.check for result in results} == {"zero_volume", "high_lt_low"}


def test_run_checks_accepts_custom_check_list() -> None:
    bars = [_make_bar(volume=Decimal("0"))]

    results = run_checks(bars, checks=[check_zero_volume])

    assert len(results) == 1
    assert results[0].check == "zero_volume"


def test_run_checks_returns_empty_list_for_no_bars() -> None:
    assert run_checks([]) == []


def test_quality_check_result_is_immutable() -> None:
    result = QualityCheckResult(
        symbol="AAPL",
        check="test_check",
        severity=CheckSeverity.info,
    )

    with pytest.raises(ValidationError):
        result.check = "other"  # type: ignore[misc]


def test_to_check_row_maps_fields_for_repository() -> None:
    affected_ts = datetime(2024, 1, 2, tzinfo=UTC)
    result = QualityCheckResult(
        symbol="AAPL",
        check="high_lt_low",
        severity=CheckSeverity.error,
        message="High less than low",
        affected_ts=affected_ts,
    )

    row = result.to_check_row(symbol_id=42, run_id=7)

    assert row == {
        "check_name": "high_lt_low",
        "severity": CheckSeverity.error,
        "message": "High less than low",
        "affected_timestamp": affected_ts,
        "run_id": 7,
        "symbol_id": 42,
    }


def test_to_check_row_uses_result_run_id_when_not_overridden() -> None:
    result = QualityCheckResult(
        symbol="AAPL",
        check="test_check",
        severity=CheckSeverity.warning,
        run_id=99,
    )

    row = result.to_check_row(symbol_id=1)

    assert row["run_id"] == 99
