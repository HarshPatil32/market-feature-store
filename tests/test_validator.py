"""Tests for data quality validation checks."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.bar import Bar
from backend.storage.models import CheckSeverity
from backend.validation.validator import (
    QualityCheckResult,
    check_duplicate_bars,
    check_high_lt_low,
    check_missing_timestamps,
    check_negative_prices,
    check_open_close_outside_range,
    check_price_jumps,
    check_stale_symbol,
    check_zero_volume,
    make_check_price_jump,
    make_check_stale_symbol,
    make_check_zero_volume,
    run_checks,
)


def _make_bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
    timeframe: str = "1d",
    open_: Decimal = Decimal("100"),
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("99"),
    close: Decimal = Decimal("103"),
    volume: Decimal = Decimal("1000"),
    skip_validation: bool = False,
) -> Bar:
    bar_ts = ts or datetime(2024, 1, 2, tzinfo=UTC)
    if skip_validation:
        return Bar.model_construct(
            symbol=symbol,
            ts=bar_ts,
            timeframe=timeframe,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            source="fake",
        )
    return Bar(
        symbol=symbol,
        ts=bar_ts,
        timeframe=timeframe,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="fake",
    )


class _FakeMarketBarRepository:
    def __init__(self, existing: set[datetime]) -> None:
        self._existing = existing
        self.calls: list[tuple[int, str, list[datetime]]] = []

    async def get_existing_timestamps(
        self,
        symbol_id: int,
        *,
        timeframe: str,
        timestamps: Sequence[datetime],
    ) -> set[datetime]:
        self.calls.append((symbol_id, timeframe, list(timestamps)))
        return {ts for ts in timestamps if ts in self._existing}


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
    message = result.message or ""
    assert "98" in message
    assert "99" in message


def test_check_open_close_outside_range_returns_none_for_valid_bar() -> None:
    assert check_open_close_outside_range(_make_bar()) is None


def test_check_open_close_outside_range_returns_none_when_open_equals_low() -> None:
    assert (
        check_open_close_outside_range(
            _make_bar(open_=Decimal("99"), low=Decimal("99"))
        )
        is None
    )


def test_check_open_close_outside_range_returns_none_when_close_equals_high() -> None:
    assert (
        check_open_close_outside_range(
            _make_bar(close=Decimal("105"), high=Decimal("105"))
        )
        is None
    )


def test_check_open_close_outside_range_returns_none_when_open_equals_high() -> None:
    assert (
        check_open_close_outside_range(
            _make_bar(open_=Decimal("105"), high=Decimal("105"))
        )
        is None
    )


def test_check_open_close_outside_range_returns_none_when_close_equals_low() -> None:
    assert (
        check_open_close_outside_range(
            _make_bar(close=Decimal("99"), low=Decimal("99"))
        )
        is None
    )


def test_check_open_close_outside_range_returns_error_when_open_below_low() -> None:
    bar = _make_bar(open_=Decimal("98"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.check == "open_close_outside_range"
    assert result.severity == CheckSeverity.error
    assert result.affected_ts == bar.ts
    assert "open=98" in (result.message or "")


def test_check_open_close_outside_range_returns_error_when_open_above_high() -> None:
    bar = _make_bar(open_=Decimal("106"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    assert result.check == "open_close_outside_range"
    assert result.severity == CheckSeverity.error
    assert "open=106" in (result.message or "")


def test_check_open_close_outside_range_returns_error_when_close_below_low() -> None:
    bar = _make_bar(close=Decimal("98"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    assert result.check == "open_close_outside_range"
    assert result.severity == CheckSeverity.error
    assert "close=98" in (result.message or "")


def test_check_open_close_outside_range_returns_error_when_close_above_high() -> None:
    bar = _make_bar(close=Decimal("106"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    assert result.check == "open_close_outside_range"
    assert result.severity == CheckSeverity.error
    assert "close=106" in (result.message or "")


def test_check_open_close_outside_range_lists_both_fields_when_both_violate() -> None:
    bar = _make_bar(open_=Decimal("98"), close=Decimal("106"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    message = result.message or ""
    assert "open=98" in message
    assert "close=106" in message


def test_check_open_close_outside_range_fires_even_when_high_lt_low() -> None:
    bar = _make_bar(high=Decimal("98"), low=Decimal("99"), open_=Decimal("100"))
    result = check_open_close_outside_range(bar)

    assert result is not None
    assert result.check == "open_close_outside_range"
    assert "open=100" in (result.message or "")


def test_check_zero_volume_returns_none_for_nonzero_volume() -> None:
    assert check_zero_volume(_make_bar()) is None


def test_check_zero_volume_returns_warning_for_zero_volume() -> None:
    bar = _make_bar(volume=Decimal("0"))
    result = check_zero_volume(bar)

    assert result is not None
    assert result.check == "zero_volume"
    assert result.severity == CheckSeverity.warning
    assert result.affected_ts == bar.ts


def test_check_zero_volume_returns_error_for_negative_volume() -> None:
    bar = _make_bar(volume=Decimal("-1"), skip_validation=True)
    result = check_zero_volume(bar)

    assert result is not None
    assert result.check == "negative_volume"
    assert result.severity == CheckSeverity.error
    assert result.affected_ts == bar.ts
    assert "-1" in (result.message or "")


def test_make_check_zero_volume_uses_custom_default_severity() -> None:
    check = make_check_zero_volume(default_severity=CheckSeverity.info)
    result = check(_make_bar(volume=Decimal("0")))

    assert result is not None
    assert result.check == "zero_volume"
    assert result.severity == CheckSeverity.info


def test_make_check_zero_volume_applies_symbol_timeframe_override() -> None:
    check = make_check_zero_volume(
        overrides={("AAPL", "1d"): CheckSeverity.error},
    )
    result = check(_make_bar(volume=Decimal("0")))

    assert result is not None
    assert result.check == "zero_volume"
    assert result.severity == CheckSeverity.error


def test_make_check_zero_volume_override_suppresses_check() -> None:
    check = make_check_zero_volume(overrides={("AAPL", "1d"): None})
    assert check(_make_bar(volume=Decimal("0"))) is None


def test_make_check_zero_volume_override_does_not_match_other_timeframe() -> None:
    check = make_check_zero_volume(
        overrides={("AAPL", "1h"): CheckSeverity.error},
        default_severity=CheckSeverity.warning,
    )
    result = check(_make_bar(volume=Decimal("0"), timeframe="1d"))

    assert result is not None
    assert result.severity == CheckSeverity.warning


def test_make_check_zero_volume_override_does_not_match_other_symbol() -> None:
    check = make_check_zero_volume(
        overrides={("MSFT", "1d"): None},
        default_severity=CheckSeverity.warning,
    )
    result = check(_make_bar(volume=Decimal("0")))

    assert result is not None
    assert result.severity == CheckSeverity.warning


def test_check_zero_volume_negative_volume_not_suppressable_by_override() -> None:
    check = make_check_zero_volume(overrides={("AAPL", "1d"): None})
    bar = _make_bar(volume=Decimal("-1"), skip_validation=True)
    result = check(bar)

    assert result is not None
    assert result.check == "negative_volume"
    assert result.severity == CheckSeverity.error


def test_make_check_zero_volume_rejects_empty_override_key() -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        make_check_zero_volume(overrides={("", "1d"): CheckSeverity.error})

    with pytest.raises(ValueError, match="timeframe must not be empty"):
        make_check_zero_volume(overrides={("AAPL", ""): CheckSeverity.error})


def test_check_negative_prices_returns_none_for_valid_bar() -> None:
    assert check_negative_prices(_make_bar()) is None


def test_check_negative_prices_returns_error_when_price_is_negative() -> None:
    bar = _make_bar(open_=Decimal("-1"), skip_validation=True)
    result = check_negative_prices(bar)

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.check == "negative_prices"
    assert result.severity == CheckSeverity.error
    assert result.affected_ts == bar.ts
    assert "open=-1" in (result.message or "")


def test_check_negative_prices_lists_all_negative_fields() -> None:
    bar = _make_bar(
        open_=Decimal("-1"),
        close=Decimal("-2"),
        skip_validation=True,
    )
    result = check_negative_prices(bar)

    assert result is not None
    message = result.message or ""
    assert "open=-1" in message
    assert "close=-2" in message


def test_run_checks_collects_results_from_multiple_bars() -> None:
    bars = [
        _make_bar(),
        _make_bar(ts=datetime(2024, 1, 3, tzinfo=UTC), volume=Decimal("0")),
        _make_bar(
            ts=datetime(2024, 1, 4, tzinfo=UTC),
            high=Decimal("98"),
            low=Decimal("99"),
        ),
        _make_bar(
            ts=datetime(2024, 1, 5, tzinfo=UTC),
            open_=Decimal("-1"),
            skip_validation=True,
        ),
    ]

    results = run_checks(bars)

    assert {result.check for result in results} == {
        "zero_volume",
        "high_lt_low",
        "open_close_outside_range",
        "negative_prices",
    }


def test_run_checks_returns_multiple_results_for_same_bar() -> None:
    bar = _make_bar(
        high=Decimal("98"),
        low=Decimal("99"),
        volume=Decimal("0"),
    )

    results = run_checks([bar])

    assert len(results) == 3
    assert {result.check for result in results} == {
        "high_lt_low",
        "open_close_outside_range",
        "zero_volume",
    }
    assert all(result.affected_ts == bar.ts for result in results)


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


@pytest.mark.asyncio
async def test_check_duplicate_bars_returns_empty_when_no_duplicates() -> None:
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    repo = _FakeMarketBarRepository(set())

    results = await check_duplicate_bars(
        [_make_bar(ts=ts)],
        symbol_id=1,
        repository=repo,
    )

    assert results == []


@pytest.mark.asyncio
async def test_check_duplicate_bars_returns_empty_for_empty_bars() -> None:
    repo = _FakeMarketBarRepository({datetime(2024, 1, 2, tzinfo=UTC)})

    results = await check_duplicate_bars([], symbol_id=1, repository=repo)

    assert results == []
    assert repo.calls == []


@pytest.mark.asyncio
async def test_check_duplicate_bars_flags_existing_bar() -> None:
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    bar = _make_bar(ts=ts)
    repo = _FakeMarketBarRepository({ts})

    results = await check_duplicate_bars([bar], symbol_id=42, repository=repo)

    assert len(results) == 1
    result = results[0]
    assert result.check == "duplicate_bar"
    assert result.severity == CheckSeverity.warning
    assert result.affected_ts == ts
    assert result.symbol == "AAPL"
    assert ts.isoformat() in (result.message or "")


@pytest.mark.asyncio
async def test_check_duplicate_bars_flags_only_duplicates_in_mixed_batch() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    ts3 = datetime(2024, 1, 3, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts1),
        _make_bar(ts=ts2),
        _make_bar(ts=ts3),
    ]
    repo = _FakeMarketBarRepository({ts1, ts3})

    results = await check_duplicate_bars(bars, symbol_id=1, repository=repo)

    assert {result.affected_ts for result in results} == {ts1, ts3}


@pytest.mark.asyncio
async def test_check_duplicate_bars_groups_queries_by_timeframe() -> None:
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts, timeframe="1d"),
        _make_bar(ts=ts, timeframe="1h"),
    ]
    repo = _FakeMarketBarRepository({ts})

    await check_duplicate_bars(bars, symbol_id=7, repository=repo)

    assert repo.calls == [
        (7, "1d", [ts]),
        (7, "1h", [ts]),
    ]


def test_check_missing_timestamps_returns_empty_for_empty_or_single_bar() -> None:
    assert check_missing_timestamps([]) == []
    assert check_missing_timestamps([_make_bar()]) == []


def test_check_missing_timestamps_returns_empty_for_consecutive_daily_bars() -> None:
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC)),
        _make_bar(ts=datetime(2024, 1, 2, tzinfo=UTC)),
    ]

    assert check_missing_timestamps(bars) == []


def test_check_missing_timestamps_ignores_weekend_between_daily_bars() -> None:
    bars = [
        _make_bar(ts=datetime(2024, 1, 5, tzinfo=UTC)),
        _make_bar(ts=datetime(2024, 1, 8, tzinfo=UTC)),
    ]

    assert check_missing_timestamps(bars) == []


def test_check_missing_timestamps_flags_skipped_weekday() -> None:
    monday = datetime(2024, 1, 1, tzinfo=UTC)
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=monday),
        _make_bar(ts=wednesday),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 1
    result = results[0]
    assert result.check == "missing_trading_days"
    assert result.severity == CheckSeverity.warning
    assert result.symbol == "AAPL"
    assert result.affected_ts == tuesday
    message = result.message or ""
    assert "1 expected trading day" in message
    assert tuesday.isoformat() in message


def test_check_missing_timestamps_flags_multiple_weekdays_before_weekend() -> None:
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    monday = datetime(2024, 1, 8, tzinfo=UTC)
    thursday = datetime(2024, 1, 4, tzinfo=UTC)
    friday = datetime(2024, 1, 5, tzinfo=UTC)
    bars = [
        _make_bar(ts=wednesday),
        _make_bar(ts=monday),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 1
    result = results[0]
    assert result.check == "missing_trading_days"
    assert result.affected_ts == thursday
    message = result.message or ""
    assert "2 expected trading day" in message
    assert thursday.isoformat() in message
    assert friday.isoformat() in message


def test_check_missing_timestamps_sorts_unordered_daily_bars() -> None:
    monday = datetime(2024, 1, 1, tzinfo=UTC)
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=wednesday),
        _make_bar(ts=monday),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 1
    assert results[0].affected_ts == tuesday


def test_check_missing_timestamps_groups_by_symbol_and_timeframe() -> None:
    monday = datetime(2024, 1, 1, tzinfo=UTC)
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(symbol="AAPL", ts=monday),
        _make_bar(symbol="AAPL", ts=wednesday),
        _make_bar(symbol="MSFT", ts=monday),
        _make_bar(symbol="MSFT", ts=wednesday),
        _make_bar(
            symbol="AAPL",
            ts=datetime(2024, 1, 1, 9, tzinfo=UTC),
            timeframe="1h",
        ),
        _make_bar(
            symbol="AAPL",
            ts=datetime(2024, 1, 1, 10, tzinfo=UTC),
            timeframe="1h",
        ),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 2
    assert {result.symbol for result in results} == {"AAPL", "MSFT"}
    assert all(result.check == "missing_trading_days" for result in results)
    assert all(result.affected_ts == tuesday for result in results)


def test_check_missing_timestamps_flags_multiple_non_contiguous_gaps() -> None:
    monday = datetime(2024, 1, 1, tzinfo=UTC)
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    friday = datetime(2024, 1, 5, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, tzinfo=UTC)
    thursday = datetime(2024, 1, 4, tzinfo=UTC)
    bars = [
        _make_bar(ts=monday),
        _make_bar(ts=wednesday),
        _make_bar(ts=friday),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 2
    assert results[0].affected_ts == tuesday
    assert results[1].affected_ts == thursday
    assert all(result.check == "missing_trading_days" for result in results)


def test_check_missing_timestamps_uses_calendar_dates_for_daily_gaps() -> None:
    monday = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    wednesday = datetime(2024, 1, 3, 10, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=monday),
        _make_bar(ts=wednesday),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 1
    assert results[0].affected_ts == tuesday


def test_check_missing_timestamps_ignores_consecutive_days_with_different_times() -> (
    None
):
    monday = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    tuesday = datetime(2024, 1, 2, 9, tzinfo=UTC)
    bars = [
        _make_bar(ts=monday),
        _make_bar(ts=tuesday),
    ]

    assert check_missing_timestamps(bars) == []


def test_check_missing_timestamps_skips_unknown_timeframe() -> None:
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC), timeframe="1wk"),
        _make_bar(ts=datetime(2024, 1, 15, tzinfo=UTC), timeframe="1wk"),
    ]

    assert check_missing_timestamps(bars) == []


def test_check_missing_timestamps_flags_same_day_intraday_gap() -> None:
    bars = [
        _make_bar(
            ts=datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
            timeframe="1h",
        ),
        _make_bar(
            ts=datetime(2024, 1, 2, 10, 30, tzinfo=UTC),
            timeframe="1h",
        ),
        _make_bar(
            ts=datetime(2024, 1, 2, 12, 30, tzinfo=UTC),
            timeframe="1h",
        ),
    ]

    results = check_missing_timestamps(bars)

    assert len(results) == 1
    result = results[0]
    assert result.check == "missing_timestamps"
    assert result.severity == CheckSeverity.warning
    assert result.affected_ts == datetime(2024, 1, 2, 11, 30, tzinfo=UTC)
    message = result.message or ""
    assert "1 expected 1h bar" in message
    assert datetime(2024, 1, 2, 11, 30, tzinfo=UTC).isoformat() in message


def test_check_missing_timestamps_ignores_overnight_intraday_gap() -> None:
    bars = [
        _make_bar(
            ts=datetime(2024, 1, 2, 16, tzinfo=UTC),
            timeframe="1h",
        ),
        _make_bar(
            ts=datetime(2024, 1, 3, 9, 30, tzinfo=UTC),
            timeframe="1h",
        ),
    ]

    assert check_missing_timestamps(bars) == []


def test_check_price_jumps_returns_empty_for_empty_or_single_bar() -> None:
    assert check_price_jumps([]) == []
    assert check_price_jumps([_make_bar()]) == []


def test_check_price_jumps_returns_empty_when_within_threshold() -> None:
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC), close=Decimal("100")),
        _make_bar(ts=datetime(2024, 1, 2, tzinfo=UTC), close=Decimal("115")),
    ]

    assert check_price_jumps(bars) == []


def test_check_price_jumps_flags_upward_move_exceeding_threshold() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts1, close=Decimal("100")),
        _make_bar(ts=ts2, close=Decimal("125")),
    ]

    results = check_price_jumps(bars)

    assert len(results) == 1
    result = results[0]
    assert result.check == "price_jump"
    assert result.severity == CheckSeverity.warning
    assert result.symbol == "AAPL"
    assert result.affected_ts == ts2
    message = result.message or ""
    assert "100" in message
    assert "125" in message
    assert "25.00%" in message


def test_check_price_jumps_flags_downward_move_exceeding_threshold() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts1, close=Decimal("100")),
        _make_bar(ts=ts2, close=Decimal("75")),
    ]

    results = check_price_jumps(bars)

    assert len(results) == 1
    assert results[0].check == "price_jump"
    assert results[0].affected_ts == ts2
    message = results[0].message or ""
    assert "100" in message
    assert "75" in message
    assert "25.00%" in message


def test_check_price_jumps_does_not_flag_change_at_exact_threshold() -> None:
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC), close=Decimal("100")),
        _make_bar(ts=datetime(2024, 1, 2, tzinfo=UTC), close=Decimal("120")),
    ]

    assert check_price_jumps(bars) == []


def test_make_check_price_jump_uses_custom_threshold() -> None:
    check = make_check_price_jump(threshold_pct=Decimal("0.10"))
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC), close=Decimal("100")),
        _make_bar(ts=datetime(2024, 1, 2, tzinfo=UTC), close=Decimal("115")),
    ]

    results = check(bars)

    assert len(results) == 1
    assert results[0].check == "price_jump"


def test_make_check_price_jump_uses_custom_severity() -> None:
    check = make_check_price_jump(severity=CheckSeverity.error)
    bars = [
        _make_bar(ts=datetime(2024, 1, 1, tzinfo=UTC), close=Decimal("100")),
        _make_bar(ts=datetime(2024, 1, 2, tzinfo=UTC), close=Decimal("125")),
    ]

    results = check(bars)

    assert len(results) == 1
    assert results[0].severity == CheckSeverity.error


def test_check_price_jumps_groups_by_symbol_and_timeframe() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(symbol="AAPL", ts=ts1, close=Decimal("100")),
        _make_bar(symbol="AAPL", ts=ts2, close=Decimal("125")),
        _make_bar(symbol="MSFT", ts=ts1, close=Decimal("100")),
        _make_bar(symbol="MSFT", ts=ts2, close=Decimal("105")),
        _make_bar(
            symbol="AAPL",
            ts=datetime(2024, 1, 1, 9, tzinfo=UTC),
            timeframe="1h",
            close=Decimal("100"),
        ),
        _make_bar(
            symbol="AAPL",
            ts=datetime(2024, 1, 1, 10, tzinfo=UTC),
            timeframe="1h",
            close=Decimal("105"),
        ),
    ]

    results = check_price_jumps(bars)

    assert len(results) == 1
    assert results[0].symbol == "AAPL"
    assert results[0].affected_ts == ts2


def test_check_price_jumps_sorts_unordered_bars() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts2, close=Decimal("125")),
        _make_bar(ts=ts1, close=Decimal("100")),
    ]

    results = check_price_jumps(bars)

    assert len(results) == 1
    assert results[0].affected_ts == ts2


def test_check_price_jumps_flags_multiple_non_contiguous_jumps() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, tzinfo=UTC)
    ts3 = datetime(2024, 1, 3, tzinfo=UTC)
    ts4 = datetime(2024, 1, 4, tzinfo=UTC)
    bars = [
        _make_bar(ts=ts1, close=Decimal("100")),
        _make_bar(ts=ts2, close=Decimal("125")),
        _make_bar(ts=ts3, close=Decimal("130")),
        _make_bar(ts=ts4, close=Decimal("100")),
    ]

    results = check_price_jumps(bars)

    assert len(results) == 2
    assert results[0].affected_ts == ts2
    assert results[1].affected_ts == ts4


def test_check_stale_symbol_returns_none_when_recent() -> None:
    now = datetime(2024, 1, 5, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 4, tzinfo=UTC)

    assert (
        check_stale_symbol(
            "AAPL",
            "1d",
            last_ingested,
            now=now,
        )
        is None
    )


def test_check_stale_symbol_returns_warning_when_stale() -> None:
    now = datetime(2024, 1, 10, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, tzinfo=UTC)

    result = check_stale_symbol("AAPL", "1d", last_ingested, now=now)

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.check == "stale_symbol"
    assert result.severity == CheckSeverity.warning
    assert result.affected_ts == last_ingested
    message = result.message or ""
    assert last_ingested.isoformat() in message
    assert "threshold 3 days" in message


def test_check_stale_symbol_returns_none_at_exact_threshold() -> None:
    now = datetime(2024, 1, 4, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, tzinfo=UTC)

    assert check_stale_symbol("AAPL", "1d", last_ingested, now=now) is None


def test_check_stale_symbol_uses_shorter_intraday_threshold() -> None:
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, 8, tzinfo=UTC)

    result = check_stale_symbol("AAPL", "1h", last_ingested, now=now)

    assert result is not None
    assert result.check == "stale_symbol"
    assert result.affected_ts == last_ingested


def test_check_stale_symbol_returns_none_for_recent_intraday_data() -> None:
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, 11, tzinfo=UTC)

    assert check_stale_symbol("AAPL", "1h", last_ingested, now=now) is None


def test_check_stale_symbol_falls_back_to_daily_threshold_for_unknown_timeframe() -> (
    None
):
    now = datetime(2024, 1, 10, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, tzinfo=UTC)

    result = check_stale_symbol("AAPL", "1wk", last_ingested, now=now)

    assert result is not None
    assert result.check == "stale_symbol"
    assert result.affected_ts == last_ingested


def test_check_stale_symbol_returns_error_when_never_ingested() -> None:
    now = datetime(2024, 1, 5, tzinfo=UTC)

    result = check_stale_symbol("AAPL", "1d", None, now=now)

    assert result is not None
    assert result.check == "stale_symbol"
    assert result.severity == CheckSeverity.error
    assert result.affected_ts is None
    assert "never been ingested" in (result.message or "")


def test_make_check_stale_symbol_respects_custom_max_age_override() -> None:
    check = make_check_stale_symbol(
        max_age_by_timeframe={"1d": timedelta(days=1)},
    )
    now = datetime(2024, 1, 3, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, tzinfo=UTC)

    result = check("AAPL", "1d", last_ingested, now=now)

    assert result is not None
    assert result.check == "stale_symbol"


def test_make_check_stale_symbol_override_does_not_match_other_timeframe() -> None:
    check = make_check_stale_symbol(
        max_age_by_timeframe={"1d": timedelta(days=1)},
    )
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, 8, tzinfo=UTC)

    result = check("AAPL", "1h", last_ingested, now=now)

    assert result is not None
    assert result.check == "stale_symbol"


def test_make_check_stale_symbol_rejects_empty_override_key() -> None:
    with pytest.raises(ValueError, match="timeframe must not be empty"):
        make_check_stale_symbol(max_age_by_timeframe={"": timedelta(days=1)})


def test_make_check_stale_symbol_rejects_non_positive_max_age() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_check_stale_symbol(max_age_by_timeframe={"1d": timedelta(0)})

    with pytest.raises(ValueError, match="must be positive"):
        make_check_stale_symbol(max_age_by_timeframe={"1d": timedelta(days=-1)})


def test_make_check_stale_symbol_respects_custom_severity() -> None:
    check = make_check_stale_symbol(severity=CheckSeverity.error)
    now = datetime(2024, 1, 10, tzinfo=UTC)
    last_ingested = datetime(2024, 1, 1, tzinfo=UTC)

    result = check("AAPL", "1d", last_ingested, now=now)

    assert result is not None
    assert result.severity == CheckSeverity.error


def test_make_check_stale_symbol_respects_custom_missing_data_severity() -> None:
    check = make_check_stale_symbol(missing_data_severity=CheckSeverity.warning)
    now = datetime(2024, 1, 5, tzinfo=UTC)

    result = check("AAPL", "1d", None, now=now)

    assert result is not None
    assert result.severity == CheckSeverity.warning
