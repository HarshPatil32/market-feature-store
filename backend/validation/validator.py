"""Data quality validation checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict

from backend.bar import Bar
from backend.storage.models import CheckSeverity
from backend.storage.schemas import Ticker


class QualityCheckResult(BaseModel):
    """Structured result from a single data quality check."""

    model_config = ConfigDict(frozen=True)

    symbol: Ticker
    check: str
    severity: CheckSeverity
    message: str | None = None
    affected_ts: AwareDatetime | None = None
    # Set by callers at persistence time; check functions leave this None.
    run_id: int | None = None

    def to_check_row(
        self,
        *,
        symbol_id: int,
        run_id: int | None = None,
    ) -> dict[str, Any]:
        return {
            "check_name": self.check,
            "severity": self.severity,
            "message": self.message,
            "affected_timestamp": self.affected_ts,
            "run_id": run_id if run_id is not None else self.run_id,
            "symbol_id": symbol_id,
        }


CheckFn = Callable[[Bar], QualityCheckResult | None]
VolumeOverrides = Mapping[tuple[str, str], CheckSeverity | None]


class BarTimestampLookup(Protocol):
    async def get_existing_timestamps(
        self,
        symbol_id: int,
        *,
        timeframe: str,
        timestamps: Sequence[datetime],
    ) -> set[datetime]: ...


def check_high_lt_low(bar: Bar) -> QualityCheckResult | None:
    if bar.high >= bar.low:
        return None
    return QualityCheckResult(
        symbol=bar.symbol,
        check="high_lt_low",
        severity=CheckSeverity.error,
        message=f"High {bar.high} is less than low {bar.low}",
        affected_ts=bar.ts,
    )


def check_open_close_outside_range(bar: Bar) -> QualityCheckResult | None:
    violations = [
        (field, value)
        for field in ("open", "close")
        if (value := getattr(bar, field)) < bar.low or value > bar.high
    ]
    if not violations:
        return None
    details = ", ".join(f"{field}={value}" for field, value in violations)
    return QualityCheckResult(
        symbol=bar.symbol,
        check="open_close_outside_range",
        severity=CheckSeverity.error,
        message=f"Open/close outside [low, high] ({bar.low}, {bar.high}): {details}",
        affected_ts=bar.ts,
    )


def _validate_volume_overrides(
    overrides: VolumeOverrides,
) -> dict[tuple[str, str], CheckSeverity | None]:
    validated: dict[tuple[str, str], CheckSeverity | None] = {}
    for (symbol, timeframe), severity in overrides.items():
        if not symbol:
            raise ValueError("volume override symbol must not be empty")
        if not timeframe:
            raise ValueError("volume override timeframe must not be empty")
        validated[(symbol, timeframe)] = severity
    return validated


def make_check_zero_volume(
    overrides: VolumeOverrides | None = None,
    *,
    default_severity: CheckSeverity = CheckSeverity.warning,
) -> CheckFn:
    resolved_overrides = _validate_volume_overrides(overrides or {})

    def _check_volume(bar: Bar) -> QualityCheckResult | None:
        if bar.volume < 0:
            return QualityCheckResult(
                symbol=bar.symbol,
                check="negative_volume",
                severity=CheckSeverity.error,
                message=f"Negative volume: {bar.volume}",
                affected_ts=bar.ts,
            )
        if bar.volume != Decimal("0"):
            return None

        severity = resolved_overrides.get((bar.symbol, bar.timeframe), default_severity)
        if severity is None:
            return None
        return QualityCheckResult(
            symbol=bar.symbol,
            check="zero_volume",
            severity=severity,
            message="Volume is zero",
            affected_ts=bar.ts,
        )

    return _check_volume


check_zero_volume: CheckFn = make_check_zero_volume()


def check_negative_prices(bar: Bar) -> QualityCheckResult | None:
    negative = [
        (field, value)
        for field in ("open", "high", "low", "close")
        if (value := getattr(bar, field)) < 0
    ]
    if not negative:
        return None
    details = ", ".join(f"{field}={value}" for field, value in negative)
    return QualityCheckResult(
        symbol=bar.symbol,
        check="negative_prices",
        severity=CheckSeverity.error,
        message=f"Negative price(s): {details}",
        affected_ts=bar.ts,
    )


# Known intraday cadences; add new timeframes here when ingestion supports them.
_INTRADAY_STEPS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
}
_DAILY_TIMEFRAME = "1d"
DEFAULT_PRICE_JUMP_THRESHOLD_PCT: Decimal = Decimal("0.20")


def _missing_weekdays(prev_ts: datetime, curr_ts: datetime) -> list[datetime]:
    """Weekdays (Mon-Fri) strictly between prev_ts and curr_ts calendar dates."""
    tz = prev_ts.tzinfo
    prev_date = prev_ts.date()
    curr_date = curr_ts.date()
    missing: list[datetime] = []
    cursor: date = prev_date + timedelta(days=1)
    while cursor < curr_date:
        if cursor.weekday() < 5:
            missing.append(datetime(cursor.year, cursor.month, cursor.day, tzinfo=tz))
        cursor += timedelta(days=1)
    return missing


def _daily_gaps(
    symbol: str,
    ordered: Sequence[Bar],
) -> list[QualityCheckResult]:
    results: list[QualityCheckResult] = []
    for prev, curr in zip(ordered, ordered[1:]):
        missing = _missing_weekdays(prev.ts, curr.ts)
        if not missing:
            continue
        results.append(
            QualityCheckResult(
                symbol=symbol,
                check="missing_trading_days",
                severity=CheckSeverity.warning,
                message=(
                    f"Missing {len(missing)} expected trading day(s) between "
                    f"{missing[0].isoformat()} and {missing[-1].isoformat()}"
                ),
                affected_ts=missing[0],
            )
        )
    return results


def _intraday_gaps(
    symbol: str,
    timeframe: str,
    ordered: Sequence[Bar],
    step: timedelta,
) -> list[QualityCheckResult]:
    results: list[QualityCheckResult] = []
    for prev, curr in zip(ordered, ordered[1:]):
        if prev.ts.date() != curr.ts.date():
            continue
        gap_count = (curr.ts - prev.ts) // step - 1
        if gap_count <= 0:
            continue
        range_start = prev.ts + step
        range_end = curr.ts - step
        results.append(
            QualityCheckResult(
                symbol=symbol,
                check="missing_timestamps",
                severity=CheckSeverity.warning,
                message=(
                    f"Missing {gap_count} expected {timeframe} bar(s) between "
                    f"{range_start.isoformat()} and {range_end.isoformat()}"
                ),
                affected_ts=range_start,
            )
        )
    return results


def check_missing_timestamps(bars: Sequence[Bar]) -> list[QualityCheckResult]:
    """Detect gaps within a batch of bars for a symbol and timeframe."""
    by_group: dict[tuple[str, str], list[Bar]] = defaultdict(list)
    for bar in bars:
        by_group[(bar.symbol, bar.timeframe)].append(bar)

    results: list[QualityCheckResult] = []
    for (symbol, timeframe), group in by_group.items():
        ordered = sorted(group, key=lambda bar: bar.ts)
        if timeframe == _DAILY_TIMEFRAME:
            results.extend(_daily_gaps(symbol, ordered))
        elif timeframe in _INTRADAY_STEPS:
            results.extend(
                _intraday_gaps(symbol, timeframe, ordered, _INTRADAY_STEPS[timeframe])
            )
    return results


def make_check_price_jump(
    threshold_pct: Decimal = DEFAULT_PRICE_JUMP_THRESHOLD_PCT,
    *,
    severity: CheckSeverity = CheckSeverity.warning,
) -> Callable[[Sequence[Bar]], list[QualityCheckResult]]:
    threshold_display = threshold_pct * 100

    def _check_price_jumps(bars: Sequence[Bar]) -> list[QualityCheckResult]:
        by_group: dict[tuple[str, str], list[Bar]] = defaultdict(list)
        for bar in bars:
            by_group[(bar.symbol, bar.timeframe)].append(bar)

        results: list[QualityCheckResult] = []
        for (symbol, _timeframe), group in by_group.items():
            ordered = sorted(group, key=lambda bar: bar.ts)
            for prev, curr in zip(ordered, ordered[1:]):
                # Bar.close is validated > 0, so prev.close is never zero.
                change_pct = abs(curr.close - prev.close) / prev.close
                if change_pct <= threshold_pct:
                    continue
                pct_display = change_pct * 100
                results.append(
                    QualityCheckResult(
                        symbol=symbol,
                        check="price_jump",
                        severity=severity,
                        message=(
                            f"Close jumped from {prev.close} to {curr.close} "
                            f"({pct_display:.2f}% change, threshold "
                            f"{threshold_display:.2f}%)"
                        ),
                        affected_ts=curr.ts,
                    )
                )
        return results

    return _check_price_jumps


check_price_jumps = make_check_price_jump()

DEFAULT_STALE_INTRADAY_MULTIPLIER: int = 3
DEFAULT_STALE_DAILY_MAX_AGE: timedelta = timedelta(days=3)


def _default_max_age(timeframe: str) -> timedelta:
    """Fallback staleness threshold when no explicit override is given for a timeframe."""
    step = _INTRADAY_STEPS.get(timeframe)
    if step is not None:
        return step * DEFAULT_STALE_INTRADAY_MULTIPLIER
    return DEFAULT_STALE_DAILY_MAX_AGE


def _validate_max_age_overrides(
    overrides: Mapping[str, timedelta],
) -> dict[str, timedelta]:
    validated: dict[str, timedelta] = {}
    for timeframe, max_age in overrides.items():
        if not timeframe:
            raise ValueError("max_age override timeframe must not be empty")
        if max_age <= timedelta(0):
            raise ValueError(f"max_age for timeframe {timeframe!r} must be positive")
        validated[timeframe] = max_age
    return validated


def make_check_stale_symbol(
    max_age_by_timeframe: Mapping[str, timedelta] | None = None,
    *,
    severity: CheckSeverity = CheckSeverity.warning,
    missing_data_severity: CheckSeverity = CheckSeverity.error,
) -> Callable[..., QualityCheckResult | None]:
    overrides = _validate_max_age_overrides(max_age_by_timeframe or {})

    def _check_stale_symbol(
        symbol: Ticker,
        timeframe: str,
        last_ingested_at: AwareDatetime | None,
        *,
        now: AwareDatetime,
    ) -> QualityCheckResult | None:
        if last_ingested_at is None:
            return QualityCheckResult(
                symbol=symbol,
                check="stale_symbol",
                severity=missing_data_severity,
                message="Symbol has never been ingested (last_ingested_at is not set)",
            )

        max_age = overrides.get(timeframe, _default_max_age(timeframe))
        age = now - last_ingested_at
        if age <= max_age:
            return None

        return QualityCheckResult(
            symbol=symbol,
            check="stale_symbol",
            severity=severity,
            message=(
                f"No data ingested in {age} (last ingested at "
                f"{last_ingested_at.isoformat()}, threshold {max_age})"
            ),
            affected_ts=last_ingested_at,
        )

    return _check_stale_symbol


check_stale_symbol = make_check_stale_symbol()


# check_duplicate_bars is async and DB-backed; check_missing_timestamps,
# check_price_jumps, and check_stale_symbol operate outside DEFAULT_CHECKS.
# The batch checks and stale-symbol check must be invoked separately.
DEFAULT_CHECKS: tuple[CheckFn, ...] = (
    check_high_lt_low,
    check_open_close_outside_range,
    check_zero_volume,
    check_negative_prices,
)


async def check_duplicate_bars(
    bars: Sequence[Bar],
    *,
    symbol_id: int,
    repository: BarTimestampLookup,
) -> list[QualityCheckResult]:
    if not bars:
        return []

    by_timeframe: dict[str, list[Bar]] = defaultdict(list)
    for bar in bars:
        by_timeframe[bar.timeframe].append(bar)

    results: list[QualityCheckResult] = []
    for timeframe, group in by_timeframe.items():
        timestamps = [bar.ts for bar in group]
        existing = await repository.get_existing_timestamps(
            symbol_id,
            timeframe=timeframe,
            timestamps=timestamps,
        )
        for bar in group:
            if bar.ts not in existing:
                continue
            results.append(
                QualityCheckResult(
                    symbol=bar.symbol,
                    check="duplicate_bar",
                    severity=CheckSeverity.warning,
                    message=(
                        f"Bar for {bar.timeframe} at {bar.ts.isoformat()} already exists"
                    ),
                    affected_ts=bar.ts,
                )
            )
    return results


def run_checks(
    bars: Sequence[Bar],
    checks: Sequence[CheckFn] = DEFAULT_CHECKS,
) -> list[QualityCheckResult]:
    results: list[QualityCheckResult] = []
    for bar in bars:
        for check in checks:
            result = check(bar)
            if result is not None:
                results.append(result)
    return results
