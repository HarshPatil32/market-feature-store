"""Data quality validation checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
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


# check_duplicate_bars is async and DB-backed; invoke it separately before upsert.
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
