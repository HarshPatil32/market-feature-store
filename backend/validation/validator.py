"""Data quality validation checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

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


def check_zero_volume(bar: Bar) -> QualityCheckResult | None:
    if bar.volume != Decimal("0"):
        return None
    return QualityCheckResult(
        symbol=bar.symbol,
        check="zero_volume",
        severity=CheckSeverity.warning,
        message="Volume is zero",
        affected_ts=bar.ts,
    )


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


DEFAULT_CHECKS: tuple[CheckFn, ...] = (
    check_high_lt_low,
    check_zero_volume,
    check_negative_prices,
)


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
