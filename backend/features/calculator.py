"""Feature computation from normalized market data."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from math import isfinite

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

import backend.features  # noqa: F401 — register feature implementations
from backend.features.base import Feature
from backend.features.registry import registry
from backend.services.features import FeatureNotFoundError
from backend.storage.models import MarketBar
from backend.storage.repository import (
    FeatureDefinitionRepository,
    FeatureValueRepository,
    MarketBarRepository,
    SymbolRepository,
)

_VALUE_QUANT = Decimal("0.00000001")
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class UnknownSymbolError(Exception):
    def __init__(self, symbol_id: int) -> None:
        self.symbol_id = symbol_id
        super().__init__(f"Symbol not found: id={symbol_id}")


class FeatureDefinitionNotFoundError(Exception):
    def __init__(self, name: str, version: int) -> None:
        self.name = name
        self.version = version
        super().__init__(
            f"Feature definition not found: name={name!r} version={version}"
        )


class FeatureDefinitionLookbackMismatchError(Exception):
    def __init__(
        self,
        name: str,
        version: int,
        *,
        expected: int,
        actual: int,
    ) -> None:
        self.name = name
        self.version = version
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Feature definition lookback_window mismatch for {name!r} v{version}: "
            f"registry={expected}, database={actual}"
        )


def _require_timezone_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


def _select_features(feature_names: list[str] | None) -> list[Feature]:
    registered = registry.all()
    if feature_names is None:
        return registered
    registered_by_name = {feature.spec.name: feature for feature in registered}
    unknown = set(feature_names) - registered_by_name.keys()
    if unknown:
        missing = sorted(unknown)[0]
        raise FeatureNotFoundError(missing)
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for name in feature_names:
        if name in seen:
            continue
        seen.add(name)
        ordered_unique.append(name)
    return [registered_by_name[name] for name in ordered_unique]


def _bars_to_frame(bars: Sequence[MarketBar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
    records: list[dict[str, float]] = []
    index: list[datetime] = []
    for bar in bars:
        index.append(bar.timestamp)
        records.append(
            {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    frame = pd.DataFrame(records, index=pd.DatetimeIndex(index))
    return frame.sort_index()


def _finite_rows(
    series: pd.Series,
    *,
    symbol_id: int,
    feature_definition_id: int,
    computed_at: datetime,
    start: datetime | None,
    end: datetime | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ts, raw in series.items():
        if not isinstance(ts, datetime):
            raise TypeError("feature series index must be datetime")
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        value = float(raw)
        if not isfinite(value):
            continue
        try:
            quantized = Decimal(str(value)).quantize(_VALUE_QUANT)
        except InvalidOperation:
            continue
        rows.append(
            {
                "symbol_id": symbol_id,
                "feature_definition_id": feature_definition_id,
                "timestamp": ts,
                "value": quantized,
                "computed_at": computed_at,
            }
        )
    return rows


async def compute_and_store_features(
    session: AsyncSession,
    symbol_id: int,
    timeframe: str,
    feature_names: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    computed_at: datetime | None = None,
) -> int:
    """Compute registered features for a symbol and upsert finite values."""
    if not timeframe:
        raise ValueError("timeframe must be non-empty")
    if start is not None:
        _require_timezone_aware(start, "start")
    if end is not None:
        _require_timezone_aware(end, "end")
    if computed_at is None:
        computed_at = datetime.now(tz=UTC)
    else:
        _require_timezone_aware(computed_at, "computed_at")

    symbol = await SymbolRepository(session).get_by_id(symbol_id)
    if symbol is None:
        raise UnknownSymbolError(symbol_id)

    features = _select_features(feature_names)
    definition_repo = FeatureDefinitionRepository(session)
    value_repo = FeatureValueRepository(session)
    bar_repo = MarketBarRepository(session)

    bars = await bar_repo.list_by_symbol(
        symbol_id,
        timeframe=timeframe,
        start=None,
        end=end,
    )
    frame = _bars_to_frame(bars)
    if frame.empty:
        return 0

    all_rows: list[dict[str, object]] = []
    for feature in features:
        spec = feature.spec
        definition = await definition_repo.get_by_name_and_version(
            spec.name,
            spec.version,
        )
        if definition is None:
            raise FeatureDefinitionNotFoundError(spec.name, spec.version)
        if definition.lookback_window != spec.lookback_window:
            raise FeatureDefinitionLookbackMismatchError(
                spec.name,
                spec.version,
                expected=spec.lookback_window,
                actual=definition.lookback_window,
            )

        series = feature.compute(frame)
        all_rows.extend(
            _finite_rows(
                series,
                symbol_id=symbol_id,
                feature_definition_id=definition.id,
                computed_at=computed_at,
                start=start,
                end=end,
            )
        )

    await value_repo.upsert_many(all_rows)
    return len(all_rows)
