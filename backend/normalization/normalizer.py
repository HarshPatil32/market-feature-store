"""Raw market data normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from pydantic import ValidationError

from backend.bar import Bar
from backend.storage.schemas import Ticker


class NormalizationError(Exception):
    """Base exception for normalization failures."""


class FieldMapper(Protocol):
    def __call__(
        self,
        entry: Mapping[str, Any],
        *,
        symbol: Ticker,
        timeframe: str,
        source: str,
    ) -> Bar: ...


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _extract_alpaca_entries(
    payload: Mapping[str, Any],
    symbol: Ticker,
) -> list[Mapping[str, Any]]:
    bars_by_symbol = payload.get("bars")
    if bars_by_symbol is None:
        return []
    if not isinstance(bars_by_symbol, dict):
        raise NormalizationError("Alpaca response bars must be an object")

    symbol_bars = bars_by_symbol.get(symbol)
    if symbol_bars is None:
        return []
    if not isinstance(symbol_bars, list):
        raise NormalizationError(f"Alpaca bars for {symbol!r} must be a list")

    extracted: list[Mapping[str, Any]] = []
    for item in symbol_bars:
        if not isinstance(item, dict):
            raise NormalizationError(
                f"Alpaca bar entry for {symbol!r} must be an object"
            )
        extracted.append(item)
    return extracted


def _map_alpaca_entry(
    entry: Mapping[str, Any],
    *,
    symbol: Ticker,
    timeframe: str,
    source: str,
) -> Bar:
    timestamp = entry["t"]
    if not isinstance(timestamp, str):
        raise TypeError("bar timestamp must be a string")

    return Bar(
        symbol=symbol,
        ts=_parse_timestamp(timestamp),
        timeframe=timeframe,
        open=Decimal(str(entry["o"])),
        high=Decimal(str(entry["h"])),
        low=Decimal(str(entry["l"])),
        close=Decimal(str(entry["c"])),
        volume=Decimal(str(entry["v"])),
        source=source,
    )


_ENTRY_EXTRACTORS: dict[
    str, Callable[[Mapping[str, Any], Ticker], list[Mapping[str, Any]]]
] = {
    "alpaca": _extract_alpaca_entries,
}

_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "alpaca": _map_alpaca_entry,
}


def normalize_bars(
    payload: Mapping[str, Any],
    *,
    symbol: Ticker,
    timeframe: str,
    source: str,
) -> list[Bar]:
    normalized_source = source.strip().lower()
    extractor = _ENTRY_EXTRACTORS.get(normalized_source)
    mapper = _FIELD_MAPPERS.get(normalized_source)
    if extractor is None or mapper is None:
        supported = ", ".join(sorted(_ENTRY_EXTRACTORS))
        raise NormalizationError(
            f"unsupported source {source!r}; supported: {supported}"
        )

    entries = extractor(payload, symbol)
    parsed: list[Bar] = []
    for entry in entries:
        try:
            parsed.append(
                mapper(
                    entry,
                    symbol=symbol,
                    timeframe=timeframe,
                    source=normalized_source,
                )
            )
        except ValidationError as exc:
            raise NormalizationError(
                f"{normalized_source} bar failed validation"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise NormalizationError(
                f"{normalized_source} bar payload is malformed"
            ) from exc
    return parsed
