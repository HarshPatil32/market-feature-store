"""Tests for raw market data normalization."""

from datetime import UTC, datetime
from decimal import Decimal
from math import inf, nan

import pytest

from backend.bar import Bar
from backend.normalization.normalizer import NormalizationError, normalize_bars


def _sample_alpaca_payload(
    *,
    symbol: str = "AAPL",
    timestamps: list[str] | None = None,
) -> dict[str, object]:
    if timestamps is None:
        timestamps = ["2024-01-02T05:00:00Z"]

    bars = [
        {
            "t": timestamp,
            "o": 187.15 + index,
            "h": 188.44 + index,
            "l": 186.89 + index,
            "c": 188.01 + index,
            "v": 45678900 + index,
            "n": 12345,
            "vw": 187.5,
        }
        for index, timestamp in enumerate(timestamps)
    ]
    return {
        "bars": {symbol: bars},
        "next_page_token": None,
    }


def _alpaca_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "t": "2024-01-02T05:00:00Z",
        "o": 187.15,
        "h": 188.44,
        "l": 186.89,
        "c": 188.01,
        "v": 45678900,
    }
    entry.update(overrides)
    return entry


def test_normalize_bars_maps_alpaca_fields_and_types() -> None:
    payload = _sample_alpaca_payload()

    bars = normalize_bars(
        payload,
        symbol="AAPL",
        timeframe="1d",
        source="alpaca",
    )

    assert len(bars) == 1
    bar = bars[0]
    assert isinstance(bar, Bar)
    assert bar.symbol == "AAPL"
    assert bar.timeframe == "1d"
    assert bar.source == "alpaca"
    assert bar.open == Decimal("187.15")
    assert bar.high == Decimal("188.44")
    assert bar.low == Decimal("186.89")
    assert bar.close == Decimal("188.01")
    assert bar.volume == Decimal("45678900")
    assert bar.ts == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


def test_normalize_bars_preserves_entry_order() -> None:
    payload = _sample_alpaca_payload(
        timestamps=[
            "2024-01-02T05:00:00Z",
            "2024-01-03T05:00:00Z",
            "2024-01-04T05:00:00Z",
        ]
    )

    bars = normalize_bars(
        payload,
        symbol="AAPL",
        timeframe="1d",
        source="alpaca",
    )

    assert len(bars) == 3
    assert bars[0].ts == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert bars[1].ts == datetime(2024, 1, 3, 5, 0, tzinfo=UTC)
    assert bars[2].ts == datetime(2024, 1, 4, 5, 0, tzinfo=UTC)
    assert bars[0].open == Decimal("187.15")
    assert bars[1].open == Decimal("188.15")
    assert bars[2].open == Decimal("189.15")


def test_normalize_bars_returns_empty_list_for_missing_symbol() -> None:
    payload = _sample_alpaca_payload(symbol="MSFT")

    bars = normalize_bars(
        payload,
        symbol="AAPL",
        timeframe="1d",
        source="alpaca",
    )

    assert bars == []


def test_normalize_bars_returns_empty_list_for_empty_bars_dict() -> None:
    payload: dict[str, object] = {"bars": {}, "next_page_token": None}

    bars = normalize_bars(
        payload,
        symbol="AAPL",
        timeframe="1d",
        source="alpaca",
    )

    assert bars == []


def test_normalize_bars_converts_non_utc_timestamp_offset_to_utc() -> None:
    payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2024-01-02T01:00:00-04:00",
                    "o": 187.15,
                    "h": 188.44,
                    "l": 186.89,
                    "c": 188.01,
                    "v": 45678900,
                }
            ]
        }
    }

    bars = normalize_bars(
        payload,
        symbol="AAPL",
        timeframe="1d",
        source="alpaca",
    )

    assert len(bars) == 1
    assert bars[0].ts == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


def test_normalize_bars_raises_for_unknown_source() -> None:
    payload = _sample_alpaca_payload()

    with pytest.raises(NormalizationError, match="unsupported source 'polygon'"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="polygon",
        )


def test_normalize_bars_raises_for_malformed_entry() -> None:
    payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2024-01-02T05:00:00Z",
                    "o": 187.15,
                    "h": 188.44,
                    "l": 186.89,
                    "v": 45678900,
                }
            ]
        }
    }

    with pytest.raises(NormalizationError, match="alpaca bar payload is malformed"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )


def test_normalize_bars_raises_for_invalid_bar_values() -> None:
    payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2024-01-02T05:00:00Z",
                    "o": -187.15,
                    "h": 188.44,
                    "l": 186.89,
                    "c": 188.01,
                    "v": 45678900,
                }
            ]
        }
    }

    with pytest.raises(NormalizationError, match="alpaca bar failed validation"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )


def test_normalize_bars_raises_when_bars_container_is_wrong_type() -> None:
    payload = {"bars": ["not-a-dict"]}

    with pytest.raises(NormalizationError, match="bars must be an object"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )


def test_normalize_bars_raises_when_bar_entry_is_wrong_type() -> None:
    payload = {"bars": {"AAPL": ["not-a-dict"]}}

    with pytest.raises(NormalizationError, match="must be an object"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("o", None),
        ("h", ""),
        ("l", "not-a-number"),
        ("c", True),
        ("v", [1, 2]),
        ("o", inf),
        ("c", nan),
    ],
)
def test_normalize_bars_raises_for_invalid_numeric_field(
    field: str, bad_value: object
) -> None:
    payload = {"bars": {"AAPL": [_alpaca_entry(**{field: bad_value})]}}

    with pytest.raises(NormalizationError, match=rf"field {field!r}"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )


def test_normalize_bars_names_offending_entry_index_in_batch() -> None:
    payload = {
        "bars": {
            "AAPL": [
                _alpaca_entry(),
                _alpaca_entry(o="not-a-number"),
            ]
        }
    }

    with pytest.raises(NormalizationError, match=r"entry 1"):
        normalize_bars(
            payload,
            symbol="AAPL",
            timeframe="1d",
            source="alpaca",
        )
