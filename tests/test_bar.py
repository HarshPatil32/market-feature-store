"""Tests for the standard internal Bar model."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.bar import Bar


def _valid_bar(**overrides: object) -> Bar:
    defaults = {
        "symbol": "AAPL",
        "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "timeframe": "1d",
        "open": Decimal("100"),
        "high": Decimal("105"),
        "low": Decimal("99"),
        "close": Decimal("103"),
        "volume": Decimal("1000"),
        "source": "alpaca",
    }
    defaults.update(overrides)
    return Bar(**defaults)


def test_bar_is_immutable() -> None:
    bar = _valid_bar()

    with pytest.raises(ValidationError):
        bar.close = Decimal("110")  # type: ignore[misc]


def test_bar_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(ts=datetime(2024, 1, 1))


def test_bar_requires_source() -> None:
    with pytest.raises(ValidationError):
        Bar.model_validate(
            {
                "symbol": "AAPL",
                "ts": datetime(2024, 1, 1, tzinfo=UTC),
                "timeframe": "1d",
                "open": Decimal("100"),
                "high": Decimal("105"),
                "low": Decimal("99"),
                "close": Decimal("103"),
                "volume": Decimal("1000"),
            }
        )


def test_bar_rejects_empty_source() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(source="")


def test_bar_rejects_invalid_source_charset() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(source="alpaca markets")


def test_bar_rejects_empty_timeframe() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(timeframe="")


def test_bar_rejects_negative_price() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(open=Decimal("-1"))


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        _valid_bar(volume=Decimal("-1"))


def test_bar_accepts_zero_volume() -> None:
    bar = _valid_bar(volume=Decimal("0"))

    assert bar.volume == Decimal("0")


def test_bar_normalizes_source_case() -> None:
    bar = _valid_bar(source="ALPACA")

    assert bar.source == "alpaca"


def test_bar_symbol_is_normalized() -> None:
    bar = _valid_bar(symbol="aapl")

    assert bar.symbol == "AAPL"
