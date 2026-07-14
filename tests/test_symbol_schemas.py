"""Tests for symbol registry Pydantic schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.storage.models import Symbol
from backend.storage.schemas import SymbolCreate, SymbolRead, SymbolUpdate


def test_symbol_create_valid_ticker() -> None:
    created = SymbolCreate(symbol="AAPL")

    assert created.symbol == "AAPL"
    assert created.asset_type == "equity"


def test_symbol_create_normalizes_lowercase_ticker() -> None:
    created = SymbolCreate(symbol="aapl")

    assert created.symbol == "AAPL"


def test_symbol_create_strips_and_normalizes_whitespace() -> None:
    created = SymbolCreate(symbol="  msft  ")

    assert created.symbol == "MSFT"


def test_symbol_create_accepts_dot_and_hyphen() -> None:
    created = SymbolCreate(symbol="brk.b")

    assert created.symbol == "BRK.B"


@pytest.mark.parametrize(
    "ticker",
    [
        "",
        "A" * 21,
        "AA PL",
        "AA$PL",
        "AAPL!",
        "   ",
    ],
)
def test_symbol_create_rejects_bad_tickers(ticker: str) -> None:
    with pytest.raises(ValidationError):
        SymbolCreate(symbol=ticker)


def test_symbol_create_rejects_non_string_ticker() -> None:
    with pytest.raises(ValidationError):
        SymbolCreate(symbol=123)


def test_symbol_update_empty_is_valid() -> None:
    updated = SymbolUpdate()

    assert updated.asset_type is None
    assert updated.active is None


def test_symbol_update_partial_active() -> None:
    updated = SymbolUpdate(active=False)

    assert updated.asset_type is None
    assert updated.active is False


def test_symbol_read_from_orm_instance() -> None:
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    updated_at = datetime(2024, 6, 1, tzinfo=UTC)
    coverage_start = datetime(2023, 1, 1, tzinfo=UTC)
    coverage_end = datetime(2023, 12, 31, tzinfo=UTC)
    last_ingested_at = datetime(2024, 5, 1, tzinfo=UTC)
    symbol = Symbol(
        id=1,
        symbol="AAPL",
        asset_type="equity",
        active=True,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        last_ingested_at=last_ingested_at,
        created_at=created_at,
        updated_at=updated_at,
    )

    read = SymbolRead.model_validate(symbol)

    assert read.id == 1
    assert read.symbol == "AAPL"
    assert read.asset_type == "equity"
    assert read.active is True
    assert read.coverage_start == coverage_start
    assert read.coverage_end == coverage_end
    assert read.last_ingested_at == last_ingested_at
    assert read.created_at == created_at
    assert read.updated_at == updated_at


def test_symbol_read_with_null_coverage_fields() -> None:
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    symbol = Symbol(
        id=2,
        symbol="TSLA",
        asset_type="equity",
        active=True,
        coverage_start=None,
        coverage_end=None,
        last_ingested_at=None,
        created_at=created_at,
        updated_at=updated_at,
    )

    read = SymbolRead.model_validate(symbol)

    assert read.symbol == "TSLA"
    assert read.coverage_start is None
    assert read.coverage_end is None
    assert read.last_ingested_at is None
