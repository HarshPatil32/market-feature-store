"""Tests for ingestion pipeline orchestration."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.bar import Bar
from backend.ingestion.pipeline import ingest_raw_data
from backend.services.ingestion_runs import trigger_backfill
from backend.services.raw_market_data import persist_raw_fetch
from backend.services.symbols import add_symbol
from backend.storage.models import CheckSeverity, RawMarketData
from backend.storage.repository import (
    DataQualityCheckRepository,
    RawMarketDataRepository,
)
from backend.storage.schemas import SymbolCreate
from backend.validation.validator import QualityCheckResult


def _make_bar(
    *,
    symbol: str = "AAPL",
    ts: datetime | None = None,
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("99"),
    open_: Decimal = Decimal("100"),
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


@pytest.mark.asyncio
async def test_ingest_raw_data_persists_raw_row_before_calling_normalize(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {
        "meta": {"symbol": "AAPL", "count": 1},
        "bars": [{"timestamp": "2024-01-02", "open": 100.0, "close": 101.5}],
    }
    normalize_calls: list[dict[str, object]] = []

    def normalize(payload: dict[str, object]) -> list[Bar]:
        normalize_calls.append(payload)
        return []

    created = await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        source="fake",
        response_payload=response_payload,
        normalize=normalize,
    )
    fetched = await RawMarketDataRepository(db_session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.run_id == run.id
    assert fetched.response_payload == response_payload
    assert normalize_calls == [response_payload]


@pytest.mark.asyncio
async def test_ingest_raw_data_raw_row_survives_normalization_failure(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="MSFT"))
    run = await trigger_backfill(db_session, "MSFT")
    response_payload: dict[str, Any] = {"bars": []}

    def normalize(_payload: dict[str, object]) -> None:
        raise ValueError("boom")

    raw_id: int | None = None

    async def capture_persist(
        session: AsyncSession,
        *,
        run_id: int,
        response_payload: dict[str, Any],
        symbol_id: int | None = None,
        source: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RawMarketData:
        nonlocal raw_id
        row = await persist_raw_fetch(
            session,
            run_id=run_id,
            symbol_id=symbol_id,
            source=source,
            request_params=request_params,
            response_payload=response_payload,
        )
        raw_id = row.id
        return row

    with patch(
        "backend.ingestion.pipeline.persist_raw_fetch",
        side_effect=capture_persist,
    ):
        with pytest.raises(ValueError, match="boom"):
            await ingest_raw_data(
                db_session,
                run_id=run.id,
                symbol_id=symbol.id,
                response_payload=response_payload,
                normalize=normalize,
            )

    assert raw_id is not None
    fetched = await RawMarketDataRepository(db_session).get_by_id(raw_id)
    assert fetched is not None
    assert fetched.response_payload == response_payload


@pytest.mark.asyncio
async def test_ingest_raw_data_normalization_failure_does_not_poison_session(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            response_payload={"bars": []},
            normalize=normalize,
        )

    created = await add_symbol(db_session, SymbolCreate(symbol="GOOG"))

    assert created.symbol == "GOOG"


@pytest.mark.asyncio
async def test_ingest_raw_data_normalization_uses_in_memory_payload_when_s3_backed(
    db_session: AsyncSession,
) -> None:
    from backend.storage.models import RawMarketData as RawMarketDataModel

    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload = {"bars": [{"timestamp": "2024-01-02", "open": 100.0}]}
    normalize_calls: list[dict[str, object]] = []

    async def fake_persist(
        session: AsyncSession,
        *,
        run_id: int,
        response_payload: dict[str, Any],
        symbol_id: int | None = None,
        source: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RawMarketData:
        return RawMarketDataModel(
            id=1,
            run_id=run_id,
            symbol_id=symbol_id,
            source=source,
            request_params=request_params,
            response_payload=None,
            payload_object_key="raw/1/abc.json",
            payload_size_bytes=123,
        )

    def normalize(payload: dict[str, object]) -> list[Bar]:
        normalize_calls.append(payload)
        return []

    with patch(
        "backend.ingestion.pipeline.persist_raw_fetch",
        side_effect=fake_persist,
    ):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            symbol_id=symbol.id,
            response_payload=response_payload,
            normalize=normalize,
        )

    assert normalize_calls == [response_payload]


@pytest.mark.asyncio
async def test_ingest_raw_data_rejects_async_normalize(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    async def normalize(_payload: dict[str, object]) -> None:
        pass

    with pytest.raises(TypeError, match="synchronous callable"):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            response_payload={"bars": []},
            normalize=normalize,
        )


@pytest.mark.asyncio
async def test_ingest_raw_data_runs_validation_after_successful_normalize(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [
            _make_bar(
                symbol="AAPL",
                high=Decimal("98"),
                low=Decimal("99"),
                close=Decimal("97"),
            )
        ]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
        normalize=normalize,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)
    high_lt_low_checks = [c for c in checks if c.check_name == "high_lt_low"]

    assert len(high_lt_low_checks) == 1
    assert high_lt_low_checks[0].severity == CheckSeverity.error
    assert high_lt_low_checks[0].symbol_id == symbol.id


@pytest.mark.asyncio
async def test_ingest_raw_data_validation_persists_multiple_checks(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [
            _make_bar(
                symbol="AAPL",
                open_=Decimal("110"),
                close=Decimal("110"),
                volume=Decimal("0"),
            )
        ]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
        normalize=normalize,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert len(checks) == 2
    assert {c.check_name for c in checks} == {
        "open_close_outside_range",
        "zero_volume",
    }


@pytest.mark.asyncio
async def test_ingest_raw_data_no_checks_created_when_bars_pass(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
        normalize=normalize,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert checks == []


@pytest.mark.asyncio
async def test_ingest_raw_data_skips_validation_without_symbol_id(
    db_session: AsyncSession,
) -> None:
    await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [
            _make_bar(
                symbol="AAPL",
                high=Decimal("98"),
                low=Decimal("99"),
                close=Decimal("97"),
            )
        ]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        response_payload={"bars": []},
        normalize=normalize,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert checks == []


@pytest.mark.asyncio
async def test_ingest_raw_data_validate_override_replaces_default_checks(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [
            _make_bar(
                symbol="AAPL",
                high=Decimal("98"),
                low=Decimal("99"),
                close=Decimal("97"),
            )
        ]

    def validate(_bars: Sequence[Bar]) -> list[QualityCheckResult]:
        return [
            QualityCheckResult(
                symbol="AAPL",
                check="custom_check",
                severity=CheckSeverity.warning,
                message="from override",
            )
        ]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
        normalize=normalize,
        validate=validate,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert len(checks) == 1
    assert checks[0].check_name == "custom_check"
    assert checks[0].message == "from override"


@pytest.mark.asyncio
async def test_ingest_raw_data_validate_none_disables_validation(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [
            _make_bar(
                symbol="AAPL",
                high=Decimal("98"),
                low=Decimal("99"),
                close=Decimal("97"),
            )
        ]

    await ingest_raw_data(
        db_session,
        run_id=run.id,
        symbol_id=symbol.id,
        response_payload={"bars": []},
        normalize=normalize,
        validate=None,
    )

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)

    assert checks == []


@pytest.mark.asyncio
async def test_ingest_raw_data_normalize_return_must_be_bar_or_sequence(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> str:
        return "not-a-bar"

    with pytest.raises(
        TypeError, match="normalize must return a Bar or iterable of Bar"
    ):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            symbol_id=symbol.id,
            response_payload={"bars": []},
            normalize=normalize,
        )


@pytest.mark.asyncio
async def test_ingest_raw_data_validate_failure_does_not_persist_checks(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")
    response_payload: dict[str, Any] = {"bars": []}

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return [_make_bar(symbol="AAPL")]

    def validate(_bars: Sequence[Bar]) -> list[QualityCheckResult]:
        raise ValueError("validate boom")

    raw_id: int | None = None

    async def capture_persist(
        session: AsyncSession,
        *,
        run_id: int,
        response_payload: dict[str, Any],
        symbol_id: int | None = None,
        source: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RawMarketData:
        nonlocal raw_id
        row = await persist_raw_fetch(
            session,
            run_id=run_id,
            symbol_id=symbol_id,
            source=source,
            request_params=request_params,
            response_payload=response_payload,
        )
        raw_id = row.id
        return row

    with patch(
        "backend.ingestion.pipeline.persist_raw_fetch",
        side_effect=capture_persist,
    ):
        with pytest.raises(ValueError, match="validate boom"):
            await ingest_raw_data(
                db_session,
                run_id=run.id,
                symbol_id=symbol.id,
                response_payload=response_payload,
                normalize=normalize,
                validate=validate,
            )

    assert raw_id is not None
    fetched = await RawMarketDataRepository(db_session).get_by_id(raw_id)
    assert fetched is not None
    assert fetched.response_payload == response_payload

    checks = await DataQualityCheckRepository(db_session).list_by_run(run.id)
    assert checks == []


@pytest.mark.asyncio
async def test_ingest_raw_data_rejects_async_validate(
    db_session: AsyncSession,
) -> None:
    symbol = await add_symbol(db_session, SymbolCreate(symbol="AAPL"))
    run = await trigger_backfill(db_session, "AAPL")

    def normalize(_payload: dict[str, object]) -> list[Bar]:
        return []

    async def validate(_bars: Sequence[Bar]) -> list[QualityCheckResult]:
        return []

    with pytest.raises(TypeError, match="synchronous callable"):
        await ingest_raw_data(
            db_session,
            run_id=run.id,
            symbol_id=symbol.id,
            response_payload={"bars": []},
            normalize=normalize,
            validate=validate,  # type: ignore[arg-type]
        )
