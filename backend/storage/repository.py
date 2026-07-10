"""Repository layer for persisted market data and features."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, TypeGuard, TypeVar

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import (
    CheckSeverity,
    DataQualityCheck,
    FeatureDefinition,
    FeatureValue,
    IngestionRun,
    MarketBar,
    RawMarketData,
    RunStatus,
    Symbol,
)

T = TypeVar("T")


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _Unset()


def _is_provided(value: T | _Unset) -> TypeGuard[T]:
    return value is not _UNSET


class SymbolRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, symbol: str, asset_type: str = "equity") -> Symbol:
        row = Symbol(symbol=symbol, asset_type=asset_type)
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, symbol_id: int) -> Symbol | None:
        return await self._session.get(Symbol, symbol_id)

    async def get_by_symbol(self, symbol: str) -> Symbol | None:
        result = await self._session.execute(
            select(Symbol).where(Symbol.symbol == symbol)
        )
        return result.scalar_one_or_none()

    async def list(self, *, active_only: bool = False) -> Sequence[Symbol]:
        stmt = select(Symbol).order_by(Symbol.symbol)
        if active_only:
            stmt = stmt.where(Symbol.active.is_(True))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_coverage(
        self,
        symbol_id: int,
        *,
        coverage_start: datetime | None | _Unset = _UNSET,
        coverage_end: datetime | None | _Unset = _UNSET,
        last_ingested_at: datetime | None | _Unset = _UNSET,
    ) -> Symbol | None:
        row = await self.get_by_id(symbol_id)
        if row is None:
            return None
        if _is_provided(coverage_start):
            row.coverage_start = coverage_start
        if _is_provided(coverage_end):
            row.coverage_end = coverage_end
        if _is_provided(last_ingested_at):
            row.last_ingested_at = last_ingested_at
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def set_active(self, symbol_id: int, active: bool) -> Symbol | None:
        row = await self.get_by_id(symbol_id)
        if row is None:
            return None
        row.active = active
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def delete(self, symbol_id: int) -> None:
        row = await self.get_by_id(symbol_id)
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()


class IngestionRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        run_type: str,
        symbol_id: int | None = None,
    ) -> IngestionRun:
        row = IngestionRun(run_type=run_type, symbol_id=symbol_id)
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, run_id: int) -> IngestionRun | None:
        return await self._session.get(IngestionRun, run_id)

    async def update(
        self,
        run_id: int,
        *,
        status: RunStatus | _Unset = _UNSET,
        fetched: int | _Unset = _UNSET,
        inserted: int | _Unset = _UNSET,
        failed: int | _Unset = _UNSET,
        error_message: str | None | _Unset = _UNSET,
        started_at: datetime | None | _Unset = _UNSET,
        finished_at: datetime | None | _Unset = _UNSET,
    ) -> IngestionRun | None:
        row = await self.get_by_id(run_id)
        if row is None:
            return None
        if _is_provided(status):
            row.status = status
        if _is_provided(fetched):
            row.fetched = fetched
        if _is_provided(inserted):
            row.inserted = inserted
        if _is_provided(failed):
            row.failed = failed
        if _is_provided(error_message):
            row.error_message = error_message
        if _is_provided(started_at):
            row.started_at = started_at
        if _is_provided(finished_at):
            row.finished_at = finished_at
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def delete(self, run_id: int) -> None:
        row = await self.get_by_id(run_id)
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()


class RawMarketDataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        response_payload: dict[str, Any],
        run_id: int | None = None,
        symbol_id: int | None = None,
        source: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RawMarketData:
        row = RawMarketData(
            response_payload=response_payload,
            run_id=run_id,
            symbol_id=symbol_id,
            source=source,
            request_params=request_params,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, raw_market_data_id: int) -> RawMarketData | None:
        return await self._session.get(RawMarketData, raw_market_data_id)


class MarketBarRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        *,
        symbol_id: int,
        timestamp: datetime,
        timeframe: str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
    ) -> MarketBar:
        insert_stmt = pg_insert(MarketBar).values(
            symbol_id=symbol_id,
            timestamp=timestamp,
            timeframe=timeframe,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=["symbol_id", "timeframe", "timestamp"],
            set_={
                "open": insert_stmt.excluded.open,
                "high": insert_stmt.excluded.high,
                "low": insert_stmt.excluded.low,
                "close": insert_stmt.excluded.close,
                "volume": insert_stmt.excluded.volume,
                "updated_at": func.now(),
            },
        ).returning(MarketBar)
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, market_bar_id: int) -> MarketBar | None:
        return await self._session.get(MarketBar, market_bar_id)

    async def list_by_symbol(
        self,
        symbol_id: int,
        *,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[MarketBar]:
        stmt = (
            select(MarketBar)
            .where(MarketBar.symbol_id == symbol_id, MarketBar.timeframe == timeframe)
            .order_by(MarketBar.timestamp)
        )
        if start is not None:
            stmt = stmt.where(MarketBar.timestamp >= start)
        if end is not None:
            stmt = stmt.where(MarketBar.timestamp <= end)
        result = await self._session.execute(stmt)
        return result.scalars().all()


class DataQualityCheckRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        check_name: str,
        severity: CheckSeverity,
        run_id: int | None = None,
        symbol_id: int | None = None,
        message: str | None = None,
        affected_timestamp: datetime | None = None,
    ) -> DataQualityCheck:
        row = DataQualityCheck(
            check_name=check_name,
            severity=severity,
            run_id=run_id,
            symbol_id=symbol_id,
            message=message,
            affected_timestamp=affected_timestamp,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def bulk_create(
        self, rows: Sequence[dict[str, Any]]
    ) -> Sequence[DataQualityCheck]:
        created = [DataQualityCheck(**row) for row in rows]
        self._session.add_all(created)
        await self._session.flush()
        # No per-row refresh: flush populates ids and server defaults via RETURNING.
        return created

    async def get_by_id(self, check_id: int) -> DataQualityCheck | None:
        return await self._session.get(DataQualityCheck, check_id)

    async def list_by_run(self, run_id: int) -> Sequence[DataQualityCheck]:
        stmt = (
            select(DataQualityCheck)
            .where(DataQualityCheck.run_id == run_id)
            .order_by(DataQualityCheck.created_at, DataQualityCheck.id)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def list_by_symbol(
        self,
        symbol_id: int,
        *,
        severity: CheckSeverity | None = None,
    ) -> Sequence[DataQualityCheck]:
        """Return checks for a symbol, newest first (for latest-quality views)."""
        stmt = (
            select(DataQualityCheck)
            .where(DataQualityCheck.symbol_id == symbol_id)
            .order_by(DataQualityCheck.created_at.desc(), DataQualityCheck.id.desc())
        )
        if severity is not None:
            stmt = stmt.where(DataQualityCheck.severity == severity)
        result = await self._session.execute(stmt)
        return result.scalars().all()


class FeatureDefinitionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        name: str,
        version: int,
        lookback_window: int,
        active: bool = True,
    ) -> FeatureDefinition:
        row = FeatureDefinition(
            name=name,
            version=version,
            lookback_window=lookback_window,
            active=active,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, feature_definition_id: int) -> FeatureDefinition | None:
        return await self._session.get(FeatureDefinition, feature_definition_id)

    async def get_by_name_and_version(
        self,
        name: str,
        version: int,
    ) -> FeatureDefinition | None:
        result = await self._session.execute(
            select(FeatureDefinition).where(
                FeatureDefinition.name == name,
                FeatureDefinition.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def list(self, *, active_only: bool = False) -> Sequence[FeatureDefinition]:
        stmt = select(FeatureDefinition).order_by(
            FeatureDefinition.name,
            FeatureDefinition.version,
        )
        if active_only:
            stmt = stmt.where(FeatureDefinition.active.is_(True))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def set_active(
        self,
        feature_definition_id: int,
        active: bool,
    ) -> FeatureDefinition | None:
        row = await self.get_by_id(feature_definition_id)
        if row is None:
            return None
        row.active = active
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def delete(self, feature_definition_id: int) -> None:
        row = await self.get_by_id(feature_definition_id)
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()


class FeatureValueRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        *,
        symbol_id: int,
        feature_definition_id: int,
        timestamp: datetime,
        value: Decimal,
    ) -> FeatureValue:
        insert_stmt = pg_insert(FeatureValue).values(
            symbol_id=symbol_id,
            feature_definition_id=feature_definition_id,
            timestamp=timestamp,
            value=value,
        )
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=["symbol_id", "timestamp", "feature_definition_id"],
            set_={
                "value": insert_stmt.excluded.value,
                "updated_at": func.now(),
            },
        ).returning(FeatureValue)
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        await self._session.refresh(row)
        return row

    async def get_by_id(self, feature_value_id: int) -> FeatureValue | None:
        return await self._session.get(FeatureValue, feature_value_id)

    async def list_by_symbol(
        self,
        symbol_id: int,
        *,
        feature_definition_id: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[FeatureValue]:
        stmt = (
            select(FeatureValue)
            .where(FeatureValue.symbol_id == symbol_id)
            .order_by(FeatureValue.timestamp)
        )
        if feature_definition_id is not None:
            stmt = stmt.where(
                FeatureValue.feature_definition_id == feature_definition_id
            )
        if start is not None:
            stmt = stmt.where(FeatureValue.timestamp >= start)
        if end is not None:
            stmt = stmt.where(FeatureValue.timestamp <= end)
        result = await self._session.execute(stmt)
        return result.scalars().all()
