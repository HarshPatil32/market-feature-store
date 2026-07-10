"""SQLAlchemy ORM models and shared declarative base."""

import enum
from datetime import datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class CheckSeverity(str, enum.Enum):
    info = "info"
    warning = "warning"
    error = "error"


class Base(DeclarativeBase):
    pass


class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    asset_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="equity",
        server_default="equity",
    )
    active: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default=sa.true(),
    )
    coverage_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    coverage_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        sa.Enum(
            RunStatus,
            native_enum=False,
            length=20,
        ),
        nullable=False,
        default=RunStatus.pending,
        server_default=RunStatus.pending.value,
    )
    symbol_id: Mapped[int | None] = mapped_column(
        ForeignKey("symbols.id", ondelete="SET NULL"),
        nullable=True,
    )
    fetched: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    inserted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RawMarketData(Base):
    __tablename__ = "raw_market_data"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    symbol_id: Mapped[int | None] = mapped_column(
        ForeignKey("symbols.id", ondelete="SET NULL"),
        nullable=True,
    )
    source: Mapped[str | None] = mapped_column(String(50))
    request_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    response_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class MarketBar(Base):
    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "timeframe",
            "timestamp",
            name="uq_market_bars_symbol_id_timeframe_timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(
        ForeignKey("symbols.id", ondelete="RESTRICT"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DataQualityCheck(Base):
    """Immutable audit record for a single validation check result.

    Rows are write-once; there is no updated_at column.
    """

    __tablename__ = "data_quality_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    symbol_id: Mapped[int | None] = mapped_column(
        ForeignKey("symbols.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Stable check identifier (e.g. negative_prices). 100 chars covers all planned VAL checks.
    check_name: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[CheckSeverity] = mapped_column(
        sa.Enum(
            CheckSeverity,
            native_enum=False,
            length=20,
        ),
        nullable=False,
    )
    message: Mapped[str | None] = mapped_column(Text)
    # Point-in-time flag for a single bar. Range checks (e.g. missing_trading_days) store
    # the range start here and encode the range end in message.
    affected_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class FeatureDefinition(Base):
    __tablename__ = "feature_definitions"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "version",
            name="uq_feature_definitions_name_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    lookback_window: Mapped[int] = mapped_column(Integer, nullable=False)
    active: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default=sa.true(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FeatureValue(Base):
    __tablename__ = "feature_values"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "timestamp",
            "feature_definition_id",
            name="uq_feature_values_symbol_id_timestamp_feature_definition_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(
        ForeignKey("symbols.id", ondelete="RESTRICT"),
        nullable=False,
    )
    feature_definition_id: Mapped[int] = mapped_column(
        ForeignKey("feature_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
