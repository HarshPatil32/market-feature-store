"""SQLAlchemy ORM models and shared declarative base."""

import enum
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


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
