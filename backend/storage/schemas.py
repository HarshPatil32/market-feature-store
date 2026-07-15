"""Pydantic schemas for symbol registry request/response contracts."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from backend.storage.models import RunStatus


def _normalize_ticker(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("symbol must be a string")
    return value.strip().upper()


Ticker = Annotated[
    str,
    BeforeValidator(_normalize_ticker),
    StringConstraints(
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9.\-]+$",
    ),
]


class SymbolCreate(BaseModel):
    symbol: Ticker
    asset_type: str = Field(default="equity", min_length=1, max_length=20)


class SymbolUpdate(BaseModel):
    asset_type: str | None = Field(default=None, min_length=1, max_length=20)
    active: bool | None = None


class SymbolRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    asset_type: str
    active: bool
    coverage_start: datetime | None
    coverage_end: datetime | None
    last_ingested_at: datetime | None
    created_at: datetime
    updated_at: datetime


class IngestionRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_type: str
    status: RunStatus
    symbol_id: int | None
    fetched: int
    inserted: int
    failed: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
