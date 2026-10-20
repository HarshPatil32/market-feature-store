"""Pydantic schemas for symbol registry request/response contracts."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from backend.storage.models import RunStatus

MAX_COMPUTE_SYMBOLS = 20
MAX_COMPUTE_FEATURES = 100

FeatureName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100),
]

# Mirrors backend.bar.Timeframe; defined here to avoid importing bar (bar imports Ticker).
ComputeTimeframe = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=10,
        pattern=r"^[a-zA-Z0-9]+$",
    ),
]


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


class FeatureDefinitionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: int
    lookback_window: int
    description: str | None
    requirements: list[str] | None
    active: bool
    created_at: datetime
    updated_at: datetime


class IncrementalRunSummary(BaseModel):
    succeeded: dict[str, IngestionRunRead] = Field(
        default_factory=dict,
        description="Symbols that completed incremental ingestion successfully.",
    )
    failed: dict[str, str] = Field(
        default_factory=dict,
        description="Symbols that failed, mapped to an error message.",
    )


class ComputeFeaturesRequest(BaseModel):
    symbols: list[Ticker] = Field(
        ...,
        min_length=1,
        max_length=MAX_COMPUTE_SYMBOLS,
    )
    features: list[FeatureName] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMPUTE_FEATURES,
    )
    timeframe: ComputeTimeframe = "1d"
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None

    @field_validator("symbols")
    @classmethod
    def dedupe_symbols(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for ticker in value:
            if ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(ticker)
        return ordered

    @model_validator(mode="after")
    def validate_datetime_range(self) -> Self:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must be <= end")
        return self


class ComputeFeaturesSummary(BaseModel):
    computed: dict[str, int] = Field(
        default_factory=dict,
        description="Symbols that completed feature computation, mapped to rows upserted.",
    )
    failed: dict[str, str] = Field(
        default_factory=dict,
        description="Symbols that failed, mapped to an error message.",
    )
