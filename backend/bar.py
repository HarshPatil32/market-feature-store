"""Standard internal bar format shared across the pipeline."""

from decimal import Decimal
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StringConstraints,
    field_validator,
)

from backend.storage.schemas import Ticker


def _normalize_source(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source must be a string")
    return value.strip().lower()


Source = Annotated[
    str,
    BeforeValidator(_normalize_source),
    StringConstraints(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_\-]+$",
    ),
]

Timeframe = Annotated[
    str,
    StringConstraints(min_length=1, max_length=10),
]


class Bar(BaseModel):
    """Standard OHLCV bar used across provider, normalization, and validation layers."""

    model_config = ConfigDict(frozen=True)

    symbol: Ticker
    ts: AwareDatetime
    timeframe: Timeframe
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: Source

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _price_must_be_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("price must be greater than zero")
        return value

    @field_validator("volume")
    @classmethod
    def _volume_must_be_non_negative(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("volume must be non-negative")
        return value
