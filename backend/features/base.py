"""Feature specification and abstract base for versioned market features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

BarColumn = Literal["open", "high", "low", "close", "volume"]


class FeatureSpec(BaseModel):
    """Immutable metadata for a versioned feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1, strict=True)
    inputs: tuple[BarColumn, ...] = Field(min_length=1)
    lookback_window: int = Field(ge=1, le=1000)

    @field_validator("inputs")
    @classmethod
    def _inputs_must_be_unique(
        cls, value: tuple[BarColumn, ...]
    ) -> tuple[BarColumn, ...]:
        if len(set(value)) != len(value):
            raise ValueError("inputs must not contain duplicates")
        return value


class Feature(ABC):
    """Versioned feature computed from an OHLCV bar frame."""

    spec: ClassVar[FeatureSpec]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", frozenset()):
            return
        if "spec" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a class-level spec")
        spec = cls.__dict__["spec"]
        if not isinstance(spec, FeatureSpec):
            raise TypeError(f"{cls.__name__}.spec must be a FeatureSpec instance")

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        """Validate input, run computation, and enforce warm-up NaNs."""
        for column in self.spec.inputs:
            if column not in frame.columns:
                raise ValueError(f"missing required column: {column}")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("frame index must be monotonic increasing")
        if not frame.index.is_unique:
            raise ValueError("frame index must be unique")

        result = self._compute(frame)
        if not isinstance(result, pd.Series):
            raise TypeError("_compute must return a pandas Series")
        if not result.index.equals(frame.index):
            raise ValueError("result index must match frame index")

        result = result.astype(float)
        warmup = self.spec.lookback_window - 1
        if warmup > 0:
            result = result.copy()
            result.iloc[:warmup] = float("nan")
        return result

    @abstractmethod
    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        """Return feature values aligned to frame.index (before warm-up masking)."""
