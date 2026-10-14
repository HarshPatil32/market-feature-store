"""Tests for FeatureRegistry."""

from __future__ import annotations

import types

import pandas as pd
import pytest

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import (
    FeatureNotRegisteredError,
    FeatureRegistry,
    register_feature,
    registry,
)


def _registry() -> FeatureRegistry:
    return FeatureRegistry()


class _CloseDoubled(Feature):
    spec = FeatureSpec(
        name="close_doubled",
        version=1,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"] * 2


class _CloseDoubledV2(Feature):
    spec = FeatureSpec(
        name="close_doubled",
        version=2,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"] * 3


class _VolumeSum(Feature):
    spec = FeatureSpec(
        name="volume_sum",
        version=1,
        inputs=("volume",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["volume"]


def test_register_then_get_returns_instance() -> None:
    reg = _registry()
    reg.register(_CloseDoubled)
    fetched = reg.get("close_doubled", 1)
    assert isinstance(fetched, _CloseDoubled)
    assert fetched.spec.version == 1


def test_register_as_decorator_returns_class() -> None:
    reg = _registry()

    @reg.register
    class _Decorated(Feature):
        spec = FeatureSpec(
            name="decorated",
            version=1,
            inputs=("open",),
            lookback_window=1,
        )

        def _compute(self, frame: pd.DataFrame) -> pd.Series:
            return frame["open"]

    fetched = reg.get("decorated", 1)
    assert isinstance(fetched, _Decorated)
    assert fetched.__class__ is _Decorated


def test_register_feature_is_bound_to_module_registry() -> None:
    assert isinstance(register_feature, types.MethodType)
    assert register_feature.__self__ is registry
    assert register_feature.__func__ is FeatureRegistry.register


def test_register_duplicate_name_version_raises() -> None:
    reg = _registry()
    reg.register(_CloseDoubled)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_CloseDoubled)


def test_same_name_different_version_allowed() -> None:
    reg = _registry()
    reg.register(_CloseDoubled)
    reg.register(_CloseDoubledV2)
    assert reg.get("close_doubled", 1).spec.version == 1
    assert reg.get("close_doubled", 2).spec.version == 2


def test_get_unknown_raises() -> None:
    reg = _registry()
    with pytest.raises(FeatureNotRegisteredError):
        reg.get("missing", 1)


def test_all_sorted_by_name_then_version() -> None:
    reg = _registry()
    reg.register(_VolumeSum)
    reg.register(_CloseDoubledV2)
    reg.register(_CloseDoubled)

    keys = [(f.spec.name, f.spec.version) for f in reg.all()]
    assert keys == [
        ("close_doubled", 1),
        ("close_doubled", 2),
        ("volume_sum", 1),
    ]
