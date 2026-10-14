"""Tests for FeatureSpec and the Feature base class."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.features.base import Feature, FeatureSpec


def _valid_spec(**overrides: Any) -> FeatureSpec:
    defaults: dict[str, Any] = {
        "name": "close_doubled",
        "version": 1,
        "inputs": ("close",),
        "lookback_window": 3,
    }
    defaults.update(overrides)
    return FeatureSpec(**defaults)


def test_feature_spec_accepts_valid_values() -> None:
    spec = _valid_spec()
    assert spec.name == "close_doubled"
    assert spec.version == 1
    assert spec.inputs == ("close",)
    assert spec.lookback_window == 3


@pytest.mark.parametrize(
    "name",
    ["", "Close", "bad name", "../x", "a" * 101],
)
def test_feature_spec_rejects_invalid_name(name: str) -> None:
    with pytest.raises(ValidationError):
        _valid_spec(name=name)


@pytest.mark.parametrize("version", [0, -1])
def test_feature_spec_rejects_non_positive_version(version: int) -> None:
    with pytest.raises(ValidationError):
        _valid_spec(version=version)


def test_feature_spec_rejects_bool_version() -> None:
    with pytest.raises(ValidationError):
        _valid_spec(version=True)


@pytest.mark.parametrize("lookback_window", [0, -1, 1001])
def test_feature_spec_rejects_invalid_lookback_window(lookback_window: int) -> None:
    with pytest.raises(ValidationError):
        _valid_spec(lookback_window=lookback_window)


def test_feature_spec_rejects_empty_inputs() -> None:
    with pytest.raises(ValidationError):
        _valid_spec(inputs=())


def test_feature_spec_rejects_unknown_input_column() -> None:
    with pytest.raises(ValidationError):
        _valid_spec(inputs=("close", "typo"))


def test_feature_spec_rejects_duplicate_inputs() -> None:
    with pytest.raises(ValidationError):
        _valid_spec(inputs=("close", "close"))


def test_feature_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FeatureSpec(
            name="x",
            version=1,
            inputs=("close",),
            lookback_window=1,
            extra=True,  # type: ignore[call-arg]
        )


def test_feature_spec_is_frozen() -> None:
    spec = _valid_spec()
    with pytest.raises(ValidationError):
        spec.version = 2  # type: ignore[misc]


class _CloseDoubled(Feature):
    spec = FeatureSpec(
        name="close_doubled",
        version=1,
        inputs=("close",),
        lookback_window=3,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"] * 2


def _sample_frame(length: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    close = pd.Series(range(1, length + 1), index=index, dtype=float)
    return pd.DataFrame({"close": close})


def test_concrete_subclass_without_spec_raises_type_error() -> None:
    with pytest.raises(TypeError, match="must define a class-level spec"):

        class _NoSpec(Feature):  # noqa: N801
            def _compute(self, frame: pd.DataFrame) -> pd.Series:
                return frame["close"]


def test_concrete_subclass_with_non_feature_spec_raises_type_error() -> None:
    with pytest.raises(TypeError, match="must be a FeatureSpec instance"):

        class _BadSpec(Feature):  # noqa: N801
            spec = {"name": "bad", "version": 1}  # type: ignore[assignment]

            def _compute(self, frame: pd.DataFrame) -> pd.Series:
                return frame["close"]


def test_subclass_without_compute_cannot_be_instantiated() -> None:
    class _Incomplete(Feature):
        spec = _valid_spec(name="incomplete")

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_compute_raises_when_input_column_missing() -> None:
    feature = _CloseDoubled()
    frame = _sample_frame().drop(columns=["close"])

    with pytest.raises(ValueError, match="missing required column"):
        feature.compute(frame)


def test_compute_raises_when_index_not_monotonic() -> None:
    feature = _CloseDoubled()
    frame = _sample_frame()
    frame = frame.iloc[[2, 0, 1, 3, 4]]

    with pytest.raises(ValueError, match="monotonic increasing"):
        feature.compute(frame)


def test_compute_raises_when_index_not_unique() -> None:
    feature = _CloseDoubled()
    frame = _sample_frame()
    dup_index = list(frame.index)
    dup_index[1] = dup_index[0]
    frame.index = pd.DatetimeIndex(dup_index)

    with pytest.raises(ValueError, match="unique"):
        feature.compute(frame)


class _ReturnsList(Feature):
    spec = FeatureSpec(
        name="returns_list",
        version=1,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return list(frame["close"])


def test_compute_raises_when_compute_returns_non_series() -> None:
    feature = _ReturnsList()
    with pytest.raises(TypeError, match="pandas Series"):
        feature.compute(_sample_frame())


class _WrongIndex(Feature):
    spec = FeatureSpec(
        name="wrong_index",
        version=1,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series([1.0] * len(frame))


def test_compute_raises_when_result_index_differs() -> None:
    feature = _WrongIndex()
    with pytest.raises(ValueError, match="index must match"):
        feature.compute(_sample_frame())


def test_compute_masks_first_lookback_minus_one_rows_as_nan() -> None:
    feature = _CloseDoubled()
    result = feature.compute(_sample_frame(length=6))

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()
    assert result.iloc[2] == pytest.approx(6.0)


def test_compute_matches_hand_computed_expectation() -> None:
    feature = _CloseDoubled()
    frame = _sample_frame(length=4)
    result = feature.compute(frame)

    expected = frame["close"] * 2
    expected.iloc[:2] = float("nan")
    pd.testing.assert_series_equal(result, expected)


def test_compute_short_frame_all_nan_no_exception() -> None:
    feature = _CloseDoubled()
    result = feature.compute(_sample_frame(length=2))

    assert len(result) == 2
    assert result.isna().all()


def test_compute_empty_frame_returns_empty_series() -> None:
    feature = _CloseDoubled()
    frame = _sample_frame(length=0)
    result = feature.compute(frame)

    assert len(result) == 0
    assert result.dtype == float


class _IntResult(Feature):
    spec = FeatureSpec(
        name="int_result",
        version=1,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return (frame["close"] * 2).astype(int)


def test_compute_coerces_result_to_float() -> None:
    feature = _IntResult()
    result = feature.compute(_sample_frame(length=3))

    assert result.dtype == float
    assert result.iloc[0] == pytest.approx(2.0)


class _TracksRawResult(Feature):
    spec = FeatureSpec(
        name="tracks_raw",
        version=1,
        inputs=("close",),
        lookback_window=3,
    )

    raw: pd.Series | None = None

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        self.raw = frame["close"] * 2
        return self.raw


def test_compute_does_not_mutate_frame_or_raw_compute_series() -> None:
    feature = _TracksRawResult()
    frame = _sample_frame(length=5)
    close_before = frame["close"].copy()

    result = feature.compute(frame)

    pd.testing.assert_series_equal(frame["close"], close_before)
    assert feature.raw is not None
    assert feature.raw.iloc[:2].notna().all()
    assert result.iloc[:2].isna().all()
