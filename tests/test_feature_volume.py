"""Tests for volume SMA and volume spike ratio features."""

from __future__ import annotations

import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.registry import registry
from backend.features.volume import VolumeSma20, VolumeSpikeRatio20

_SERIES_ATOL = 1e-10
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"

_LOOKBACK = 20


def _frame(volumes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(volumes), freq="D")
    return pd.DataFrame({"volume": volumes}, index=index)


def _decimal_frame(volumes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(volumes), freq="D")
    return pd.DataFrame(
        {"volume": [Decimal(str(v)) for v in volumes]},
        index=index,
    )


def _long_volumes(n: int = 320) -> list[float]:
    rng = np.random.default_rng(42)
    return [float(x) for x in rng.integers(1000, 100_000, size=n)]


def test_volume_sma_reference_value_matches_hand_computed() -> None:
    volumes = [float(i) for i in range(1, 21)]
    result = VolumeSma20().compute(_frame(volumes))
    assert result.iloc[:19].isna().all()
    assert result.iloc[19] == pytest.approx(10.5, rel=1e-9, abs=_SERIES_ATOL)


def test_volume_spike_ratio_reference_value_matches_hand_computed() -> None:
    volumes = [float(i) for i in range(1, 21)]
    result = VolumeSpikeRatio20().compute(_frame(volumes))
    assert result.iloc[:19].isna().all()
    assert result.iloc[19] == pytest.approx(20.0 / 10.5, rel=1e-9, abs=_SERIES_ATOL)


def test_volume_sma_matches_pandas_rolling_mean() -> None:
    volumes = _long_volumes()
    frame = _frame(volumes)
    result = VolumeSma20().compute(frame)
    clean = frame["volume"].astype(float)
    expected = clean.rolling(_LOOKBACK, min_periods=_LOOKBACK).mean()
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_volume_spike_ratio_matches_volume_over_sma() -> None:
    volumes = _long_volumes()
    frame = _frame(volumes)
    result = VolumeSpikeRatio20().compute(frame)
    sma = VolumeSma20().compute(frame)
    volume = frame["volume"].astype(float)
    expected = volume / sma
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_leading_edge_nan() -> None:
    volumes = _long_volumes()
    sma = VolumeSma20().compute(_frame(volumes))
    ratio = VolumeSpikeRatio20().compute(_frame(volumes))
    assert sma.iloc[: _LOOKBACK - 1].isna().all()
    assert np.isfinite(sma.iloc[_LOOKBACK - 1])
    assert ratio.iloc[: _LOOKBACK - 1].isna().all()
    assert np.isfinite(ratio.iloc[_LOOKBACK - 1])

    short_sma = VolumeSma20().compute(_frame(volumes[: _LOOKBACK - 1]))
    assert short_sma.isna().all()

    empty = VolumeSma20().compute(_frame([]))
    assert len(empty) == 0
    assert empty.dtype == float


def test_constant_volume_gives_spike_ratio_one() -> None:
    volumes = [1000.0] * 25
    result = VolumeSpikeRatio20().compute(_frame(volumes))
    assert result.iloc[_LOOKBACK - 1] == pytest.approx(1.0, rel=1e-9, abs=_SERIES_ATOL)
    assert result.iloc[-1] == pytest.approx(1.0, rel=1e-9, abs=_SERIES_ATOL)


def test_spike_and_quiet_bar_sign() -> None:
    base = [1000.0] * (_LOOKBACK - 1)
    spike_volumes = base + [10_000.0]
    spike_result = VolumeSpikeRatio20().compute(_frame(spike_volumes))
    assert spike_result.iloc[-1] > 1.0

    quiet_volumes = base + [100.0]
    quiet_result = VolumeSpikeRatio20().compute(_frame(quiet_volumes))
    assert quiet_result.iloc[-1] < 1.0


def test_zero_current_volume_gives_ratio_zero() -> None:
    volumes = [1000.0] * (_LOOKBACK - 1) + [0.0]
    result = VolumeSpikeRatio20().compute(_frame(volumes))
    assert result.iloc[-1] == pytest.approx(0.0, rel=1e-9, abs=_SERIES_ATOL)


def test_zero_mid_window_volume_sma_finite_and_hand_computed() -> None:
    volumes = [1000.0] * _LOOKBACK
    volumes[10] = 0.0
    frame = _frame(volumes)
    sma = VolumeSma20().compute(frame)
    expected_mean = (19 * 1000.0) / _LOOKBACK
    assert np.isfinite(sma.iloc[_LOOKBACK - 1])
    assert sma.iloc[_LOOKBACK - 1] == pytest.approx(
        expected_mean, rel=1e-9, abs=_SERIES_ATOL
    )
    ratio = VolumeSpikeRatio20().compute(frame)
    assert ratio.iloc[_LOOKBACK - 1] == pytest.approx(
        1000.0 / expected_mean, rel=1e-9, abs=_SERIES_ATOL
    )


def test_all_zero_window_gives_nan_ratio_without_warning() -> None:
    volumes = [0.0] * _LOOKBACK
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = VolumeSpikeRatio20().compute(_frame(volumes))
    assert pd.isna(result.iloc[-1])
    assert not np.isinf(result.iloc[-1])


@pytest.mark.parametrize(
    "bad_volume",
    [float("nan"), float("inf"), -1.0],
)
def test_invalid_volume_produces_nan_without_warning(bad_volume: float) -> None:
    bad_index = 30
    volumes = _long_volumes(max(320, bad_index + _LOOKBACK + 5))
    volumes[bad_index] = bad_volume
    frame = _frame(volumes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sma = VolumeSma20().compute(frame)
        ratio = VolumeSpikeRatio20().compute(frame)
    for offset in range(0, _LOOKBACK):
        assert pd.isna(sma.iloc[bad_index + offset])
        assert pd.isna(ratio.iloc[bad_index + offset])
    recovery = bad_index + _LOOKBACK
    assert recovery < len(sma)
    assert np.isfinite(sma.iloc[recovery])
    assert np.isfinite(ratio.iloc[recovery])


def test_missing_volume_column_raises() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    with pytest.raises(ValueError, match="missing required column"):
        VolumeSma20().compute(frame)


def test_unsorted_index_raises() -> None:
    volumes = _long_volumes()
    frame = _frame(volumes).iloc[[2, 0, 1, 3, 4] + list(range(5, len(volumes)))]
    with pytest.raises(ValueError, match="monotonic"):
        VolumeSma20().compute(frame)


def test_duplicate_index_raises() -> None:
    volumes = _long_volumes()[:10]
    frame = _frame(volumes)
    dup_index = frame.index.tolist()
    dup_index[3] = dup_index[2]
    frame.index = pd.DatetimeIndex(dup_index)
    with pytest.raises(ValueError, match="unique"):
        VolumeSma20().compute(frame)


@pytest.mark.parametrize(
    "feature_cls",
    [VolumeSma20, VolumeSpikeRatio20],
)
def test_decimal_volume_matches_float(feature_cls: type) -> None:
    volumes = _long_volumes()
    float_result = feature_cls().compute(_frame(volumes))
    decimal_result = feature_cls().compute(_decimal_frame(volumes))
    pd.testing.assert_series_equal(
        float_result,
        decimal_result,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_int_volume_column() -> None:
    volumes = _long_volumes()[:50]
    int_volumes = [int(v) for v in volumes]
    index = pd.date_range("2024-01-01", periods=len(volumes), freq="D")
    frame = pd.DataFrame({"volume": int_volumes}, index=index)
    result = VolumeSma20().compute(frame)
    float_frame = _frame([float(v) for v in int_volumes])
    expected = VolumeSma20().compute(float_frame)
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_compute_does_not_mutate_frame() -> None:
    volumes = _long_volumes()
    frame = _frame(volumes)
    before = frame["volume"].copy()
    VolumeSma20().compute(frame)
    VolumeSpikeRatio20().compute(frame)
    pd.testing.assert_series_equal(frame["volume"], before)


def test_past_values_do_not_depend_on_future_bars() -> None:
    volumes = _long_volumes()
    full = VolumeSpikeRatio20().compute(_frame(volumes))
    prefix_len = 40
    prefix = VolumeSpikeRatio20().compute(_frame(volumes[:prefix_len]))
    pd.testing.assert_series_equal(
        full.iloc[:prefix_len],
        prefix,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )

    frame = _frame(volumes)
    before = VolumeSpikeRatio20().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("volume")] = 999_999.0
    after = VolumeSpikeRatio20().compute(mutated)
    pd.testing.assert_series_equal(
        before.iloc[:-1],
        after.iloc[:-1],
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


@pytest.mark.parametrize(
    "name,feature_cls",
    [
        ("volume_sma_20", VolumeSma20),
        ("volume_spike_ratio_20", VolumeSpikeRatio20),
    ],
)
def test_registered_in_global_registry(name: str, feature_cls: type) -> None:
    import backend.features  # noqa: F401

    feature = registry.get(name, 1)
    assert isinstance(feature, feature_cls)
    assert feature.spec.name == name
    assert feature.spec.version == 1
    assert feature.spec.inputs == ("volume",)
    assert feature.spec.lookback_window == _LOOKBACK


@pytest.mark.parametrize("name", ["volume_sma_20", "volume_spike_ratio_20"])
def test_seed_entry_matches_registry_spec(name: str) -> None:
    import backend.features  # noqa: F401

    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    matches = [entry for entry in raw if entry["name"] == name]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["description"].strip()
    feature = registry.get(entry["name"], entry["version"])
    assert feature.spec.lookback_window == entry["lookback_window"]
    assert feature.spec.version == entry["version"]
    assert feature.spec.inputs == ("volume",)
