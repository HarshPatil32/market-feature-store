"""Tests for simple moving average features on close."""

from __future__ import annotations

import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.moving_averages import Sma5, Sma20, Sma50, Sma200
from backend.features.registry import registry

_SERIES_ATOL = 1e-10
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"

_SMA_FEATURES: list[tuple[type, int]] = [
    (Sma5, 5),
    (Sma20, 20),
    (Sma50, 50),
    (Sma200, 200),
]

REFERENCE_SMA5_CLOSES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=index)


def _decimal_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"close": [Decimal(str(c)) for c in closes]},
        index=index,
    )


def _long_closes(n: int = 320) -> list[float]:
    rng = np.random.default_rng(42)
    path = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, size=n))
    return [float(x) for x in path]


def test_sma_5_reference_value_matches_hand_computed() -> None:
    result = Sma5().compute(_frame(REFERENCE_SMA5_CLOSES))
    assert result.iloc[:4].isna().all()
    assert result.iloc[4] == pytest.approx(3.0, rel=1e-9, abs=_SERIES_ATOL)
    assert result.iloc[5] == pytest.approx(4.0, rel=1e-9, abs=_SERIES_ATOL)


@pytest.mark.parametrize("feature_cls,window", _SMA_FEATURES)
def test_matches_numpy_mean_at_every_valid_index(
    feature_cls: type,
    window: int,
) -> None:
    closes = _long_closes()
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    close = frame["close"].astype(float).to_numpy()
    for i in range(len(result)):
        if pd.isna(result.iloc[i]):
            continue
        window_closes = close[i - window + 1 : i + 1]
        assert len(window_closes) == window
        assert np.all(np.isfinite(window_closes))
        assert np.all(window_closes > 0)
        expected = float(np.mean(window_closes))
        assert result.iloc[i] == pytest.approx(expected, rel=1e-9, abs=_SERIES_ATOL)


@pytest.mark.parametrize("feature_cls,window", _SMA_FEATURES)
def test_leading_edge_nan(
    feature_cls: type,
    window: int,
) -> None:
    closes = _long_closes()
    result = feature_cls().compute(_frame(closes))
    assert result.iloc[: window - 1].isna().all()
    assert np.isfinite(result.iloc[window - 1])

    short = feature_cls().compute(_frame(closes[: window - 1]))
    assert short.isna().all()

    empty = feature_cls().compute(_frame([]))
    assert len(empty) == 0
    assert empty.dtype == float


@pytest.mark.parametrize("feature_cls,window", _SMA_FEATURES)
def test_constant_price_series(feature_cls: type, window: int) -> None:
    closes = [100.0] * max(25, window + 5)
    result = feature_cls().compute(_frame(closes))
    assert result.iloc[window - 1] == pytest.approx(100.0, rel=1e-9, abs=_SERIES_ATOL)


def test_missing_price_no_forward_fill() -> None:
    window = 5
    closes = _long_closes(80)
    gap_index = 10
    closes[gap_index] = float("nan")
    result = Sma5().compute(_frame(closes))
    assert pd.isna(result.iloc[gap_index])
    for offset in range(1, window):
        assert pd.isna(result.iloc[gap_index + offset])
    recovery = gap_index + window
    assert np.isfinite(result.iloc[recovery])


@pytest.mark.parametrize(
    "bad_close",
    [0.0, -5.0, float("inf")],
)
@pytest.mark.parametrize("feature_cls,window", _SMA_FEATURES)
def test_invalid_price_produces_nan_without_warning(
    bad_close: float,
    feature_cls: type,
    window: int,
) -> None:
    bad_index = 8
    closes = _long_closes(max(320, bad_index + window + 5))
    closes[bad_index] = bad_close
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_cls().compute(frame)
    for offset in range(0, window):
        assert pd.isna(result.iloc[bad_index + offset])
    recovery = bad_index + window
    assert recovery < len(result)
    assert np.isfinite(result.iloc[recovery])


@pytest.mark.parametrize("feature_cls", [Sma5, Sma20, Sma50, Sma200])
def test_unsorted_index_raises(feature_cls: type) -> None:
    closes = _long_closes()
    frame = _frame(closes).iloc[[2, 0, 1, 3, 4] + list(range(5, len(closes)))]
    with pytest.raises(ValueError, match="monotonic"):
        feature_cls().compute(frame)


@pytest.mark.parametrize("feature_cls", [Sma5, Sma20, Sma50, Sma200])
def test_duplicate_index_raises(feature_cls: type) -> None:
    closes = _long_closes()[:10]
    frame = _frame(closes)
    dup_index = frame.index.tolist()
    dup_index[3] = dup_index[2]
    frame.index = pd.DatetimeIndex(dup_index)
    with pytest.raises(ValueError, match="unique"):
        feature_cls().compute(frame)


@pytest.mark.parametrize("feature_cls", [Sma5, Sma20, Sma50, Sma200])
def test_decimal_close_matches_float(feature_cls: type) -> None:
    closes = _long_closes()
    float_result = feature_cls().compute(_frame(closes))
    decimal_result = feature_cls().compute(_decimal_frame(closes))
    pd.testing.assert_series_equal(
        float_result,
        decimal_result,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


@pytest.mark.parametrize("feature_cls", [Sma5, Sma20, Sma50, Sma200])
def test_int_close_column(feature_cls: type) -> None:
    closes = _long_closes()[:50]
    int_closes = [int(round(c)) for c in closes]
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    frame = pd.DataFrame({"close": int_closes}, index=index)
    result = feature_cls().compute(frame)
    float_frame = _frame([float(c) for c in int_closes])
    expected = feature_cls().compute(float_frame)
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_compute_does_not_mutate_frame() -> None:
    closes = _long_closes()
    frame = _frame(closes)
    before = frame["close"].copy()
    Sma5().compute(frame)
    Sma20().compute(frame)
    Sma50().compute(frame)
    Sma200().compute(frame)
    pd.testing.assert_series_equal(frame["close"], before)


def test_no_lookahead_on_prefix() -> None:
    closes = _long_closes()
    full = Sma5().compute(_frame(closes))
    prefix_len = 15
    prefix = Sma5().compute(_frame(closes[:prefix_len]))
    pd.testing.assert_series_equal(
        full.iloc[:prefix_len],
        prefix,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_future_close_change_does_not_affect_past() -> None:
    closes = _long_closes()
    frame = _frame(closes)
    before = Sma5().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
    after = Sma5().compute(mutated)
    pd.testing.assert_series_equal(
        before.iloc[:-1],
        after.iloc[:-1],
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_registered_in_global_registry() -> None:
    import backend.features  # noqa: F401

    expected_lookback = {
        "sma_5": 5,
        "sma_20": 20,
        "sma_50": 50,
        "sma_200": 200,
    }
    for name, lookback in expected_lookback.items():
        feature = registry.get(name, 1)
        assert feature.spec.name == name
        assert feature.spec.version == 1
        assert feature.spec.inputs == ("close",)
        assert feature.spec.lookback_window == lookback


def test_sma_seed_entries_match_registry_specs() -> None:
    import backend.features  # noqa: F401

    expected_names = {cls().spec.name for cls, _ in _SMA_FEATURES}
    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    sma_entries = [entry for entry in raw if entry["name"] in expected_names]
    assert {entry["name"] for entry in sma_entries} == expected_names

    for entry in sma_entries:
        feature = registry.get(entry["name"], entry["version"])
        assert feature.spec.lookback_window == entry["lookback_window"]
        assert feature.spec.version == entry["version"]
        assert feature.spec.inputs == ("close",)
