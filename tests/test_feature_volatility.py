"""Tests for rolling volatility features (sample std of daily returns)."""

from __future__ import annotations

import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.registry import registry
from backend.features.returns import DailyReturn
from backend.features.volatility import Volatility5D, Volatility20D, Volatility60D

_SERIES_ATOL = 1e-10
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"

_VOL_FEATURES: list[tuple[type, int]] = [
    (Volatility5D, 5),
    (Volatility20D, 20),
    (Volatility60D, 60),
]

REFERENCE_VOL_CLOSES = [100.0, 110.0, 99.0, 99.0, 108.9, 108.9]
REFERENCE_VOL_5D = (0.028 / 4) ** 0.5


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=index)


def _decimal_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"close": [Decimal(str(c)) for c in closes]},
        index=index,
    )


def _long_closes(n: int = 100) -> list[float]:
    rng = np.random.default_rng(42)
    path = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, size=n))
    return [float(x) for x in path]


def _hand_volatility(
    closes: list[float],
    window: int,
    lookback_window: int,
) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    close = close.where(np.isfinite(close) & (close > 0))
    returns = (close / close.shift(1) - 1).astype(float)
    result = returns.rolling(window, min_periods=window).std(ddof=1)
    warmup = lookback_window - 1
    if warmup > 0:
        result = result.copy()
        result.iloc[:warmup] = float("nan")
    return result


def test_volatility_5d_reference_value_matches_hand_computed() -> None:
    result = Volatility5D().compute(_frame(REFERENCE_VOL_CLOSES))
    assert result.iloc[:5].isna().all()
    assert result.iloc[5] == pytest.approx(REFERENCE_VOL_5D, rel=1e-9, abs=_SERIES_ATOL)


@pytest.mark.parametrize("feature_cls,window", _VOL_FEATURES)
def test_matches_numpy_std_at_every_valid_index(
    feature_cls: type,
    window: int,
) -> None:
    closes = _long_closes()
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    returns = DailyReturn().compute(frame).to_numpy()
    for i in range(len(result)):
        if pd.isna(result.iloc[i]):
            continue
        window_returns = returns[i - window + 1 : i + 1]
        assert len(window_returns) == window
        assert np.all(np.isfinite(window_returns))
        expected = float(np.std(window_returns, ddof=1))
        assert result.iloc[i] == pytest.approx(expected, rel=1e-9, abs=_SERIES_ATOL)


@pytest.mark.parametrize(
    "feature_cls,window,lookback",
    [
        (Volatility5D, 5, 6),
        (Volatility20D, 20, 21),
        (Volatility60D, 60, 61),
    ],
)
def test_matches_hand_pandas_reference(
    feature_cls: type,
    window: int,
    lookback: int,
) -> None:
    closes = _long_closes()
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    expected = _hand_volatility(closes, window, lookback)
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )
    assert np.isfinite(result.iloc[window])


@pytest.mark.parametrize("feature_cls,window", _VOL_FEATURES)
def test_uses_sample_std_not_population(
    feature_cls: type,
    window: int,
) -> None:
    closes = _long_closes(max(100, window + 10))
    frame = _frame(closes)
    feature = feature_cls().compute(frame)
    returns = DailyReturn().compute(frame)
    pop = returns.rolling(window, min_periods=window).std(ddof=0)
    lookback = feature_cls().spec.lookback_window
    pop.iloc[: lookback - 1] = float("nan")
    ratio = feature / pop
    expected_ratio = (window / (window - 1)) ** 0.5
    for value in ratio.dropna():
        assert value == pytest.approx(expected_ratio, rel=1e-9)


@pytest.mark.parametrize(
    "feature_cls,window,lookback",
    [
        (Volatility5D, 5, 6),
        (Volatility20D, 20, 21),
        (Volatility60D, 60, 61),
    ],
)
def test_leading_edge_nan(
    feature_cls: type,
    window: int,
    lookback: int,
) -> None:
    closes = _long_closes()
    result = feature_cls().compute(_frame(closes))
    assert result.iloc[:window].isna().all()
    assert np.isfinite(result.iloc[window])

    short = feature_cls().compute(_frame(closes[:window]))
    assert short.isna().all()

    empty = feature_cls().compute(_frame([]))
    assert len(empty) == 0
    assert empty.dtype == float


def test_constant_price_series_zero_volatility() -> None:
    closes = [100.0] * 25
    result = Volatility5D().compute(_frame(closes))
    assert result.iloc[5] == pytest.approx(0.0, abs=1e-15)


def test_missing_price_no_forward_fill() -> None:
    window = 5
    closes = _long_closes(40)
    gap_index = 10
    closes[gap_index] = float("nan")
    result = Volatility5D().compute(_frame(closes))
    assert pd.isna(result.iloc[gap_index])
    for offset in range(1, window + 1):
        assert pd.isna(result.iloc[gap_index + offset])
    recovery = gap_index + window + 1
    assert np.isfinite(result.iloc[recovery])


@pytest.mark.parametrize(
    "bad_close",
    [0.0, -5.0, float("inf")],
)
@pytest.mark.parametrize("feature_cls,window", _VOL_FEATURES)
def test_invalid_price_produces_nan_without_warning(
    bad_close: float,
    feature_cls: type,
    window: int,
) -> None:
    bad_index = 8
    closes = _long_closes(max(100, bad_index + window + 5))
    closes[bad_index] = bad_close
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_cls().compute(frame)
    for offset in range(0, window + 1):
        assert pd.isna(result.iloc[bad_index + offset])
    recovery = bad_index + window + 1
    assert recovery < len(result)
    assert np.isfinite(result.iloc[recovery])


@pytest.mark.parametrize("feature_cls", [Volatility5D, Volatility20D, Volatility60D])
def test_unsorted_index_raises(feature_cls: type) -> None:
    closes = _long_closes()
    frame = _frame(closes).iloc[[2, 0, 1, 3, 4] + list(range(5, len(closes)))]
    with pytest.raises(ValueError, match="monotonic"):
        feature_cls().compute(frame)


@pytest.mark.parametrize("feature_cls", [Volatility5D, Volatility20D, Volatility60D])
def test_duplicate_index_raises(feature_cls: type) -> None:
    closes = _long_closes()[:10]
    frame = _frame(closes)
    dup_index = frame.index.tolist()
    dup_index[3] = dup_index[2]
    frame.index = pd.DatetimeIndex(dup_index)
    with pytest.raises(ValueError, match="unique"):
        feature_cls().compute(frame)


@pytest.mark.parametrize("feature_cls", [Volatility5D, Volatility20D, Volatility60D])
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


def test_compute_does_not_mutate_frame() -> None:
    closes = _long_closes()
    frame = _frame(closes)
    before = frame["close"].copy()
    Volatility5D().compute(frame)
    Volatility20D().compute(frame)
    Volatility60D().compute(frame)
    pd.testing.assert_series_equal(frame["close"], before)


def test_no_lookahead_on_prefix() -> None:
    closes = _long_closes()
    full = Volatility5D().compute(_frame(closes))
    prefix_len = 15
    prefix = Volatility5D().compute(_frame(closes[:prefix_len]))
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
    before = Volatility5D().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
    after = Volatility5D().compute(mutated)
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
        "volatility_5d": 6,
        "volatility_20d": 21,
        "volatility_60d": 61,
    }
    for name, lookback in expected_lookback.items():
        feature = registry.get(name, 1)
        assert feature.spec.name == name
        assert feature.spec.version == 1
        assert feature.spec.inputs == ("close",)
        assert feature.spec.lookback_window == lookback


def test_volatility_seed_entries_match_registry_specs() -> None:
    import backend.features  # noqa: F401

    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    vol_entries = [entry for entry in raw if entry["name"].startswith("volatility_")]
    assert len(vol_entries) == 3

    for entry in vol_entries:
        feature = registry.get(entry["name"], entry["version"])
        assert feature.spec.lookback_window == entry["lookback_window"]
        assert feature.spec.version == entry["version"]
        assert feature.spec.inputs == ("close",)
