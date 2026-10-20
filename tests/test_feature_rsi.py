"""Tests for Wilder RSI (14-period) feature."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.registry import registry
from backend.features.rsi import Rsi14

_SERIES_ATOL = 1e-10
_PUBLISHED_RSI_ATOL = 0.005
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"
_PERIOD = 14
_LOOKBACK = 15
_RECOVERY_BARS_AFTER_GAP = _PERIOD + 1

# StockCharts ChartSchool RSI worked example (full-precision closes).
_STOCKCHARTS_EXAMPLE_CLOSES = [
    44.3389,
    44.0902,
    44.1497,
    43.6124,
    44.3278,
    44.8264,
    45.0955,
    45.4245,
    45.8433,
    46.0826,
    45.8931,
    46.0328,
    45.6140,
    46.2820,
    46.2820,
    46.0028,
    46.0328,
    46.4116,
    46.2222,
    45.6439,
    46.2122,
    46.2521,
    45.7137,
    46.4515,
    45.7835,
    45.3548,
    44.0288,
    44.1783,
    44.2181,
    44.5672,
    43.4205,
    42.6628,
    43.1314,
]

_STOCKCHARTS_PUBLISHED_RSI_FROM_INDEX_14 = [
    70.53,
    66.32,
    66.55,
    69.41,
    66.36,
    57.97,
    62.93,
    63.26,
    56.06,
    62.38,
    54.71,
    50.42,
    39.99,
    41.46,
    41.87,
    45.46,
    37.30,
    33.08,
    37.77,
]


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=index)


def _long_closes(n: int = 70) -> list[float]:
    return [100.0 * (1.01**i) for i in range(n)]


def _series_long_enough() -> list[float]:
    return _long_closes(max(70, _PERIOD * 2 + 15))


def _reference_wilder_rsi_via_rs(
    closes: list[float], period: int = _PERIOD
) -> pd.Series:
    """Wilder RSI via RS = avg_gain/avg_loss and 100 - 100/(1+RS)."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    values = np.asarray(closes, dtype=float)
    n = len(values)
    raw = np.full(n, np.nan, dtype=float)

    avg_gain: float | None = None
    avg_loss: float | None = None
    seed_gains: list[float] = []
    seed_losses: list[float] = []

    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if not np.isfinite(delta):
            avg_gain = None
            avg_loss = None
            seed_gains.clear()
            seed_losses.clear()
            continue

        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if avg_gain is None:
            seed_gains.append(gain)
            seed_losses.append(loss)
            if len(seed_gains) < period:
                continue
            avg_gain = sum(seed_gains) / period
            avg_loss = sum(seed_losses) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_gain == 0.0 and avg_loss == 0.0:
            continue
        if avg_loss == 0.0:
            raw[i] = 100.0
        elif avg_gain == 0.0:
            raw[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            raw[i] = 100.0 - 100.0 / (1.0 + rs)

    result = pd.Series(raw, index=index, dtype=float)
    warmup = _LOOKBACK - 1
    if warmup > 0:
        result = result.copy()
        result.iloc[:warmup] = float("nan")
    return result


def test_rsi_stockcharts_example_matches_published_values() -> None:
    frame = _frame(_STOCKCHARTS_EXAMPLE_CLOSES)
    result = Rsi14().compute(frame)
    expected_tail = pd.Series(
        _STOCKCHARTS_PUBLISHED_RSI_FROM_INDEX_14,
        index=frame.index[_PERIOD:],
        dtype=float,
    )
    pd.testing.assert_series_equal(
        result.iloc[_PERIOD:],
        expected_tail,
        check_exact=False,
        atol=_PUBLISHED_RSI_ATOL,
        check_names=False,
    )


def test_rsi_matches_rs_formula_reference() -> None:
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 1.0, size=120)
    closes = (100.0 * np.exp(np.cumsum(steps * 0.01))).tolist()
    frame = _frame(closes)
    result = Rsi14().compute(frame)
    expected = _reference_wilder_rsi_via_rs(closes)
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_rsi_leading_edge_nan() -> None:
    closes = _long_closes()
    result = Rsi14().compute(_frame(closes))
    assert result.iloc[:_PERIOD].isna().all()
    assert np.isfinite(result.iloc[_PERIOD])

    short = Rsi14().compute(_frame(closes[:_PERIOD]))
    assert short.isna().all()

    one_finite = Rsi14().compute(_frame(closes[: _PERIOD + 1]))
    assert one_finite.iloc[:-1].isna().all()
    assert np.isfinite(one_finite.iloc[-1])


def test_rsi_empty_frame() -> None:
    frame = _frame([])
    result = Rsi14().compute(frame)
    assert len(result) == 0


def test_rsi_all_gains() -> None:
    closes = [100.0 + i for i in range(_LOOKBACK + 5)]
    result = Rsi14().compute(_frame(closes))
    assert result.iloc[:_PERIOD].isna().all()
    assert (result.iloc[_PERIOD:] == 100.0).all()


def test_rsi_all_losses() -> None:
    closes = [200.0 - i for i in range(_LOOKBACK + 5)]
    result = Rsi14().compute(_frame(closes))
    assert result.iloc[:_PERIOD].isna().all()
    assert (result.iloc[_PERIOD:] == 0.0).all()


def test_rsi_flat_prices_nan_without_warning() -> None:
    closes = [50.0] * (_LOOKBACK + 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = Rsi14().compute(_frame(closes))
    assert result.isna().all()


def test_rsi_flat_then_up_move() -> None:
    closes = [50.0] * _LOOKBACK + [51.0]
    result = Rsi14().compute(_frame(closes))
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_then_down_move() -> None:
    closes = [50.0] * _LOOKBACK + [49.0]
    result = Rsi14().compute(_frame(closes))
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_values_bounded() -> None:
    result = Rsi14().compute(_frame(_STOCKCHARTS_EXAMPLE_CLOSES))
    finite = result.dropna()
    assert (finite >= 0.0).all()
    assert (finite <= 100.0).all()


def test_rsi_missing_price_no_forward_fill() -> None:
    closes = _series_long_enough()
    gap_index = _PERIOD + 3
    closes[gap_index] = float("nan")
    frame = _frame(closes)
    result = Rsi14().compute(frame)
    assert pd.isna(result.iloc[gap_index])
    assert pd.isna(result.iloc[gap_index + _PERIOD])
    recovery_index = gap_index + _RECOVERY_BARS_AFTER_GAP
    assert np.isfinite(result.iloc[recovery_index])
    filled = closes.copy()
    filled[gap_index] = 100.0
    without_gap = Rsi14().compute(_frame(filled))
    assert np.isfinite(without_gap.iloc[recovery_index])


@pytest.mark.parametrize("bad_close", [0.0, -5.0, float("inf")])
def test_rsi_invalid_price_produces_nan_without_warning(bad_close: float) -> None:
    closes = _series_long_enough()
    bad_index = _PERIOD + 3
    closes[bad_index] = bad_close
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = Rsi14().compute(frame)
    assert pd.isna(result.iloc[bad_index])
    assert pd.isna(result.iloc[bad_index + _PERIOD])
    assert np.isfinite(result.iloc[bad_index + _RECOVERY_BARS_AFTER_GAP])


def test_rsi_no_lookahead_on_prefix() -> None:
    closes = _long_closes()
    full = Rsi14().compute(_frame(closes))
    prefix_len = _PERIOD + 15
    prefix = Rsi14().compute(_frame(closes[:prefix_len]))
    pd.testing.assert_series_equal(
        full.iloc[:prefix_len],
        prefix,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_rsi_future_close_change_does_not_affect_past() -> None:
    closes = _long_closes()
    frame = _frame(closes)
    before = Rsi14().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
    after = Rsi14().compute(mutated)
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

    feature = registry.get("rsi_14", 1)
    assert feature.spec.name == "rsi_14"
    assert feature.spec.version == 1
    assert feature.spec.inputs == ("close",)
    assert feature.spec.lookback_window == _LOOKBACK


def test_rsi_seed_entry_matches_registry_spec() -> None:
    import backend.features  # noqa: F401

    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    matches = [entry for entry in raw if entry["name"] == "rsi_14"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["description"].strip()
    feature = registry.get(entry["name"], entry["version"])
    assert feature.spec.lookback_window == entry["lookback_window"]
    assert feature.spec.version == entry["version"]
    assert feature.spec.inputs == ("close",)
