"""Tests for price vs SMA features on close."""

from __future__ import annotations

import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.moving_averages import PriceVsSma20, Sma20
from backend.features.registry import registry

_SERIES_ATOL = 1e-10
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"

_LOOKBACK = 20


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


def test_reference_value_matches_hand_computed() -> None:
    closes_20 = [float(i) for i in range(1, 21)]
    result_20 = PriceVsSma20().compute(_frame(closes_20))
    assert result_20.iloc[:19].isna().all()
    assert result_20.iloc[19] == pytest.approx(
        (20.0 - 10.5) / 10.5, rel=1e-9, abs=_SERIES_ATOL
    )

    closes_21 = [float(i) for i in range(1, 22)]
    result_21 = PriceVsSma20().compute(_frame(closes_21))
    assert result_21.iloc[20] == pytest.approx(
        (21.0 - 11.5) / 11.5, rel=1e-9, abs=_SERIES_ATOL
    )


def test_matches_close_minus_sma_over_sma() -> None:
    closes = _long_closes()
    frame = _frame(closes)
    result = PriceVsSma20().compute(frame)
    sma = Sma20().compute(frame)
    close = frame["close"].astype(float)
    expected = (close - sma) / sma
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_leading_edge_nan() -> None:
    closes = _long_closes()
    result = PriceVsSma20().compute(_frame(closes))
    assert result.iloc[: _LOOKBACK - 1].isna().all()
    assert np.isfinite(result.iloc[_LOOKBACK - 1])

    short = PriceVsSma20().compute(_frame(closes[: _LOOKBACK - 1]))
    assert short.isna().all()

    empty = PriceVsSma20().compute(_frame([]))
    assert len(empty) == 0
    assert empty.dtype == float


def test_constant_price_series() -> None:
    closes = [100.0] * 25
    result = PriceVsSma20().compute(_frame(closes))
    assert result.iloc[_LOOKBACK - 1] == pytest.approx(0.0, rel=1e-9, abs=_SERIES_ATOL)
    assert result.iloc[-1] == pytest.approx(0.0, rel=1e-9, abs=_SERIES_ATOL)


def test_price_above_and_below_ma_sign() -> None:
    rising = [float(i) for i in range(1, 30)]
    rising_result = PriceVsSma20().compute(_frame(rising))
    assert (rising_result.iloc[_LOOKBACK - 1 :] > 0).all()

    falling = [float(30 - i) for i in range(30)]
    falling_result = PriceVsSma20().compute(_frame(falling))
    assert (falling_result.iloc[_LOOKBACK - 1 :] < 0).all()


@pytest.mark.parametrize(
    "bad_close",
    [0.0, -5.0, float("inf"), float("nan")],
)
def test_invalid_price_produces_nan_without_warning(bad_close: float) -> None:
    bad_index = 30
    closes = _long_closes(max(320, bad_index + _LOOKBACK + 5))
    closes[bad_index] = bad_close
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = PriceVsSma20().compute(frame)
    for offset in range(0, _LOOKBACK):
        assert pd.isna(result.iloc[bad_index + offset])
    recovery = bad_index + _LOOKBACK
    assert recovery < len(result)
    assert np.isfinite(result.iloc[recovery])


def test_missing_price_no_forward_fill() -> None:
    closes = _long_closes(80)
    gap_index = 30
    closes[gap_index] = float("nan")
    result = PriceVsSma20().compute(_frame(closes))
    assert pd.isna(result.iloc[gap_index])
    for offset in range(1, _LOOKBACK):
        assert pd.isna(result.iloc[gap_index + offset])
    recovery = gap_index + _LOOKBACK
    assert np.isfinite(result.iloc[recovery])


def test_invalid_close_nan_at_row_prior_valid_window_unchanged() -> None:
    bad_index = 30
    closes = _long_closes(80)
    closes[bad_index] = 0.0
    result = PriceVsSma20().compute(_frame(closes))
    assert np.isfinite(result.iloc[bad_index - 1])
    assert pd.isna(result.iloc[bad_index])


def test_unsorted_index_raises() -> None:
    closes = _long_closes()
    frame = _frame(closes).iloc[[2, 0, 1, 3, 4] + list(range(5, len(closes)))]
    with pytest.raises(ValueError, match="monotonic"):
        PriceVsSma20().compute(frame)


def test_duplicate_index_raises() -> None:
    closes = _long_closes()[:10]
    frame = _frame(closes)
    dup_index = frame.index.tolist()
    dup_index[3] = dup_index[2]
    frame.index = pd.DatetimeIndex(dup_index)
    with pytest.raises(ValueError, match="unique"):
        PriceVsSma20().compute(frame)


def test_decimal_close_matches_float() -> None:
    closes = _long_closes()
    float_result = PriceVsSma20().compute(_frame(closes))
    decimal_result = PriceVsSma20().compute(_decimal_frame(closes))
    pd.testing.assert_series_equal(
        float_result,
        decimal_result,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_int_close_column() -> None:
    closes = _long_closes()[:50]
    int_closes = [int(round(c)) for c in closes]
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    frame = pd.DataFrame({"close": int_closes}, index=index)
    result = PriceVsSma20().compute(frame)
    float_frame = _frame([float(c) for c in int_closes])
    expected = PriceVsSma20().compute(float_frame)
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
    PriceVsSma20().compute(frame)
    pd.testing.assert_series_equal(frame["close"], before)


def test_past_values_do_not_depend_on_future_bars() -> None:
    closes = _long_closes()
    full = PriceVsSma20().compute(_frame(closes))
    prefix_len = 40
    prefix = PriceVsSma20().compute(_frame(closes[:prefix_len]))
    pd.testing.assert_series_equal(
        full.iloc[:prefix_len],
        prefix,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )

    frame = _frame(closes)
    before = PriceVsSma20().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
    after = PriceVsSma20().compute(mutated)
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

    feature = registry.get("price_vs_sma_20", 1)
    assert feature.spec.name == "price_vs_sma_20"
    assert feature.spec.version == 1
    assert feature.spec.inputs == ("close",)
    assert feature.spec.lookback_window == _LOOKBACK


def test_seed_entry_matches_registry_spec() -> None:
    import backend.features  # noqa: F401

    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    matches = [entry for entry in raw if entry["name"] == "price_vs_sma_20"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["description"].strip()
    feature = registry.get(entry["name"], entry["version"])
    assert feature.spec.lookback_window == entry["lookback_window"]
    assert feature.spec.version == entry["version"]
    assert feature.spec.inputs == ("close",)
