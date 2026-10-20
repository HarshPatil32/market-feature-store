"""Tests for rolling momentum (close-to-close return) features."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.registry import registry
from backend.features.returns import (
    Momentum20D,
    Momentum60D,
    Return20D,
    clean_close,
)

_SERIES_ATOL = 1e-10
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_DEFINITIONS_PATH = _REPO_ROOT / "config" / "feature_definitions.json"

_MOMENTUM_FEATURES: list[tuple[type, int, int]] = [
    (Momentum20D, 20, 21),
    (Momentum60D, 60, 61),
]


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=index)


def _long_closes(n: int = 70) -> list[float]:
    return [100.0 * (1.01**i) for i in range(n)]


def _series_long_enough(periods: int) -> list[float]:
    return _long_closes(max(70, periods * 2 + 15))


def _hand_rolling(closes: list[float], periods: int, lookback_window: int) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    close = close.where(np.isfinite(close) & (close > 0))
    result = (close / close.shift(periods) - 1).astype(float)
    warmup = lookback_window - 1
    if warmup > 0:
        result = result.copy()
        result.iloc[:warmup] = float("nan")
    return result


@pytest.mark.parametrize(
    "feature_cls,periods,lookback",
    _MOMENTUM_FEATURES,
)
def test_momentum_matches_hand_computed(
    feature_cls: type,
    periods: int,
    lookback: int,
) -> None:
    closes = _long_closes()
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    expected = _hand_rolling(closes, periods, lookback)
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )
    assert np.isfinite(result.iloc[periods])
    assert result.iloc[periods] == pytest.approx(1.01**periods - 1, rel=1e-9)


@pytest.mark.parametrize(
    "feature_cls,periods,base_close,final_close,expected_return",
    [
        (Momentum20D, 20, 100.0, 150.0, 0.5),
        (Momentum60D, 60, 200.0, 300.0, 0.5),
    ],
)
def test_momentum_literal_constant_then_jump(
    feature_cls: type,
    periods: int,
    base_close: float,
    final_close: float,
    expected_return: float,
) -> None:
    closes = [base_close] * periods + [final_close]
    result = feature_cls().compute(_frame(closes))
    assert result.iloc[:periods].isna().all()
    assert result.iloc[periods] == pytest.approx(expected_return)


@pytest.mark.parametrize(
    "feature_cls,periods",
    [(Momentum20D, 20), (Momentum60D, 60)],
)
def test_momentum_leading_edge_nan(
    feature_cls: type,
    periods: int,
) -> None:
    closes = _long_closes()
    result = feature_cls().compute(_frame(closes))
    assert result.iloc[:periods].isna().all()
    assert np.isfinite(result.iloc[periods])

    short = feature_cls().compute(_frame(closes[:periods]))
    assert short.isna().all()

    one_finite = feature_cls().compute(_frame(closes[: periods + 1]))
    assert one_finite.iloc[:-1].isna().all()
    assert np.isfinite(one_finite.iloc[-1])


@pytest.mark.parametrize("feature_cls,periods", [(Momentum20D, 20), (Momentum60D, 60)])
def test_momentum_off_by_one_guard(feature_cls: type, periods: int) -> None:
    closes = [100.0, 102.0, 101.0, 105.0, 103.0, 110.0, 108.0, 115.0]
    closes = closes + _long_closes(max(70, periods + 10))
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    close = clean_close(frame)
    i = periods + 5
    assert result.iloc[i] == pytest.approx(
        close.iloc[i] / close.iloc[i - periods] - 1,
        rel=1e-9,
    )
    four = close / close.shift(periods - 1) - 1
    six = close / close.shift(periods + 1) - 1
    assert result.iloc[i] != pytest.approx(four.iloc[i], rel=1e-6)
    assert result.iloc[i] != pytest.approx(six.iloc[i], rel=1e-6)


@pytest.mark.parametrize("feature_cls,periods", [(Momentum20D, 20), (Momentum60D, 60)])
def test_momentum_missing_price_no_forward_fill(
    feature_cls: type,
    periods: int,
) -> None:
    closes = _series_long_enough(periods)
    gap_index = periods + 3
    closes[gap_index] = float("nan")
    frame = _frame(closes)
    result = feature_cls().compute(frame)
    assert pd.isna(result.iloc[gap_index])
    assert pd.isna(result.iloc[gap_index + periods])
    filled = closes.copy()
    filled[gap_index] = 100.0
    without_gap = feature_cls().compute(_frame(filled))
    assert np.isfinite(without_gap.iloc[gap_index + periods])


@pytest.mark.parametrize(
    "bad_close",
    [0.0, -5.0, float("inf")],
)
@pytest.mark.parametrize("feature_cls,periods", [(Momentum20D, 20), (Momentum60D, 60)])
def test_momentum_invalid_price_produces_nan_without_warning(
    bad_close: float,
    feature_cls: type,
    periods: int,
) -> None:
    closes = _series_long_enough(periods)
    bad_index = periods + 3
    closes[bad_index] = bad_close
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_cls().compute(frame)
    assert pd.isna(result.iloc[bad_index])
    assert pd.isna(result.iloc[bad_index + periods])


@pytest.mark.parametrize("feature_cls,periods", [(Momentum20D, 20), (Momentum60D, 60)])
def test_momentum_no_lookahead_on_prefix(feature_cls: type, periods: int) -> None:
    closes = _long_closes()
    full = feature_cls().compute(_frame(closes))
    prefix_len = periods + 15
    prefix = feature_cls().compute(_frame(closes[:prefix_len]))
    pd.testing.assert_series_equal(
        full.iloc[:prefix_len],
        prefix,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


@pytest.mark.parametrize("feature_cls", [Momentum20D, Momentum60D])
def test_momentum_future_close_change_does_not_affect_past(feature_cls: type) -> None:
    closes = _long_closes()
    frame = _frame(closes)
    before = feature_cls().compute(frame)
    mutated = frame.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
    after = feature_cls().compute(mutated)
    pd.testing.assert_series_equal(
        before.iloc[:-1],
        after.iloc[:-1],
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_momentum_20d_equals_return_20d() -> None:
    frame = _frame(_long_closes())
    momentum = Momentum20D().compute(frame)
    rolling_return = Return20D().compute(frame)
    pd.testing.assert_series_equal(
        momentum,
        rolling_return,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_registered_in_global_registry() -> None:
    import backend.features  # noqa: F401

    expected_lookback = {
        "momentum_20d": 21,
        "momentum_60d": 61,
    }
    for name, lookback in expected_lookback.items():
        feature = registry.get(name, 1)
        assert feature.spec.name == name
        assert feature.spec.version == 1
        assert feature.spec.inputs == ("close",)
        assert feature.spec.lookback_window == lookback


@pytest.mark.parametrize("name", ["momentum_20d", "momentum_60d"])
def test_momentum_seed_entries_match_registry_specs(name: str) -> None:
    import backend.features  # noqa: F401

    raw = json.loads(_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    matches = [entry for entry in raw if entry["name"] == name]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["description"].strip()
    feature = registry.get(entry["name"], entry["version"])
    assert feature.spec.lookback_window == entry["lookback_window"]
    assert feature.spec.version == entry["version"]
    assert feature.spec.inputs == ("close",)
