"""Tests for daily, log, and cumulative return features."""

from __future__ import annotations

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from backend.features.registry import registry
from backend.features.returns import CumulativeReturn, DailyReturn, LogReturn

LN_1_1 = 0.0953101798
LN_0_9 = -0.1053605157

REFERENCE_CLOSES = [100.0, 110.0, 99.0, 99.0, 108.9]

_SERIES_ATOL = 1e-10


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=index)


def _decimal_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"close": [Decimal(str(c)) for c in closes]},
        index=index,
    )


@pytest.mark.parametrize(
    "feature_cls,expected",
    [
        (
            DailyReturn,
            [float("nan"), 0.10, -0.10, 0.0, 0.10],
        ),
        (
            LogReturn,
            [float("nan"), LN_1_1, LN_0_9, 0.0, LN_1_1],
        ),
        (
            CumulativeReturn,
            [0.0, 0.10, -0.01, -0.01, 0.089],
        ),
    ],
)
def test_reference_values_match_hand_computed(
    feature_cls: type,
    expected: list[float],
) -> None:
    feature = feature_cls()
    result = feature.compute(_frame(REFERENCE_CLOSES))
    expected_series = pd.Series(expected, index=result.index, dtype=float)
    pd.testing.assert_series_equal(
        result,
        expected_series,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_exp_log_return_minus_one_matches_daily_return() -> None:
    frame = _frame(REFERENCE_CLOSES)
    daily = DailyReturn().compute(frame)
    log_r = LogReturn().compute(frame)
    reconstructed = np.exp(log_r) - 1
    pd.testing.assert_series_equal(
        daily,
        reconstructed,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
        check_names=False,
    )


def test_cumulative_matches_cumprod_of_daily_returns() -> None:
    frame = _frame(REFERENCE_CLOSES)
    daily = DailyReturn().compute(frame)
    cum = CumulativeReturn().compute(frame)

    daily_filled = daily.copy()
    daily_filled.iloc[0] = 0.0
    from_cumprod = (1 + daily_filled).cumprod() - 1
    pd.testing.assert_series_equal(
        cum,
        from_cumprod,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
        check_names=False,
    )


def test_missing_price_no_forward_fill() -> None:
    closes = [100.0, 110.0, float("nan"), 99.0, 108.9]
    frame = _frame(closes)
    daily = DailyReturn().compute(frame)
    log_r = LogReturn().compute(frame)
    cum = CumulativeReturn().compute(frame)

    assert pd.isna(daily.iloc[2])
    assert pd.isna(daily.iloc[3])
    assert daily.iloc[4] == pytest.approx(0.10)
    assert pd.isna(log_r.iloc[2])
    assert pd.isna(log_r.iloc[3])
    assert log_r.iloc[4] == pytest.approx(LN_1_1, abs=_SERIES_ATOL)
    assert pd.isna(cum.iloc[2])
    assert cum.iloc[4] == pytest.approx(108.9 / 100.0 - 1)


def test_cumulative_anchors_on_first_valid_with_leading_nan() -> None:
    closes = [float("nan"), 100.0, 110.0]
    cum = CumulativeReturn().compute(_frame(closes))
    assert pd.isna(cum.iloc[0])
    assert cum.iloc[1] == pytest.approx(0.0)
    assert cum.iloc[2] == pytest.approx(0.10)


def test_first_invalid_close_daily_log_nan_cumulative_uses_next_valid() -> None:
    closes = [-5.0, 100.0, 110.0]
    frame = _frame(closes)
    daily = DailyReturn().compute(frame)
    log_r = LogReturn().compute(frame)
    cum = CumulativeReturn().compute(frame)

    assert pd.isna(daily.iloc[0])
    assert daily.iloc[2] == pytest.approx(0.10)
    assert pd.isna(log_r.iloc[0])
    assert log_r.iloc[2] == pytest.approx(LN_1_1, abs=_SERIES_ATOL)
    assert cum.iloc[1] == pytest.approx(0.0)
    assert cum.iloc[2] == pytest.approx(0.10)


@pytest.mark.parametrize(
    "bad_close",
    [0.0, -5.0, float("inf")],
)
def test_invalid_price_produces_nan_without_warning(bad_close: float) -> None:
    closes = [100.0, bad_close, 110.0]
    frame = _frame(closes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        daily = DailyReturn().compute(frame)
        log_r = LogReturn().compute(frame)
        cum = CumulativeReturn().compute(frame)

    assert pd.isna(daily.iloc[1])
    assert pd.isna(daily.iloc[2])
    assert pd.isna(log_r.iloc[1])
    assert pd.isna(log_r.iloc[2])
    assert pd.isna(cum.iloc[1])
    assert cum.iloc[2] == pytest.approx(0.10)


@pytest.mark.parametrize("feature_cls", [DailyReturn, LogReturn, CumulativeReturn])
def test_empty_frame_returns_empty_float_series(feature_cls: type) -> None:
    result = feature_cls().compute(_frame([]))
    assert len(result) == 0
    assert result.dtype == float


@pytest.mark.parametrize("feature_cls", [DailyReturn, LogReturn])
def test_all_nan_frame_all_nan(feature_cls: type) -> None:
    closes = [float("nan"), float("nan")]
    result = feature_cls().compute(_frame(closes))
    assert result.isna().all()


def test_cumulative_all_nan_frame() -> None:
    result = CumulativeReturn().compute(_frame([float("nan"), float("nan")]))
    assert result.isna().all()


def test_cumulative_single_valid_row_is_zero() -> None:
    result = CumulativeReturn().compute(_frame([100.0]))
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize("feature_cls", [DailyReturn, LogReturn, CumulativeReturn])
def test_unsorted_index_raises(feature_cls: type) -> None:
    frame = _frame(REFERENCE_CLOSES).iloc[[2, 0, 1, 3, 4]]
    with pytest.raises(ValueError, match="monotonic"):
        feature_cls().compute(frame)


@pytest.mark.parametrize("feature_cls", [DailyReturn, LogReturn, CumulativeReturn])
def test_decimal_close_matches_float(feature_cls: type) -> None:
    float_result = feature_cls().compute(_frame(REFERENCE_CLOSES))
    decimal_result = feature_cls().compute(_decimal_frame(REFERENCE_CLOSES))
    pd.testing.assert_series_equal(
        float_result,
        decimal_result,
        check_exact=False,
        rtol=1e-9,
        atol=_SERIES_ATOL,
        check_names=False,
    )


def test_int_close_column_computes_as_float() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    frame = pd.DataFrame({"close": [100, 110]}, index=index)
    result = DailyReturn().compute(frame)
    assert result.dtype == float
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)


def test_compute_does_not_mutate_frame() -> None:
    frame = _frame(REFERENCE_CLOSES)
    before = frame["close"].copy()
    DailyReturn().compute(frame)
    LogReturn().compute(frame)
    CumulativeReturn().compute(frame)
    pd.testing.assert_series_equal(frame["close"], before)


def test_registered_in_global_registry() -> None:
    import backend.features  # noqa: F401

    for name in ("daily_return", "log_return", "cumulative_return"):
        feature = registry.get(name, 1)
        assert feature.spec.name == name
        assert feature.spec.version == 1
        assert feature.spec.inputs == ("close",)
