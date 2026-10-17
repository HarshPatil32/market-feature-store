"""Simple moving averages of close; windows are N bar rows, not calendar days."""

from __future__ import annotations

import pandas as pd

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import register_feature
from backend.features.returns import clean_close


def _sma(frame: pd.DataFrame, window: int) -> pd.Series:
    close = clean_close(frame)
    return close.rolling(window, min_periods=window).mean()


@register_feature
class Sma5(Feature):
    """Simple mean of last 5 valid closes; NaN if any in the window is invalid."""

    spec = FeatureSpec(
        name="sma_5",
        version=1,
        inputs=("close",),
        lookback_window=5,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _sma(frame, self.spec.lookback_window)


@register_feature
class Sma20(Feature):
    """Simple mean of last 20 valid closes; NaN if any in the window is invalid."""

    spec = FeatureSpec(
        name="sma_20",
        version=1,
        inputs=("close",),
        lookback_window=20,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _sma(frame, self.spec.lookback_window)


@register_feature
class Sma50(Feature):
    """Simple mean of last 50 valid closes; NaN if any in the window is invalid."""

    spec = FeatureSpec(
        name="sma_50",
        version=1,
        inputs=("close",),
        lookback_window=50,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _sma(frame, self.spec.lookback_window)


@register_feature
class Sma200(Feature):
    """Simple mean of last 200 valid closes; NaN if any in the window is invalid."""

    spec = FeatureSpec(
        name="sma_200",
        version=1,
        inputs=("close",),
        lookback_window=200,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _sma(frame, self.spec.lookback_window)


@register_feature
class PriceVsSma20(Feature):
    """Close relative to the 20-bar SMA as a fraction; NaN if close or any window close is invalid."""

    spec = FeatureSpec(
        name="price_vs_sma_20",
        version=1,
        inputs=("close",),
        lookback_window=20,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        close = clean_close(frame)
        return close / _sma(frame, self.spec.lookback_window) - 1
