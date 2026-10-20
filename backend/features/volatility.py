"""Rolling volatility (sample std of daily returns) from close prices."""

from __future__ import annotations

import pandas as pd

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import register_feature
from backend.features.returns import DailyReturn


def _rolling_volatility(frame: pd.DataFrame, window: int) -> pd.Series:
    returns = DailyReturn().compute(frame)
    return returns.rolling(window, min_periods=window).std(ddof=1)


@register_feature
class Volatility5D(Feature):
    """Sample std (ddof=1) of simple daily returns over 5 bars; not annualized."""

    spec = FeatureSpec(
        name="volatility_5d",
        version=1,
        inputs=("close",),
        lookback_window=6,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _rolling_volatility(frame, 5)


@register_feature
class Volatility20D(Feature):
    """Sample std (ddof=1) of simple daily returns over 20 bars; not annualized."""

    spec = FeatureSpec(
        name="volatility_20d",
        version=1,
        inputs=("close",),
        lookback_window=21,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _rolling_volatility(frame, 20)


@register_feature
class Volatility60D(Feature):
    """Sample std (ddof=1) of simple daily returns over 60 bars; not annualized."""

    spec = FeatureSpec(
        name="volatility_60d",
        version=1,
        inputs=("close",),
        lookback_window=61,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _rolling_volatility(frame, 60)
