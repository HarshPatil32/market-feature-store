"""Bar-to-bar and cumulative return features from close prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import register_feature


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"].astype(float)
    return close.where(np.isfinite(close) & (close > 0))


@register_feature
class DailyReturn(Feature):
    spec = FeatureSpec(
        name="daily_return",
        version=1,
        inputs=("close",),
        lookback_window=2,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        close = _clean_close(frame)
        prev = close.shift(1)
        return close / prev - 1


@register_feature
class LogReturn(Feature):
    spec = FeatureSpec(
        name="log_return",
        version=1,
        inputs=("close",),
        lookback_window=2,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        close = _clean_close(frame)
        prev = close.shift(1)
        return np.log(close / prev)


@register_feature
class CumulativeReturn(Feature):
    """Return vs the first valid close in the supplied frame (not global history)."""

    spec = FeatureSpec(
        name="cumulative_return",
        version=1,
        inputs=("close",),
        lookback_window=1,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        close = _clean_close(frame)
        first_valid = close.first_valid_index()
        if first_valid is None:
            return close
        base = close.loc[first_valid]
        return close / base - 1
