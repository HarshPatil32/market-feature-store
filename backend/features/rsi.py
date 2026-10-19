"""Wilder relative strength index (RSI) from close prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import register_feature
from backend.features.returns import clean_close


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. Non-finite price changes reset state; zero net movement is NaN."""
    values = close.to_numpy(dtype=float, copy=False)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    averages: tuple[float, float] | None = None
    seed_gains: list[float] = []
    seed_losses: list[float] = []

    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if not np.isfinite(delta):
            averages = None
            seed_gains.clear()
            seed_losses.clear()
            continue

        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if averages is None:
            seed_gains.append(gain)
            seed_losses.append(loss)
            if len(seed_gains) < period:
                continue
            averages = (
                sum(seed_gains) / period,
                sum(seed_losses) / period,
            )
        else:
            avg_gain, avg_loss = averages
            averages = (
                (avg_gain * (period - 1) + gain) / period,
                (avg_loss * (period - 1) + loss) / period,
            )

        avg_gain, avg_loss = averages
        denom = avg_gain + avg_loss
        if denom == 0.0:
            continue
        out[i] = 100.0 * avg_gain / denom

    return pd.Series(out, index=close.index)


@register_feature
class Rsi14(Feature):
    """Wilder's RSI over 14 bars; 15 closes required. Bad prices reset smoothing."""

    spec = FeatureSpec(
        name="rsi_14",
        version=1,
        inputs=("close",),
        lookback_window=15,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        close = clean_close(frame)
        return _wilder_rsi(close, 14)
