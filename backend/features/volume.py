"""Rolling average volume and volume spike ratio; windows are N bar rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import Feature, FeatureSpec
from backend.features.registry import register_feature


def clean_volume(frame: pd.DataFrame) -> pd.Series:
    volume = frame["volume"].astype(float)
    return volume.where(np.isfinite(volume) & (volume >= 0))


def _volume_sma(frame: pd.DataFrame, window: int) -> pd.Series:
    volume = clean_volume(frame)
    return volume.rolling(window, min_periods=window).mean()


@register_feature
class VolumeSma20(Feature):
    """Simple mean of last 20 valid volumes; NaN if any in the window is invalid."""

    spec = FeatureSpec(
        name="volume_sma_20",
        version=1,
        inputs=("volume",),
        lookback_window=20,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        return _volume_sma(frame, self.spec.lookback_window)


@register_feature
class VolumeSpikeRatio20(Feature):
    """Volume divided by its 20-bar SMA; NaN if any window volume is invalid or avg is zero."""

    spec = FeatureSpec(
        name="volume_spike_ratio_20",
        version=1,
        inputs=("volume",),
        lookback_window=20,
    )

    def _compute(self, frame: pd.DataFrame) -> pd.Series:
        volume = clean_volume(frame)
        avg = _volume_sma(frame, self.spec.lookback_window)
        return volume / avg.where(avg > 0)
