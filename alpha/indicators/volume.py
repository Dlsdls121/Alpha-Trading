"""Volume indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Session VWAP. Expects intraday bars for a single session; for multi-day
    input the caller must group by session first, or this is just a running
    cumulative average and means nothing."""
    tp = (high + low + close) / 3.0
    cum_v = volume.cumsum()
    return (tp * volume).cumsum() / cum_v.replace(0, np.nan)


def relative_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    """Volume as a multiple of its own n-bar average. A breakout on 0.7x volume
    is a different animal from the same breakout on 2.5x."""
    avg = volume.rolling(n, min_periods=n).mean()
    return volume / avg.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()
