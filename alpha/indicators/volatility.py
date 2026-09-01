"""Volatility indicators. ATR drives stop placement and position sizing advice."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR as a percent of price -- comparable across instruments."""
    return atr(high, low, close, n) / close * 100.0


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised close-to-close realised volatility, in percent.

    Compared against implied vol this is the option buyer's core question:
    am I paying more for movement than the index has actually been delivering?
    """
    logret = np.log(close / close.shift(1))
    return logret.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year) * 100.0


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan) * 100.0
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width})
