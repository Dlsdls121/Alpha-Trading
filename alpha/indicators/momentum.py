"""Momentum indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (smoothed, not simple-average)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # all-gain window -> RSI 100 rather than NaN
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ef = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def roc(close: pd.Series, n: int = 21) -> pd.Series:
    """Rate of change in percent over n bars."""
    return (close / close.shift(n) - 1.0) * 100.0


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14, d: int = 3) -> pd.DataFrame:
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"k": k, "d": k.rolling(d, min_periods=d).mean()})
