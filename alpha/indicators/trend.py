"""Trend indicators. All functions take/return pandas Series and never mutate input."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def slope_pct(s: pd.Series, n: int = 5) -> pd.Series:
    """Percent change of a series over n bars -- used to ask whether a moving
    average is rising or merely sitting above price by accident."""
    return (s / s.shift(n) - 1.0) * 100.0


def ema_stack(close: pd.Series, fast: int = 20, mid: int = 50, slow: int = 200) -> dict:
    """Classify the moving-average structure.

    Stage analysis in one number: a stock with price > 20 > 50 > 200 and all
    rising is in a different regime from one where those are tangled.
    """
    e_f, e_m, e_s = ema(close, fast), ema(close, mid), ema(close, slow)
    px = close.iloc[-1]
    f, m, s = e_f.iloc[-1], e_m.iloc[-1], e_s.iloc[-1]

    if any(pd.isna(v) for v in (f, m, s)):
        return {"stage": "unknown", "aligned": 0.0, "px": float(px),
                "fast": None, "mid": None, "slow": None, "slow_slope": None}

    up = px > f > m > s
    down = px < f < m < s
    slow_slope = slope_pct(e_s, 20).iloc[-1]

    if up:
        stage = "uptrend"
    elif down:
        stage = "downtrend"
    elif px > s:
        stage = "above_long_term"
    else:
        stage = "below_long_term"

    # aligned in [-1, 1]: how cleanly the stack is ordered
    checks = [px > f, f > m, m > s]
    aligned = (sum(checks) - 1.5) / 1.5

    return {"stage": stage, "aligned": float(aligned), "px": float(px),
            "fast": float(f), "mid": float(m), "slow": float(s),
            "slow_slope": None if pd.isna(slow_slope) else float(slow_slope)}


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI.

    ADX answers "is there a trend at all", which matters more than direction for
    option buyers: a directional bet inside a range bleeds theta and dies.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    # Wilder smoothing == ewm(alpha=1/n)
    atr_w = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_w

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})
