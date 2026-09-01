"""Shared feature extraction used by both engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.indicators import adx, atr, atr_pct, ema, macd, realized_vol, roc, rsi, vwap
from alpha.indicators.trend import ema_stack, slope_pct
from alpha.indicators.volume import relative_volume


def last_session(intraday: pd.DataFrame) -> pd.DataFrame:
    """Slice the most recent trading session out of an intraday frame.

    VWAP is a session statistic. Running it over a multi-day frame produces a
    slowly-drifting cumulative average that looks like VWAP and is not.
    """
    if intraday.empty:
        return intraday
    last_day = intraday.index[-1].date()
    return intraday[intraday.index.date == last_day]


@dataclass
class TrendFeatures:
    stack: dict
    adx: float | None
    plus_di: float | None
    minus_di: float | None
    rsi: float | None
    rsi_prev: float | None
    macd_hist: float | None
    macd_hist_prev: float | None
    roc21: float | None
    atr: float | None
    atr_pct: float | None
    realized_vol: float | None
    close: float
    prev_close: float | None
    ema20_slope: float | None

    @property
    def rsi_rising(self) -> bool | None:
        if self.rsi is None or self.rsi_prev is None:
            return None
        return self.rsi > self.rsi_prev


def _last(series: pd.Series, offset: int = 0):
    """Last (or n-back) non-NaN-safe scalar, or None."""
    if series is None or len(series) <= offset:
        return None
    val = series.iloc[-1 - offset]
    return None if pd.isna(val) else float(val)


def extract_trend(daily: pd.DataFrame) -> TrendFeatures:
    """Compute the daily technical picture once, so factors just read it."""
    close, high, low = daily["close"], daily["high"], daily["low"]
    adx_df = adx(high, low, close, 14)
    macd_df = macd(close)
    r = rsi(close, 14)

    return TrendFeatures(
        stack=ema_stack(close),
        adx=_last(adx_df["adx"]),
        plus_di=_last(adx_df["plus_di"]),
        minus_di=_last(adx_df["minus_di"]),
        rsi=_last(r),
        rsi_prev=_last(r, 1),
        macd_hist=_last(macd_df["hist"]),
        macd_hist_prev=_last(macd_df["hist"], 1),
        roc21=_last(roc(close, 21)),
        atr=_last(atr(high, low, close, 14)),
        atr_pct=_last(atr_pct(high, low, close, 14)),
        realized_vol=_last(realized_vol(close, 20)),
        close=float(close.iloc[-1]),
        prev_close=_last(close, 1),
        ema20_slope=_last(slope_pct(ema(close, 20), 5)),
    )


@dataclass
class IntradayFeatures:
    vwap: float | None
    close: float | None
    pct_from_vwap: float | None
    session_high: float | None
    session_low: float | None
    bars: int
    rel_volume: float | None


def extract_intraday(intraday: pd.DataFrame | None) -> IntradayFeatures:
    if intraday is None or intraday.empty:
        return IntradayFeatures(None, None, None, None, None, 0, None)

    sess = last_session(intraday)
    if sess.empty or len(sess) < 2:
        return IntradayFeatures(None, None, None, None, None, len(sess), None)

    v = vwap(sess["high"], sess["low"], sess["close"], sess["volume"])
    vw = _last(v)
    px = float(sess["close"].iloc[-1])
    rv = relative_volume(intraday["volume"], 20)

    return IntradayFeatures(
        vwap=vw, close=px,
        pct_from_vwap=None if not vw else (px / vw - 1.0) * 100.0,
        session_high=float(sess["high"].max()), session_low=float(sess["low"].min()),
        bars=len(sess), rel_volume=_last(rv),
    )


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0 if old else 0.0


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def scale(value: float, lo: float, hi: float) -> float:
    """Map ``value`` from [lo, hi] onto [-1, +1], clamped at the ends."""
    if hi == lo:
        return 0.0
    return clamp(2.0 * (value - lo) / (hi - lo) - 1.0)
