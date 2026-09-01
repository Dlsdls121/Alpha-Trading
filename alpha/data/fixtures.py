"""Deterministic offline provider.

This exists for three reasons, in order of importance:

1. Tests must not depend on a live exchange feed or on the market being open.
2. The dashboard has to be demonstrable at any hour, including weekends.
3. Some environments simply cannot reach NSE (locked-down networks, CI).

The generated series are *plausible*, not real. Every snapshot it produces is
tagged ``source="fixture"`` and carries a ``stale`` flag so the UI can shout
about it -- a synthetic signal must never be mistakable for a live one.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from alpha.calendar import next_expiry
from alpha.data.base import ChainRow, ChainSnapshot, Quote

# Roughly realistic anchors so charts and strikes look sane.
_ANCHORS: dict[str, tuple[float, float, float]] = {
    # symbol: (spot, annual vol, strike step)
    "NIFTY": (25_000.0, 0.13, 50.0),
    "BANKNIFTY": (55_000.0, 0.16, 100.0),
    "FINNIFTY": (26_500.0, 0.14, 50.0),
    "RELIANCE": (2_950.0, 0.22, 0.0),
    "TCS": (4_100.0, 0.20, 0.0),
    "HDFCBANK": (1_680.0, 0.19, 0.0),
    "INFY": (1_880.0, 0.23, 0.0),
    "ICICIBANK": (1_240.0, 0.21, 0.0),
    "SBIN": (830.0, 0.26, 0.0),
    "LT": (3_640.0, 0.22, 0.0),
    "AXISBANK": (1_150.0, 0.24, 0.0),
    "BHARTIARTL": (1_620.0, 0.21, 0.0),
    "ITC": (470.0, 0.17, 0.0),
    "MARUTI": (12_400.0, 0.22, 0.0),
    "TITAN": (3_380.0, 0.24, 0.0),
    "SUNPHARMA": (1_790.0, 0.20, 0.0),
    "TATAMOTORS": (980.0, 0.30, 0.0),
    "BAJFINANCE": (7_200.0, 0.26, 0.0),
    "HINDUNILVR": (2_480.0, 0.16, 0.0),
    "KOTAKBANK": (1_790.0, 0.20, 0.0),
    "ASIANPAINT": (2_920.0, 0.21, 0.0),
    "WIPRO": (545.0, 0.24, 0.0),
    "ULTRACEMCO": (11_300.0, 0.21, 0.0),
    "NESTLEIND": (2_530.0, 0.16, 0.0),
    "POWERGRID": (330.0, 0.20, 0.0),
    "NTPC": (400.0, 0.22, 0.0),
    "TATASTEEL": (165.0, 0.30, 0.0),
    "JSWSTEEL": (960.0, 0.27, 0.0),
    "ADANIENT": (2_780.0, 0.38, 0.0),
    "COALINDIA": (470.0, 0.26, 0.0),
    "INDIAVIX": (13.5, 0.0, 0.0),
}


def _seed_for(symbol: str, salt: str = "") -> int:
    """Stable per-symbol seed so the same symbol always yields the same series."""
    h = hashlib.sha256(f"{symbol}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


class FixtureProvider:
    """Synthetic but self-consistent market data."""

    name = "fixture"

    def __init__(self, as_of: date | None = None, seed_salt: str = "v1",
                 drift_bias: float | None = None):
        self.as_of = as_of or date.today()
        self.seed_salt = seed_salt
        self.drift_bias = drift_bias

    # -- helpers ---------------------------------------------------------

    def _anchor(self, symbol: str) -> tuple[float, float, float]:
        return _ANCHORS.get(symbol.upper(), (1_000.0, 0.25, 0.0))

    def _path(self, symbol: str, n: int, interval: str) -> np.ndarray:
        """Geometric Brownian motion with a mild per-symbol drift and a slow
        cyclical component, so trend/momentum indicators have something with
        actual structure to read rather than pure noise."""
        spot, vol, _ = self._anchor(symbol)
        rng = np.random.default_rng(_seed_for(symbol, self.seed_salt + interval))

        per_year = {"1d": 252, "60m": 252 * 7, "15m": 252 * 25, "5m": 252 * 75}.get(interval, 252)
        dt = 1.0 / per_year

        drift = self.drift_bias if self.drift_bias is not None else rng.normal(0.10, 0.30)
        shocks = rng.normal(0.0, 1.0, n)
        cycle = 0.35 * vol * np.sin(np.linspace(0, rng.uniform(2, 6) * np.pi, n))

        logret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks + cycle * dt
        path = spot * np.exp(np.cumsum(logret))
        # End the path at the anchor so "spot" is recognisable.
        return path * (spot / path[-1])

    # -- provider interface ----------------------------------------------

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        close = self._path(symbol, lookback, interval)
        rng = np.random.default_rng(_seed_for(symbol, self.seed_salt + "bars" + interval))
        _, vol, _ = self._anchor(symbol)

        noise = np.abs(rng.normal(0, vol / np.sqrt(252) * 0.6, lookback)) * close
        high = close + noise
        low = close - np.abs(rng.normal(0, vol / np.sqrt(252) * 0.6, lookback)) * close
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum.reduce([high, close, open_])
        low = np.minimum.reduce([low, close, open_])

        base_vol = {"NIFTY": 2.4e8, "BANKNIFTY": 1.6e8}.get(symbol.upper(), 4.0e6)
        volume = np.abs(rng.lognormal(np.log(base_vol), 0.45, lookback))

        if interval == "1d":
            idx = pd.bdate_range(end=pd.Timestamp(self.as_of), periods=lookback)
        else:
            minutes = {"60m": 60, "15m": 15, "5m": 5}.get(interval, 60)
            idx = pd.date_range(end=pd.Timestamp(self.as_of) + pd.Timedelta(hours=15, minutes=30),
                                periods=lookback, freq=f"{minutes}min")

        return pd.DataFrame({"open": open_, "high": high, "low": low,
                             "close": close, "volume": volume}, index=idx)

    def quote(self, symbol: str) -> Quote:
        df = self.ohlcv(symbol, "1d", 30)
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        return Quote(symbol=symbol.upper(), last=last,
                     change_pct=(last / prev - 1) * 100, timestamp=datetime.now(),
                     prev_close=prev, day_high=float(df["high"].iloc[-1]),
                     day_low=float(df["low"].iloc[-1]), day_open=float(df["open"].iloc[-1]))

    def india_vix(self) -> float:
        rng = np.random.default_rng(_seed_for("INDIAVIX", self.seed_salt))
        return float(np.clip(rng.normal(13.5, 3.0), 8.0, 35.0))

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        """Chain with a volatility smile and a plausible OI distribution.

        OI is placed as real chains tend to look: put OI concentrated below spot
        and call OI above it, both peaking at round strikes, because that is
        where writers sit. This gives the support/resistance and max-pain code
        something structurally realistic to chew on.
        """
        from alpha.indicators.options import bs_price

        sym = symbol.upper()
        spot, base_vol, step = self._anchor(sym)
        if step == 0:
            step = 50.0
        expiry = expiry or next_expiry(sym, self.as_of)
        t = max((expiry - self.as_of).days, 0) / 365.0 or (2 / 24) / 365.0

        atm = round(spot / step) * step
        strikes = [atm + i * step for i in range(-15, 16)]
        rng = np.random.default_rng(_seed_for(sym, self.seed_salt + "chain" + str(expiry)))

        # Slight bullish skew in OI placement, varying by symbol.
        skew = rng.uniform(-0.35, 0.35)
        rows: list[ChainRow] = []

        for k in strikes:
            m = (k - spot) / spot                                   # moneyness
            # Smile: IV rises away from ATM, puts a touch richer (crash skew).
            iv = base_vol * (1.0 + 6.0 * m**2 + (-0.55 * m))
            iv = float(np.clip(iv, 0.05, 1.2))

            # OI peaks ~1.5% out on each side, decaying with distance.
            call_w = np.exp(-((m - 0.015 - 0.004 * skew) ** 2) / (2 * 0.018**2))
            put_w = np.exp(-((m + 0.015 - 0.004 * skew) ** 2) / (2 * 0.018**2))
            round_bonus = 1.6 if abs(k % (step * 10)) < 1e-6 else 1.0

            scale = 9.0e5 if sym in ("NIFTY", "BANKNIFTY") else 1.2e5
            ce_oi = int(call_w * scale * round_bonus * rng.uniform(0.7, 1.3))
            pe_oi = int(put_w * scale * round_bonus * rng.uniform(0.7, 1.3))

            ce_ltp = round(bs_price(spot, k, t, iv, 0.065, "CE"), 2)
            pe_ltp = round(bs_price(spot, k, t, iv, 0.065, "PE"), 2)

            rows.append(ChainRow(
                strike=float(k),
                ce_oi=ce_oi, ce_change_oi=int(ce_oi * rng.uniform(-0.18, 0.28)),
                ce_ltp=ce_ltp, ce_iv=round(iv * 100, 2),
                ce_volume=int(ce_oi * rng.uniform(0.15, 0.9)),
                ce_bid=round(ce_ltp * 0.995, 2), ce_ask=round(ce_ltp * 1.005, 2),
                pe_oi=pe_oi, pe_change_oi=int(pe_oi * rng.uniform(-0.18, 0.28)),
                pe_ltp=pe_ltp, pe_iv=round(iv * 100, 2),
                pe_volume=int(pe_oi * rng.uniform(0.15, 0.9)),
                pe_bid=round(pe_ltp * 0.995, 2), pe_ask=round(pe_ltp * 1.005, 2),
            ))

        return ChainSnapshot(symbol=sym, expiry=expiry, spot=spot,
                             timestamp=datetime.now(), rows=rows,
                             source="fixture", stale=True)
