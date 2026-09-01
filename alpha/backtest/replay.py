"""Point-in-time data replay.

The whole value of a backtest rests on one property: **at simulated date T the
engine must not be able to see anything that happened after T.** Lookahead bias
is the reason most home-made backtests show an edge that evaporates live, and it
is easy to introduce by accident -- one `.iloc[-1]` on an untruncated frame is
enough.

So it is prevented structurally rather than by care. :class:`PointInTimeProvider`
owns the full history and hands out only the slice up to ``as_of``. The engines
are given no way to reach the underlying frame, and
``test_backtest_replay.py`` asserts the guarantee directly by comparing what the
provider returns against the future rows it is hiding.

A limitation stated up front, because it bounds what any result here can mean:
**historical NSE option chains are not freely available.** Underlying prices are
real history; the option chain at date T is *synthesised* from the underlying's
trailing realised volatility. That means the price-based factors (trend,
momentum, VWAP, structure) are genuinely tested, while the OI-based positioning
factors (PCR, max pain, OI support/resistance, OI buildup) are being scored
against invented data and are therefore **not validated**. Use
``exclude_synthetic_factors=True`` to run without them and get an honest read on
the part that is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from alpha.data.base import ChainRow, ChainSnapshot, ProviderError, Quote

# Factors whose inputs are synthetic under replay, so any result involving them
# says nothing about the real world.
SYNTHETIC_CHAIN_FACTORS = frozenset({"pcr", "max_pain", "oi_levels", "oi_buildup"})


class LookaheadError(RuntimeError):
    """Raised when something asks for data at or after the simulated date."""


@dataclass
class HistoryStore:
    """Full history for every symbol, loaded once and replayed many times.

    Loading is the slow part of a backtest -- a 2-year daily scan over 27 names
    is thousands of provider calls if done naively. This loads each series once
    and every simulated date slices the same frame.
    """

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    vix: pd.Series | None = None
    source_notes: list[str] = field(default_factory=list)
    synthetic: bool = False
    """True when the underlying price history is generated rather than real.

    This is load-bearing, not bookkeeping. Generated paths contain structure that
    real markets do not -- the bundled fixture generator adds a smooth sinusoidal
    cycle -- and a trend-following engine predicts that structure almost
    perfectly. A backtest over such data produces spectacular numbers that mean
    exactly nothing, so the runner refuses to present them as findings.
    """

    @classmethod
    def load(cls, provider, symbols: list[str], lookback: int = 1200) -> "HistoryStore":
        frames: dict[str, pd.DataFrame] = {}
        notes: list[str] = []
        for sym in symbols:
            try:
                df = provider.ohlcv(sym, "1d", lookback)
            except Exception as exc:                          # noqa: BLE001
                notes.append(f"{sym}: history unavailable ({exc})")
                continue
            if df is None or df.empty:
                notes.append(f"{sym}: empty history")
                continue
            df = df.sort_index()
            df.index = pd.to_datetime(df.index).normalize()
            frames[sym.upper()] = df

        notes.extend(getattr(provider, "degraded", []))

        # Detect generated history from the provider itself, and from the
        # degradation notes when a composite silently fell back to fixtures.
        synthetic = (getattr(provider, "name", "") == "fixture"
                     or any("SIMULATED" in n or "FIXTURE" in n.upper() for n in notes))
        if synthetic:
            notes.insert(0, "PRICE HISTORY IS GENERATED, NOT REAL. Any performance "
                            "number from this run is meaningless.")

        return cls(frames=frames, source_notes=list(dict.fromkeys(notes)),
                   synthetic=synthetic)

    def trading_dates(self, symbol: str) -> list[date]:
        df = self.frames.get(symbol.upper())
        if df is None:
            return []
        return [d.date() for d in df.index]

    def common_dates(self, symbols: list[str] | None = None) -> list[date]:
        """Dates present for every requested symbol."""
        syms = [s.upper() for s in (symbols or self.frames.keys())]
        sets = [set(self.trading_dates(s)) for s in syms if s in self.frames]
        if not sets:
            return []
        return sorted(set.intersection(*sets))

    def future(self, symbol: str, after: date) -> pd.DataFrame:
        """Bars strictly after ``after`` -- for outcome evaluation ONLY.

        Deliberately not reachable from a :class:`PointInTimeProvider`: only the
        evaluator, which runs after a signal is fixed, may call this.
        """
        df = self.frames.get(symbol.upper())
        if df is None:
            return pd.DataFrame()
        return df[df.index.date > after]


class PointInTimeProvider:
    """Provider view frozen at ``as_of``.

    Satisfies the same interface the engines already use, so no engine code
    changes for backtesting -- which matters, because a backtest that exercises
    different code from production tests the wrong thing.
    """

    name = "point-in-time"

    def __init__(self, store: HistoryStore, as_of: date,
                 synthesise_chains: bool = True, rate: float = 0.065):
        self.store = store
        self.as_of = as_of
        self.synthesise_chains = synthesise_chains
        self.rate = rate
        self.degraded: list[str] = []
        if synthesise_chains:
            self._note("Option chains under replay are SYNTHETIC (built from trailing "
                       "realised volatility). Historical NSE chains are not freely "
                       "available, so OI-based factors are not validated by this run.")

    def _note(self, msg: str) -> None:
        if msg not in self.degraded:
            self.degraded.append(msg)

    # -- the guarantee ---------------------------------------------------

    def _visible(self, symbol: str) -> pd.DataFrame:
        df = self.store.frames.get(symbol.upper())
        if df is None:
            raise ProviderError(f"no history loaded for {symbol}")
        # Inclusive of as_of: a decision made after the close of day T may use
        # day T's bar. Nothing after it is reachable.
        return df[df.index.date <= self.as_of]

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        if interval != "1d":
            # Intraday history is not stored for replay. The VWAP factor already
            # degrades to weight 0 when this raises, which is the correct
            # outcome -- far better than fabricating an intraday path.
            raise ProviderError(
                f"replay has no {interval} history; only daily bars are stored")
        visible = self._visible(symbol)
        if visible.empty:
            raise ProviderError(f"no data for {symbol} at or before {self.as_of}")
        return visible.tail(lookback)

    def quote(self, symbol: str) -> Quote:
        df = self.ohlcv(symbol, "1d", 5)
        if len(df) < 2:
            raise ProviderError(f"not enough history to quote {symbol} at {self.as_of}")
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        return Quote(symbol=symbol.upper(), last=last, change_pct=(last / prev - 1) * 100,
                     timestamp=datetime.combine(self.as_of, datetime.min.time()),
                     prev_close=prev, day_high=float(df["high"].iloc[-1]),
                     day_low=float(df["low"].iloc[-1]), day_open=float(df["open"].iloc[-1]))

    def india_vix(self) -> float | None:
        """Trailing realised volatility as a VIX stand-in.

        India VIX history is not stored here. Realised vol is a genuine proxy at
        index level but it is not the same series, and the note says so rather
        than letting a factor quote it as if it were the real index.
        """
        try:
            df = self.ohlcv("NIFTY", "1d", 40)
        except ProviderError:
            return None
        from alpha.indicators import realized_vol

        rv = realized_vol(df["close"], 20).dropna()
        if rv.empty:
            return None
        self._note("India VIX under replay is a realised-volatility proxy, not the "
                   "actual index.")
        return float(rv.iloc[-1])

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        """Synthesise a chain from the underlying's state at ``as_of``.

        Priced with Black-Scholes off trailing realised vol plus a smile. The
        *prices* are therefore internally consistent with the real underlying
        path, which is what strike selection and theta need. The *open interest*
        is invented, which is why the OI factors cannot be validated here.
        """
        if not self.synthesise_chains:
            raise ProviderError("chain synthesis disabled for this replay")

        from alpha.calendar import next_expiry
        from alpha.indicators import realized_vol
        from alpha.indicators.options import bs_price

        sym = symbol.upper()
        df = self.ohlcv(sym, "1d", 60)
        spot = float(df["close"].iloc[-1])

        rv = realized_vol(df["close"], 20).dropna()
        base_vol = float(rv.iloc[-1]) / 100.0 if not rv.empty else 0.15
        base_vol = float(np.clip(base_vol, 0.05, 1.0))

        expiry = expiry or next_expiry(sym, self.as_of)
        t = max((expiry - self.as_of).days, 0) / 365.0 or (2 / 24) / 365.0

        step = 100.0 if sym == "BANKNIFTY" else 50.0
        atm = round(spot / step) * step
        # Seeded by date so a replayed run is reproducible.
        rng = np.random.default_rng(abs(hash((sym, self.as_of.toordinal()))) % (2**32))

        rows: list[ChainRow] = []
        for i in range(-15, 16):
            k = atm + i * step
            if k <= 0:
                continue
            m = (k - spot) / spot
            iv = float(np.clip(base_vol * (1.0 + 6.0 * m**2 - 0.55 * m), 0.05, 1.5))

            call_w = float(np.exp(-((m - 0.015) ** 2) / (2 * 0.018**2)))
            put_w = float(np.exp(-((m + 0.015) ** 2) / (2 * 0.018**2)))
            scale = 9.0e5 if sym in ("NIFTY", "BANKNIFTY") else 1.2e5
            ce_oi = int(call_w * scale * rng.uniform(0.7, 1.3))
            pe_oi = int(put_w * scale * rng.uniform(0.7, 1.3))

            ce = round(bs_price(spot, k, t, iv, self.rate, "CE"), 2)
            pe = round(bs_price(spot, k, t, iv, self.rate, "PE"), 2)

            rows.append(ChainRow(
                strike=float(k),
                ce_oi=ce_oi, ce_change_oi=int(ce_oi * rng.uniform(-0.15, 0.25)),
                ce_ltp=ce, ce_iv=round(iv * 100, 2),
                ce_volume=int(ce_oi * rng.uniform(0.2, 0.8)),
                ce_bid=round(ce * 0.995, 2), ce_ask=round(ce * 1.005, 2),
                pe_oi=pe_oi, pe_change_oi=int(pe_oi * rng.uniform(-0.15, 0.25)),
                pe_ltp=pe, pe_iv=round(iv * 100, 2),
                pe_volume=int(pe_oi * rng.uniform(0.2, 0.8)),
                pe_bid=round(pe * 0.995, 2), pe_ask=round(pe * 1.005, 2),
            ))

        return ChainSnapshot(symbol=sym, expiry=expiry, spot=spot,
                             timestamp=datetime.combine(self.as_of, datetime.min.time()),
                             rows=rows, source="replay-synthetic", stale=True)
