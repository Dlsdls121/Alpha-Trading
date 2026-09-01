"""Data provider interface.

Every provider returns the same shapes so engines never know or care where the
numbers came from. That is what lets the same signal code run against a live NSE
feed, a cached snapshot, or deterministic fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class Quote:
    symbol: str
    last: float
    change_pct: float
    timestamp: datetime
    prev_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    day_open: float | None = None


@dataclass
class ChainRow:
    """One strike, both sides. Any field may be None -- illiquid strikes really
    do come back empty, and pretending otherwise produces confident nonsense."""

    strike: float
    ce_oi: int | None = None
    ce_change_oi: int | None = None
    ce_ltp: float | None = None
    ce_iv: float | None = None
    ce_volume: int | None = None
    ce_bid: float | None = None
    ce_ask: float | None = None
    pe_oi: int | None = None
    pe_change_oi: int | None = None
    pe_ltp: float | None = None
    pe_iv: float | None = None
    pe_volume: int | None = None
    pe_bid: float | None = None
    pe_ask: float | None = None

    def as_dict(self) -> dict:
        return {
            "strike": self.strike,
            "ce_oi": self.ce_oi, "ce_change_oi": self.ce_change_oi, "ce_ltp": self.ce_ltp,
            "ce_iv": self.ce_iv, "ce_volume": self.ce_volume,
            "pe_oi": self.pe_oi, "pe_change_oi": self.pe_change_oi, "pe_ltp": self.pe_ltp,
            "pe_iv": self.pe_iv, "pe_volume": self.pe_volume,
        }


@dataclass
class ChainSnapshot:
    symbol: str
    expiry: date
    spot: float
    timestamp: datetime
    rows: list[ChainRow] = field(default_factory=list)
    source: str = "unknown"
    stale: bool = False

    def as_dicts(self) -> list[dict]:
        return [r.as_dict() for r in self.rows]

    def atm_strike(self) -> float | None:
        if not self.rows:
            return None
        return min((r.strike for r in self.rows), key=lambda k: abs(k - self.spot))

    def row(self, strike: float) -> ChainRow | None:
        for r in self.rows:
            if abs(r.strike - strike) < 1e-6:
                return r
        return None

    def near_atm(self, n: int = 10) -> list[ChainRow]:
        """The n strikes closest to spot.

        Chain-wide PCR is dominated by far strikes that nobody trades; the
        near-ATM band is where positioning actually reflects a view.
        """
        return sorted(self.rows, key=lambda r: abs(r.strike - self.spot))[:n]


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        """DatetimeIndex, columns: open, high, low, close, volume."""
        ...

    def quote(self, symbol: str) -> Quote: ...

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot: ...

    def india_vix(self) -> float | None: ...


class ProviderError(RuntimeError):
    """Raised when a provider cannot supply data. Engines must degrade with a
    visible data-quality note rather than silently substituting a default."""
