"""Compose providers so each does what it is actually good at.

NSE has the option chain and India VIX; Yahoo has clean historical OHLCV.
Neither alone is enough, and in a locked-down network neither may be reachable
at all -- hence the fixture fallback, which is *always* announced rather than
silently substituted.
"""

from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd

from alpha.data.base import ChainSnapshot, ProviderError, Quote
from alpha.data.fixtures import FixtureProvider

log = logging.getLogger(__name__)


class CompositeProvider:
    """Chain/VIX from ``chain_provider``, history from ``history_provider``.

    ``degraded`` accumulates human-readable notes whenever a fallback fires.
    Engines copy those into ``Signal.data_quality`` so the dashboard can show
    exactly which parts of a signal rest on real data.
    """

    name = "composite"

    def __init__(self, chain_provider=None, history_provider=None,
                 fallback: FixtureProvider | None = None):
        self.chain_provider = chain_provider
        self.history_provider = history_provider
        self.fallback = fallback or FixtureProvider()
        self.degraded: list[str] = []

    def _note(self, msg: str) -> None:
        if msg not in self.degraded:
            self.degraded.append(msg)
        log.warning(msg)

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        if self.history_provider is not None:
            try:
                return self.history_provider.ohlcv(symbol, interval, lookback)
            except ProviderError as exc:
                self._note(f"History for {symbol} is SIMULATED - live fetch failed: {exc}")
        else:
            self._note(f"History for {symbol} is SIMULATED - no live history provider configured.")
        return self.fallback.ohlcv(symbol, interval, lookback)

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        if self.chain_provider is not None:
            try:
                return self.chain_provider.option_chain(symbol, expiry)
            except ProviderError as exc:
                self._note(f"Option chain for {symbol} is SIMULATED - live fetch failed: {exc}")
        else:
            self._note(f"Option chain for {symbol} is SIMULATED - no live chain provider configured.")
        return self.fallback.option_chain(symbol, expiry)

    def quote(self, symbol: str) -> Quote:
        for p in (self.chain_provider, self.history_provider):
            if p is None:
                continue
            try:
                return p.quote(symbol)
            except ProviderError:
                continue
        self._note(f"Quote for {symbol} is SIMULATED.")
        return self.fallback.quote(symbol)

    def india_vix(self) -> float | None:
        for p in (self.chain_provider, self.history_provider):
            if p is None:
                continue
            try:
                v = p.india_vix()
                if v is not None:
                    return v
            except ProviderError:
                continue
        self._note("India VIX is SIMULATED - could not fetch the live value.")
        return self.fallback.india_vix()

    @property
    def is_live(self) -> bool:
        return not self.degraded


def build_provider(mode: str | None = None) -> CompositeProvider:
    """Build a provider from ``ALPHA_DATA_MODE``.

    ``live``     - NSE chains + Yahoo history, fixtures only on failure
    ``fixture``  - fully offline and deterministic (default; safe anywhere)
    """
    mode = (mode or os.getenv("ALPHA_DATA_MODE", "fixture")).lower()

    if mode == "fixture":
        fx = FixtureProvider()
        cp = CompositeProvider(fallback=fx)
        cp._note("Running in FIXTURE mode: all market data is simulated. "
                 "Set ALPHA_DATA_MODE=live for real data.")
        return cp

    if mode == "live":
        chain = history = None
        try:
            from alpha.data.nse import NSEProvider

            chain = NSEProvider()
        except Exception as exc:                       # noqa: BLE001
            log.warning("NSE provider unavailable: %s", exc)
        try:
            from alpha.data.yahoo import YahooProvider

            history = YahooProvider()
        except Exception as exc:                       # noqa: BLE001
            log.warning("Yahoo provider unavailable: %s", exc)
        return CompositeProvider(chain_provider=chain, history_provider=history)

    raise ValueError(f"unknown ALPHA_DATA_MODE {mode!r}; expected 'live' or 'fixture'")
