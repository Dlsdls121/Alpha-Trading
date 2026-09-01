"""Historical OHLCV from Yahoo Finance's chart endpoint.

Used for daily and intraday history, which NSE does not serve conveniently.
NSE symbols map as ``RELIANCE`` -> ``RELIANCE.NS``; indices have their own
tickers (``^NSEI``, ``^NSEBANK``).

No API key is needed. This is an undocumented endpoint and can change, so
failures raise :class:`ProviderError` loudly.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from alpha.data.base import ProviderError, Quote

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

INDEX_TICKERS = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MIDCAP_100.NS",
}

_INTERVAL_MAP = {"1d": "1d", "1wk": "1wk", "60m": "60m", "15m": "15m", "5m": "5m"}
# Yahoo caps intraday history; ask for a range that is actually allowed.
_RANGE_FOR = {"1d": "2y", "1wk": "5y", "60m": "60d", "15m": "60d", "5m": "60d"}


def to_ticker(symbol: str) -> str:
    s = symbol.upper().replace(" ", "")
    if s in INDEX_TICKERS:
        return INDEX_TICKERS[s]
    return s if s.startswith("^") or "." in s else f"{s}.NS"


class YahooProvider:
    name = "yahoo"

    def __init__(self, timeout: float = 20.0, cache_dir: str = ".cache",
                 ttl: float = 300.0):
        import httpx

        self._httpx = httpx
        self.timeout = timeout
        from alpha.data.cache import DiskCache

        self.cache = DiskCache(cache_dir)
        self.ttl = ttl

    def _fetch(self, symbol: str, interval: str) -> dict:
        ticker = to_ticker(symbol)
        iv = _INTERVAL_MAP.get(interval, "1d")
        rng = _RANGE_FOR.get(iv, "2y")
        key = f"yf_{ticker}_{iv}"

        cached = self.cache.get(key, self.ttl)
        if cached is not None:
            return cached

        url = CHART.format(ticker=ticker)
        try:
            r = self._httpx.get(url, params={"range": rng, "interval": iv},
                                timeout=self.timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
        except Exception as exc:
            raise ProviderError(f"Yahoo request failed for {ticker}: {exc}") from exc

        if r.status_code != 200:
            raise ProviderError(f"Yahoo returned HTTP {r.status_code} for {ticker}")

        payload = r.json()
        err = (payload.get("chart") or {}).get("error")
        if err:
            raise ProviderError(f"Yahoo error for {ticker}: {err}")

        self.cache.set(key, payload)
        return payload

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        payload = self._fetch(symbol, interval)
        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            raise ProviderError(f"Yahoo returned no data for {symbol}")

        res = results[0]
        stamps = res.get("timestamp") or []
        quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        if not stamps:
            raise ProviderError(f"Yahoo returned an empty series for {symbol}")

        df = pd.DataFrame({
            "open": quote.get("open"), "high": quote.get("high"),
            "low": quote.get("low"), "close": quote.get("close"),
            "volume": quote.get("volume"),
        }, index=pd.to_datetime(stamps, unit="s"))

        # Yahoo emits nulls for halted/holiday bars; a NaN close breaks every
        # indicator downstream, so drop those rows rather than forward-filling
        # a price that never traded.
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0.0)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df.empty:
            raise ProviderError(f"Yahoo data for {symbol} was entirely null")
        return df.tail(lookback)

    def quote(self, symbol: str) -> Quote:
        df = self.ohlcv(symbol, "1d", 5)
        if len(df) < 2:
            raise ProviderError(f"not enough history to quote {symbol}")
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        return Quote(symbol=symbol.upper(), last=last,
                     change_pct=(last / prev - 1) * 100,
                     timestamp=datetime.fromtimestamp(df.index[-1].timestamp()),
                     prev_close=prev, day_high=float(df["high"].iloc[-1]),
                     day_low=float(df["low"].iloc[-1]), day_open=float(df["open"].iloc[-1]))

    def india_vix(self) -> float | None:
        try:
            return float(self.ohlcv("INDIAVIX", "1d", 5)["close"].iloc[-1])
        except ProviderError:
            return None

    def option_chain(self, symbol: str, expiry: date | None = None):
        raise ProviderError(
            "Yahoo does not provide NSE option chains. Use NSEProvider for chains."
        )
