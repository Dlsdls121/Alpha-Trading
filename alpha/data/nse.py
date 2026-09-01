"""Live NSE India provider.

NSE has no public documented API. What exists is the JSON the website itself
calls, and it is defended: requests without a browser-like header set and a
prior session cookie get 401/403, and it rate-limits aggressively.

The handling below reflects that reality:

* a session is primed by hitting the homepage to collect cookies before any
  API call, and re-primed automatically on the first auth failure;
* responses are cached, because repeated uncached polling is what gets a client
  blocked;
* every failure raises :class:`ProviderError` rather than returning empty data,
  so a signal is never computed from silence.

NSE serves the option chain well but not convenient historical OHLCV, so pair
this with :class:`~alpha.data.yahoo.YahooProvider` via
:class:`~alpha.data.composite.CompositeProvider`.

NOTE ON VERIFICATION: this code could not be exercised against the live
endpoint in the environment it was written in (egress to nseindia.com was
blocked by network policy). The request shape follows NSE's documented-by-
observation behaviour, but treat the first live run as the real test --
``python -m alpha.data.nse selftest`` exists for exactly that.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from alpha.data.base import ChainRow, ChainSnapshot, ProviderError, Quote
from alpha.data.cache import DiskCache

BASE = "https://www.nseindia.com"

# NSE rejects anything that does not look like a browser.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": f"{BASE}/option-chain",
    "Connection": "keep-alive",
}

INDEX_API_NAMES = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
    "INDIAVIX": "INDIA VIX",
}
INDEX_SYMBOLS = set(INDEX_API_NAMES)


class NSEProvider:
    """Option chain, spot quotes and India VIX from nseindia.com."""

    name = "nse"

    def __init__(self, cache_dir: str = ".cache", chain_ttl: float = 60.0,
                 timeout: float = 15.0):
        import httpx                       # imported lazily: fixtures need no httpx

        self._httpx = httpx
        self.cache = DiskCache(cache_dir)
        self.chain_ttl = chain_ttl
        self.timeout = timeout
        self._client: Any = None

    # -- session handling ------------------------------------------------

    def _new_client(self):
        client = self._httpx.Client(headers=HEADERS, timeout=self.timeout,
                                    follow_redirects=True)
        try:
            # Prime cookies. NSE sets them on the HTML pages, not the API.
            client.get(BASE, timeout=self.timeout)
            client.get(f"{BASE}/option-chain", timeout=self.timeout)
        except Exception as exc:
            client.close()
            raise ProviderError(f"could not reach NSE to establish a session: {exc}") from exc
        return client

    def _client_or_new(self):
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def _get_json(self, path: str, retry: bool = True) -> dict:
        client = self._client_or_new()
        try:
            resp = client.get(f"{BASE}{path}")
        except Exception as exc:
            raise ProviderError(f"NSE request failed for {path}: {exc}") from exc

        if resp.status_code in (401, 403) and retry:
            # Cookies expire; rebuild the session once and try again.
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            return self._get_json(path, retry=False)

        if resp.status_code != 200:
            raise ProviderError(f"NSE returned HTTP {resp.status_code} for {path}")

        try:
            return resp.json()
        except ValueError as exc:
            # Usually an interstitial/blocked HTML page rather than JSON.
            raise ProviderError(
                f"NSE returned non-JSON for {path} (likely rate-limited or blocked)"
            ) from exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    # -- provider interface ----------------------------------------------

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        sym = symbol.upper()
        is_index = sym in INDEX_SYMBOLS
        path = (f"/api/option-chain-indices?symbol={sym}" if is_index
                else f"/api/option-chain-equities?symbol={sym}")

        key = f"chain_{sym}"
        raw = self.cache.get(key, self.chain_ttl)
        cached = raw is not None
        if raw is None:
            raw = self._get_json(path)
            self.cache.set(key, raw)

        records = raw.get("records") or {}
        data = records.get("data") or []
        if not data:
            raise ProviderError(f"NSE option chain for {sym} came back empty")

        spot = records.get("underlyingValue")
        if not spot:
            raise ProviderError(f"NSE option chain for {sym} has no underlying value")

        expiries = records.get("expiryDates") or []
        target = expiry or (self._parse_nse_date(expiries[0]) if expiries else None)
        if target is None:
            raise ProviderError(f"no expiry dates in NSE response for {sym}")

        rows: list[ChainRow] = []
        for item in data:
            if self._parse_nse_date(item.get("expiryDate", "")) != target:
                continue
            ce, pe = item.get("CE") or {}, item.get("PE") or {}
            rows.append(ChainRow(
                strike=float(item["strikePrice"]),
                ce_oi=ce.get("openInterest"), ce_change_oi=ce.get("changeinOpenInterest"),
                ce_ltp=ce.get("lastPrice"), ce_iv=ce.get("impliedVolatility") or None,
                ce_volume=ce.get("totalTradedVolume"),
                ce_bid=ce.get("bidprice"), ce_ask=ce.get("askPrice"),
                pe_oi=pe.get("openInterest"), pe_change_oi=pe.get("changeinOpenInterest"),
                pe_ltp=pe.get("lastPrice"), pe_iv=pe.get("impliedVolatility") or None,
                pe_volume=pe.get("totalTradedVolume"),
                pe_bid=pe.get("bidprice"), pe_ask=pe.get("askPrice"),
            ))

        if not rows:
            raise ProviderError(f"no strikes for {sym} expiry {target}")

        rows.sort(key=lambda r: r.strike)
        return ChainSnapshot(symbol=sym, expiry=target, spot=float(spot),
                             timestamp=datetime.now(), rows=rows,
                             source="nse-cache" if cached else "nse", stale=cached)

    def all_indices(self) -> dict[str, dict]:
        raw = self.cache.get("all_indices", 60.0)
        if raw is None:
            raw = self._get_json("/api/allIndices")
            self.cache.set("all_indices", raw)
        return {d.get("index", ""): d for d in raw.get("data", [])}

    def india_vix(self) -> float | None:
        try:
            row = self.all_indices().get("INDIA VIX")
        except ProviderError:
            return None
        return float(row["last"]) if row and row.get("last") is not None else None

    def quote(self, symbol: str) -> Quote:
        sym = symbol.upper()
        if sym in INDEX_SYMBOLS:
            row = self.all_indices().get(INDEX_API_NAMES[sym])
            if not row:
                raise ProviderError(f"index {sym} not present in allIndices response")
            return Quote(symbol=sym, last=float(row["last"]),
                         change_pct=float(row.get("percentChange") or 0.0),
                         timestamp=datetime.now(), prev_close=_f(row.get("previousClose")),
                         day_high=_f(row.get("high")), day_low=_f(row.get("low")),
                         day_open=_f(row.get("open")))

        raw = self._get_json(f"/api/quote-equity?symbol={sym}")
        info = raw.get("priceInfo") or {}
        if not info.get("lastPrice"):
            raise ProviderError(f"no price in NSE quote for {sym}")
        intra = info.get("intraDayHighLow") or {}
        return Quote(symbol=sym, last=float(info["lastPrice"]),
                     change_pct=float(info.get("pChange") or 0.0),
                     timestamp=datetime.now(), prev_close=_f(info.get("previousClose")),
                     day_high=_f(intra.get("max")), day_low=_f(intra.get("min")),
                     day_open=_f(info.get("open")))

    def ohlcv(self, symbol: str, interval: str = "1d", lookback: int = 400) -> pd.DataFrame:
        raise ProviderError(
            "NSEProvider does not serve historical OHLCV. Compose it with "
            "YahooProvider (see alpha.data.composite.build_provider)."
        )

    @staticmethod
    def _parse_nse_date(value: str) -> date | None:
        """NSE formats expiries as '26-Sep-2026'."""
        if not value:
            return None
        try:
            return datetime.strptime(value.strip(), "%d-%b-%Y").date()
        except ValueError:
            return None


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":       # pragma: no cover - manual live check
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        p = NSEProvider()
        try:
            vix = p.india_vix()
            print(f"India VIX: {vix}")
            ch = p.option_chain("NIFTY")
            print(f"NIFTY spot={ch.spot} expiry={ch.expiry} strikes={len(ch.rows)}")
            print(f"ATM strike: {ch.atm_strike()}")
            print("OK - live NSE access is working.")
        except ProviderError as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)
        finally:
            p.close()
