"""HTTP API and dashboard host.

ADVISORY ONLY. Every endpoint here is read-only analysis. There is deliberately
no order-placement route, no broker credential handling and no position state --
not as an oversight but as the product boundary.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from alpha.calendar import EXPIRY_RULES, expiry_context
from alpha.data import build_provider
from alpha.engines import equity_positional as eq
from alpha.engines import index_options as io
from alpha.universe import Universe

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web"
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]

DISCLAIMER = (
    "Analysis and education only. These are heuristic signals produced by rules, "
    "not predictions, not investment advice, and not a solicitation to trade. "
    "No backtested edge is claimed. Options can lose 100% of premium. "
    "Verify everything independently before risking money."
)

app = FastAPI(
    title="Alpha Trading - Signal Advisor",
    description=DISCLAIMER,
    version="0.1.0",
)


def _provider():
    return build_provider(os.getenv("ALPHA_DATA_MODE", "fixture"))


def _as_of(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, f"as_of must be YYYY-MM-DD, got {value!r}")


@app.get("/api/health")
def health() -> dict:
    p = _provider()
    return {
        "status": "ok",
        "server_time": datetime.now().isoformat(),
        "data_mode": os.getenv("ALPHA_DATA_MODE", "fixture"),
        "live_data": p.is_live,
        "degraded": p.degraded,
        "places_orders": False,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/expiries")
def expiries() -> dict:
    """Expiry rules per symbol -- these changed twice recently, so they are
    surfaced rather than buried."""
    today = date.today()
    out = []
    for sym, rule in EXPIRY_RULES.items():
        try:
            ctx = expiry_context(sym, today)
        except Exception:                                    # noqa: BLE001
            continue
        out.append({
            "symbol": sym, "exchange": rule.exchange,
            "has_weekly": rule.has_weekly, "note": rule.note,
            "next_expiry": ctx.expiry.isoformat(),
            "days_to_expiry": ctx.calendar_days,
            "sessions_to_expiry": ctx.trading_days,
            "is_expiry_day": ctx.is_expiry_day,
            "warning": ctx.warning,
        })
    return {"as_of": today.isoformat(), "expiries": out}


@app.get("/api/signals/options")
def option_signals(symbols: str = Query(",".join(INDEX_SYMBOLS)),
                   as_of: str | None = None) -> dict:
    p = _provider()
    when = _as_of(as_of)
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    signals, errors = [], []
    for sym in wanted:
        try:
            signals.append(io.build_signal(sym, p, when).to_dict())
        except Exception as exc:                              # noqa: BLE001
            log.exception("option signal failed for %s", sym)
            errors.append({"symbol": sym, "error": str(exc)})

    return {"as_of": when.isoformat(), "generated_at": datetime.now().isoformat(),
            "signals": signals, "errors": errors,
            "live_data": p.is_live, "degraded": p.degraded,
            "disclaimer": DISCLAIMER}


@app.get("/api/signals/equity")
def equity_signals(top: int = Query(5, ge=1, le=50),
                   include_rejected: bool = False,
                   as_of: str | None = None) -> dict:
    p = _provider()
    when = _as_of(as_of)
    cfg = eq.EquityEngineConfig(top_n=top)
    try:
        sigs = eq.scan(p, when, cfg, include_rejected=include_rejected)
    except Exception as exc:                                  # noqa: BLE001
        log.exception("equity scan failed")
        raise HTTPException(500, f"equity scan failed: {exc}")

    return {"as_of": when.isoformat(), "generated_at": datetime.now().isoformat(),
            "signals": [s.to_dict() for s in sigs],
            "live_data": p.is_live, "degraded": p.degraded,
            "disclaimer": DISCLAIMER}


@app.get("/api/sectors")
def sectors(as_of: str | None = None) -> dict:
    p = _provider()
    when = _as_of(as_of)
    try:
        rows = eq.sector_table(p, when)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(500, f"sector scan failed: {exc}")
    return {"as_of": when.isoformat(), "sectors": rows,
            "live_data": p.is_live, "degraded": p.degraded}


@app.get("/api/universe")
def universe() -> dict:
    u = Universe.load()
    return {"benchmark": u.benchmark,
            "verified_through": u.verified_through.isoformat() if u.verified_through else None,
            "count": len(u.symbols),
            "constituents": [{"symbol": c.symbol, "name": c.name, "sector": c.sector}
                             for c in u.constituents]}


@app.get("/api/brief")
def brief(as_of: str | None = None, top: int = 5) -> dict:
    """Everything the dashboard needs in one round trip.

    A tablet on a mobile connection should not make five sequential requests to
    render one screen.
    """
    p = _provider()
    when = _as_of(as_of)

    opts, errors = [], []
    for sym in INDEX_SYMBOLS:
        try:
            opts.append(io.build_signal(sym, p, when).to_dict())
        except Exception as exc:                              # noqa: BLE001
            log.exception("option signal failed for %s", sym)
            errors.append({"symbol": sym, "error": str(exc)})

    try:
        equities = [s.to_dict() for s in eq.scan(p, when, eq.EquityEngineConfig(top_n=top))]
    except Exception as exc:                                  # noqa: BLE001
        log.exception("equity scan failed")
        equities, _ = [], errors.append({"symbol": "equity_scan", "error": str(exc)})

    try:
        sector_rows = eq.sector_table(p, when)
    except Exception:                                         # noqa: BLE001
        sector_rows = []

    return {
        "as_of": when.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "options": opts,
        "equities": equities,
        "sectors": sector_rows,
        "expiries": expiries()["expiries"],
        "errors": errors,
        "live_data": p.is_live,
        "degraded": p.degraded,
        "places_orders": False,
        "disclaimer": DISCLAIMER,
    }


# -- dashboard -------------------------------------------------------------

if (WEB_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "dashboard not built", "api": "/api/brief"},
                            status_code=404)
    return FileResponse(index)


def main() -> None:                                           # pragma: no cover
    import uvicorn

    uvicorn.run("alpha.api:app", host=os.getenv("ALPHA_HOST", "0.0.0.0"),
                port=int(os.getenv("ALPHA_PORT", "8000")), reload=False)


if __name__ == "__main__":                                    # pragma: no cover
    main()
