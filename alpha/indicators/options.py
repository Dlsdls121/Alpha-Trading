"""Options mathematics and option-chain analytics.

Black-Scholes here uses ``math.erf`` for the normal CDF rather than scipy, so
the whole package installs without a compiler toolchain.

A note on what these numbers are worth. Greeks computed from a mid price on an
illiquid strike are close to meaningless -- the model is exact, the input is
not. Every consumer of these functions should check open interest and spread
before trusting the output, which is why ``bs_greeks`` returns the inputs it
used alongside the results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float) -> tuple[float, float]:
    vt = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / vt
    return d1, d1 - vt


def bs_price(spot: float, strike: float, t_years: float, vol: float,
             rate: float = 0.065, option_type: str = "CE") -> float:
    """Black-Scholes price. ``vol`` is a decimal (0.14 == 14%).

    Default rate is a rough Indian risk-free proxy; for the short tenors these
    signals deal in, the rate barely moves the price.
    """
    option_type = option_type.upper()
    if t_years <= 0 or vol <= 0:
        # At expiry (or with no vol) an option is worth only its intrinsic value.
        return max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)

    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    disc = math.exp(-rate * t_years)
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float          # per 1 percentage point of IV
    theta: float         # per calendar day, in rupees
    theta_pct: float     # theta as a percent of the option's own price per day
    spot: float
    strike: float
    t_years: float
    vol: float

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def bs_greeks(spot: float, strike: float, t_years: float, vol: float,
              rate: float = 0.065, option_type: str = "CE") -> Greeks:
    """Full greeks for one contract.

    ``theta_pct`` is the number an option *buyer* should actually look at: a
    theta of -40 on a 900-rupee premium (-4.4%/day) is survivable, the same -40
    on a 90-rupee premium (-44%/day) is not.
    """
    option_type = option_type.upper()
    if t_years <= 0 or vol <= 0:
        price = bs_price(spot, strike, t_years, vol, rate, option_type)
        intrinsic_delta = (1.0 if spot > strike else 0.0) if option_type == "CE" \
            else (-1.0 if spot < strike else 0.0)
        return Greeks(price, intrinsic_delta, 0.0, 0.0, 0.0, 0.0, spot, strike, t_years, vol)

    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    disc = math.exp(-rate * t_years)
    pdf = _norm_pdf(d1)
    sqrt_t = math.sqrt(t_years)

    price = bs_price(spot, strike, t_years, vol, rate, option_type)
    delta = _norm_cdf(d1) if option_type == "CE" else _norm_cdf(d1) - 1.0
    gamma = pdf / (spot * vol * sqrt_t)
    vega = spot * pdf * sqrt_t / 100.0

    common = -(spot * pdf * vol) / (2.0 * sqrt_t)
    if option_type == "CE":
        theta_year = common - rate * strike * disc * _norm_cdf(d2)
    else:
        theta_year = common + rate * strike * disc * _norm_cdf(-d2)
    theta_day = theta_year / 365.0
    theta_pct = (theta_day / price * 100.0) if price > 1e-9 else 0.0

    return Greeks(price, delta, gamma, vega, theta_day, theta_pct, spot, strike, t_years, vol)


def implied_vol(market_price: float, spot: float, strike: float, t_years: float,
                rate: float = 0.065, option_type: str = "CE",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6) -> float | None:
    """Back out IV from a traded price by bisection.

    Bisection rather than Newton-Raphson: vega collapses toward zero for deep
    ITM/OTM strikes and near expiry, which is exactly where Newton diverges.
    Bisection is slower and always converges on a bracketed root.

    Returns ``None`` when the price is outside no-arbitrage bounds -- typically
    a stale or crossed quote on a dead strike, which is worth knowing about
    rather than silently getting a garbage IV.
    """
    if market_price <= 0 or t_years <= 0 or spot <= 0 or strike <= 0:
        return None

    intrinsic = max(0.0, spot - strike) if option_type.upper() == "CE" \
        else max(0.0, strike - spot)
    if market_price < intrinsic - 1e-6:
        return None                      # below intrinsic: bad quote

    if bs_price(spot, strike, t_years, hi, rate, option_type) < market_price:
        return None                      # richer than 500% vol: bad quote

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        diff = bs_price(spot, strike, t_years, mid, rate, option_type) - market_price
        if abs(diff) < tol or (hi - lo) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# -- option chain analytics ------------------------------------------------


def put_call_ratio(rows: Sequence[dict], field: str = "oi") -> float | None:
    """PCR across a chain. ``field`` is "oi" or "volume".

    Convention: put total / call total. Above 1 means more puts outstanding.
    Read it as a *contrarian* gauge at extremes and a trend gauge in the middle
    -- and always alongside its own recent range, since the neutral band drifts.
    """
    calls = sum(float(r.get(f"ce_{field}") or 0) for r in rows)
    puts = sum(float(r.get(f"pe_{field}") or 0) for r in rows)
    if calls <= 0:
        return None
    return puts / calls


def max_pain(rows: Sequence[dict]) -> float | None:
    """Strike at which option writers collectively lose the least.

    For each candidate settlement strike K, total writer payout is
    sum(CE_OI * max(0, K - strike)) + sum(PE_OI * max(0, strike - K)).
    The minimising K is max pain.

    It is a gravity point near expiry, not a forecast. Far from expiry it says
    very little, and it moves as OI moves.
    """
    strikes = sorted({float(r["strike"]) for r in rows})
    if not strikes:
        return None

    ce = {float(r["strike"]): float(r.get("ce_oi") or 0) for r in rows}
    pe = {float(r["strike"]): float(r.get("pe_oi") or 0) for r in rows}

    best_k, best_pain = None, None
    for k in strikes:
        pain = 0.0
        for s in strikes:
            if k > s:
                pain += ce.get(s, 0.0) * (k - s)      # calls below settlement pay out
            if s > k:
                pain += pe.get(s, 0.0) * (s - k)      # puts above settlement pay out
        if best_pain is None or pain < best_pain:
            best_k, best_pain = k, pain
    return best_k


def classify_oi_buildup(price_change_pct: float, oi_change_pct: float,
                        price_eps: float = 0.05, oi_eps: float = 1.0) -> dict:
    """The four-quadrant read of price versus open interest.

    OI measures commitment, price says who is winning:

    ==================  ==========  ===========================================
    price / OI          label       reading
    ==================  ==========  ===========================================
    up   / up           long buildup      fresh longs -- bullish, strongest
    down / up           short buildup     fresh shorts -- bearish, strongest
    up   / down         short covering    forced buying -- bullish but weaker
    down / down         long unwinding    longs leaving -- bearish but weaker
    ==================  ==========  ===========================================

    Covering and unwinding are deliberately scored lower than fresh buildup:
    they are positions closing, which exhausts, rather than opening, which
    sustains.
    """
    p_up = price_change_pct > price_eps
    p_dn = price_change_pct < -price_eps
    o_up = oi_change_pct > oi_eps
    o_dn = oi_change_pct < -oi_eps

    if p_up and o_up:
        return {"label": "long_buildup", "bias": "bullish", "score": 1.0,
                "note": "price up on rising OI - fresh longs being added"}
    if p_dn and o_up:
        return {"label": "short_buildup", "bias": "bearish", "score": -1.0,
                "note": "price down on rising OI - fresh shorts being added"}
    if p_up and o_dn:
        return {"label": "short_covering", "bias": "bullish", "score": 0.45,
                "note": "price up on falling OI - shorts covering, not fresh buying"}
    if p_dn and o_dn:
        return {"label": "long_unwinding", "bias": "bearish", "score": -0.45,
                "note": "price down on falling OI - longs exiting, not fresh selling"}
    return {"label": "indecisive", "bias": "neutral", "score": 0.0,
            "note": "no meaningful price or OI change"}


def oi_support_resistance(rows: Sequence[dict], spot: float, top_n: int = 3) -> dict:
    """Strikes where writers have the most at stake.

    Heavy call OI *above* spot caps rallies; heavy put OI *below* spot cushions
    falls -- writers defend those strikes because that is where their money is.
    Only strikes on the correct side of spot are considered; a huge call OI
    below spot is deep ITM and tells you nothing about resistance.
    """
    calls = [(float(r["strike"]), float(r.get("ce_oi") or 0)) for r in rows
             if float(r["strike"]) > spot]
    puts = [(float(r["strike"]), float(r.get("pe_oi") or 0)) for r in rows
            if float(r["strike"]) < spot]

    calls.sort(key=lambda x: x[1], reverse=True)
    puts.sort(key=lambda x: x[1], reverse=True)

    return {
        "resistance": [{"strike": k, "oi": int(v)} for k, v in calls[:top_n] if v > 0],
        "support": [{"strike": k, "oi": int(v)} for k, v in puts[:top_n] if v > 0],
        "immediate_resistance": calls[0][0] if calls and calls[0][1] > 0 else None,
        "immediate_support": puts[0][0] if puts and puts[0][1] > 0 else None,
    }


def percentile_rank(value: float, history: Iterable[float]) -> float | None:
    """Percentile of ``value`` within ``history``, 0-100.

    Used for IV percentile: an absolute IV of 14 means nothing on its own, but
    "14, which is the 8th percentile of the last year" means options are cheap
    and buying premium is comparatively attractive.
    """
    hist = [h for h in history if h is not None and not math.isnan(h)]
    if not hist:
        return None
    below = sum(1 for h in hist if h < value)
    equal = sum(1 for h in hist if h == value)
    return 100.0 * (below + 0.5 * equal) / len(hist)
