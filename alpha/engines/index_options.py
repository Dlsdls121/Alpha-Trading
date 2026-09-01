"""NIFTY / BANKNIFTY option-buying signal engine.

ADVISORY ONLY. This module describes a contract and the reasoning behind it.
It cannot place an order; nothing in this codebase can.

Why option *buying* needs its own engine
----------------------------------------
A long option loses on three independent axes and only wins on one. Being right
about direction is necessary and nowhere near sufficient:

* **theta** -- premium bleeds every day, and the bleed accelerates into expiry.
  Percentage decay on the final day is brutal, which is why expiry-day buying is
  a hard veto here rather than a small negative score.
* **vega** -- buying when implied vol is rich means an IV crush can take money
  even on a correct call. The sharpest test is IV against *realised* vol: if the
  market charges 22% for movement that has been running at 11%, the buyer is
  paying double for what the index actually delivers.
* **direction** -- the only axis that pays.

So the scorecard is split. Directional factors (trend, momentum, OI positioning)
carry weight and vote on which way. Cost factors (VIX regime, IV vs realised, IV
percentile, days to expiry, liquidity) carry **weight 0** -- they are displayed
as evidence and can raise vetoes, but they never push the direction around,
because expensive premium does not make the market bearish. It just makes buying
a bad idea in either direction.

Instrument differences that matter (as of Sep 2026)
--------------------------------------------------
NIFTY has weekly expiries (Tuesday), so days-to-expiry is 0-7 and theta dominates.
BANKNIFTY lost its weeklies in Nov 2024 and is monthly-only, so DTE runs 1-35 and
strike selection can afford to be different. The engine reads this from
:mod:`alpha.calendar` rather than assuming.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from alpha.calendar import ExpiryContext, expiry_context
from alpha.data.base import ChainSnapshot
from alpha.engines.features import (
    IntradayFeatures, TrendFeatures, extract_intraday, extract_trend, scale,
)
from alpha.indicators.options import (
    Greeks, bs_greeks, classify_oi_buildup, implied_vol, max_pain,
    oi_support_resistance, percentile_rank, put_call_ratio,
)
from alpha.models import (
    Category, Direction, Factor, OptionLeg, Scorecard, Signal, Verdict,
)


@dataclass
class OptionEngineConfig:
    """Every threshold the engine uses, in one place so they can be tuned and
    argued with rather than hidden in the code."""

    # -- direction thresholds
    direction_threshold: float = 0.15
    min_conviction: int = 25

    # -- volatility / cost gates
    iv_rv_rich: float = 1.7          # IV / realised vol above this = overpaying
    iv_rv_block: float = 2.2         # ...and above this it is a hard veto
    ivp_rich: float = 85.0           # IV percentile considered expensive
    vix_low: float = 12.0
    vix_high: float = 20.0

    # -- expiry gates
    block_on_expiry_day: bool = True
    warn_dte_at_or_below: int = 1
    theta_pct_warn: float = 8.0      # premium bleed per day, in %
    theta_pct_block: float = 20.0

    # -- trend regime
    adx_trending: float = 22.0
    adx_choppy: float = 15.0

    # -- strike selection
    target_delta_lo: float = 0.42
    target_delta_hi: float = 0.62
    min_strike_oi: int = 1_000
    min_strike_volume: int = 100
    max_spread_pct: float = 2.5      # (ask-bid)/mid


@dataclass
class OptionSignalInputs:
    """Everything gathered before any scoring happens, kept together so the
    reasoning can quote the exact values that produced it."""

    symbol: str
    chain: ChainSnapshot
    trend: TrendFeatures
    intraday: IntradayFeatures
    expiry: ExpiryContext
    india_vix: float | None
    iv_history: list[float] = field(default_factory=list)
    data_quality: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Directional factors -- these carry weight and vote on direction.
# ---------------------------------------------------------------------------


def _factor_trend_stack(t: TrendFeatures) -> Factor:
    st = t.stack
    aligned = st.get("aligned") or 0.0
    stage = st.get("stage", "unknown")

    if stage == "unknown":
        return Factor("trend_stack", "Moving-average structure", Category.TREND,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "Not enough history to build the 20/50/200 EMA stack.",
                      value="insufficient data")

    verdict = (Verdict.BULLISH if aligned > 0.3 else
               Verdict.BEARISH if aligned < -0.3 else Verdict.NEUTRAL)
    detail = (
        f"Spot {st['px']:,.0f} against EMA20 {st['fast']:,.0f}, EMA50 {st['mid']:,.0f}, "
        f"EMA200 {st['slow']:,.0f} - classified as {stage.replace('_', ' ')}. "
    )
    if st.get("slow_slope") is not None:
        d = "rising" if st["slow_slope"] > 0 else "falling"
        detail += f"The 200 EMA is {d} ({st['slow_slope']:+.2f}% over 20 sessions)."

    return Factor("trend_stack", "Moving-average structure", Category.TREND,
                  verdict, aligned, 2.0, detail, value=stage.replace("_", " "))


def _factor_adx(t: TrendFeatures, cfg: OptionEngineConfig) -> Factor:
    """Trend *strength*, which for an option buyer matters as much as direction:
    a directional bet inside a range pays theta and gets nothing back."""
    if t.adx is None or t.plus_di is None or t.minus_di is None:
        return Factor("adx", "Trend strength (ADX)", Category.TREND,
                      Verdict.NEUTRAL, 0.0, 0.0, "ADX unavailable.", value="n/a")

    di_spread = t.plus_di - t.minus_di
    strength = scale(t.adx, cfg.adx_choppy, 35.0)          # -1 chop .. +1 strong
    score = scale(di_spread, -25, 25) * max(0.15, (strength + 1) / 2)

    if t.adx >= cfg.adx_trending:
        regime = "trending"
    elif t.adx <= cfg.adx_choppy:
        regime = "range-bound"
    else:
        regime = "weakly trending"

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    detail = (
        f"ADX {t.adx:.1f} - {regime}. +DI {t.plus_di:.1f} vs -DI {t.minus_di:.1f} "
        f"({di_spread:+.1f}). "
    )
    if t.adx <= cfg.adx_choppy:
        detail += ("Below 15 the index is not trending, and directional premium "
                   "buying tends to bleed here regardless of which way it is bought.")

    return Factor("adx", "Trend strength (ADX)", Category.TREND, verdict,
                  score, 1.5, detail, value=f"ADX {t.adx:.1f} ({regime})")


def _factor_rsi(t: TrendFeatures) -> Factor:
    if t.rsi is None:
        return Factor("rsi", "RSI(14)", Category.MOMENTUM, Verdict.NEUTRAL,
                      0.0, 0.0, "RSI unavailable.", value="n/a")

    score = scale(t.rsi, 30, 70)
    note = ""
    # Extremes are cut back rather than flipped: RSI can pin >70 for weeks in a
    # real trend, so treating 72 as "sell" is how you fade a runaway market.
    if t.rsi > 78:
        score *= 0.45
        note = " Above 78 this is stretched; momentum is strong but late, so the reading is discounted."
    elif t.rsi < 22:
        score *= 0.45
        note = " Below 22 this is washed out; the downside reading is discounted."

    direction_word = "rising" if t.rsi_rising else "falling" if t.rsi_rising is False else "flat"
    verdict = (Verdict.BULLISH if score > 0.12 else
               Verdict.BEARISH if score < -0.12 else Verdict.NEUTRAL)
    detail = (f"RSI(14) at {t.rsi:.1f} and {direction_word}"
              + (f" (from {t.rsi_prev:.1f})" if t.rsi_prev is not None else "") + f".{note}")

    return Factor("rsi", "RSI(14)", Category.MOMENTUM, verdict, score, 1.2,
                  detail, value=f"{t.rsi:.1f}")


def _factor_macd(t: TrendFeatures) -> Factor:
    if t.macd_hist is None:
        return Factor("macd", "MACD histogram", Category.MOMENTUM, Verdict.NEUTRAL,
                      0.0, 0.0, "MACD unavailable.", value="n/a")

    ref = max(abs(t.macd_hist), 1e-9) * 3
    score = scale(t.macd_hist, -ref, ref)
    expanding = (t.macd_hist_prev is not None
                 and abs(t.macd_hist) > abs(t.macd_hist_prev))
    if expanding:
        score *= 1.15

    verdict = (Verdict.BULLISH if t.macd_hist > 0 else
               Verdict.BEARISH if t.macd_hist < 0 else Verdict.NEUTRAL)
    detail = (f"MACD histogram {t.macd_hist:+.1f}, "
              f"{'expanding' if expanding else 'contracting'}"
              + (f" from {t.macd_hist_prev:+.1f}" if t.macd_hist_prev is not None else "")
              + ". Expanding histograms mean the momentum impulse is still building.")

    return Factor("macd", "MACD histogram", Category.MOMENTUM, verdict,
                  max(-1.0, min(1.0, score)), 1.0, detail, value=f"{t.macd_hist:+.1f}")


def _factor_vwap(i: IntradayFeatures) -> Factor:
    """Intraday acceptance. A common option-buying discipline is to take longs
    only while price holds above session VWAP, since that is where the day's
    average participant is break-even."""
    if i.vwap is None or i.pct_from_vwap is None:
        return Factor("vwap", "Price vs session VWAP", Category.TREND,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "No intraday data available, so session VWAP could not be computed.",
                      value="n/a")

    score = scale(i.pct_from_vwap, -0.6, 0.6)
    side = "above" if i.pct_from_vwap > 0 else "below"
    verdict = (Verdict.BULLISH if i.pct_from_vwap > 0.05 else
               Verdict.BEARISH if i.pct_from_vwap < -0.05 else Verdict.NEUTRAL)
    detail = (f"Spot {i.close:,.0f} is {abs(i.pct_from_vwap):.2f}% {side} session VWAP "
              f"{i.vwap:,.0f} ({i.bars} bars into the session).")
    if i.rel_volume is not None:
        detail += (f" Volume is running {i.rel_volume:.2f}x its 20-bar average"
                   + (", which confirms the move." if i.rel_volume > 1.2
                      else ", which is thin confirmation." if i.rel_volume < 0.8 else "."))

    return Factor("vwap", "Price vs session VWAP", Category.TREND, verdict,
                  score, 1.3, detail, value=f"{i.pct_from_vwap:+.2f}% vs VWAP")


def _factor_pcr(chain: ChainSnapshot) -> Factor:
    """PCR read near the money, with contrarian handling at the extremes.

    Chain-wide PCR is dominated by far strikes nobody trades, so this uses the
    near-ATM band where positioning reflects an actual view.
    """
    rows = [r.as_dict() for r in chain.near_atm(20)]
    pcr = put_call_ratio(rows, "oi")
    if pcr is None:
        return Factor("pcr", "Put-call ratio (OI)", Category.POSITIONING,
                      Verdict.NEUTRAL, 0.0, 0.0, "No call OI to compute PCR.", value="n/a")

    # Mid-range: higher PCR = more put writing = bullish support.
    # Extremes: read as exhaustion and fade.
    if pcr > 1.8:
        score, reading = -0.45, ("extreme put writing - historically an exhaustion "
                                 "zone that has preceded sharp reversals, so this is "
                                 "read contrarian rather than bullish")
    elif pcr < 0.5:
        score, reading = 0.45, ("extreme call writing / capitulation - read contrarian, "
                                "these levels have tended to get bought")
    else:
        score = scale(pcr, 0.7, 1.3)
        if pcr >= 1.3:
            reading = "strong put writing, an institutional bullish tilt"
        elif pcr >= 1.0:
            reading = "neutral-to-bullish"
        elif pcr >= 0.7:
            reading = "neutral-to-mildly-bearish"
        else:
            reading = "heavy call writing, a bearish tilt"

    verdict = (Verdict.BULLISH if score > 0.12 else
               Verdict.BEARISH if score < -0.12 else Verdict.NEUTRAL)
    return Factor("pcr", "Put-call ratio (OI)", Category.POSITIONING, verdict,
                  score, 1.2,
                  f"Near-the-money PCR is {pcr:.2f} - {reading}. "
                  f"Computed across the 20 strikes closest to spot, since chain-wide "
                  f"PCR is skewed by far strikes with no real participation.",
                  value=f"PCR {pcr:.2f}", data={"pcr": round(pcr, 3)})


def _factor_oi_levels(chain: ChainSnapshot) -> Factor:
    """Where spot sits between the strikes writers are defending."""
    levels = oi_support_resistance([r.as_dict() for r in chain.rows], chain.spot)
    sup, res = levels["immediate_support"], levels["immediate_resistance"]

    if sup is None or res is None or res <= sup:
        return Factor("oi_levels", "OI support / resistance", Category.POSITIONING,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "Could not identify OI-based support and resistance around spot.",
                      value="n/a")

    # Position within the band: 0 at support, 1 at resistance.
    pos = (chain.spot - sup) / (res - sup)
    room_up = (res / chain.spot - 1) * 100
    room_dn = (1 - sup / chain.spot) * 100
    # Nearer support = more room to rally = mildly bullish for a call buyer.
    score = scale(1.0 - pos, 0.25, 0.75) * 0.8

    verdict = (Verdict.BULLISH if score > 0.12 else
               Verdict.BEARISH if score < -0.12 else Verdict.NEUTRAL)
    detail = (
        f"Heaviest put OI sits at {sup:,.0f} ({room_dn:.2f}% below spot) and heaviest "
        f"call OI at {res:,.0f} ({room_up:.2f}% above). Spot is {pos * 100:.0f}% of the way "
        f"through that band. Writers defend these strikes, so they act as a floor and a "
        f"ceiling until OI shifts."
    )
    return Factor("oi_levels", "OI support / resistance", Category.POSITIONING,
                  verdict, score, 1.3, detail,
                  value=f"{sup:,.0f} / {res:,.0f}",
                  data={"support": sup, "resistance": res,
                        "room_up_pct": round(room_up, 2), "room_down_pct": round(room_dn, 2)})


def _factor_oi_buildup(chain: ChainSnapshot, t: TrendFeatures) -> Factor:
    """Four-quadrant price/OI read across the near-ATM band."""
    rows = chain.near_atm(20)
    tot_oi = sum((r.ce_oi or 0) + (r.pe_oi or 0) for r in rows)
    tot_chg = sum((r.ce_change_oi or 0) + (r.pe_change_oi or 0) for r in rows)

    if tot_oi <= 0 or t.prev_close is None:
        return Factor("oi_buildup", "Price vs OI buildup", Category.POSITIONING,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "Insufficient OI-change data to classify buildup.", value="n/a")

    prev_oi = tot_oi - tot_chg
    oi_change_pct = (tot_chg / prev_oi * 100.0) if prev_oi > 0 else 0.0
    price_change_pct = (t.close / t.prev_close - 1.0) * 100.0

    cls = classify_oi_buildup(price_change_pct, oi_change_pct)
    verdict = {"bullish": Verdict.BULLISH, "bearish": Verdict.BEARISH,
               "neutral": Verdict.NEUTRAL}[cls["bias"]]

    detail = (f"Underlying {price_change_pct:+.2f}% with near-ATM open interest "
              f"{oi_change_pct:+.1f}% - {cls['label'].replace('_', ' ')}: {cls['note']}.")
    if cls["label"] in ("short_covering", "long_unwinding"):
        detail += (" Positions closing tends to exhaust, so this is scored below a "
                   "fresh buildup of the same direction.")

    return Factor("oi_buildup", "Price vs OI buildup", Category.POSITIONING,
                  verdict, cls["score"], 1.5, detail,
                  value=cls["label"].replace("_", " "),
                  data={"price_change_pct": round(price_change_pct, 2),
                        "oi_change_pct": round(oi_change_pct, 2)})


def _factor_max_pain(chain: ChainSnapshot, exp: ExpiryContext) -> Factor:
    """Max pain, weighted down hard when expiry is far away.

    Max pain is a gravity point in the last day or two of a contract's life. A
    month out it is mostly noise, so its weight scales with proximity to expiry
    instead of being a constant.
    """
    mp = max_pain([r.as_dict() for r in chain.rows])
    if mp is None:
        return Factor("max_pain", "Max pain", Category.POSITIONING, Verdict.NEUTRAL,
                      0.0, 0.0, "Max pain could not be computed.", value="n/a")

    gap_pct = (mp / chain.spot - 1.0) * 100.0
    # Full weight at 0-1 DTE, negligible beyond ~6 sessions.
    proximity = max(0.0, 1.0 - exp.trading_days / 6.0)
    weight = 1.4 * proximity
    score = scale(gap_pct, -1.5, 1.5)

    verdict = (Verdict.BULLISH if score > 0.12 and weight > 0.1 else
               Verdict.BEARISH if score < -0.12 and weight > 0.1 else Verdict.NEUTRAL)

    detail = (f"Max pain is {mp:,.0f}, {gap_pct:+.2f}% from spot {chain.spot:,.0f}. ")
    if exp.trading_days <= 2:
        detail += ("With expiry within two sessions this pull is at its strongest - "
                   "writers actively defend it.")
    elif exp.trading_days <= 6:
        detail += f"With {exp.trading_days} sessions left the pull is weakening."
    else:
        detail += (f"With {exp.trading_days} sessions to expiry this carries little "
                   f"information, so it is weighted to near zero here.")

    return Factor("max_pain", "Max pain", Category.POSITIONING, verdict, score,
                  round(weight, 3), detail, value=f"{mp:,.0f} ({gap_pct:+.2f}%)",
                  data={"max_pain": mp, "gap_pct": round(gap_pct, 2)})


# ---------------------------------------------------------------------------
# Cost / volatility factors.
#
# These carry weight 0 by design: they are shown as evidence and can raise
# vetoes, but they must not move the *direction*. Rich premium does not make the
# market bearish -- it makes buying premium a bad idea whichever way you lean.
# ---------------------------------------------------------------------------


def atm_iv(chain: ChainSnapshot, exp: ExpiryContext) -> tuple[float | None, str]:
    """ATM implied vol in percent, preferring the exchange's own figure.

    NSE publishes IV per strike. When it is missing or zero (common on thin
    strikes) we solve it from the traded price instead, and say which was used
    so the reasoning is not silently resting on a derived number.
    """
    atm = chain.atm_strike()
    if atm is None:
        return None, "no ATM strike"
    row = chain.row(atm)
    if row is None:
        return None, "no ATM row"

    ivs = [v for v in (row.ce_iv, row.pe_iv) if v]
    if ivs:
        return sum(ivs) / len(ivs), "exchange-published IV"

    solved = []
    for ltp, opt in ((row.ce_ltp, "CE"), (row.pe_ltp, "PE")):
        if ltp and ltp > 0:
            v = implied_vol(ltp, chain.spot, atm, exp.t_years, 0.065, opt)
            if v:
                solved.append(v * 100)
    if solved:
        return sum(solved) / len(solved), "solved from traded price (exchange IV missing)"
    return None, "could not determine ATM IV"


def _factor_iv_vs_realized(iv: float | None, iv_src: str, t: TrendFeatures,
                           sc: Scorecard, cfg: OptionEngineConfig) -> Factor:
    """The single most important test for an option buyer.

    Implied vol is the price of movement. Realised vol is the movement actually
    delivered. Buying when implied badly exceeds realised means paying for
    motion the index has not been producing -- and getting the direction right
    can still lose after the IV crush.
    """
    if iv is None or t.realized_vol is None or t.realized_vol <= 0:
        return Factor("iv_vs_rv", "Implied vs realised volatility", Category.VOLATILITY,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "Could not compare implied against realised volatility "
                      f"({iv_src}).", value="n/a")

    ratio = iv / t.realized_vol
    if ratio >= cfg.iv_rv_block:
        sc.veto("iv_crush", "Implied volatility is far above realised",
                f"ATM IV {iv:.1f}% is {ratio:.2f}x the 20-day realised volatility of "
                f"{t.realized_vol:.1f}%. Buying here means paying roughly double for "
                f"movement the index has not been delivering; an IV crush can lose "
                f"money even if the direction is right.")
        assess = "severely overpriced"
    elif ratio >= cfg.iv_rv_rich:
        sc.veto("iv_rich", "Implied volatility is rich versus realised",
                f"ATM IV {iv:.1f}% is {ratio:.2f}x realised {t.realized_vol:.1f}%. "
                f"Premium is expensive relative to actual movement.", severity="warn")
        assess = "expensive"
    elif ratio <= 0.9:
        assess = "cheap - the index has been moving more than options are pricing"
    else:
        assess = "fairly priced"

    return Factor("iv_vs_rv", "Implied vs realised volatility", Category.VOLATILITY,
                  Verdict.NEUTRAL, 0.0, 0.0,
                  f"ATM IV {iv:.1f}% ({iv_src}) against 20-day realised volatility "
                  f"{t.realized_vol:.1f}% - a ratio of {ratio:.2f}x, i.e. {assess}. "
                  f"For a buyer this is the cost of the bet, not its direction.",
                  value=f"{ratio:.2f}x", data={"iv": round(iv, 2),
                                               "realized_vol": round(t.realized_vol, 2),
                                               "ratio": round(ratio, 3)})


def _factor_iv_percentile(iv: float | None, history: list[float],
                          sc: Scorecard, cfg: OptionEngineConfig) -> Factor:
    if iv is None or not history:
        return Factor("iv_percentile", "IV percentile", Category.VOLATILITY,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "No IV history available, so IV percentile cannot be computed. "
                      "An absolute IV number alone does not say whether options are "
                      "cheap or dear.", value="n/a")

    ivp = percentile_rank(iv, history)
    if ivp is None:
        return Factor("iv_percentile", "IV percentile", Category.VOLATILITY,
                      Verdict.NEUTRAL, 0.0, 0.0, "IV percentile unavailable.", value="n/a")

    if ivp >= cfg.ivp_rich:
        sc.veto("ivp_high", "IV is near the top of its own range",
                f"ATM IV {iv:.1f}% sits at the {ivp:.0f}th percentile of the last "
                f"{len(history)} observations. Buying premium at the top of its range "
                f"means the most likely move in IV is down.", severity="warn")
        note = "expensive - options have rarely been dearer"
    elif ivp <= 20:
        note = "cheap - options have rarely been this inexpensive, which favours buyers"
    else:
        note = "mid-range"

    return Factor("iv_percentile", "IV percentile", Category.VOLATILITY,
                  Verdict.NEUTRAL, 0.0, 0.0,
                  f"ATM IV {iv:.1f}% is at the {ivp:.0f}th percentile of its last "
                  f"{len(history)} readings - {note}.",
                  value=f"{ivp:.0f}th pct", data={"ivp": round(ivp, 1)})


def _factor_vix(vix: float | None, cfg: OptionEngineConfig) -> Factor:
    if vix is None:
        return Factor("vix", "India VIX regime", Category.VOLATILITY, Verdict.NEUTRAL,
                      0.0, 0.0, "India VIX unavailable.", value="n/a")

    if vix < cfg.vix_low:
        note = ("a low-volatility regime. Premium is cheap, which helps buyers, but "
                "the index also tends to move less - a cheap option on a still market "
                "still decays to nothing")
    elif vix > cfg.vix_high:
        note = ("a high-volatility regime. Premium is rich and the edge historically "
                "shifts toward sellers; buyers are paying up for movement already priced in")
    else:
        note = "a normal-volatility regime"

    return Factor("vix", "India VIX regime", Category.VOLATILITY, Verdict.NEUTRAL,
                  0.0, 0.0, f"India VIX at {vix:.2f} - {note}.",
                  value=f"{vix:.2f}", data={"vix": round(vix, 2)})


def _factor_expiry(exp: ExpiryContext, sc: Scorecard, cfg: OptionEngineConfig) -> Factor:
    """Days to expiry, and the hard gate on expiry-day buying."""
    if exp.is_expiry_day and cfg.block_on_expiry_day:
        sc.veto("expiry_day", "Expiry day - no premium buying",
                f"{exp.symbol} expires today ({exp.expiry:%d-%b-%Y}). On the final day "
                f"time value collapses to zero within hours; percentage theta is at its "
                f"most violent and any option that does not finish in the money is worth "
                f"nothing. Directional buying here is a coin flip against a decaying "
                f"clock, so this engine will not issue a buy signal.")
    elif exp.calendar_days <= cfg.warn_dte_at_or_below:
        sc.veto("near_expiry", "One session to expiry",
                f"Expiry is {exp.expiry:%d-%b-%Y}, {exp.calendar_days} calendar day away. "
                f"Decay is at its sharpest in the final sessions; the move has to happen "
                f"almost immediately to pay.", severity="warn")

    detail = (f"Nearest {exp.symbol} expiry is {exp.expiry:%d-%b-%Y} - "
              f"{exp.calendar_days} calendar days, {exp.trading_days} trading sessions away. "
              f"{exp.rule_note}")
    if not exp.has_weekly:
        detail += (" Because this symbol no longer has weekly contracts, days-to-expiry "
                   "can be long and there is more room for a directional view to play out.")

    return Factor("expiry", "Days to expiry", Category.COST, Verdict.NEUTRAL,
                  0.0, 0.0, detail, value=exp.label,
                  data={"dte": exp.calendar_days, "sessions": exp.trading_days})


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------


@dataclass
class StrikeChoice:
    leg: OptionLeg
    greeks: Greeks
    rejected: list[str] = field(default_factory=list)


def select_strike(chain: ChainSnapshot, exp: ExpiryContext, direction: Direction,
                  cfg: OptionEngineConfig) -> StrikeChoice | None:
    """Pick the contract to buy, targeting delta rather than cheapness.

    The instinct to buy the cheap far-OTM strike is what makes most option
    buying lose. A 0.15-delta option needs a large *and fast* move merely to
    break even, and its percentage decay per day is savage. Targeting roughly
    0.45-0.60 delta (at the money to slightly in the money) buys an option that
    actually tracks the underlying, at the cost of a bigger ticket.

    Liquidity is a filter, not a preference: greeks computed off a stale quote
    on a dead strike are precise and wrong.
    """
    if direction is Direction.NO_TRADE:
        return None

    opt_type = "CE" if direction is Direction.LONG else "PE"
    rejected: list[str] = []
    candidates: list[tuple[float, OptionLeg, Greeks]] = []

    for row in chain.rows:
        ltp = row.ce_ltp if opt_type == "CE" else row.pe_ltp
        oi = row.ce_oi if opt_type == "CE" else row.pe_oi
        vol = row.ce_volume if opt_type == "CE" else row.pe_volume
        iv_pct = row.ce_iv if opt_type == "CE" else row.pe_iv
        bid = row.ce_bid if opt_type == "CE" else row.pe_bid
        ask = row.ce_ask if opt_type == "CE" else row.pe_ask

        if not ltp or ltp <= 0:
            continue

        iv = (iv_pct / 100.0) if iv_pct else implied_vol(
            ltp, chain.spot, row.strike, exp.t_years, 0.065, opt_type)
        if not iv or iv <= 0:
            continue

        g = bs_greeks(chain.spot, row.strike, exp.t_years, iv, 0.065, opt_type)
        adelta = abs(g.delta)

        if not (cfg.target_delta_lo <= adelta <= cfg.target_delta_hi):
            continue
        if (oi or 0) < cfg.min_strike_oi:
            rejected.append(f"{row.strike:,.0f} {opt_type}: open interest {oi or 0:,} "
                            f"is below the {cfg.min_strike_oi:,} minimum")
            continue
        if (vol or 0) < cfg.min_strike_volume:
            rejected.append(f"{row.strike:,.0f} {opt_type}: volume {vol or 0:,} "
                            f"is below the {cfg.min_strike_volume:,} minimum")
            continue
        if bid and ask and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999
            if spread_pct > cfg.max_spread_pct:
                rejected.append(f"{row.strike:,.0f} {opt_type}: bid-ask spread "
                                f"{spread_pct:.1f}% exceeds {cfg.max_spread_pct}%")
                continue

        leg = OptionLeg(symbol=chain.symbol, expiry=exp.expiry, strike=row.strike,
                        option_type=opt_type, ltp=ltp, delta=round(g.delta, 4),
                        iv=round(iv * 100, 2), oi=oi, volume=vol)
        # Prefer the most liquid strike inside the acceptable delta band.
        candidates.append((float(oi or 0), leg, g))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, leg, g = candidates[0]
    moneyness = ("at the money" if abs(leg.strike - chain.spot) < (chain.spot * 0.002)
                 else "in the money" if (leg.strike < chain.spot) == (opt_type == "CE")
                 else "out of the money")
    leg.rationale = (
        f"{leg.tradingsymbol} at {leg.ltp:,.2f}. Delta {g.delta:+.2f} means the premium "
        f"moves about {abs(g.delta) * 100:.0f} paise per 1 point of {chain.symbol}. "
        f"This strike is {moneyness}, chosen for delta rather than for being cheap: "
        f"far out-of-the-money strikes cost less but need a much larger and faster move "
        f"merely to break even. Theta is {g.theta:,.2f} per day, which is "
        f"{abs(g.theta_pct):.1f}% of the premium every day it is held. "
        f"Open interest {leg.oi:,} and volume {leg.volume:,}."
    )
    return StrikeChoice(leg=leg, greeks=g, rejected=rejected[:5])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def gather_inputs(symbol: str, provider, as_of: date | None = None) -> OptionSignalInputs:
    """Fetch everything the engine needs, recording any degradation."""
    as_of = as_of or date.today()
    dq: list[str] = []

    exp = expiry_context(symbol, as_of)
    if exp.warning:
        dq.append(exp.warning)

    chain = provider.option_chain(symbol, exp.expiry)
    if chain.stale or chain.source.startswith("fixture"):
        dq.append(f"Option chain source is '{chain.source}' - not a live quote.")

    daily = provider.ohlcv(symbol, "1d", 400)
    try:
        intraday_df = provider.ohlcv(symbol, "15m", 200)
    except Exception:                                   # noqa: BLE001
        intraday_df = None
        dq.append("Intraday bars unavailable; the VWAP factor is skipped.")

    vix = provider.india_vix()
    if vix is None:
        dq.append("India VIX unavailable; the volatility-regime factor is skipped.")

    dq.extend(getattr(provider, "degraded", []))

    return OptionSignalInputs(
        symbol=symbol.upper(), chain=chain, trend=extract_trend(daily),
        intraday=extract_intraday(intraday_df), expiry=exp, india_vix=vix,
        iv_history=_iv_history_proxy(daily), data_quality=list(dict.fromkeys(dq)),
    )


def _iv_history_proxy(daily) -> list[float]:
    """IV percentile needs an IV history that no free source publishes per-strike.

    Realised volatility over a rolling year is used as a stand-in. It is a
    genuine proxy -- implied and realised track each other closely at the index
    level -- but it is *not* the same series, and the factor text says so rather
    than implying a real IV history was used.
    """
    from alpha.indicators import realized_vol

    rv = realized_vol(daily["close"], 20).dropna()
    return [float(v) for v in rv.tail(252)]


def build_signal(symbol: str, provider, as_of: date | None = None,
                 cfg: OptionEngineConfig | None = None) -> Signal:
    """Produce one option-buying signal, reasoning included.

    ADVISORY ONLY -- this returns an analysis, never an order.
    """
    cfg = cfg or OptionEngineConfig()
    inp = gather_inputs(symbol, provider, as_of)
    sc = Scorecard()

    # -- directional evidence (carries weight)
    sc.add(_factor_trend_stack(inp.trend))
    sc.add(_factor_adx(inp.trend, cfg))
    sc.add(_factor_rsi(inp.trend))
    sc.add(_factor_macd(inp.trend))
    sc.add(_factor_vwap(inp.intraday))
    sc.add(_factor_oi_buildup(inp.chain, inp.trend))
    sc.add(_factor_pcr(inp.chain))
    sc.add(_factor_oi_levels(inp.chain))
    sc.add(_factor_max_pain(inp.chain, inp.expiry))

    # -- cost gates (weight 0, may veto)
    iv, iv_src = atm_iv(inp.chain, inp.expiry)
    sc.add(_factor_expiry(inp.expiry, sc, cfg))
    sc.add(_factor_vix(inp.india_vix, cfg))
    sc.add(_factor_iv_vs_realized(iv, iv_src, inp.trend, sc, cfg))
    sc.add(_factor_iv_percentile(iv, inp.iv_history, sc, cfg))

    # -- regime warning: buying direction inside a range is a slow bleed
    if inp.trend.adx is not None and inp.trend.adx < cfg.adx_choppy:
        sc.veto("no_trend", "Index is range-bound",
                f"ADX is {inp.trend.adx:.1f}, below {cfg.adx_choppy:.0f}. There is no "
                f"established trend to ride, and long premium in a range decays without "
                f"a payoff.", severity="warn")

    direction = sc.direction(cfg.direction_threshold)
    conviction = sc.conviction()

    if direction is not Direction.NO_TRADE and conviction < cfg.min_conviction:
        sc.veto("low_conviction", "Evidence too weak",
                f"Conviction scored {conviction}, below the {cfg.min_conviction} floor. "
                f"The factors do not line up well enough to justify paying premium.")
        direction, conviction = Direction.NO_TRADE, 0

    # -- instrument selection
    choice = select_strike(inp.chain, inp.expiry, direction, cfg)
    if direction is not Direction.NO_TRADE and choice is None:
        sc.veto("no_tradable_strike", "No liquid strike in the target delta band",
                f"No {'call' if direction is Direction.LONG else 'put'} strike between "
                f"{cfg.target_delta_lo:.2f} and {cfg.target_delta_hi:.2f} delta passed the "
                f"liquidity filters (minimum {cfg.min_strike_oi:,} OI, "
                f"{cfg.min_strike_volume:,} volume, spread under {cfg.max_spread_pct}%).")
        direction, conviction = Direction.NO_TRADE, 0

    if choice is not None:
        bleed = abs(choice.greeks.theta_pct)
        if bleed >= cfg.theta_pct_block:
            sc.veto("theta_burn", "Time decay is prohibitive",
                    f"The selected contract loses {bleed:.1f}% of its premium per day to "
                    f"theta alone. At that rate the position needs to be right almost "
                    f"immediately to survive the carry.")
            direction, conviction, choice = Direction.NO_TRADE, 0, None
        elif bleed >= cfg.theta_pct_warn:
            sc.veto("theta_high", "Time decay is heavy",
                    f"The contract bleeds {bleed:.1f}% of premium per day. Hold it only "
                    f"while the move is working.", severity="warn")

    # Vetoes raised after the first read can flip the direction; recompute.
    if sc.blocking_vetoes:
        direction, conviction, choice = Direction.NO_TRADE, 0, None

    return _assemble(inp, sc, direction, conviction, choice, cfg)


def _assemble(inp: OptionSignalInputs, sc: Scorecard, direction: Direction,
              conviction: int, choice: StrikeChoice | None,
              cfg: OptionEngineConfig) -> Signal:
    spot = inp.chain.spot
    atr_v = inp.trend.atr or (spot * 0.008)
    levels = oi_support_resistance([r.as_dict() for r in inp.chain.rows], spot)

    if direction is Direction.NO_TRADE:
        blockers = sc.blocking_vetoes
        headline = f"{inp.symbol}: stand aside"
        if blockers:
            summary = ("No option-buying signal. " + " ".join(v.detail for v in blockers[:2]))
        else:
            summary = (f"No option-buying signal. The evidence is mixed - weighted score "
                       f"{sc.raw_score:+.2f} against a {cfg.direction_threshold:+.2f} "
                       f"threshold, with only {sc.agreement * 100:.0f}% of directional "
                       f"weight on one side. Paying premium needs a clearer picture.")
        return Signal(
            signal_id=str(uuid.uuid4())[:8], kind="index_option", symbol=inp.symbol,
            generated_at=datetime.now(), direction=direction, conviction=0,
            headline=headline, summary=summary, scorecard=sc, spot=spot,
            horizon="-", data_quality=inp.data_quality,
            invalidated_by=[v.label for v in sc.vetoes],
        )

    bullish = direction is Direction.LONG
    side = "calls" if bullish else "puts"

    if bullish:
        invalidation = min(levels["immediate_support"] or (spot - 1.2 * atr_v),
                           spot - 1.2 * atr_v)
        targets = [t for t in [levels["immediate_resistance"], spot + 2.0 * atr_v] if t]
    else:
        invalidation = max(levels["immediate_resistance"] or (spot + 1.2 * atr_v),
                           spot + 1.2 * atr_v)
        targets = [t for t in [levels["immediate_support"], spot - 2.0 * atr_v] if t]
    targets = sorted(set(round(float(t), 1) for t in targets), reverse=not bullish)

    reasons = sc.top_reasons(3)
    reason_text = "; ".join(f"{f.label.lower()} ({f.value})" for f in reasons if f.value)

    headline = (f"{inp.symbol}: buy {side} - conviction {conviction}/100")
    summary = (
        f"Weighted evidence leans {'bullish' if bullish else 'bearish'} "
        f"({sc.raw_score:+.2f}, {sc.agreement * 100:.0f}% of directional weight agreeing). "
        f"The heaviest contributors are {reason_text}. "
    )
    if choice is not None:
        summary += (f"Suggested contract: {choice.leg.tradingsymbol} near "
                    f"{choice.leg.ltp:,.2f}, bleeding {abs(choice.greeks.theta_pct):.1f}% "
                    f"per day to time decay. ")
    summary += (f"The view is wrong below {invalidation:,.0f}." if bullish
                else f"The view is wrong above {invalidation:,.0f}.")
    if sc.warnings:
        summary += " Caveats: " + " ".join(w.label.lower() + "." for w in sc.warnings)

    horizon = ("intraday to 2 sessions" if inp.expiry.trading_days <= 3
               else f"up to {min(inp.expiry.trading_days, 10)} sessions")

    return Signal(
        signal_id=str(uuid.uuid4())[:8], kind="index_option", symbol=inp.symbol,
        generated_at=datetime.now(), direction=direction, conviction=conviction,
        headline=headline, summary=summary, scorecard=sc, spot=spot,
        horizon=horizon, entry_zone=(round(spot - 0.15 * atr_v, 1),
                                     round(spot + 0.15 * atr_v, 1)),
        invalidation=round(float(invalidation), 1), targets=targets,
        leg=choice.leg if choice else None, data_quality=inp.data_quality,
        invalidated_by=[
            f"{inp.symbol} {'closing below' if bullish else 'closing above'} "
            f"{invalidation:,.0f}",
            "a sharp fall in India VIX (an IV crush hurts long premium even if direction is right)",
            "open interest shifting so the defended strikes move against the view",
        ],
    )
