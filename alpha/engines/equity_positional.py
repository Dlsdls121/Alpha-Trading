"""Positional equity signal engine -- multi-day to multi-week holds.

ADVISORY ONLY. This ranks candidates and explains the ranking. It never trades.

What this engine is built around
--------------------------------
Positional selection is a *relative* problem, not an absolute one. In any given
month a third of a large-cap index is going up; the question is which names are
being accumulated rather than merely carried along. So the two heaviest factors
are relative strength against the benchmark and the momentum of the stock's own
sector, with absolute trend and structure supporting them.

Sector momentum is computed from the constituents themselves -- the mean
relative strength of every stock in that sector -- rather than from sector index
data. That keeps the engine self-contained and means a sector's reading always
describes the same names being scanned.

A deliberate asymmetry: this engine only proposes longs. Positional shorting in
Indian cash equity is not available to most retail participants beyond intraday,
so a "buy puts on a weak stock" suggestion would be advice the reader mostly
cannot act on. Weak stocks are reported as avoid, not as short.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from alpha.engines.features import TrendFeatures, extract_trend, scale
from alpha.indicators import ema, relative_volume, rsi
from alpha.models import (
    Category, Direction, Factor, Scorecard, Signal, Verdict,
)
from alpha.universe import Universe


@dataclass
class EquityEngineConfig:
    direction_threshold: float = 0.20
    min_conviction: int = 30

    rs_lookback_short: int = 63          # ~3 months
    rs_lookback_long: int = 126          # ~6 months

    near_high_pct: float = 8.0           # "within 5-8% of the 52w high"
    max_extension_atr: float = 4.0       # too far above EMA20 = chasing

    min_turnover_cr: float = 25.0        # average daily traded value, Rs crore
    max_atr_pct: float = 6.0             # above this, sizing gets dangerous

    rsi_overbought: float = 80.0
    stop_atr_multiple: float = 2.0
    # Targets must clear the stop distance or the trade is not worth taking:
    # 3 and 5 ATR against a 2-ATR stop gives 1.5:1 on the first target, 2.5:1 on the second.
    target_atr_multiples: tuple[float, float] = (3.0, 5.0)
    min_reward_risk: float = 1.3

    top_n: int = 5


@dataclass
class EquityContext:
    """Per-symbol data, plus the benchmark and sector aggregates."""

    symbol: str
    name: str
    sector: str
    daily: pd.DataFrame
    trend: TrendFeatures
    benchmark: pd.DataFrame
    rs_short: float | None
    rs_long: float | None
    rs_slope: float | None
    sector_rs: float | None
    sector_rank: int | None
    sector_count: int | None
    sector_n: int | None               # constituents of this sector in the scan
    high_52w: float | None
    low_52w: float | None
    turnover_cr: float | None
    data_quality: list[str] = field(default_factory=list)


def relative_strength(stock: pd.Series, bench: pd.Series, lookback: int) -> float | None:
    """Stock return minus benchmark return over ``lookback`` sessions, in points.

    Positive means the stock outperformed. This is the plainest possible RS and
    it is the one that matters: a stock up 4% while the index is up 9% is weak,
    however green the chart looks.
    """
    if len(stock) <= lookback or len(bench) <= lookback:
        return None
    s = stock.iloc[-1] / stock.iloc[-1 - lookback] - 1.0
    b = bench.iloc[-1] / bench.iloc[-1 - lookback] - 1.0
    if pd.isna(s) or pd.isna(b):
        return None
    return float((s - b) * 100.0)


def rs_line_slope(stock: pd.Series, bench: pd.Series, window: int = 21) -> float | None:
    """Slope of the stock/benchmark ratio -- is outperformance still building?

    A stock can carry a great 6-month RS number while having quietly stopped
    outperforming six weeks ago. The slope catches that; the level does not.
    """
    if len(stock) < window + 1 or len(bench) < window + 1:
        return None
    aligned = pd.concat([stock, bench], axis=1, join="inner").dropna()
    if len(aligned) < window + 1:
        return None
    ratio = (aligned.iloc[:, 0] / aligned.iloc[:, 1]).tail(window)
    x = np.arange(len(ratio), dtype=float)
    slope = np.polyfit(x, ratio.to_numpy(dtype=float), 1)[0]
    return float(slope / ratio.mean() * 100.0 * window)     # % drift over the window


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------


def _factor_relative_strength(ctx: EquityContext) -> Factor:
    if ctx.rs_short is None:
        return Factor("rel_strength", "Relative strength vs benchmark", Category.RELATIVE,
                      Verdict.NEUTRAL, 0.0, 0.0,
                      "Not enough overlapping history to measure relative strength.",
                      value="n/a")

    long_part = ctx.rs_long if ctx.rs_long is not None else ctx.rs_short
    blended = 0.6 * ctx.rs_short + 0.4 * long_part
    score = scale(blended, -12.0, 12.0)

    slope_txt = ""
    if ctx.rs_slope is not None:
        if ctx.rs_slope > 0.5:
            slope_txt = (f" The relative-strength line is still rising "
                         f"({ctx.rs_slope:+.2f}% over the last month), so the "
                         f"outperformance is current rather than historical.")
            score = min(1.0, score + 0.12)
        elif ctx.rs_slope < -0.5:
            slope_txt = (f" But the relative-strength line has rolled over "
                         f"({ctx.rs_slope:+.2f}% over the last month) - the stock has "
                         f"stopped outperforming even if the longer number is still good.")
            score -= 0.25

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    detail = (f"Over 3 months the stock has {'beaten' if ctx.rs_short >= 0 else 'lagged'} "
              f"the benchmark by {abs(ctx.rs_short):.1f} percentage points"
              + (f", and over 6 months by {abs(long_part):.1f} points "
                 f"({'ahead' if long_part >= 0 else 'behind'})" if ctx.rs_long is not None else "")
              + f".{slope_txt}")

    return Factor("rel_strength", "Relative strength vs benchmark", Category.RELATIVE,
                  verdict, max(-1.0, min(1.0, score)), 2.0, detail,
                  value=f"{ctx.rs_short:+.1f}pp (3m)",
                  data={"rs_3m": round(ctx.rs_short, 2),
                        "rs_6m": round(long_part, 2) if long_part is not None else None,
                        "rs_slope": round(ctx.rs_slope, 3) if ctx.rs_slope is not None else None})


def _factor_sector(ctx: EquityContext) -> Factor:
    if ctx.sector_rs is None or ctx.sector_rank is None:
        return Factor("sector", "Sector rotation", Category.RELATIVE, Verdict.NEUTRAL,
                      0.0, 0.0, "Sector strength could not be computed.", value="n/a")

    # A sector represented by a single scanned name has a "sector strength" that
    # is just that stock's own relative strength under another name. Scoring it
    # would count the same evidence twice, so it is shown and given zero weight.
    if (ctx.sector_n or 0) < 2:
        return Factor("sector", "Sector rotation", Category.RELATIVE, Verdict.NEUTRAL,
                      0.0, 0.0,
                      f"{ctx.sector} has only one representative in the scan universe, so a "
                      f"sector reading here would just restate this stock's own relative "
                      f"strength. Carried as context, not counted as evidence. Add more "
                      f"{ctx.sector} names to alpha/reference/universe.json to make this "
                      f"factor meaningful.",
                      value=f"{ctx.sector} (n=1)",
                      data={"sector_rs": round(ctx.sector_rs, 2), "sector_n": ctx.sector_n})

    score = scale(ctx.sector_rs, -8.0, 8.0)
    pos = f"{ctx.sector_rank} of {ctx.sector_count}"
    if ctx.sector_rank <= max(1, (ctx.sector_count or 1) // 3):
        standing = "a leading sector"
    elif ctx.sector_rank >= (ctx.sector_count or 1) - max(1, (ctx.sector_count or 1) // 3):
        standing = "a lagging sector"
    else:
        standing = "a middling sector"

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    return Factor("sector", "Sector rotation", Category.RELATIVE, verdict, score, 1.2,
                  f"{ctx.sector} is {standing}, ranked {pos} by the mean 3-month relative "
                  f"strength of its {ctx.sector_n} scanned constituents ({ctx.sector_rs:+.1f} points vs the "
                  f"benchmark). Money rotates by sector, so a strong name in a weak sector "
                  f"is fighting its own group.",
                  value=f"{ctx.sector} #{ctx.sector_rank}",
                  data={"sector_rs": round(ctx.sector_rs, 2), "rank": ctx.sector_rank})


def _factor_trend_stage(ctx: EquityContext) -> Factor:
    st = ctx.trend.stack
    if st.get("stage") == "unknown":
        return Factor("stage", "Trend stage", Category.TREND, Verdict.NEUTRAL, 0.0, 0.0,
                      "Not enough history for the 20/50/200 EMA stack.", value="n/a")

    aligned = st.get("aligned") or 0.0
    score = aligned
    slope_note = ""
    if st.get("slow_slope") is not None:
        rising = st["slow_slope"] > 0
        slope_note = (f" The 200 EMA is {'rising' if rising else 'falling'} "
                      f"({st['slow_slope']:+.2f}% over 20 sessions).")
        if not rising:
            score -= 0.2

    verdict = (Verdict.BULLISH if score > 0.3 else
               Verdict.BEARISH if score < -0.3 else Verdict.NEUTRAL)
    return Factor("stage", "Trend stage", Category.TREND, verdict,
                  max(-1.0, min(1.0, score)), 2.0,
                  f"Price {st['px']:,.2f} vs EMA20 {st['fast']:,.2f}, EMA50 {st['mid']:,.2f}, "
                  f"EMA200 {st['slow']:,.2f} - {st['stage'].replace('_', ' ')}.{slope_note} "
                  f"Positional longs work best when the stack is ordered and the long-term "
                  f"average is rising.",
                  value=st["stage"].replace("_", " "))


def _factor_high_proximity(ctx: EquityContext, cfg: EquityEngineConfig) -> Factor:
    if ctx.high_52w is None or ctx.high_52w <= 0:
        return Factor("high_proximity", "Position in 52-week range", Category.STRUCTURE,
                      Verdict.NEUTRAL, 0.0, 0.0, "No 52-week high available.", value="n/a")

    px = ctx.trend.close
    off_high = (1 - px / ctx.high_52w) * 100.0
    rng_pos = ((px - ctx.low_52w) / (ctx.high_52w - ctx.low_52w) * 100.0
               if ctx.low_52w is not None and ctx.high_52w > ctx.low_52w else None)

    # Near the high is strength, not risk: breakouts come from stocks already there.
    score = scale(-off_high, -25.0, -1.0)
    if off_high <= cfg.near_high_pct:
        note = (f"Within {cfg.near_high_pct:.0f}% of the 52-week high, which is where "
                f"breakouts start - there is no overhead supply of trapped holders.")
    elif off_high > 25:
        note = ("Far below the 52-week high, so any rally has to work through layers of "
                "holders waiting to get out at break-even.")
    else:
        note = "Mid-range - neither breaking out nor deeply out of favour."

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    return Factor("high_proximity", "Position in 52-week range", Category.STRUCTURE,
                  verdict, score, 1.5,
                  f"Trading {off_high:.1f}% below its 52-week high of {ctx.high_52w:,.2f}"
                  + (f", which is {rng_pos:.0f}% of the way up the 52-week range" if rng_pos is not None else "")
                  + f". {note}",
                  value=f"{off_high:.1f}% off high",
                  data={"off_high_pct": round(off_high, 2), "high_52w": ctx.high_52w})


def _factor_momentum(ctx: EquityContext) -> Factor:
    t = ctx.trend
    if t.roc21 is None:
        return Factor("momentum", "Price momentum", Category.MOMENTUM, Verdict.NEUTRAL,
                      0.0, 0.0, "Momentum unavailable.", value="n/a")

    score = scale(t.roc21, -10.0, 10.0)
    rsi_note = ""
    if t.rsi is not None:
        rsi_note = f" RSI(14) is {t.rsi:.1f}"
        if t.rsi > 80:
            rsi_note += " - extended, so entries are better on a pullback than at market."
            score *= 0.6
        elif t.rsi < 35:
            rsi_note += " - weak."
        else:
            rsi_note += "."

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    return Factor("momentum", "Price momentum", Category.MOMENTUM, verdict, score, 1.2,
                  f"Up {t.roc21:+.1f}% over the last 21 sessions.{rsi_note}",
                  value=f"{t.roc21:+.1f}% (21d)")


def _factor_volume(ctx: EquityContext) -> Factor:
    vol = ctx.daily["volume"]
    rv = relative_volume(vol, 20)
    if rv.dropna().empty:
        return Factor("volume", "Volume confirmation", Category.STRUCTURE, Verdict.NEUTRAL,
                      0.0, 0.0, "No volume data.", value="n/a")

    recent = float(rv.tail(5).mean())
    if pd.isna(recent):
        return Factor("volume", "Volume confirmation", Category.STRUCTURE, Verdict.NEUTRAL,
                      0.0, 0.0, "No volume data.", value="n/a")

    up = ctx.trend.roc21 is not None and ctx.trend.roc21 > 0
    # Heavy volume confirms whichever way price is going.
    score = scale(recent, 0.7, 1.8) * (1.0 if up else -1.0)

    if recent > 1.3:
        note = "well above average - the move is being backed by real participation"
    elif recent < 0.8:
        note = "below average - the move lacks conviction behind it"
    else:
        note = "about average"

    verdict = (Verdict.BULLISH if score > 0.15 else
               Verdict.BEARISH if score < -0.15 else Verdict.NEUTRAL)
    return Factor("volume", "Volume confirmation", Category.STRUCTURE, verdict, score, 1.0,
                  f"Volume over the last 5 sessions is running {recent:.2f}x its 20-day "
                  f"average - {note}.", value=f"{recent:.2f}x")


def _factor_extension(ctx: EquityContext, cfg: EquityEngineConfig, sc: Scorecard) -> Factor:
    """How far price has run from its own mean. Chasing is a real cost."""
    st = ctx.trend.stack
    if st.get("fast") is None or not ctx.trend.atr:
        return Factor("extension", "Extension from mean", Category.STRUCTURE,
                      Verdict.NEUTRAL, 0.0, 0.0, "Extension unavailable.", value="n/a")

    dist_atr = (ctx.trend.close - st["fast"]) / ctx.trend.atr
    if dist_atr > cfg.max_extension_atr:
        sc.veto("extended", "Price is stretched from its mean",
                f"The stock is {dist_atr:.1f} ATRs above its 20 EMA. Entering here means "
                f"buying after the move, with a stop that has to sit far away.",
                severity="warn")
        note = "stretched - better entered on a pullback toward the mean"
        score = -0.3
    elif dist_atr < -2.0:
        note = "well below its short-term mean"
        score = -0.2
    else:
        note = "a reasonable distance from its mean, so entry risk is contained"
        score = 0.25

    return Factor("extension", "Extension from mean", Category.STRUCTURE, Verdict.NEUTRAL,
                  score, 0.8,
                  f"Price sits {dist_atr:+.1f} ATRs from the 20 EMA - {note}.",
                  value=f"{dist_atr:+.1f} ATR")


# -- gates (weight 0) ------------------------------------------------------


def _factor_liquidity(ctx: EquityContext, cfg: EquityEngineConfig, sc: Scorecard) -> Factor:
    if ctx.turnover_cr is None:
        return Factor("liquidity", "Liquidity", Category.LIQUIDITY, Verdict.NEUTRAL,
                      0.0, 0.0, "Turnover could not be computed.", value="n/a")

    if ctx.turnover_cr < cfg.min_turnover_cr:
        sc.veto("illiquid", "Too thinly traded",
                f"Average daily turnover is about Rs {ctx.turnover_cr:,.0f} crore, below the "
                f"Rs {cfg.min_turnover_cr:,.0f} crore floor. Slippage and gap risk in a name "
                f"this thin can exceed the edge being chased.")

    return Factor("liquidity", "Liquidity", Category.LIQUIDITY, Verdict.NEUTRAL, 0.0, 0.0,
                  f"Average daily turnover is roughly Rs {ctx.turnover_cr:,.0f} crore over "
                  f"the last 20 sessions.", value=f"Rs {ctx.turnover_cr:,.0f} cr")


def _factor_risk(ctx: EquityContext, cfg: EquityEngineConfig, sc: Scorecard) -> Factor:
    if ctx.trend.atr_pct is None:
        return Factor("risk", "Volatility / sizing", Category.COST, Verdict.NEUTRAL,
                      0.0, 0.0, "ATR unavailable.", value="n/a")

    ap = ctx.trend.atr_pct
    if ap > cfg.max_atr_pct:
        sc.veto("volatile", "Unusually volatile",
                f"Daily ATR is {ap:.1f}% of price. A {cfg.stop_atr_multiple:.0f}-ATR stop "
                f"sits {ap * cfg.stop_atr_multiple:.1f}% away, so position size has to be "
                f"cut hard for the risk to stay sane.", severity="warn")

    return Factor("risk", "Volatility / sizing", Category.COST, Verdict.NEUTRAL, 0.0, 0.0,
                  f"Daily ATR is {ap:.2f}% of price, so a {cfg.stop_atr_multiple:.0f}-ATR "
                  f"stop sits about {ap * cfg.stop_atr_multiple:.1f}% below entry. Size the "
                  f"position from that distance, not from a fixed rupee amount.",
                  value=f"ATR {ap:.2f}%")


# ---------------------------------------------------------------------------
# Scan orchestration
# ---------------------------------------------------------------------------


def _turnover_crore(df: pd.DataFrame, window: int = 20) -> float | None:
    """Average daily traded value in Rs crore (1 crore = 10 million)."""
    if df.empty or "volume" not in df:
        return None
    tail = df.tail(window)
    value = (tail["close"] * tail["volume"]).mean()
    return None if pd.isna(value) else float(value) / 1e7


def build_contexts(provider, universe: Universe, as_of: date | None = None,
                   cfg: EquityEngineConfig | None = None) -> list[EquityContext]:
    """Load every symbol once, then compute the cross-sectional aggregates.

    Sector strength is genuinely cross-sectional -- it cannot be known from one
    stock -- so the whole universe is loaded before any factor is scored.
    """
    cfg = cfg or EquityEngineConfig()
    bench_df = provider.ohlcv(universe.benchmark, "1d", 400)
    bench_close = bench_df["close"]

    raw: list[EquityContext] = []
    for sym in universe.symbols:
        dq: list[str] = []
        try:
            df = provider.ohlcv(sym, "1d", 400)
        except Exception as exc:                              # noqa: BLE001
            dq.append(f"Skipped {sym}: {exc}")
            continue
        if len(df) < 210:
            continue

        close = df["close"]
        win = close.tail(252)
        raw.append(EquityContext(
            symbol=sym, name=universe.name_of(sym), sector=universe.sector_of(sym),
            daily=df, trend=extract_trend(df), benchmark=bench_df,
            rs_short=relative_strength(close, bench_close, cfg.rs_lookback_short),
            rs_long=relative_strength(close, bench_close, cfg.rs_lookback_long),
            rs_slope=rs_line_slope(close, bench_close, 21),
            sector_rs=None, sector_rank=None, sector_count=None, sector_n=None,
            high_52w=float(win.max()), low_52w=float(win.min()),
            turnover_cr=_turnover_crore(df), data_quality=dq,
        ))

    # Cross-sectional pass: mean 3m RS per sector, then rank sectors.
    by_sector: dict[str, list[float]] = {}
    for c in raw:
        if c.rs_short is not None:
            by_sector.setdefault(c.sector, []).append(c.rs_short)

    sector_rs = {s: float(np.mean(v)) for s, v in by_sector.items() if v}
    ranked = sorted(sector_rs.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {s: i + 1 for i, (s, _) in enumerate(ranked)}

    for c in raw:
        c.sector_rs = sector_rs.get(c.sector)
        c.sector_rank = rank_of.get(c.sector)
        c.sector_count = len(ranked)
        c.sector_n = len(by_sector.get(c.sector, []))
    return raw


def score_symbol(ctx: EquityContext, cfg: EquityEngineConfig) -> tuple[Scorecard, Direction, int]:
    sc = Scorecard()
    sc.add(_factor_relative_strength(ctx))
    sc.add(_factor_trend_stage(ctx))
    sc.add(_factor_high_proximity(ctx, cfg))
    sc.add(_factor_sector(ctx))
    sc.add(_factor_momentum(ctx))
    sc.add(_factor_volume(ctx))
    sc.add(_factor_extension(ctx, cfg, sc))
    sc.add(_factor_liquidity(ctx, cfg, sc))
    sc.add(_factor_risk(ctx, cfg, sc))

    # Reward-to-risk is a property of the setup, not an opinion about it, so it
    # gates the trade rather than nudging the score.
    px = ctx.trend.close
    atr_v = ctx.trend.atr or px * 0.02
    stop_dist = cfg.stop_atr_multiple * atr_v
    rr = (cfg.target_atr_multiples[0] * atr_v) / stop_dist if stop_dist > 0 else 0.0
    if rr < cfg.min_reward_risk:
        sc.veto("poor_rr", "Reward does not justify the risk",
                f"A {cfg.stop_atr_multiple:.0f}-ATR stop against a "
                f"{cfg.target_atr_multiples[0]:.0f}-ATR first target is only {rr:.1f}:1, "
                f"below the {cfg.min_reward_risk:.1f}:1 floor.")

    direction = sc.direction(cfg.direction_threshold)
    conviction = sc.conviction()

    # Long-only by design -- see the module docstring.
    if direction is Direction.SHORT:
        direction, conviction = Direction.NO_TRADE, 0
    if direction is Direction.LONG and conviction < cfg.min_conviction:
        direction, conviction = Direction.NO_TRADE, 0
    if sc.blocking_vetoes:
        direction, conviction = Direction.NO_TRADE, 0

    return sc, direction, conviction


def _build_signal(ctx: EquityContext, sc: Scorecard, direction: Direction,
                  conviction: int, cfg: EquityEngineConfig,
                  extra_dq: list[str]) -> Signal:
    px = ctx.trend.close
    atr_v = ctx.trend.atr or px * 0.02
    stop = px - cfg.stop_atr_multiple * atr_v
    targets = [round(px + m * atr_v, 2) for m in cfg.target_atr_multiples]

    reasons = sc.top_reasons(3)
    reason_text = "; ".join(f"{f.label.lower()} ({f.value})" for f in reasons if f.value)
    rr = (targets[0] - px) / (px - stop) if px > stop else 0.0

    if direction is Direction.LONG:
        headline = f"{ctx.symbol}: accumulate - conviction {conviction}/100"
        summary = (
            f"{ctx.name} ({ctx.sector}) scores {sc.raw_score:+.2f} with "
            f"{sc.agreement * 100:.0f}% of directional weight agreeing. Leading reasons: "
            f"{reason_text}. Around {px:,.2f}, a {cfg.stop_atr_multiple:.0f}-ATR stop sits "
            f"at {stop:,.2f} ({(1 - stop / px) * 100:.1f}% away) with a first target of "
            f"{targets[0]:,.2f} - roughly {rr:.1f}:1 on the first target. "
            f"The thesis breaks on a close below {stop:,.2f}."
        )
    else:
        blockers = sc.blocking_vetoes
        headline = f"{ctx.symbol}: avoid"
        summary = ((" ".join(v.detail for v in blockers[:2])) if blockers else
                   f"{ctx.name} scores {sc.raw_score:+.2f}, short of the "
                   f"{cfg.direction_threshold:+.2f} bar for a positional long. "
                   f"Leading reasons: {reason_text}.")

    return Signal(
        signal_id=str(uuid.uuid4())[:8], kind="equity_positional", symbol=ctx.symbol,
        generated_at=datetime.now(), direction=direction, conviction=conviction,
        headline=headline, summary=summary, scorecard=sc, spot=round(px, 2),
        horizon="5-20 sessions",
        entry_zone=(round(px - 0.35 * atr_v, 2), round(px + 0.35 * atr_v, 2)),
        invalidation=round(stop, 2),
        targets=targets if direction is Direction.LONG else [],
        data_quality=list(dict.fromkeys(ctx.data_quality + extra_dq)),
        invalidated_by=[
            f"a daily close below {stop:,.2f} ({cfg.stop_atr_multiple:.0f} ATR)",
            "the relative-strength line rolling over versus the benchmark",
            f"{ctx.sector} losing its sector leadership",
        ],
    )


def scan(provider, as_of: date | None = None, cfg: EquityEngineConfig | None = None,
         universe: Universe | None = None, include_rejected: bool = False) -> list[Signal]:
    """Rank the universe and return the best positional candidates.

    Returns signals sorted by conviction. With ``include_rejected`` the
    stand-aside names come back too, each still carrying its full reasoning --
    knowing why a stock was passed over is often more useful than the picks.
    """
    cfg = cfg or EquityEngineConfig()
    universe = universe or Universe.load()

    contexts = build_contexts(provider, universe, as_of, cfg)
    if not contexts:
        return []

    # Read the provider's degradation notes *after* fetching, not before: most
    # fallbacks only announce themselves at the moment a fetch fails, so a
    # snapshot taken up front misses exactly the ones worth reporting.
    extra_dq = list(getattr(provider, "degraded", []))

    signals: list[Signal] = []
    for ctx in contexts:
        sc, direction, conviction = score_symbol(ctx, cfg)
        sig = _build_signal(ctx, sc, direction, conviction, cfg, extra_dq)
        if direction is Direction.LONG or include_rejected:
            signals.append(sig)

    signals.sort(key=lambda s: (s.direction is Direction.LONG, s.conviction,
                                s.scorecard.raw_score), reverse=True)
    if include_rejected:
        return signals
    return signals[:cfg.top_n]


def sector_table(provider, as_of: date | None = None,
                 universe: Universe | None = None,
                 cfg: EquityEngineConfig | None = None) -> list[dict]:
    """Sector leadership board -- the rotation picture behind the stock picks."""
    cfg = cfg or EquityEngineConfig()
    universe = universe or Universe.load()
    contexts = build_contexts(provider, universe, as_of, cfg)

    agg: dict[str, list[float]] = {}
    for c in contexts:
        if c.rs_short is not None:
            agg.setdefault(c.sector, []).append(c.rs_short)

    rows = [{"sector": s, "mean_rs_3m": round(float(np.mean(v)), 2),
             "constituents": len(v),
             "leaders": [c.symbol for c in sorted(
                 [x for x in contexts if x.sector == s and x.rs_short is not None],
                 key=lambda x: x.rs_short, reverse=True)][:3]}
            for s, v in agg.items()]
    rows.sort(key=lambda r: r["mean_rs_3m"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
