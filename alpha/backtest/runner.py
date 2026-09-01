"""Backtest orchestration.

Walks a date range, generates signals against a point-in-time view, resolves each
against what followed, and compares the result to baselines that make the number
mean something.

Two baselines are computed, because they answer different questions:

* **Buy-and-hold the same instrument** over the same horizon, with no stop and no
  target. Isolates whether the *exits* added anything.
* **Universe average** over the same horizon. Isolates whether the *selection*
  added anything -- picks returning 2% in a month when the average name returned
  2% is a strategy that has done nothing except take on risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from alpha.backtest.evaluate import CostModel, Outcome, evaluate, evaluate_equity
from alpha.backtest.metrics import Stats, Verdict, summarise, verdict
from alpha.backtest.replay import (
    SYNTHETIC_CHAIN_FACTORS, HistoryStore, PointInTimeProvider,
)
from alpha.engines import equity_positional as eq
from alpha.engines import index_options as io
from alpha.models import Direction, Scorecard, Signal
from alpha.universe import Universe

log = logging.getLogger(__name__)


FIXTURE_WARNING = (
    "THESE RESULTS ARE MEANINGLESS: the price history is generated, not real. "
    "The bundled fixture generator builds paths from a drift plus a smooth "
    "sinusoidal cycle, and a trend-following engine predicts a sine wave nearly "
    "perfectly - which is why a run like this can show a huge, entirely fake "
    "edge. Use this mode only to check that the machinery runs. Point it at real "
    "history (ALPHA_DATA_MODE=live) before reading any number as evidence."
)


def _apply_data_guard(result: "BacktestResult", store: HistoryStore) -> "BacktestResult":
    """Overwrite the verdict when the underlying data cannot support one."""
    if not store.synthetic:
        return result
    result.caveats.insert(0, FIXTURE_WARNING)
    result.verdict = Verdict(
        conclusion="Not a real result - generated data",
        detail=FIXTURE_WARNING,
        beats_chance=None, beats_baseline=None, sample_adequate=False)
    return result


@dataclass
class BacktestConfig:
    start: date | None = None
    end: date | None = None
    step_days: int = 5                    # generate signals weekly by default
    horizon_bars: int = 10                # option holding window
    equity_horizon_bars: int = 20         # positional holding window
    costs: CostModel = field(default_factory=CostModel)
    exclude_synthetic_factors: bool = True
    index_symbols: tuple[str, ...] = ("NIFTY", "BANKNIFTY")
    equity_top_n: int = 3
    warmup_bars: int = 250                # indicators need history before day one


@dataclass
class BacktestResult:
    kind: str
    outcomes: list[Outcome]
    stats: Stats
    baseline_hold: Stats
    baseline_universe: Stats | None
    verdict: Verdict
    factor_attribution: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    config_note: str = ""
    signals_generated: int = 0
    no_trade_count: int = 0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "signals_generated": self.signals_generated,
            "no_trade_count": self.no_trade_count,
            "stats": self.stats.as_dict(),
            "baseline_hold": self.baseline_hold.as_dict(),
            "baseline_universe": self.baseline_universe.as_dict() if self.baseline_universe else None,
            "verdict": {
                "conclusion": self.verdict.conclusion, "detail": self.verdict.detail,
                "beats_chance": self.verdict.beats_chance,
                "beats_baseline": self.verdict.beats_baseline,
                "sample_adequate": self.verdict.sample_adequate,
            },
            "factor_attribution": self.factor_attribution,
            "caveats": self.caveats,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def _sample_dates(store: HistoryStore, symbol: str, cfg: BacktestConfig) -> list[date]:
    """Decision dates, spaced by ``step_days`` and leaving room to evaluate."""
    all_dates = store.trading_dates(symbol)
    if len(all_dates) <= cfg.warmup_bars:
        return []

    horizon = max(cfg.horizon_bars, cfg.equity_horizon_bars)
    usable = all_dates[cfg.warmup_bars: len(all_dates) - horizon - 1]
    if cfg.start:
        usable = [d for d in usable if d >= cfg.start]
    if cfg.end:
        usable = [d for d in usable if d <= cfg.end]

    step = max(1, cfg.step_days)
    return usable[::step]


def _hold_baseline(store: HistoryStore, symbol: str, signal_date: date,
                   horizon: int, costs: CostModel, kind: str) -> Outcome | None:
    """Buy at the next open, hold to the horizon, no stop and no target."""
    fut = store.future(symbol, signal_date).head(horizon + 1)
    if fut.empty or len(fut) < 2:
        return None
    entry = float(fut.iloc[0]["open"])
    exit_px = float(fut.iloc[-1]["close"])
    gross = (exit_px / entry - 1.0) * 100.0
    return Outcome(
        signal_id=f"hold-{symbol}-{signal_date}", symbol=symbol, kind=kind,
        direction="long", conviction=0, signal_date=signal_date,
        entry_date=fut.index[0].date(), entry_price=entry,
        exit_date=fut.index[-1].date(), exit_price=exit_px, exit_reason="horizon",
        bars_held=len(fut) - 1, gross_return_pct=round(gross, 4),
        net_return_pct=round(gross - costs.for_kind(kind) / 100.0, 4),
        underlying_return_pct=round(gross, 4), mfe_pct=0.0, mae_pct=0.0,
        directionally_right=gross > 0)


def _factor_attribution(pairs: Sequence[tuple[Scorecard, float]]) -> list[dict]:
    """Correlate each factor's score with the realised return.

    Deliberately reported with its sample size attached and never ranked as if it
    were reliable: with a few dozen observations these correlations are extremely
    noisy, and treating them as a tuning signal is how a strategy gets overfitted
    to its own backtest.
    """
    by_key: dict[str, list[tuple[float, float]]] = {}
    for sc, ret in pairs:
        for f in sc.factors:
            if f.weight > 0:
                by_key.setdefault(f.key, []).append((f.score, ret))

    rows = []
    for key, obs in by_key.items():
        if len(obs) < 5:
            continue
        scores = np.array([o[0] for o in obs], dtype=float)
        rets = np.array([o[1] for o in obs], dtype=float)
        if scores.std() < 1e-9 or rets.std() < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(scores, rets)[0, 1])
        rows.append({"factor": key, "n": len(obs), "correlation": round(corr, 3),
                     "mean_score": round(float(scores.mean()), 3)})

    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return rows


def run_options(store: HistoryStore, cfg: BacktestConfig | None = None,
                progress: Callable[[str], None] | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    ocfg = io.OptionEngineConfig(
        disabled_factors=SYNTHETIC_CHAIN_FACTORS if cfg.exclude_synthetic_factors
        else frozenset())

    outcomes: list[Outcome] = []
    holds: list[Outcome] = []
    attribution: list[tuple[Scorecard, float]] = []
    generated = no_trade = 0

    for symbol in cfg.index_symbols:
        if symbol not in store.frames:
            continue
        for d in _sample_dates(store, symbol, cfg):
            provider = PointInTimeProvider(store, d)
            try:
                sig = io.build_signal(symbol, provider, d, ocfg)
            except Exception as exc:                       # noqa: BLE001
                log.debug("signal failed for %s on %s: %s", symbol, d, exc)
                continue

            generated += 1
            if sig.direction is Direction.NO_TRADE:
                no_trade += 1
                continue

            out = evaluate(sig, store, cfg.horizon_bars, cfg.costs, as_of=d)
            if out.exit_reason == "no_data":
                continue
            outcomes.append(out)
            attribution.append((sig.scorecard, out.net_return_pct))

            hb = _hold_baseline(store, symbol, d, cfg.horizon_bars, cfg.costs,
                                "index_option")
            if hb:
                holds.append(hb)
            if progress:
                progress(f"{symbol} {d}: {sig.direction.value} -> {out.net_return_pct:+.2f}%")

    stats = summarise(outcomes)
    baseline = summarise(holds)

    caveats = [
        "Historical NSE option chains are not freely available, so the option chain "
        "at each decision date was SYNTHESISED from the underlying's trailing realised "
        "volatility. Chain prices are consistent with the real index path, but open "
        "interest is invented.",
        "Option P&L is modelled with Black-Scholes at constant implied volatility. "
        "Real IV crush after events makes outcomes worse than shown, never better. "
        "The 'directional accuracy' figure is the honest one; modelled returns are "
        "an upper bound.",
        "Intraday bars are not replayed, so the session-VWAP factor was inactive "
        "throughout. Live behaviour will differ.",
        "India VIX was proxied by trailing realised volatility.",
    ]
    if cfg.exclude_synthetic_factors:
        caveats.insert(1, "OI-based factors (PCR, max pain, OI support/resistance, OI "
                          "buildup) were EXCLUDED from scoring because their inputs are "
                          "synthetic here. This run therefore tests only the price-based "
                          "factors - which is the part that can honestly be tested.")
    else:
        caveats.insert(1, "WARNING: OI-based factors were INCLUDED but scored against "
                          "invented open interest. These results are not meaningful. "
                          "Re-run with exclude_synthetic_factors=True.")

    return _apply_data_guard(BacktestResult(
        kind="index_option", outcomes=outcomes, stats=stats, baseline_hold=baseline,
        baseline_universe=None, verdict=verdict(stats, baseline),
        factor_attribution=_factor_attribution(attribution), caveats=caveats,
        config_note=f"step {cfg.step_days}d, horizon {cfg.horizon_bars} bars",
        signals_generated=generated, no_trade_count=no_trade), store)


def run_equity(store: HistoryStore, cfg: BacktestConfig | None = None,
               universe: Universe | None = None,
               progress: Callable[[str], None] | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    universe = universe or Universe.load()
    ecfg = eq.EquityEngineConfig(top_n=cfg.equity_top_n)

    outcomes: list[Outcome] = []
    holds: list[Outcome] = []
    universe_returns: list[Outcome] = []
    attribution: list[tuple[Scorecard, float]] = []
    generated = no_trade = 0

    anchor = universe.benchmark if universe.benchmark in store.frames else next(
        iter(store.frames), None)
    if anchor is None:
        raise ValueError("history store is empty")

    for d in _sample_dates(store, anchor, cfg):
        provider = PointInTimeProvider(store, d, synthesise_chains=False)
        try:
            sigs = eq.scan(provider, d, ecfg, universe=universe)
        except Exception as exc:                            # noqa: BLE001
            log.debug("equity scan failed on %s: %s", d, exc)
            continue

        generated += 1
        if not sigs:
            no_trade += 1

        for sig in sigs:
            out = evaluate_equity(sig, store, cfg.equity_horizon_bars, cfg.costs,
                                  as_of=d)
            if out.exit_reason == "no_data":
                continue
            outcomes.append(out)
            attribution.append((sig.scorecard, out.net_return_pct))
            hb = _hold_baseline(store, sig.symbol, d, cfg.equity_horizon_bars,
                                cfg.costs, "equity_positional")
            if hb:
                holds.append(hb)
            if progress:
                progress(f"{d} {sig.symbol}: {out.net_return_pct:+.2f}%")

        # Selection baseline: what the average universe member did from here.
        for sym in universe.symbols:
            if sym in store.frames:
                ub = _hold_baseline(store, sym, d, cfg.equity_horizon_bars,
                                    cfg.costs, "equity_positional")
                if ub:
                    universe_returns.append(ub)

    stats = summarise(outcomes)
    baseline = summarise(holds)
    uni = summarise(universe_returns) if universe_returns else None

    caveats = [
        "The scan universe is TODAY's constituent list applied to past dates. That is "
        "survivorship bias: names that were dropped from the index are absent, and the "
        "survivors did better than the full historical set. Real results would be worse.",
        "No corporate-action adjustment (splits, bonuses, dividends). A split in the "
        "sample window shows up as a crash and will distort those outcomes.",
        "Positions are treated one at a time and sized equally; the drawdown figure is "
        "sequential, not portfolio-level.",
    ]
    if uni is not None:
        caveats.append("The 'universe average' baseline is the honest test of stock "
                       "selection: beating it is the only evidence that picking helped.")

    return _apply_data_guard(BacktestResult(
        kind="equity_positional", outcomes=outcomes, stats=stats,
        baseline_hold=baseline, baseline_universe=uni,
        verdict=verdict(stats, uni or baseline),
        factor_attribution=_factor_attribution(attribution), caveats=caveats,
        config_note=f"step {cfg.step_days}d, horizon {cfg.equity_horizon_bars} bars, "
                    f"top {cfg.equity_top_n}",
        signals_generated=generated, no_trade_count=no_trade), store)
