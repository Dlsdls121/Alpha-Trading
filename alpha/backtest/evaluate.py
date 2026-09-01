"""Forward outcome evaluation.

Every modelling choice here is deliberately the pessimistic one, because a
backtest's job is to try to talk you out of a strategy, not to sell it to you.
The choices, and why:

* **Entry on the next bar's open, never the signal bar's close.** A signal
  computed from day T's close cannot be filled at day T's close. Using it is the
  most common way a backtest invents an edge that does not exist.
* **Stop wins ties.** When a bar's high reaches the target *and* its low reaches
  the stop, intrabar order is unknowable from daily data. Assuming the stop hit
  first is the assumption that costs money, so it is the one used.
* **Gaps fill at the open, not the level.** If price gaps straight through a
  stop, the fill is the open, which is worse than the stop. Modelling it as the
  stop level would be a fiction that flatters every result.
* **Costs come off every trade** -- brokerage, taxes and slippage as a round-trip
  charge in basis points, with a wider default for options because the spread is
  the real cost there.
* **Option IV is held constant** while the position is open. This is *optimistic*
  and is flagged everywhere it is reported: a real IV crush after an event makes
  option outcomes worse than modelled, never better. That is why underlying
  directional accuracy is reported as the primary honest number and modelled
  option P&L as a clearly-labelled secondary.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from alpha.indicators.options import bs_price
from alpha.models import Direction, Signal

ExitReason = Literal["target", "stop", "expiry", "horizon", "no_data"]


@dataclass
class CostModel:
    """Round-trip transaction costs in basis points of traded value.

    Defaults are deliberately not optimistic. Indian equity delivery carries
    brokerage, STT, exchange fees, GST and stamp duty; index options additionally
    cross a bid-ask spread that is usually the largest single cost for a retail
    buyer.
    """

    equity_bps: float = 35.0        # ~0.35% round trip, all in
    option_bps: float = 120.0       # ~1.2% round trip, spread-dominated
    slippage_bps: float = 10.0

    def for_kind(self, kind: str) -> float:
        base = self.option_bps if kind == "index_option" else self.equity_bps
        return base + self.slippage_bps


@dataclass
class Outcome:
    """What actually happened after a signal."""

    signal_id: str
    symbol: str
    kind: str
    direction: str
    conviction: int
    signal_date: date
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    exit_reason: ExitReason
    bars_held: int
    gross_return_pct: float
    net_return_pct: float
    underlying_return_pct: float
    mfe_pct: float                  # max favourable excursion
    mae_pct: float                  # max adverse excursion
    directionally_right: bool | None
    modelled: bool = False          # True when option P&L came from a model
    notes: list[str] = field(default_factory=list)

    @property
    def win(self) -> bool:
        return self.net_return_pct > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("signal_date", "entry_date", "exit_date"):
            d[key] = d[key].isoformat() if d[key] else None
        return d


def _future_bars(store, symbol: str, after: date, limit: int) -> pd.DataFrame:
    fut = store.future(symbol, after)
    return fut.head(limit) if not fut.empty else fut


def _walk(bars: pd.DataFrame, entry: float, stop: float, target: float,
          long: bool) -> tuple[int, float, str, float, float]:
    """Walk bars forward and resolve the first exit.

    Returns ``(bar_index, exit_price, reason, mfe_pct, mae_pct)``. ``bar_index``
    is -1 when neither level is touched, meaning the caller exits on time.
    """
    mfe = mae = 0.0
    for i, (_, bar) in enumerate(bars.iterrows()):
        hi, lo, op = float(bar["high"]), float(bar["low"]), float(bar["open"])

        fav = (hi - entry) / entry * 100 if long else (entry - lo) / entry * 100
        adv = (lo - entry) / entry * 100 if long else (entry - hi) / entry * 100
        mfe, mae = max(mfe, fav), min(mae, adv)

        if long:
            gapped_through_stop = op <= stop
            hit_stop, hit_target = lo <= stop, hi >= target
        else:
            gapped_through_stop = op >= stop
            hit_stop, hit_target = hi >= stop, lo <= target

        # A gap through the stop fills at the open, which is worse than the stop.
        if gapped_through_stop:
            return i, op, "stop", mfe, mae
        # Stop wins ties: intrabar order is unknowable from daily bars.
        if hit_stop:
            return i, stop, "stop", mfe, mae
        if hit_target:
            return i, target, "target", mfe, mae

    return -1, float(bars["close"].iloc[-1]) if not bars.empty else entry, "horizon", mfe, mae


def evaluate_equity(sig: Signal, store, horizon_bars: int = 20,
                    costs: CostModel | None = None,
                    as_of: date | None = None) -> Outcome:
    """Resolve a positional equity signal against what followed.

    ``as_of`` is the *simulated decision date*. It must be passed under replay:
    ``sig.generated_at`` is wall-clock time, so relying on it silently searches
    for bars after today and finds none. The signal has no idea it is being
    backtested, and should not need to.
    """
    costs = costs or CostModel()
    decided = as_of or sig.generated_at.date()
    bars = _future_bars(store, sig.symbol, decided, horizon_bars + 1)

    if bars.empty:
        return _no_data(sig, as_of=decided)

    entry_bar = bars.iloc[0]
    entry_date = bars.index[0].date()
    entry = float(entry_bar["open"])            # next bar's open, never today's close

    stop = float(sig.invalidation) if sig.invalidation else entry * 0.95
    target = float(sig.targets[0]) if sig.targets else entry * 1.05

    # The remaining bars are where the trade actually plays out.
    walk_bars = bars.iloc[1:] if len(bars) > 1 else bars
    idx, exit_px, reason, mfe, mae = _walk(walk_bars, entry, stop, target, long=True)

    if idx >= 0:
        exit_date, held = walk_bars.index[idx].date(), idx + 1
    else:
        exit_date, held = walk_bars.index[-1].date(), len(walk_bars)

    gross = (exit_px / entry - 1.0) * 100.0
    net = gross - costs.for_kind(sig.kind) / 100.0
    return Outcome(
        signal_id=sig.signal_id, symbol=sig.symbol, kind=sig.kind,
        direction=sig.direction.value, conviction=sig.conviction,
        signal_date=decided, entry_date=entry_date, entry_price=entry,
        exit_date=exit_date, exit_price=exit_px, exit_reason=reason, bars_held=held,
        gross_return_pct=round(gross, 4), net_return_pct=round(net, 4),
        underlying_return_pct=round(gross, 4), mfe_pct=round(mfe, 3), mae_pct=round(mae, 3),
        directionally_right=gross > 0,
    )


def evaluate_option(sig: Signal, store, horizon_bars: int = 10,
                    costs: CostModel | None = None,
                    hold_iv_constant: bool = True,
                    as_of: date | None = None) -> Outcome:
    """Resolve an index option signal.

    Two numbers come out of this and they must not be conflated:

    ``underlying_return_pct``  -- what the index did, signed by the signal's
    direction. Real, and the honest measure of whether the *call* was right.

    ``net_return_pct`` -- modelled option P&L, repricing the contract with
    Black-Scholes as spot moves and time decays. Real underlying path, real
    theta, but **constant IV**, which is optimistic. Marked ``modelled=True``.
    """
    costs = costs or CostModel()
    decided = as_of or sig.generated_at.date()
    leg = sig.leg
    bars = _future_bars(store, sig.symbol, decided, horizon_bars + 1)

    if bars.empty or leg is None:
        return _no_data(sig, as_of=decided)

    # Truncate at expiry BEFORE walking. Clamping only the exit date while taking
    # the spot from a later bar prices the option at t=0 using a price from after
    # it expired -- and because intrinsic value is max(0, S-K), a convex and
    # non-negative function, feeding it a higher-variance future price inflates
    # the payoff systematically. That single mistake manufactured a ~+19% mean
    # return out of pure random-walk data. A contract cannot be held past its own
    # expiry, so the bars after it must not exist for this trade.
    bars = bars[bars.index.date <= leg.expiry]
    if len(bars) < 2:
        return _no_data(sig, "expiry falls at or before the first tradeable bar",
                        as_of=decided)

    entry_date = bars.index[0].date()
    entry_spot = float(bars.iloc[0]["open"])
    long_underlying = sig.direction is Direction.LONG

    stop = float(sig.invalidation) if sig.invalidation else (
        entry_spot * (0.99 if long_underlying else 1.01))
    target = float(sig.targets[0]) if sig.targets else (
        entry_spot * (1.01 if long_underlying else 0.99))

    walk_bars = bars.iloc[1:] if len(bars) > 1 else bars
    idx, exit_spot, reason, mfe, mae = _walk(
        walk_bars, entry_spot, stop, target, long=long_underlying)

    if idx >= 0:
        exit_date, held = walk_bars.index[idx].date(), idx + 1
    else:
        exit_date, held = walk_bars.index[-1].date(), len(walk_bars)

    # Bars are already truncated at expiry, so reaching the last one without
    # touching a level means the contract expired rather than timed out.
    notes: list[str] = []
    if exit_date >= leg.expiry and reason == "horizon":
        reason = "expiry"
        notes.append("Position ran to expiry; settled at intrinsic value.")

    underlying_move = (exit_spot / entry_spot - 1.0) * 100.0
    signed_underlying = underlying_move if long_underlying else -underlying_move

    iv = (leg.iv or 15.0) / 100.0
    t_entry = max((leg.expiry - entry_date).days, 0) / 365.0
    t_exit = max((leg.expiry - exit_date).days, 0) / 365.0

    entry_prem = bs_price(entry_spot, leg.strike, t_entry, iv, 0.065, leg.option_type)
    exit_prem = bs_price(exit_spot, leg.strike, t_exit, iv, 0.065, leg.option_type)

    if entry_prem <= 0.01:
        return _no_data(sig, "entry premium modelled at ~0; contract not priceable",
                        as_of=decided)

    gross = (exit_prem / entry_prem - 1.0) * 100.0
    net = gross - costs.for_kind(sig.kind) / 100.0
    if hold_iv_constant:
        notes.append("Option P&L modelled with constant IV - a real IV crush would "
                     "make this worse, never better.")

    return Outcome(
        signal_id=sig.signal_id, symbol=sig.symbol, kind=sig.kind,
        direction=sig.direction.value, conviction=sig.conviction,
        signal_date=decided, entry_date=entry_date,
        entry_price=round(entry_prem, 2), exit_date=exit_date,
        exit_price=round(exit_prem, 2), exit_reason=reason, bars_held=held,
        gross_return_pct=round(gross, 4), net_return_pct=round(net, 4),
        underlying_return_pct=round(signed_underlying, 4),
        mfe_pct=round(mfe, 3), mae_pct=round(mae, 3),
        directionally_right=signed_underlying > 0, modelled=True, notes=notes,
    )


def _no_data(sig: Signal, note: str = "no forward bars available",
             as_of: date | None = None) -> Outcome:
    return Outcome(
        signal_id=sig.signal_id, symbol=sig.symbol, kind=sig.kind,
        direction=sig.direction.value, conviction=sig.conviction,
        signal_date=as_of or sig.generated_at.date(), entry_date=None, entry_price=None,
        exit_date=None, exit_price=None, exit_reason="no_data", bars_held=0,
        gross_return_pct=0.0, net_return_pct=0.0, underlying_return_pct=0.0,
        mfe_pct=0.0, mae_pct=0.0, directionally_right=None, notes=[note],
    )


def evaluate(sig: Signal, store, horizon_bars: int | None = None,
             costs: CostModel | None = None, as_of: date | None = None) -> Outcome:
    """Dispatch on signal kind. Pass ``as_of`` whenever replaying history."""
    if sig.kind == "index_option":
        return evaluate_option(sig, store, horizon_bars or 10, costs, as_of=as_of)
    return evaluate_equity(sig, store, horizon_bars or 20, costs, as_of=as_of)
