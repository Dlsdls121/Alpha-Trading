"""Performance statistics, with the error bars attached.

The failure mode this module exists to prevent: running 40 signals, seeing a 60%
hit rate, and concluding the strategy works. With 40 samples a 60% hit rate has a
95% confidence interval of roughly 44%-74%. It is not distinguishable from a coin
flip, and reporting the point estimate alone actively misleads.

So every headline number here carries an interval, and
:func:`verdict` refuses to call a result meaningful when the interval spans the
baseline. A backtest whose honest answer is "this tells you nothing yet" should
say exactly that.

Two comparisons matter and both are computed:

* **against chance** -- is the hit rate distinguishable from 50%?
* **against buy-and-hold** -- did selecting these names on these dates beat simply
  holding them over the same horizon? A signal that is right 65% of the time in a
  market that rose 65% of the time has added nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np

from alpha.backtest.evaluate import Outcome

Z95 = 1.959963985


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion, as percentages.

    Preferred over the normal approximation because it behaves correctly for
    small n and for proportions near 0 or 1 -- exactly the regime a backtest with
    a few dozen signals lives in.
    """
    if n == 0:
        return (0.0, 100.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half)) * 100, min(1.0, (centre + half)) * 100)


def bootstrap_mean_ci(values: Sequence[float], iterations: int = 5000,
                      seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Trade returns are skewed and fat-tailed -- a few large winners dominate -- so
    a t-interval understates uncertainty. Resampling makes no distributional
    assumption.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(iterations, arr.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def max_drawdown(returns_pct: Sequence[float]) -> float:
    """Worst peak-to-trough decline of a compounded equity curve, in percent.

    Sequential, not portfolio -- it assumes one position at a time, which
    overstates capacity but keeps the number interpretable.
    """
    if not len(returns_pct):
        return 0.0
    curve = np.cumprod(1.0 + np.asarray(returns_pct, dtype=float) / 100.0)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min() * 100.0)


@dataclass
class Stats:
    n: int
    n_evaluable: int
    hit_rate: float
    hit_rate_ci: tuple[float, float]
    mean_return: float
    mean_return_ci: tuple[float, float]
    median_return: float
    total_return: float
    best: float
    worst: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    avg_bars_held: float
    directional_accuracy: float | None
    directional_ci: tuple[float, float] | None
    exit_breakdown: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def summarise(outcomes: Sequence[Outcome]) -> Stats:
    usable = [o for o in outcomes if o.exit_reason != "no_data"]
    n_all, n = len(outcomes), len(usable)

    if n == 0:
        return Stats(n_all, 0, 0.0, (0.0, 100.0), 0.0, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, {})

    rets = [o.net_return_pct for o in usable]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    gross_win, gross_loss = sum(wins), abs(sum(losses))
    hit = len(wins) / n

    directional = [o for o in usable if o.directionally_right is not None]
    dir_right = sum(1 for o in directional if o.directionally_right)

    exits: dict[str, int] = {}
    for o in usable:
        exits[o.exit_reason] = exits.get(o.exit_reason, 0) + 1

    return Stats(
        n=n_all, n_evaluable=n,
        hit_rate=round(hit * 100, 2),
        hit_rate_ci=tuple(round(x, 2) for x in wilson_interval(len(wins), n)),
        mean_return=round(float(np.mean(rets)), 4),
        mean_return_ci=tuple(round(x, 4) for x in bootstrap_mean_ci(rets)),
        median_return=round(float(np.median(rets)), 4),
        total_return=round(float((np.cumprod(1 + np.asarray(rets) / 100)[-1] - 1) * 100), 3),
        best=round(max(rets), 3), worst=round(min(rets), 3),
        avg_win=round(float(np.mean(wins)), 3) if wins else 0.0,
        avg_loss=round(float(np.mean(losses)), 3) if losses else 0.0,
        profit_factor=round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        expectancy=round(float(np.mean(rets)), 4),
        max_drawdown=round(max_drawdown(rets), 3),
        avg_bars_held=round(float(np.mean([o.bars_held for o in usable])), 2),
        directional_accuracy=round(dir_right / len(directional) * 100, 2) if directional else None,
        directional_ci=(tuple(round(x, 2) for x in wilson_interval(dir_right, len(directional)))
                        if directional else None),
        exit_breakdown=exits,
    )


@dataclass
class Verdict:
    conclusion: str
    detail: str
    beats_chance: bool | None
    beats_baseline: bool | None
    sample_adequate: bool
    signals_needed: int | None = None


MIN_USEFUL_N = 30


def verdict(stats: Stats, baseline: Stats | None = None) -> Verdict:
    """State plainly what the numbers do and do not support.

    Written to be hard to over-read. When the confidence interval spans the
    comparison point the conclusion is "not distinguishable", never "slightly
    better".
    """
    n = stats.n_evaluable

    if n == 0:
        return Verdict("No data", "No signal produced an evaluable outcome. Nothing "
                       "can be concluded.", None, None, False)

    lo, hi = stats.hit_rate_ci
    mlo, mhi = stats.mean_return_ci

    beats_chance = None if lo <= 50.0 <= hi else lo > 50.0
    beats_baseline = None
    if baseline is not None and baseline.n_evaluable > 0:
        blo, bhi = baseline.mean_return_ci
        if mlo > baseline.mean_return:
            beats_baseline = True
        elif mhi < baseline.mean_return:
            beats_baseline = False
        else:
            beats_baseline = None

    if n < MIN_USEFUL_N:
        return Verdict(
            "Sample too small to conclude anything",
            f"Only {n} evaluable signals. The hit rate of {stats.hit_rate:.1f}% has a 95% "
            f"confidence interval of {lo:.1f}%-{hi:.1f}%, which is far too wide to "
            f"distinguish skill from luck. Treat this run as a smoke test that the "
            f"machinery works, not as evidence about the strategy. At least "
            f"{MIN_USEFUL_N} signals are needed before the numbers mean much, and "
            f"several hundred before they mean a lot.",
            beats_chance, beats_baseline, False, MIN_USEFUL_N - n)

    parts = [
        f"{n} evaluable signals. Hit rate {stats.hit_rate:.1f}% "
        f"(95% CI {lo:.1f}%-{hi:.1f}%). Mean net return per signal "
        f"{stats.mean_return:+.2f}% (95% CI {mlo:+.2f}% to {mhi:+.2f}%)."
    ]

    if beats_chance is None:
        parts.append("The hit-rate interval spans 50%, so this is NOT distinguishable "
                     "from a coin flip.")
        headline = "Not distinguishable from chance"
    elif beats_chance:
        parts.append("The hit-rate interval sits above 50%.")
        headline = "Better than chance on hit rate"
    else:
        parts.append("The hit-rate interval sits below 50% - worse than a coin flip.")
        headline = "Worse than chance"

    if mlo <= 0.0 <= mhi:
        parts.append("The mean-return interval includes zero, so no profitable edge is "
                     "demonstrated even where the hit rate looks favourable.")
        headline = "No demonstrated edge"
    elif mhi < 0:
        parts.append("The mean-return interval is entirely below zero: this lost money.")
        headline = "Loses money"

    if baseline is not None and baseline.n_evaluable > 0:
        parts.append(f"Buy-and-hold over the same names and horizons returned "
                     f"{baseline.mean_return:+.2f}% per instance.")
        if beats_baseline is None:
            parts.append("The signal is NOT distinguishable from simply holding.")
            if headline.startswith("Better"):
                headline = "No advantage over buy-and-hold"
        elif beats_baseline is False:
            parts.append("Buy-and-hold did better.")
            headline = "Worse than buy-and-hold"

    return Verdict(headline, " ".join(parts), beats_chance, beats_baseline, True)
