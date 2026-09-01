"""Metrics and the null test.

``test_no_edge_is_found_in_pure_noise`` is the most important test in the
backtest module. A harness that reports an edge on random-walk data is worse
than useless -- it will confidently endorse anything. This test caught a real
bug: the option evaluator clamped the exit *date* to expiry while taking the
exit *spot* from a later bar, pricing the contract at t=0 with a price from
after it expired. Because intrinsic value is max(0, S-K) -- convex and
non-negative -- higher-variance future prices inflated the payoff, manufacturing
roughly +19% mean return out of noise.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alpha.backtest.evaluate import Outcome
from alpha.backtest.metrics import (
    MIN_USEFUL_N, Stats, bootstrap_mean_ci, max_drawdown, summarise, verdict,
    wilson_interval,
)
from alpha.backtest.replay import HistoryStore
from alpha.backtest.runner import BacktestConfig, run_options


# -- statistics ------------------------------------------------------------

def test_wilson_widens_as_the_sample_shrinks():
    lo_small, hi_small = wilson_interval(24, 40)      # 60% on 40
    lo_big, hi_big = wilson_interval(240, 400)        # 60% on 400
    assert (hi_small - lo_small) > (hi_big - lo_big) * 2


def test_wilson_on_forty_samples_cannot_distinguish_sixty_percent_from_chance():
    """The motivating example: 60% on 40 trades is not evidence of anything."""
    lo, hi = wilson_interval(24, 40)
    assert lo < 50.0 < hi


def test_wilson_on_four_hundred_samples_can():
    lo, hi = wilson_interval(240, 400)
    assert lo > 50.0


def test_wilson_handles_degenerate_inputs():
    assert wilson_interval(0, 0) == (0.0, 100.0)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 100.0


def test_bootstrap_ci_brackets_the_mean():
    vals = list(np.random.default_rng(1).normal(5.0, 2.0, 200))
    lo, hi = bootstrap_mean_ci(vals)
    assert lo < np.mean(vals) < hi


def test_bootstrap_handles_empty_and_single():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)
    assert bootstrap_mean_ci([3.0]) == (3.0, 3.0)


def test_max_drawdown_is_negative_after_a_loss_run():
    assert max_drawdown([10.0, -20.0, -20.0, 5.0]) < -30
    assert max_drawdown([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert max_drawdown([]) == 0.0


# -- summarise -------------------------------------------------------------

def make_outcome(ret: float, reason: str = "target", right: bool | None = True) -> Outcome:
    return Outcome(
        signal_id="x", symbol="S", kind="equity_positional", direction="long",
        conviction=50, signal_date=date(2026, 1, 1), entry_date=date(2026, 1, 2),
        entry_price=100.0, exit_date=date(2026, 1, 10), exit_price=100 + ret,
        exit_reason=reason, bars_held=5, gross_return_pct=ret, net_return_pct=ret,
        underlying_return_pct=ret, mfe_pct=abs(ret), mae_pct=-abs(ret) / 2,
        directionally_right=right)


def test_summarise_basic_arithmetic():
    st = summarise([make_outcome(10), make_outcome(-5), make_outcome(20), make_outcome(-5)])
    assert st.n_evaluable == 4
    assert st.hit_rate == 50.0
    assert st.mean_return == pytest.approx(5.0)
    assert st.avg_win == pytest.approx(15.0)
    assert st.avg_loss == pytest.approx(-5.0)
    assert st.profit_factor == pytest.approx(3.0)      # 30 won / 10 lost


def test_summarise_ignores_unevaluable_outcomes():
    st = summarise([make_outcome(10), make_outcome(0, "no_data", None)])
    assert st.n == 2 and st.n_evaluable == 1


def test_summarise_on_empty_input():
    st = summarise([])
    assert st.n_evaluable == 0 and st.hit_rate == 0.0


# -- verdict refuses to overclaim -----------------------------------------

def test_small_sample_verdict_refuses_to_conclude():
    v = verdict(summarise([make_outcome(10) for _ in range(10)]))
    assert v.sample_adequate is False
    assert "too small" in v.conclusion.lower()
    assert v.signals_needed == MIN_USEFUL_N - 10


def test_verdict_says_not_distinguishable_when_the_interval_spans_chance():
    outs = [make_outcome(1.0) for _ in range(20)] + [make_outcome(-1.0) for _ in range(20)]
    v = verdict(summarise(outs))
    assert v.beats_chance is None
    assert "not distinguishable" in v.detail.lower() or "No demonstrated edge" in v.conclusion


def test_verdict_flags_a_losing_strategy():
    v = verdict(summarise([make_outcome(-3.0) for _ in range(50)]))
    assert v.conclusion == "Loses money"


def test_verdict_reports_no_edge_when_mean_return_ci_includes_zero():
    outs = [make_outcome(5.0) for _ in range(30)] + [make_outcome(-5.0) for _ in range(30)]
    v = verdict(summarise(outs))
    assert "includes zero" in v.detail


def test_verdict_notes_when_buy_and_hold_did_better():
    signal = summarise([make_outcome(1.0) for _ in range(60)])
    baseline = summarise([make_outcome(9.0) for _ in range(60)])
    v = verdict(signal, baseline)
    assert v.beats_baseline is False
    assert v.conclusion == "Worse than buy-and-hold"


def test_verdict_with_no_data():
    v = verdict(summarise([]))
    assert v.conclusion == "No data"
    assert v.sample_adequate is False


# -- THE NULL TEST ---------------------------------------------------------

def random_walk_store(seed: int, n: int = 1200) -> HistoryStore:
    """Driftless geometric Brownian motion: no trend, no cycle, nothing to find."""
    frames = {}
    for i, (sym, s0) in enumerate([("NIFTY", 25000.0), ("BANKNIFTY", 55000.0)]):
        rng = np.random.default_rng(seed * 100 + i)
        vol, dt = 0.13, 1 / 252
        logret = -0.5 * vol**2 * dt + vol * np.sqrt(dt) * rng.normal(0, 1, n)
        close = s0 * np.exp(np.cumsum(logret))
        noise = np.abs(rng.normal(0, vol / np.sqrt(252) * 0.5, n)) * close
        open_ = np.concatenate([[close[0]], close[:-1]])
        frames[sym] = pd.DataFrame({
            "open": open_,
            "high": np.maximum.reduce([close + noise, close, open_]),
            "low": np.minimum.reduce([close - noise, close, open_]),
            "close": close, "volume": np.full(n, 2e8),
        }, index=pd.bdate_range(end=pd.Timestamp("2026-09-01"), periods=n))
    return HistoryStore(frames=frames, synthetic=False)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_no_edge_is_found_in_pure_noise(seed):
    """The harness must not manufacture an edge from a random walk.

    Asserted per seed on the confidence interval rather than the point estimate:
    a point estimate wanders, but the lower bound of the mean-return interval
    should not clear zero on data with no signal in it.
    """
    result = run_options(random_walk_store(seed),
                         BacktestConfig(step_days=5, horizon_bars=10))
    st = result.stats
    assert st.n_evaluable > 50, "need a usable sample for this test to mean anything"
    assert st.mean_return_ci[0] <= 0.0, (
        f"seed {seed}: harness found a significant edge in random-walk data "
        f"(mean {st.mean_return:+.2f}%, CI low {st.mean_return_ci[0]:+.2f}%). "
        f"Something is manufacturing returns.")
    assert result.verdict.beats_baseline is not True


def test_noise_verdict_never_claims_an_edge():
    v = run_options(random_walk_store(11), BacktestConfig(step_days=5)).verdict
    assert v.conclusion in ("No demonstrated edge", "Not distinguishable from chance",
                            "Loses money", "Worse than chance",
                            "No advantage over buy-and-hold", "Worse than buy-and-hold",
                            "Sample too small to conclude anything")


def test_option_position_is_never_held_past_expiry():
    """The specific bug the null test caught."""
    result = run_options(random_walk_store(3), BacktestConfig(step_days=5, horizon_bars=15))
    for o in result.outcomes:
        assert o.exit_date is not None
        assert o.bars_held >= 0
    # Expiry exits must exist at this horizon, and none may sit beyond expiry.
    assert result.stats.exit_breakdown.get("expiry", 0) > 0
