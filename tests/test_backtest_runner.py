"""Runner, guards and report."""

from datetime import date

import pytest

from alpha.backtest import SYNTHETIC_CHAIN_FACTORS, BacktestConfig, HistoryStore
from alpha.backtest.report import render
from alpha.backtest.runner import FIXTURE_WARNING, run_equity, run_options
from alpha.data.fixtures import FixtureProvider
from alpha.engines.index_options import OptionEngineConfig, build_signal
from alpha.backtest.replay import PointInTimeProvider
from tests.test_backtest_metrics import random_walk_store


@pytest.fixture(scope="module")
def fixture_store():
    return HistoryStore.load(FixtureProvider(as_of=date(2026, 9, 1)),
                             ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"], 700)


# -- the generated-data guard ---------------------------------------------

def test_fixture_history_is_detected_as_synthetic(fixture_store):
    assert fixture_store.synthetic is True
    assert any("GENERATED" in n for n in fixture_store.source_notes)


def test_real_looking_store_is_not_flagged():
    assert random_walk_store(1).synthetic is False


def test_generated_data_verdict_is_overridden(fixture_store):
    """A spectacular number from a sine wave must not be presented as a finding."""
    res = run_options(fixture_store, BacktestConfig(step_days=20))
    assert res.verdict.conclusion == "Not a real result - generated data"
    assert res.verdict.sample_adequate is False
    assert res.caveats[0] == FIXTURE_WARNING


def test_raw_stats_survive_the_guard(fixture_store):
    """The guard replaces the verdict, not the data -- inspection stays possible."""
    res = run_options(fixture_store, BacktestConfig(step_days=20))
    assert res.stats.n_evaluable > 0


# -- synthetic factor exclusion -------------------------------------------

def test_oi_factors_are_excluded_by_default(fixture_store):
    res = run_options(fixture_store, BacktestConfig(step_days=40))
    assert any("EXCLUDED" in c for c in res.caveats)
    assert all(r["factor"] not in SYNTHETIC_CHAIN_FACTORS
               for r in res.factor_attribution)


def test_including_oi_factors_is_loudly_flagged(fixture_store):
    res = run_options(fixture_store,
                      BacktestConfig(step_days=40, exclude_synthetic_factors=False))
    assert any("not meaningful" in c for c in res.caveats)


def test_disabled_factors_are_shown_but_not_scored(fixture_store):
    as_of = fixture_store.trading_dates("BANKNIFTY")[-120]
    p = PointInTimeProvider(fixture_store, as_of)
    sig = build_signal("BANKNIFTY", p, as_of,
                       OptionEngineConfig(disabled_factors=SYNTHETIC_CHAIN_FACTORS))
    disabled = [f for f in sig.scorecard.factors if f.key in SYNTHETIC_CHAIN_FACTORS]
    assert disabled, "the OI factors should still be present as context"
    assert all(f.weight == 0.0 for f in disabled)
    assert all("Excluded from scoring" in f.detail for f in disabled)


# -- runner mechanics ------------------------------------------------------

def test_options_runner_produces_outcomes(fixture_store):
    res = run_options(fixture_store, BacktestConfig(step_days=20))
    assert res.signals_generated > 0
    assert res.stats.n_evaluable > 0
    assert all(o.exit_reason != "no_data" for o in res.outcomes)


def test_every_outcome_has_a_signal_date_inside_the_history(fixture_store):
    res = run_options(fixture_store, BacktestConfig(step_days=20))
    dates = set(fixture_store.trading_dates("NIFTY")) | set(
        fixture_store.trading_dates("BANKNIFTY"))
    for o in res.outcomes:
        assert o.signal_date in dates
        assert o.entry_date > o.signal_date       # entry is always the NEXT bar


def test_equity_runner_produces_outcomes_and_baselines(fixture_store):
    res = run_equity(fixture_store, BacktestConfig(step_days=40, equity_top_n=2))
    assert res.stats.n_evaluable > 0
    assert res.baseline_universe is not None
    assert res.baseline_universe.n_evaluable > res.stats.n_evaluable


def test_warmup_is_respected(fixture_store):
    """No signal may be generated before indicators have enough history."""
    cfg = BacktestConfig(step_days=10, warmup_bars=400)
    res = run_options(fixture_store, cfg)
    earliest = fixture_store.trading_dates("NIFTY")[400]
    assert all(o.signal_date >= earliest for o in res.outcomes)


def test_step_days_controls_sample_count(fixture_store):
    few = run_options(fixture_store, BacktestConfig(step_days=40)).signals_generated
    many = run_options(fixture_store, BacktestConfig(step_days=10)).signals_generated
    assert many > few


def test_survivorship_caveat_is_present(fixture_store):
    res = run_equity(fixture_store, BacktestConfig(step_days=60, equity_top_n=2))
    assert any("survivorship" in c.lower() for c in res.caveats)


def test_result_serialises(fixture_store):
    import json
    json.dumps(run_options(fixture_store, BacktestConfig(step_days=60)).as_dict())


# -- report ----------------------------------------------------------------

def test_report_leads_with_the_verdict_then_caveats(fixture_store):
    text = render(run_options(fixture_store, BacktestConfig(step_days=40)))
    v_at = text.index("VERDICT")
    c_at = text.index("READ BEFORE THE NUMBERS")
    r_at = text.index("COVERAGE")
    assert v_at < c_at < r_at, "verdict and caveats must precede the numbers"


def test_report_shows_confidence_intervals(fixture_store):
    text = render(run_options(fixture_store, BacktestConfig(step_days=20)))
    assert "95% CI" in text


def test_report_warns_against_tuning_on_attribution(fixture_store):
    text = render(run_options(fixture_store, BacktestConfig(step_days=20)))
    if "FACTOR ATTRIBUTION" in text:
        assert "overfitted" in text


def test_report_handles_an_empty_result(fixture_store):
    res = run_options(fixture_store, BacktestConfig(step_days=5, warmup_bars=100_000))
    text = render(res)
    assert "No evaluable outcomes" in text or "COVERAGE" in text
