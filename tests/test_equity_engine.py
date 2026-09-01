"""Positional equity engine behaviour."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alpha.data.composite import CompositeProvider
from alpha.data.fixtures import FixtureProvider
from alpha.engines.equity_positional import (
    EquityEngineConfig, build_contexts, relative_strength, rs_line_slope, scan,
    score_symbol, sector_table,
)
from alpha.models import Category, Direction, Verdict
from alpha.universe import Universe

AS_OF = date(2026, 9, 1)


@pytest.fixture
def provider():
    return CompositeProvider(fallback=FixtureProvider(as_of=AS_OF))


@pytest.fixture
def universe():
    return Universe.load()


# -- relative strength primitives -----------------------------------------

def test_relative_strength_sign_and_magnitude():
    stock = pd.Series(np.linspace(100, 120, 200))      # +20%
    bench = pd.Series(np.linspace(100, 110, 200))      # +10%
    rs = relative_strength(stock, bench, 199)
    assert rs == pytest.approx(10.0, abs=0.2)

    assert relative_strength(bench, stock, 199) < 0


def test_relative_strength_needs_enough_history():
    s = pd.Series([1.0, 2.0, 3.0])
    assert relative_strength(s, s, 100) is None


def test_rs_slope_positive_when_outperformance_is_building():
    stock = pd.Series(np.linspace(100, 150, 120))
    bench = pd.Series(np.linspace(100, 105, 120))
    assert rs_line_slope(stock, bench, 21) > 0


def test_rs_slope_negative_when_outperformance_fades():
    """A stock that led and then stalled: the level stays good, the slope turns."""
    lead = np.linspace(100, 150, 100)
    stall = np.linspace(150, 145, 40)
    stock = pd.Series(np.concatenate([lead, stall]))
    bench = pd.Series(np.linspace(100, 120, 140))
    assert rs_line_slope(stock, bench, 21) < 0


# -- scanning --------------------------------------------------------------

def test_scan_returns_ranked_longs(provider):
    sigs = scan(provider, AS_OF)
    assert sigs
    assert all(s.direction is Direction.LONG for s in sigs)
    assert all(s.kind == "equity_positional" for s in sigs)
    convictions = [s.conviction for s in sigs]
    assert convictions == sorted(convictions, reverse=True)


def test_scan_respects_top_n(provider):
    sigs = scan(provider, AS_OF, EquityEngineConfig(top_n=3))
    assert len(sigs) <= 3


def test_include_rejected_returns_the_whole_universe_with_reasons(provider, universe):
    sigs = scan(provider, AS_OF, include_rejected=True)
    assert len(sigs) > len(scan(provider, AS_OF))
    for s in sigs:
        assert s.scorecard.factors
        assert s.summary.strip()


def test_engine_never_proposes_a_short(provider):
    """Long-only by design: positional cash shorting is not available to most
    retail participants, so a short call would be unactionable advice."""
    for s in scan(provider, AS_OF, include_rejected=True):
        assert s.direction in (Direction.LONG, Direction.NO_TRADE)


def test_every_long_has_a_stop_below_entry_and_targets_above(provider):
    for s in scan(provider, AS_OF):
        assert s.invalidation < s.spot
        assert s.targets and all(t > s.spot for t in s.targets)
        assert s.targets == sorted(s.targets)


def test_reward_to_risk_clears_the_configured_floor(provider):
    cfg = EquityEngineConfig()
    for s in scan(provider, AS_OF, cfg):
        rr = (s.targets[0] - s.spot) / (s.spot - s.invalidation)
        assert rr >= cfg.min_reward_risk - 1e-6


def test_poor_reward_risk_is_vetoed(provider):
    """Make the first target smaller than the stop and nothing should qualify."""
    cfg = EquityEngineConfig(target_atr_multiples=(1.0, 2.0), min_reward_risk=1.3)
    assert scan(provider, AS_OF, cfg) == []


def test_liquidity_floor_blocks_everything_when_absurd(provider):
    assert scan(provider, AS_OF, EquityEngineConfig(min_turnover_cr=10**9)) == []


def test_conviction_floor_is_enforced(provider):
    assert scan(provider, AS_OF, EquityEngineConfig(min_conviction=100)) == []


# -- the double-counting fix ----------------------------------------------

def test_single_constituent_sector_is_not_counted_as_evidence(provider, universe):
    """A one-stock sector's 'sector strength' is that stock's own relative
    strength restated. Counting both would double-count the same evidence."""
    contexts = {c.symbol: c for c in build_contexts(provider, universe, AS_OF)}
    singles = [c for c in contexts.values() if (c.sector_n or 0) < 2]
    assert singles, "fixture universe should contain at least one single-name sector"

    cfg = EquityEngineConfig()
    for ctx in singles:
        sc, _, _ = score_symbol(ctx, cfg)
        sector_f = next(f for f in sc.factors if f.key == "sector")
        assert sector_f.weight == 0.0
        assert "only one representative" in sector_f.detail


def test_multi_constituent_sector_is_counted(provider, universe):
    contexts = {c.symbol: c for c in build_contexts(provider, universe, AS_OF)}
    multi = next(c for c in contexts.values() if (c.sector_n or 0) >= 3)
    sc, _, _ = score_symbol(multi, EquityEngineConfig())
    assert next(f for f in sc.factors if f.key == "sector").weight > 0


# -- gates are non-directional --------------------------------------------

def test_liquidity_and_risk_factors_do_not_vote(provider, universe):
    ctx = build_contexts(provider, universe, AS_OF)[0]
    sc, _, _ = score_symbol(ctx, EquityEngineConfig())
    for key in ("liquidity", "risk"):
        f = next(x for x in sc.factors if x.key == key)
        assert f.weight == 0.0
        assert f.verdict is Verdict.NEUTRAL


# -- sector board ----------------------------------------------------------

def test_sector_table_is_ranked_and_complete(provider):
    rows = sector_table(provider, AS_OF)
    assert rows
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows == sorted(rows, key=lambda r: r["mean_rs_3m"], reverse=True)
    assert all(r["leaders"] for r in rows)


# -- explainability --------------------------------------------------------

def test_every_factor_explains_itself(provider):
    for s in scan(provider, AS_OF):
        for f in s.scorecard.factors:
            assert len(f.detail) > 25, f"{s.symbol}/{f.key}: {f.detail!r}"


def test_signals_carry_invalidation_conditions(provider):
    for s in scan(provider, AS_OF):
        assert len(s.invalidated_by) >= 2


def test_signal_serialises(provider):
    import json
    for s in scan(provider, AS_OF):
        json.dumps(s.to_dict())


def test_fixture_mode_is_flagged(provider):
    for s in scan(provider, AS_OF):
        assert any("SIMULATED" in q for q in s.data_quality)
