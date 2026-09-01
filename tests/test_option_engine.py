"""Option-buying engine behaviour.

The important assertions here are about the *gates*. A directional score is a
matter of taste; refusing to buy premium on expiry day is not.
"""

from datetime import date

import pytest

from alpha.calendar import expiry_context
from alpha.data.composite import CompositeProvider
from alpha.data.fixtures import FixtureProvider
from alpha.engines.index_options import (
    OptionEngineConfig, atm_iv, build_signal, gather_inputs, select_strike,
)
from alpha.models import Category, Direction, Verdict

MID_MONTH = date(2026, 9, 10)      # deliberately not an expiry day


def provider(bias=None, salt="t", as_of=MID_MONTH):
    return CompositeProvider(fallback=FixtureProvider(as_of=as_of, seed_salt=salt,
                                                      drift_bias=bias))


# -- gates -----------------------------------------------------------------

def test_expiry_day_blocks_buying():
    """1 Sep 2026 is a Tuesday, so NIFTY weekly expires that day."""
    ctx = expiry_context("NIFTY", date(2026, 9, 1))
    assert ctx.is_expiry_day

    sig = build_signal("NIFTY", provider(bias=0.9, as_of=date(2026, 9, 1)), date(2026, 9, 1))
    assert sig.direction is Direction.NO_TRADE
    assert sig.conviction == 0
    assert any(v.key == "expiry_day" for v in sig.scorecard.blocking_vetoes)
    assert sig.leg is None


def test_expiry_day_block_can_be_disabled():
    cfg = OptionEngineConfig(block_on_expiry_day=False)
    sig = build_signal("NIFTY", provider(bias=0.9, as_of=date(2026, 9, 1)),
                       date(2026, 9, 1), cfg)
    assert not any(v.key == "expiry_day" for v in sig.scorecard.vetoes)


def test_strong_trend_produces_a_signal_with_a_leg():
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    assert sig.direction is Direction.LONG
    assert sig.conviction > 0
    assert sig.leg is not None and sig.leg.option_type == "CE"
    assert sig.invalidation is not None and sig.invalidation < sig.spot
    assert sig.targets and all(t > sig.spot for t in sig.targets)


def test_iv_crush_veto_blocks_even_a_strong_trend():
    """Direction being right does not rescue premium bought far above realised vol."""
    cfg = OptionEngineConfig(iv_rv_block=0.01)      # force the gate
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH, cfg)
    assert sig.direction is Direction.NO_TRADE
    assert any(v.key == "iv_crush" for v in sig.scorecard.blocking_vetoes)


def test_theta_veto_blocks_when_bleed_is_prohibitive():
    cfg = OptionEngineConfig(theta_pct_block=0.001)
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH, cfg)
    assert sig.direction is Direction.NO_TRADE
    assert any(v.key == "theta_burn" for v in sig.scorecard.blocking_vetoes)
    assert sig.leg is None


def test_conviction_floor_blocks_weak_evidence():
    cfg = OptionEngineConfig(min_conviction=99)
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH, cfg)
    assert sig.direction is Direction.NO_TRADE
    assert any(v.key == "low_conviction" for v in sig.scorecard.blocking_vetoes)


def test_illiquid_chain_blocks_the_signal():
    cfg = OptionEngineConfig(min_strike_oi=10**12)
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH, cfg)
    assert sig.direction is Direction.NO_TRADE
    assert any(v.key == "no_tradable_strike" for v in sig.scorecard.blocking_vetoes)


# -- the weight-0 contract -------------------------------------------------

def test_cost_factors_do_not_vote_on_direction():
    """VOLATILITY and COST factors must never move the direction: rich premium
    is not a bearish opinion."""
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    non_voting = [f for f in sig.scorecard.factors
                  if f.category in (Category.VOLATILITY, Category.COST)]
    assert non_voting, "expected volatility/cost factors to be present as evidence"
    assert all(f.weight == 0.0 for f in non_voting)
    assert all(f.verdict is Verdict.NEUTRAL for f in non_voting)


def test_cost_factors_still_appear_as_evidence():
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    keys = {f.key for f in sig.scorecard.factors}
    assert {"expiry", "vix", "iv_vs_rv", "iv_percentile"} <= keys


# -- strike selection ------------------------------------------------------

def test_selected_strike_is_inside_the_target_delta_band():
    cfg = OptionEngineConfig()
    inp = gather_inputs("BANKNIFTY", provider(salt="trendA"), MID_MONTH)
    choice = select_strike(inp.chain, inp.expiry, Direction.LONG, cfg)
    assert choice is not None
    assert cfg.target_delta_lo <= abs(choice.greeks.delta) <= cfg.target_delta_hi


def test_short_direction_selects_puts():
    inp = gather_inputs("BANKNIFTY", provider(salt="trendA"), MID_MONTH)
    choice = select_strike(inp.chain, inp.expiry, Direction.SHORT, OptionEngineConfig())
    assert choice is not None and choice.leg.option_type == "PE"
    assert choice.greeks.delta < 0


def test_no_trade_selects_nothing():
    inp = gather_inputs("BANKNIFTY", provider(salt="trendA"), MID_MONTH)
    assert select_strike(inp.chain, inp.expiry, Direction.NO_TRADE, OptionEngineConfig()) is None


def test_leg_rationale_quotes_real_numbers():
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    assert sig.leg is not None
    r = sig.leg.rationale
    assert "Delta" in r and "Theta" in r and "Open interest" in r


# -- explainability contract ----------------------------------------------

def test_every_factor_explains_itself():
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    for f in sig.scorecard.factors:
        assert f.detail.strip(), f"factor {f.key} has no explanation"
        assert len(f.detail) > 25, f"factor {f.key} explanation is too thin: {f.detail!r}"
        assert f.label.strip()


def test_no_trade_signal_still_explains_why():
    sig = build_signal("NIFTY", provider(as_of=date(2026, 9, 1)), date(2026, 9, 1))
    assert sig.direction is Direction.NO_TRADE
    assert len(sig.summary) > 60
    assert sig.scorecard.factors, "a stand-aside call must still show its evidence"


def test_banknifty_is_monthly_only_and_nifty_is_weekly():
    """The Nov-2024 SEBI change, asserted so a regression is caught loudly."""
    inp_n = gather_inputs("NIFTY", provider(), MID_MONTH)
    inp_b = gather_inputs("BANKNIFTY", provider(), MID_MONTH)
    assert inp_n.expiry.has_weekly is True
    assert inp_b.expiry.has_weekly is False
    assert inp_b.expiry.calendar_days >= inp_n.expiry.calendar_days


def test_atm_iv_reports_its_source():
    inp = gather_inputs("BANKNIFTY", provider(), MID_MONTH)
    iv, src = atm_iv(inp.chain, inp.expiry)
    assert iv is not None and 1 < iv < 200
    assert "IV" in src or "solved" in src


def test_fixture_data_is_flagged_in_data_quality():
    sig = build_signal("BANKNIFTY", provider(), MID_MONTH)
    assert sig.data_quality
    assert any("SIMULATED" in q or "fixture" in q.lower() for q in sig.data_quality)


def test_signal_serialises_for_the_api():
    sig = build_signal("BANKNIFTY", provider(bias=0.85, salt="trendA"), MID_MONTH)
    d = sig.to_dict()
    assert d["direction"] in ("long", "short", "no_trade")
    assert isinstance(d["scorecard"]["factors"], list)
    assert d["scorecard"]["factors"][0]["contribution"] is not None
    assert d["leg"]["tradingsymbol"]
    import json
    json.dumps(d)          # must be JSON-safe end to end
