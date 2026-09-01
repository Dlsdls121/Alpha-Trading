"""Indicator correctness tests.

These check properties and hand-computed values rather than snapshots, so they
stay meaningful if the implementation is rewritten.
"""

import math

import numpy as np
import pandas as pd
import pytest

from alpha.indicators import (
    ema, sma, adx, rsi, macd, roc, atr, true_range, realized_vol,
    vwap, relative_volume, bs_price, bs_greeks, implied_vol,
    put_call_ratio, max_pain, classify_oi_buildup, oi_support_resistance,
    percentile_rank,
)
from alpha.indicators.trend import ema_stack


@pytest.fixture
def trending_up():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 200, n) + np.sin(np.arange(n) / 5) * 2, index=idx)
    high = close * 1.01
    low = close * 0.99
    vol = pd.Series(np.full(n, 1_000_000.0), index=idx)
    return high, low, close, vol


# -- moving averages -------------------------------------------------------

def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert sma(s, 3).iloc[-1] == pytest.approx(4.0)       # (3+4+5)/3
    assert pd.isna(sma(s, 3).iloc[1])                      # warm-up is NaN


def test_ema_of_constant_is_the_constant():
    s = pd.Series([7.0] * 50)
    assert ema(s, 10).iloc[-1] == pytest.approx(7.0)


def test_ema_reacts_faster_than_sma():
    """On a cleanly rising series the EMA tracks recent price more closely and
    so sits above the equal-weighted SMA. (Superimpose an oscillation and this
    stops holding at an arbitrary bar, which is why the series here is monotonic.)"""
    close = pd.Series(np.linspace(100, 200, 300))
    assert ema(close, 50).iloc[-1] > sma(close, 50).iloc[-1]

    falling = pd.Series(np.linspace(200, 100, 300))
    assert ema(falling, 50).iloc[-1] < sma(falling, 50).iloc[-1]


# -- RSI -------------------------------------------------------------------

def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1.0, 60.0))
    assert rsi(s, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_zero():
    s = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(s, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_stays_in_bounds(trending_up):
    _, _, close, _ = trending_up
    r = rsi(close, 14).dropna()
    assert r.between(0, 100).all()


def test_rsi_uptrend_above_50(trending_up):
    _, _, close, _ = trending_up
    assert rsi(close, 14).iloc[-1] > 50


# -- ATR / true range ------------------------------------------------------

def test_true_range_uses_gap():
    high = pd.Series([10.0, 20.0])
    low = pd.Series([9.0, 19.0])
    close = pd.Series([9.5, 19.5])
    # bar 2: high-low=1, |high-prevclose|=10.5 -> gap dominates
    assert true_range(high, low, close).iloc[1] == pytest.approx(10.5)


def test_atr_positive_and_scales_with_range(trending_up):
    high, low, close, _ = trending_up
    a = atr(high, low, close, 14).dropna()
    assert (a > 0).all()
    wide = atr(high * 1.05, low * 0.95, close, 14).iloc[-1]
    assert wide > a.iloc[-1]


def test_realized_vol_of_flat_series_is_zero():
    s = pd.Series([100.0] * 60)
    assert realized_vol(s, 20).iloc[-1] == pytest.approx(0.0)


# -- ADX -------------------------------------------------------------------

def test_adx_high_in_strong_trend(trending_up):
    high, low, close, _ = trending_up
    out = adx(high, low, close, 14)
    assert out["adx"].iloc[-1] > 20
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]


def test_adx_low_in_chop():
    n = 300
    close = pd.Series([100.0 + (1.0 if i % 2 else -1.0) for i in range(n)])
    out = adx(close * 1.005, close * 0.995, close, 14)
    assert out["adx"].iloc[-1] < 30


# -- MACD / ROC ------------------------------------------------------------

def test_macd_positive_in_uptrend(trending_up):
    _, _, close, _ = trending_up
    assert macd(close)["macd"].iloc[-1] > 0


def test_roc_matches_manual():
    s = pd.Series([100.0, 110.0, 121.0])
    assert roc(s, 2).iloc[-1] == pytest.approx(21.0)


# -- volume ----------------------------------------------------------------

def test_vwap_between_high_and_low(trending_up):
    high, low, close, vol = trending_up
    v = vwap(high, low, close, vol).dropna()
    assert (v > low.min()).all() and (v < high.max()).all()


def test_relative_volume_is_one_for_flat_volume():
    v = pd.Series([1000.0] * 40)
    assert relative_volume(v, 20).iloc[-1] == pytest.approx(1.0)


# -- Black-Scholes ---------------------------------------------------------

def test_put_call_parity():
    S, K, t, vol, r = 25000, 25000, 30 / 365, 0.15, 0.065
    c = bs_price(S, K, t, vol, r, "CE")
    p = bs_price(S, K, t, vol, r, "PE")
    assert c - p == pytest.approx(S - K * math.exp(-r * t), abs=1e-6)


def test_price_is_monotonic_in_vol():
    prices = [bs_price(25000, 25000, 0.08, v, 0.065, "CE") for v in (0.10, 0.15, 0.25)]
    assert prices[0] < prices[1] < prices[2]


def test_expiry_price_is_intrinsic():
    assert bs_price(25100, 25000, 0.0, 0.2, 0.065, "CE") == pytest.approx(100.0)
    assert bs_price(25100, 25000, 0.0, 0.2, 0.065, "PE") == pytest.approx(0.0)


def test_atm_delta_near_half():
    g = bs_greeks(25000, 25000, 7 / 365, 0.14, option_type="CE")
    assert 0.45 < g.delta < 0.60


def test_deep_itm_and_otm_deltas():
    assert bs_greeks(25000, 20000, 7 / 365, 0.14, option_type="CE").delta > 0.97
    assert bs_greeks(25000, 30000, 7 / 365, 0.14, option_type="CE").delta < 0.03


def test_put_delta_is_negative():
    assert bs_greeks(25000, 25000, 7 / 365, 0.14, option_type="PE").delta < 0


def test_theta_is_negative_for_long_options():
    for ot in ("CE", "PE"):
        assert bs_greeks(25000, 25000, 7 / 365, 0.14, option_type=ot).theta < 0


def test_theta_pct_accelerates_into_expiry():
    """The core reason DTE gates exist: percentage decay explodes near expiry."""
    far = abs(bs_greeks(25000, 25000, 30 / 365, 0.14, option_type="CE").theta_pct)
    near = abs(bs_greeks(25000, 25000, 1 / 365, 0.14, option_type="CE").theta_pct)
    assert near > far * 4


def test_implied_vol_round_trip():
    for vol in (0.08, 0.15, 0.35):
        px = bs_price(25000, 25200, 10 / 365, vol, 0.065, "CE")
        assert implied_vol(px, 25000, 25200, 10 / 365, 0.065, "CE") == pytest.approx(vol, abs=1e-4)


def test_implied_vol_rejects_impossible_quotes():
    assert implied_vol(0.05, 25000, 24000, 1 / 365, 0.065, "CE") is None   # below intrinsic
    assert implied_vol(-5, 25000, 25000, 0.1) is None
    assert implied_vol(100, 25000, 25000, 0.0) is None                     # no time left


# -- chain analytics -------------------------------------------------------

@pytest.fixture
def chain():
    return [
        {"strike": 24800, "ce_oi": 10_000, "pe_oi": 90_000, "ce_volume": 500, "pe_volume": 4000},
        {"strike": 24900, "ce_oi": 20_000, "pe_oi": 70_000, "ce_volume": 900, "pe_volume": 3000},
        {"strike": 25000, "ce_oi": 50_000, "pe_oi": 50_000, "ce_volume": 5000, "pe_volume": 5000},
        {"strike": 25100, "ce_oi": 70_000, "pe_oi": 20_000, "ce_volume": 3000, "pe_volume": 900},
        {"strike": 25200, "ce_oi": 95_000, "pe_oi": 8_000, "ce_volume": 4000, "pe_volume": 400},
    ]


def test_pcr_matches_manual(chain):
    expected = (90_000 + 70_000 + 50_000 + 20_000 + 8_000) / (10_000 + 20_000 + 50_000 + 70_000 + 95_000)
    assert put_call_ratio(chain, "oi") == pytest.approx(expected)


def test_pcr_none_when_no_calls():
    assert put_call_ratio([{"strike": 1, "ce_oi": 0, "pe_oi": 10}]) is None


def test_max_pain_hand_computed():
    """Two strikes only, so the writer payout is checkable by hand.

    Settle at 100: calls at 100 pay 0, puts at 110 pay 10*OI -> 10*100 = 1000.
    Settle at 110: calls at 100 pay 10*OI -> 10*500 = 5000, puts pay 0.
    Minimum pain is therefore 100.
    """
    rows = [
        {"strike": 100, "ce_oi": 500, "pe_oi": 0},
        {"strike": 110, "ce_oi": 0, "pe_oi": 100},
    ]
    assert max_pain(rows) == 100


def test_max_pain_sits_inside_strike_range(chain):
    mp = max_pain(chain)
    assert 24800 <= mp <= 25200


def test_max_pain_empty_chain():
    assert max_pain([]) is None


@pytest.mark.parametrize("dp,do,label", [
    (1.5, 12.0, "long_buildup"),
    (-1.5, 12.0, "short_buildup"),
    (1.5, -12.0, "short_covering"),
    (-1.5, -12.0, "long_unwinding"),
    (0.0, 0.0, "indecisive"),
])
def test_oi_buildup_quadrants(dp, do, label):
    assert classify_oi_buildup(dp, do)["label"] == label


def test_fresh_buildup_scores_stronger_than_covering():
    fresh = classify_oi_buildup(1.5, 12.0)["score"]
    covering = classify_oi_buildup(1.5, -12.0)["score"]
    assert fresh > covering > 0


def test_oi_levels_only_use_correct_side_of_spot(chain):
    lv = oi_support_resistance(chain, spot=25000)
    assert all(r["strike"] > 25000 for r in lv["resistance"])
    assert all(s["strike"] < 25000 for s in lv["support"])
    assert lv["immediate_resistance"] == 25200      # biggest call OI above spot
    assert lv["immediate_support"] == 24800         # biggest put OI below spot


def test_percentile_rank():
    hist = list(range(100))
    assert percentile_rank(50, hist) == pytest.approx(50.5)
    assert percentile_rank(-1, hist) == 0.0
    assert percentile_rank(1000, hist) == 100.0
    assert percentile_rank(5, []) is None


def test_ema_stack_detects_uptrend(trending_up):
    _, _, close, _ = trending_up
    st = ema_stack(close)
    assert st["stage"] == "uptrend"
    assert st["aligned"] == pytest.approx(1.0)


def test_ema_stack_unknown_when_too_short():
    assert ema_stack(pd.Series([1.0, 2.0, 3.0]))["stage"] == "unknown"
