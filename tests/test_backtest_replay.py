"""Lookahead-bias tests.

Everything a backtest claims depends on the engine being unable to see the
future. These tests attack that property directly rather than trusting it.

The strongest of them is ``test_mutating_the_future_does_not_change_the_signal``:
it generates a signal, then rewrites every bar after the decision date to
something wildly different, regenerates, and asserts the output is identical. If
any future value leaks into any factor, that test fails.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alpha.backtest.replay import (
    HistoryStore, LookaheadError, PointInTimeProvider, SYNTHETIC_CHAIN_FACTORS,
)
from alpha.data.base import ProviderError
from alpha.data.fixtures import FixtureProvider
from alpha.engines import index_options as io


@pytest.fixture(scope="module")
def store():
    fx = FixtureProvider(as_of=date(2026, 9, 1))
    return HistoryStore.load(fx, ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"], lookback=800)


def test_store_loads_symbols(store):
    assert set(store.frames) >= {"NIFTY", "BANKNIFTY", "RELIANCE"}
    assert len(store.trading_dates("NIFTY")) > 500


# -- the core guarantee ----------------------------------------------------

@pytest.mark.parametrize("offset", [0, 50, 200, 400])
def test_provider_never_returns_data_after_as_of(store, offset):
    dates = store.trading_dates("NIFTY")
    as_of = dates[-1 - offset]
    p = PointInTimeProvider(store, as_of)
    df = p.ohlcv("NIFTY", "1d", 400)
    assert df.index.max().date() <= as_of


def test_visible_slice_matches_the_true_prefix(store):
    dates = store.trading_dates("RELIANCE")
    as_of = dates[-100]
    full = store.frames["RELIANCE"]
    seen = PointInTimeProvider(store, as_of).ohlcv("RELIANCE", "1d", 10_000)
    expected = full[full.index.date <= as_of]
    pd.testing.assert_frame_equal(seen, expected)


def test_as_of_bar_itself_is_visible(store):
    """A decision made after the close of day T may use day T's bar."""
    as_of = store.trading_dates("NIFTY")[-50]
    df = PointInTimeProvider(store, as_of).ohlcv("NIFTY", "1d", 400)
    assert df.index.max().date() == as_of


def test_mutating_the_future_does_not_change_the_signal(store):
    """The decisive test. Rewrite everything after the decision date; the signal
    must come out identical."""
    as_of = store.trading_dates("BANKNIFTY")[-120]

    p1 = PointInTimeProvider(store, as_of)
    sig1 = io.build_signal("BANKNIFTY", p1, as_of)

    # Corrupt the future beyond recognition.
    poisoned = HistoryStore(frames={k: v.copy() for k, v in store.frames.items()})
    for sym, df in poisoned.frames.items():
        mask = df.index.date > as_of
        for col in ("open", "high", "low", "close"):
            df.loc[mask, col] = df.loc[mask, col] * 100.0
        df.loc[mask, "volume"] = 1.0

    p2 = PointInTimeProvider(poisoned, as_of)
    sig2 = io.build_signal("BANKNIFTY", p2, as_of)

    assert sig1.direction == sig2.direction
    assert sig1.conviction == sig2.conviction
    assert sig1.spot == sig2.spot
    assert [f.detail for f in sig1.scorecard.factors] == \
           [f.detail for f in sig2.scorecard.factors]
    assert [f.contribution for f in sig1.scorecard.factors] == \
           [f.contribution for f in sig2.scorecard.factors]


def test_indicators_see_no_future_values(store):
    """Belt and braces: the last visible close must equal the true close on
    as_of, not any later one."""
    as_of = store.trading_dates("TCS")[-77]
    seen = PointInTimeProvider(store, as_of).ohlcv("TCS", "1d", 400)
    truth = store.frames["TCS"]
    assert seen["close"].iloc[-1] == truth.loc[truth.index.date == as_of, "close"].iloc[0]


# -- future() is for evaluation only --------------------------------------

def test_future_returns_only_later_bars(store):
    as_of = store.trading_dates("NIFTY")[-60]
    fut = store.future("NIFTY", as_of)
    assert not fut.empty
    assert fut.index.min().date() > as_of


def test_provider_exposes_no_route_to_the_future(store):
    """PointInTimeProvider must not offer a future() of its own -- the evaluator
    reaches the store directly, the engine cannot."""
    p = PointInTimeProvider(store, store.trading_dates("NIFTY")[-60])
    assert not hasattr(p, "future")


# -- degradation rather than fabrication -----------------------------------

def test_intraday_raises_instead_of_inventing_a_path(store):
    p = PointInTimeProvider(store, store.trading_dates("NIFTY")[-60])
    with pytest.raises(ProviderError, match="no 15m history"):
        p.ohlcv("NIFTY", "15m", 200)


def test_vwap_factor_degrades_to_zero_weight_under_replay(store):
    as_of = store.trading_dates("NIFTY")[-60]
    sig = io.build_signal("NIFTY", PointInTimeProvider(store, as_of), as_of)
    vwap = next(f for f in sig.scorecard.factors if f.key == "vwap")
    assert vwap.weight == 0.0
    assert "No intraday data" in vwap.detail


def test_missing_symbol_raises(store):
    p = PointInTimeProvider(store, date(2026, 8, 1))
    with pytest.raises(ProviderError, match="no history loaded"):
        p.ohlcv("NOSUCHSYMBOL")


def test_date_before_all_history_raises(store):
    p = PointInTimeProvider(store, date(1990, 1, 1))
    with pytest.raises(ProviderError, match="no data for"):
        p.ohlcv("NIFTY")


# -- synthetic chain honesty -----------------------------------------------

def test_synthetic_chain_is_labelled(store):
    as_of = store.trading_dates("NIFTY")[-60]
    p = PointInTimeProvider(store, as_of)
    chain = p.option_chain("NIFTY")
    assert chain.source == "replay-synthetic"
    assert chain.stale is True
    assert any("SYNTHETIC" in d for d in p.degraded)


def test_synthetic_chain_spot_matches_the_real_underlying(store):
    """Chain prices must be consistent with the true historical spot, or strike
    selection and theta are measuring nothing."""
    as_of = store.trading_dates("BANKNIFTY")[-60]
    p = PointInTimeProvider(store, as_of)
    truth = store.frames["BANKNIFTY"]
    expected = float(truth.loc[truth.index.date == as_of, "close"].iloc[0])
    assert p.option_chain("BANKNIFTY").spot == pytest.approx(expected)


def test_replay_is_reproducible(store):
    as_of = store.trading_dates("NIFTY")[-60]
    a = PointInTimeProvider(store, as_of).option_chain("NIFTY")
    b = PointInTimeProvider(store, as_of).option_chain("NIFTY")
    assert [r.ce_oi for r in a.rows] == [r.ce_oi for r in b.rows]
    assert [r.ce_ltp for r in a.rows] == [r.ce_ltp for r in b.rows]


def test_synthetic_factor_list_covers_the_oi_factors():
    assert SYNTHETIC_CHAIN_FACTORS == {"pcr", "max_pain", "oi_levels", "oi_buildup"}


def test_vix_proxy_is_flagged(store):
    p = PointInTimeProvider(store, store.trading_dates("NIFTY")[-60])
    assert p.india_vix() is not None
    assert any("realised-volatility proxy" in d for d in p.degraded)
