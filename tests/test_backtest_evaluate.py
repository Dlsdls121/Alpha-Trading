"""Outcome-resolution tests.

Bars are hand-built so every rule (next-bar entry, stop-wins-ties, gap fills at
the open, cost deduction) is checked against an arithmetic answer rather than a
snapshot.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from alpha.backtest.evaluate import CostModel, evaluate, evaluate_equity, evaluate_option
from alpha.backtest.replay import HistoryStore
from alpha.models import Direction, OptionLeg, Scorecard, Signal

SIG_DATE = date(2026, 3, 2)          # a Monday


def make_store(symbol: str, bars: list[tuple[float, float, float, float]],
               start: date = SIG_DATE) -> HistoryStore:
    """bars = [(open, high, low, close), ...] starting the day AFTER `start`."""
    idx = pd.bdate_range(start=pd.Timestamp(start), periods=len(bars) + 1)[1:]
    df = pd.DataFrame(
        {"open": [b[0] for b in bars], "high": [b[1] for b in bars],
         "low": [b[2] for b in bars], "close": [b[3] for b in bars],
         "volume": [1e6] * len(bars)}, index=idx)
    # Include a bar on the signal date itself so `future()` has something to cut.
    hist = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0],
                         "close": [100.0], "volume": [1e6]},
                        index=[pd.Timestamp(start)])
    return HistoryStore(frames={symbol: pd.concat([hist, df])})


def make_equity_signal(symbol="TEST", spot=100.0, stop=95.0, target=110.0) -> Signal:
    return Signal(
        signal_id="t1", kind="equity_positional", symbol=symbol,
        generated_at=datetime.combine(SIG_DATE, datetime.min.time()),
        direction=Direction.LONG, conviction=60, headline="", summary="",
        scorecard=Scorecard(), spot=spot, invalidation=stop, targets=[target, 120.0])


# -- entry rule ------------------------------------------------------------

def test_entry_is_the_next_bar_open_not_the_signal_close():
    """The single most common way a backtest invents an edge."""
    store = make_store("TEST", [(102.0, 103.0, 101.0, 102.5),
                                (102.5, 104.0, 102.0, 103.0)])
    out = evaluate_equity(make_equity_signal(), store, horizon_bars=5)
    assert out.entry_price == 102.0          # next bar's open
    assert out.entry_date == date(2026, 3, 3)


def test_signal_day_bar_is_not_tradeable():
    store = make_store("TEST", [(102.0, 103.0, 101.0, 102.5)])
    out = evaluate_equity(make_equity_signal(), store, horizon_bars=5)
    assert out.entry_date > SIG_DATE


# -- exit resolution -------------------------------------------------------

def test_target_hit_exits_at_target():
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.5),
                                (100.5, 112.0, 100.0, 111.0)])
    out = evaluate_equity(make_equity_signal(target=110.0), store, horizon_bars=5)
    assert out.exit_reason == "target"
    assert out.exit_price == 110.0
    assert out.gross_return_pct == pytest.approx(10.0, abs=0.01)


def test_stop_hit_exits_at_stop():
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.5),
                                (100.0, 100.5, 94.0, 94.5)])
    out = evaluate_equity(make_equity_signal(stop=95.0), store, horizon_bars=5)
    assert out.exit_reason == "stop"
    assert out.exit_price == 95.0
    assert out.gross_return_pct == pytest.approx(-5.0, abs=0.01)


def test_stop_wins_when_both_levels_are_touched_in_one_bar():
    """Intrabar order is unknowable from daily data, so the losing assumption
    is the one taken."""
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.0),
                                (100.0, 115.0, 90.0, 100.0)])     # both hit
    out = evaluate_equity(make_equity_signal(stop=95.0, target=110.0), store, horizon_bars=5)
    assert out.exit_reason == "stop"
    assert out.exit_price == 95.0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Modelling this as the stop level would be a fiction that flatters results."""
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.0),
                                (88.0, 89.0, 85.0, 86.0)])        # gapped below stop
    out = evaluate_equity(make_equity_signal(stop=95.0), store, horizon_bars=5)
    assert out.exit_reason == "stop"
    assert out.exit_price == 88.0                                  # the open, worse
    assert out.gross_return_pct < -10


def test_horizon_timeout_exits_at_the_last_close():
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.0)] * 4)
    out = evaluate_equity(make_equity_signal(), store, horizon_bars=3)
    assert out.exit_reason == "horizon"
    assert out.exit_price == 100.0


def test_no_forward_data_is_reported_not_silently_zero():
    store = HistoryStore(frames={"TEST": pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
         "volume": [1.0]}, index=[pd.Timestamp(SIG_DATE)])})
    out = evaluate_equity(make_equity_signal(), store, horizon_bars=5)
    assert out.exit_reason == "no_data"
    assert out.directionally_right is None
    assert out.notes


# -- excursions ------------------------------------------------------------

def test_mfe_and_mae_track_the_best_and_worst_points():
    store = make_store("TEST", [(100.0, 100.0, 100.0, 100.0),
                                (100.0, 106.0, 97.0, 101.0),
                                (101.0, 103.0, 99.0, 102.0)])
    out = evaluate_equity(make_equity_signal(stop=90.0, target=130.0), store, horizon_bars=5)
    assert out.mfe_pct == pytest.approx(6.0, abs=0.1)     # best +6%
    assert out.mae_pct == pytest.approx(-3.0, abs=0.1)    # worst -3%


# -- costs -----------------------------------------------------------------

def test_costs_are_deducted_from_every_trade():
    store = make_store("TEST", [(100.0, 101.0, 99.0, 100.0),
                                (100.0, 112.0, 100.0, 111.0)])
    costs = CostModel(equity_bps=35.0, slippage_bps=15.0)     # 0.50% round trip
    out = evaluate_equity(make_equity_signal(), store, horizon_bars=5, costs=costs)
    assert out.net_return_pct == pytest.approx(out.gross_return_pct - 0.50, abs=1e-6)
    assert out.net_return_pct < out.gross_return_pct


def test_options_are_charged_more_than_equities():
    c = CostModel()
    assert c.for_kind("index_option") > c.for_kind("equity_positional")


def test_a_small_win_can_be_a_net_loss_after_costs():
    store = make_store("TEST", [(100.0, 100.0, 100.0, 100.0),
                                (100.0, 100.2, 99.9, 100.1)])
    out = evaluate_equity(make_equity_signal(stop=90.0, target=130.0), store,
                          horizon_bars=2, costs=CostModel(equity_bps=50.0, slippage_bps=10.0))
    assert out.gross_return_pct > 0
    assert out.net_return_pct < 0
    assert out.win is False


# -- options ---------------------------------------------------------------

def make_option_signal(direction=Direction.LONG, strike=25000.0,
                       expiry_offset=30) -> Signal:
    leg = OptionLeg(symbol="NIFTY", expiry=SIG_DATE + timedelta(days=expiry_offset),
                    strike=strike, option_type="CE" if direction is Direction.LONG else "PE",
                    ltp=300.0, delta=0.5, iv=15.0, oi=100000, volume=5000)
    return Signal(
        signal_id="o1", kind="index_option", symbol="NIFTY",
        generated_at=datetime.combine(SIG_DATE, datetime.min.time()),
        direction=direction, conviction=60, headline="", summary="",
        scorecard=Scorecard(), spot=25000.0, invalidation=24700.0,
        targets=[25400.0], leg=leg)


def test_option_outcome_is_marked_as_modelled():
    store = make_store("NIFTY", [(25000.0, 25100.0, 24900.0, 25050.0),
                                 (25050.0, 25500.0, 25000.0, 25450.0)])
    out = evaluate_option(make_option_signal(), store, horizon_bars=5)
    assert out.modelled is True
    assert any("constant IV" in n for n in out.notes)


def test_option_reports_underlying_move_separately_from_modelled_pnl():
    store = make_store("NIFTY", [(25000.0, 25100.0, 24900.0, 25050.0),
                                 (25050.0, 25500.0, 25000.0, 25450.0)])
    out = evaluate_option(make_option_signal(), store, horizon_bars=5)
    assert out.underlying_return_pct > 0
    # Leverage: the option should move much further than the index.
    assert abs(out.gross_return_pct) > abs(out.underlying_return_pct)


def test_put_signal_is_right_when_the_index_falls():
    store = make_store("NIFTY", [(25000.0, 25050.0, 24900.0, 24950.0),
                                 (24950.0, 25000.0, 24500.0, 24550.0)])
    sig = make_option_signal(direction=Direction.SHORT, strike=25000.0)
    sig.invalidation, sig.targets = 25300.0, [24600.0]
    out = evaluate_option(sig, store, horizon_bars=5)
    assert out.underlying_return_pct > 0        # signed by direction
    assert out.directionally_right is True


def test_theta_alone_loses_money_on_a_flat_index():
    """A correct-but-stationary view still bleeds. This is the whole reason the
    engine has expiry vetoes."""
    flat = [(25000.0, 25010.0, 24990.0, 25000.0)] * 9
    store = make_store("NIFTY", flat)
    out = evaluate_option(make_option_signal(expiry_offset=20), store, horizon_bars=8)
    assert out.underlying_return_pct == pytest.approx(0.0, abs=0.2)
    assert out.gross_return_pct < 0              # decayed anyway


def test_expiry_caps_the_holding_period():
    store = make_store("NIFTY", [(25000.0, 25010.0, 24990.0, 25000.0)] * 12)
    out = evaluate_option(make_option_signal(expiry_offset=5), store, horizon_bars=10)
    assert out.exit_date <= SIG_DATE + timedelta(days=5)


def test_dispatch_picks_the_right_evaluator():
    store = make_store("NIFTY", [(25000.0, 25100.0, 24900.0, 25050.0)] * 3)
    assert evaluate(make_option_signal(), store).modelled is True
    store2 = make_store("TEST", [(100.0, 101.0, 99.0, 100.0)] * 3)
    assert evaluate(make_equity_signal(), store2).modelled is False
