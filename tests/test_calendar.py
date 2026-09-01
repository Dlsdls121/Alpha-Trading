"""Expiry and holiday logic.

These assertions encode exchange policy that changed twice recently. If a future
circular moves things again these tests are meant to fail loudly rather than let
signals be computed against a stale assumption.
"""

from datetime import date, timedelta

import pytest

from alpha.calendar import (
    EXPIRY_RULES, HolidayCalendar, expiry_context, expiry_series, monthly_expiry,
    next_expiry,
)


def test_nifty_is_weekly_tuesday():
    rule = EXPIRY_RULES["NIFTY"]
    assert rule.has_weekly is True
    assert rule.weekday == 1                    # Tuesday


@pytest.mark.parametrize("sym", ["BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
def test_weeklies_were_discontinued(sym):
    """SEBI's Oct-2024 single-weekly-expiry rule; effective 20 Nov 2024."""
    assert EXPIRY_RULES[sym].has_weekly is False


def test_nifty_weekly_expiry_lands_on_tuesday():
    for offset in range(0, 21):
        d = date(2026, 9, 1) + timedelta(days=offset)
        exp = next_expiry("NIFTY", d)
        assert exp >= d
        # Tuesday, unless rolled back off a holiday.
        assert exp.weekday() <= 1


def test_monthly_expiry_is_last_tuesday():
    exp = monthly_expiry("BANKNIFTY", 2026, 10)
    assert exp.weekday() == 1
    assert (exp + timedelta(days=7)).month != 10       # nothing later in the month


def test_banknifty_expiry_is_further_out_than_nifty():
    """The practical consequence of the rule change: different theta problems."""
    d = date(2026, 9, 2)
    assert (next_expiry("BANKNIFTY", d) - d).days >= (next_expiry("NIFTY", d) - d).days


def test_expiry_series_is_ordered_and_distinct():
    s = expiry_series("NIFTY", 5, date(2026, 9, 1))
    assert len(s) == len(set(s)) == 5
    assert s == sorted(s)


def test_holiday_rolls_expiry_backwards():
    cal = HolidayCalendar(holidays={date(2026, 10, 27)},
                          verified_through=date(2027, 1, 1))
    exp = monthly_expiry("BANKNIFTY", 2026, 10, cal)
    assert exp == date(2026, 10, 26)               # rolled back to Monday
    assert exp.weekday() == 0


def test_weekend_is_never_a_trading_day():
    cal = HolidayCalendar()
    assert not cal.is_trading_day(date(2026, 9, 5))    # Saturday
    assert not cal.is_trading_day(date(2026, 9, 6))    # Sunday
    assert cal.is_trading_day(date(2026, 9, 7))        # Monday


def test_trading_days_between_excludes_weekends():
    cal = HolidayCalendar()
    # Tue 1 Sep -> Tue 8 Sep 2026 is 5 sessions (Wed,Thu,Fri,Mon,Tue)
    assert cal.trading_days_between(date(2026, 9, 1), date(2026, 9, 8)) == 5
    assert cal.trading_days_between(date(2026, 9, 8), date(2026, 9, 1)) == 0


def test_coverage_warning_fires_past_verified_date():
    cal = HolidayCalendar(holidays=set(), verified_through=date(2026, 9, 1))
    assert cal.coverage_warning(date(2026, 8, 1)) is None
    assert "verified through" in cal.coverage_warning(date(2026, 12, 1))


def test_missing_holiday_list_warns_rather_than_pretending():
    assert "No holiday list" in HolidayCalendar().coverage_warning(date(2026, 9, 1))


def test_expiry_day_is_detected():
    ctx = expiry_context("NIFTY", date(2026, 9, 1))     # a Tuesday
    assert ctx.is_expiry_day
    assert ctx.calendar_days == 0
    assert ctx.t_years > 0                # never zero: greeks must stay finite


def test_unknown_symbol_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no expiry rule"):
        next_expiry("RELIANCE")
