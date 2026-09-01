"""NSE expiry and holiday logic.

Getting this wrong quietly corrupts every option signal, because days-to-expiry
drives theta, strike choice and the veto rules. Two things changed recently and
are encoded here as *data* rather than assumptions baked into code:

* SEBI's Oct-2024 circular limited each exchange to one weekly index expiry.
  NSE kept NIFTY; **BANKNIFTY, FINNIFTY and MIDCPNIFTY lost their weeklies on
  20 Nov 2024 and are monthly-only.**
* NSE moved its expiry day from Thursday to **Tuesday** (effective 1 Sep 2025).

Both are exchange policy and have already changed twice. They live in
``EXPIRY_RULES`` so a future circular is a one-line edit, not a refactor.
"""

from __future__ import annotations

import calendar as _cal
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = 0, 1, 2, 3, 4

_HOLIDAY_FILE = Path(__file__).parent / "reference" / "holidays.json"


@dataclass(frozen=True)
class ExpiryRule:
    """How one symbol's contracts expire."""

    symbol: str
    has_weekly: bool
    weekday: int                  # 0=Mon .. 4=Fri
    exchange: str = "NSE"
    note: str = ""


# As of September 2026. See module docstring for the two rule changes.
EXPIRY_RULES: dict[str, ExpiryRule] = {
    "NIFTY": ExpiryRule("NIFTY", True, TUESDAY, "NSE",
                        "Weekly every Tuesday; monthly on the last Tuesday."),
    "BANKNIFTY": ExpiryRule("BANKNIFTY", False, TUESDAY, "NSE",
                            "Monthly only since 20-Nov-2024 (SEBI single-weekly rule). "
                            "Last Tuesday of the month."),
    "FINNIFTY": ExpiryRule("FINNIFTY", False, TUESDAY, "NSE",
                           "Monthly only since 20-Nov-2024. Last Tuesday."),
    "MIDCPNIFTY": ExpiryRule("MIDCPNIFTY", False, TUESDAY, "NSE",
                             "Monthly only since 20-Nov-2024. Last Tuesday."),
    "SENSEX": ExpiryRule("SENSEX", True, THURSDAY, "BSE",
                         "BSE retained the weekly under the SEBI rule; Thursday cycle."),
}


@dataclass
class HolidayCalendar:
    holidays: set[date] = field(default_factory=set)
    verified_through: date | None = None
    source: str = "bundled"

    @classmethod
    def load(cls, path: Path | None = None) -> "HolidayCalendar":
        path = path or _HOLIDAY_FILE
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        days = {date.fromisoformat(h["date"]) for h in raw.get("holidays", [])}
        vt = raw.get("verified_through")
        return cls(holidays=days,
                   verified_through=date.fromisoformat(vt) if vt else None,
                   source=str(path))

    def is_holiday(self, d: date) -> bool:
        return d in self.holidays

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self.is_holiday(d)

    def previous_trading_day(self, d: date) -> date:
        cur = d
        for _ in range(30):
            cur -= timedelta(days=1)
            if self.is_trading_day(cur):
                return cur
        raise RuntimeError(f"no trading day found within 30 days before {d}")

    def roll_back_to_trading_day(self, d: date) -> date:
        """NSE convention: an expiry landing on a holiday moves *earlier*."""
        cur = d
        for _ in range(30):
            if self.is_trading_day(cur):
                return cur
            cur -= timedelta(days=1)
        raise RuntimeError(f"no trading day at or before {d}")

    def trading_days_between(self, start: date, end: date) -> int:
        """Trading sessions strictly after ``start``, up to and including ``end``."""
        if end <= start:
            return 0
        n, cur = 0, start
        while cur < end:
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                n += 1
        return n

    def coverage_warning(self, target: date) -> str | None:
        """Flag when we are computing past the point the holiday list was checked."""
        if self.verified_through is None:
            return ("No holiday list is loaded, so expiry dates are not adjusted for "
                    "exchange holidays and days-to-expiry may be overstated.")
        if target > self.verified_through:
            return (f"Holiday list is only verified through "
                    f"{self.verified_through:%d-%b-%Y}; an unlisted holiday on or before "
                    f"{target:%d-%b-%Y} would shift this expiry earlier. "
                    f"Refresh alpha/reference/holidays.json from the NSE circular.")
        return None


def _nth_weekday_of_month(year: int, month: int, weekday: int, last: bool = True) -> date:
    """First or last occurrence of ``weekday`` in a month."""
    if last:
        last_day = _cal.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def monthly_expiry(symbol: str, year: int, month: int,
                   cal: HolidayCalendar | None = None) -> date:
    """Last <expiry weekday> of the month, rolled back off holidays."""
    rule = _rule(symbol)
    cal = cal or HolidayCalendar.load()
    return cal.roll_back_to_trading_day(_nth_weekday_of_month(year, month, rule.weekday))


def _rule(symbol: str) -> ExpiryRule:
    key = symbol.upper().replace(" ", "")
    if key not in EXPIRY_RULES:
        raise KeyError(
            f"no expiry rule for {symbol!r}; known symbols: {sorted(EXPIRY_RULES)}. "
            f"Add one to EXPIRY_RULES rather than assuming a weekly Tuesday cycle."
        )
    return EXPIRY_RULES[key]


def next_expiry(symbol: str, on: date | None = None,
                cal: HolidayCalendar | None = None) -> date:
    """The nearest tradable expiry at or after ``on``.

    For a monthly-only symbol this can be up to ~5 weeks out, which is exactly
    why BANKNIFTY and NIFTY need different strike and theta handling.
    """
    on = on or date.today()
    cal = cal or HolidayCalendar.load()
    rule = _rule(symbol)

    if rule.has_weekly:
        ahead = (rule.weekday - on.weekday()) % 7
        cand = cal.roll_back_to_trading_day(on + timedelta(days=ahead))
        if cand >= on:
            return cand
        # Rolled back past today (holiday week) -- take next week's.
        return cal.roll_back_to_trading_day(on + timedelta(days=ahead + 7))

    this_month = monthly_expiry(symbol, on.year, on.month, cal)
    if this_month >= on:
        return this_month
    nxt = date(on.year + (on.month == 12), (on.month % 12) + 1, 1)
    return monthly_expiry(symbol, nxt.year, nxt.month, cal)


def expiry_series(symbol: str, count: int = 4, on: date | None = None,
                  cal: HolidayCalendar | None = None) -> list[date]:
    """The next ``count`` expiries, nearest first."""
    on = on or date.today()
    cal = cal or HolidayCalendar.load()
    out: list[date] = []
    cursor = on
    for _ in range(count):
        e = next_expiry(symbol, cursor, cal)
        out.append(e)
        cursor = e + timedelta(days=1)
    return out


@dataclass
class ExpiryContext:
    """Everything the option engine needs to know about time."""

    symbol: str
    expiry: date
    as_of: date
    calendar_days: int
    trading_days: int
    t_years: float
    is_expiry_day: bool
    has_weekly: bool
    rule_note: str
    warning: str | None = None

    @property
    def label(self) -> str:
        kind = "weekly" if self.has_weekly else "monthly"
        return f"{self.expiry:%d-%b-%Y} ({kind}, {self.calendar_days}d / {self.trading_days} sessions)"


def expiry_context(symbol: str, on: date | None = None,
                   cal: HolidayCalendar | None = None) -> ExpiryContext:
    on = on or date.today()
    cal = cal or HolidayCalendar.load()
    rule = _rule(symbol)
    exp = next_expiry(symbol, on, cal)

    cd = (exp - on).days
    td = cal.trading_days_between(on, exp)
    # Time to expiry in years. Floor at ~2 hours so expiry-day greeks stay finite
    # instead of dividing by zero; the expiry-day veto is what really guards this.
    t_years = max(cd, 0.0) / 365.0 if cd > 0 else (2.0 / 24.0) / 365.0

    return ExpiryContext(
        symbol=symbol.upper(), expiry=exp, as_of=on, calendar_days=cd,
        trading_days=td, t_years=t_years, is_expiry_day=(cd == 0),
        has_weekly=rule.has_weekly, rule_note=rule.note,
        warning=cal.coverage_warning(exp),
    )
