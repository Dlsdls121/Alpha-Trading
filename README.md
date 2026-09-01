# Alpha Trading — signal advisor for NSE

An **advisory** signal engine for Indian markets. It analyses NIFTY and BANKNIFTY
for option-buying setups, scans equities for multi-day positional candidates, and
shows you **the full reasoning behind every call** on a tablet-friendly dashboard.

> **It does not trade.** There is no broker integration, no credential handling,
> no order routing and no position or P&L tracking anywhere in this codebase.
> The API is read-only (a test asserts this). You get analysis; what you do with
> it is entirely your decision.

---

## Read this before you use it

This produces **heuristic signals, not predictions.**

- **No backtested edge is claimed.** The factors and weights are drawn from
  widely-used market practice, not from a validated study. Nobody has
  demonstrated that this combination makes money.
- **"Conviction 72" is not "72% likely to be right."** It is the magnitude of
  the weighted score, discounted for factor disagreement. It has no calibration
  behind it and must not be read as a probability.
- **Option buying loses money most of the time**, even for people who are good
  at it. You can be right about direction and still lose everything to time decay
  or an IV crush. The engine's vetoes exist because of this, not despite it.
- The sensible way to use this is to **paper-trade it for months** and keep
  score, before any money is involved.

---

## What it actually looks at

### Index options (NIFTY, BANKNIFTY)

A long option loses on three axes and wins on one, so the scorecard is split in
two, and the split is load-bearing:

**Directional factors** — these carry weight and vote on which way:

| Factor | What it reads |
|---|---|
| Moving-average structure | 20/50/200 EMA stack and slope — regime, not just direction |
| Trend strength (ADX) | Whether there is a trend *at all*; long premium bleeds in a range |
| RSI(14) | Momentum, with extremes discounted rather than flipped |
| MACD histogram | Whether the impulse is still expanding |
| Price vs session VWAP | Intraday acceptance, with volume confirmation |
| Price vs OI buildup | Long buildup / short buildup / covering / unwinding |
| Put-call ratio | Near-ATM only, read contrarian at the extremes |
| OI support / resistance | The strikes writers are defending |
| Max pain | Weighted by proximity to expiry — near-worthless a month out |

**Cost factors** — shown as evidence, **weight zero, no vote on direction**:

| Factor | Why it can't vote |
|---|---|
| India VIX regime | Expensive premium is a cost, not a bearish opinion |
| **Implied vs realised vol** | The sharpest test for a buyer: are you paying 22% for movement running at 11%? |
| IV percentile | Where IV sits in its own range |
| Days to expiry | Theta, and the expiry-day gate |

**Vetoes block; they do not subtract.** Being directionally right does not rescue
a position bought on expiry day or into an IV crush, so those are hard stops:

- expiry day, or one session out (warning)
- IV far above realised vol
- IV at the top of its own range (warning)
- prohibitive theta bleed on the selected contract
- no liquid strike in the target delta band
- conviction below the floor
- index range-bound on ADX (warning)

**Strike selection targets delta, not cheapness.** The instinct to buy the cheap
far-OTM strike is what makes most option buying lose: a 0.15-delta option needs a
large *and fast* move merely to break even. The engine targets 0.42–0.62 delta and
filters on open interest, volume and spread, then reports theta as a **percentage
of premium per day** — the number a buyer should actually watch.

### Positional equity

Selection is treated as a *relative* problem, because in any given month much of
a large-cap index is going up anyway:

- **Relative strength vs NIFTY** (3m and 6m) plus the **slope** of the RS line —
  a stock can carry a strong six-month number while having quietly stopped
  outperforming weeks ago
- **Sector rotation** — mean RS of the sector's scanned constituents
- Trend stage, position in the 52-week range, 21-day momentum, volume
  confirmation, extension from the mean
- **Gates:** turnover floor, ATR-based sizing warning, and a reward-to-risk veto
  (2-ATR stop against a 3-ATR first target = 1.5:1 minimum)

Long-only by design: positional cash shorting isn't available to most retail
participants, so a short call would be advice you couldn't act on.

---

## Exchange rules encoded here (these changed recently)

Two changes materially affect option timing, and both are **data, not
assumptions baked into code** — see `alpha/calendar.py`:

- **BANKNIFTY, FINNIFTY and MIDCPNIFTY lost their weekly contracts on
  20 Nov 2024** under SEBI's one-weekly-expiry-per-exchange rule. They are
  **monthly-only**, expiring the last Tuesday.
- **NSE moved expiry from Thursday to Tuesday** (effective 1 Sep 2025).

The practical consequence: NIFTY is a 0–7 day theta problem, BANKNIFTY is a 1–35
day one. They are different trades and the engine treats them that way.

Holidays live in `alpha/reference/holidays.json` with a `verified_through` date.
**The bundled list is nearly empty** — expiries computed past that date raise a
visible warning rather than pretending to be certain. Refresh it from the NSE
circular before relying on expiry maths.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m alpha.cli brief --explain      # everything, with full reasoning
python -m alpha.cli serve                # dashboard at http://localhost:8000
pytest                                   # 110 tests
```

It runs in **fixture mode by default**: fully offline, deterministic, simulated
data. Nothing is live and everything says so, loudly, in the UI and the CLI.

### Going live

```bash
export ALPHA_DATA_MODE=live
python -m alpha.data.nse selftest        # verify NSE is reachable from your network
python -m alpha.cli brief --mode live
```

Live mode uses **NSE** for option chains, spot and India VIX, and **Yahoo
Finance** for historical OHLCV. No API keys, no paid subscription.

**Verification status:** the NSE provider was written against NSE's
observed request/response behaviour but **could not be exercised against the live
endpoint** — the network this was built on blocks `nseindia.com` by policy. Treat
your first live run as the real test; that is what `selftest` is for. NSE has no
documented public API, defends it with cookie and header checks, and rate-limits
aggressively, so responses are cached and every failure raises loudly instead of
returning empty data.

If a provider fails, the engine falls back to fixtures and **says so** in
`Signal.data_quality`, in the dashboard banner and in the CLI footer. Simulated
data is never silently substituted for live.

---

## Dashboard

Tablet-first (designed at 834px, works from phone to desktop), no build step —
vanilla JS and CSS, so there is nothing to compile and little to break.

Every card shows the call, then the **factor-by-factor reasoning**: each factor's
verdict, the actual numbers it read, a plain-English explanation, and a diverging
bar for its contribution. Cost factors are marked `context` so it is obvious they
inform but do not vote. Every signal also carries **"what would make this wrong"**.

Run it on any machine on your network and open it from the tablet:

```bash
ALPHA_HOST=0.0.0.0 python -m alpha.cli serve
```

---

## Layout

```
alpha/
  models.py               Factor / Veto / Scorecard / Signal — the explainability core
  calendar.py             expiry rules, holidays, days-to-expiry
  universe.py             scan universe + sector map
  indicators/             trend, momentum, volatility, volume, options maths
  data/                   provider protocol, NSE, Yahoo, fixtures, composite, cache
  engines/                index_options.py, equity_positional.py, features.py
  api.py  cli.py
  reference/              holidays.json, universe.json  (edit these, not code)
web/                      dashboard (index.html + static/)
tests/                    110 tests
```

**Tuning:** thresholds live in `OptionEngineConfig` and `EquityEngineConfig` —
one place, so they can be argued with rather than hunted for.

---

## Known limitations

- **No backtest.** There is no harness here to measure whether these signals have
  ever worked. That is the single biggest gap and the most valuable next thing to
  build; until it exists, treat every output as a hypothesis.
- **IV percentile uses realised vol as a proxy** for an IV history, because no
  free source publishes per-strike IV history. The factor text says so rather
  than passing it off as real.
- **The equity universe is 27 names.** Several sectors have one constituent, and
  for those the sector factor is shown with zero weight — with one name, "sector
  strength" is just that stock's own relative strength restated, and counting
  both would be double-counting. Add names to `universe.json` to fix this
  properly.
- **Index constituents and lot sizes change** at every exchange review. The
  reference files carry `verified_through` dates for a reason.
- No corporate-action handling (splits, bonuses) on the equity history.
- No intraday scheduling or alerting; it computes on request.

---

## Licence and responsibility

Educational and personal use. Markets can and do take your money. Nothing here is
investment advice, and no one involved in writing it is responsible for what you
do with it.
