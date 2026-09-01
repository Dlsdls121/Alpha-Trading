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
pytest                                   # 201 tests
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

## Backtesting: does any of this work?

```bash
python -m alpha.cli backtest --which both --step 5
```

The harness replays history day by day, generates signals against a **point-in-time
view**, and resolves each against what actually followed.

### How it avoids lying to you

**Lookahead bias is prevented structurally, not by care.** `PointInTimeProvider`
owns the history and hands out only the slice up to the decision date; the
engines have no route to the rest. The test that proves it takes a signal,
rewrites every future bar to something wildly different, regenerates, and asserts
the output is byte-identical.

**Every modelling choice is the pessimistic one:**

| Choice | Why |
|---|---|
| Entry at the **next bar's open** | A signal from today's close cannot be filled at today's close |
| **Stop wins ties** | When a bar touches both stop and target, intrabar order is unknowable — so assume the one that costs money |
| Gaps fill at the **open**, not the level | Modelling a gap-through as filling at the stop is a fiction that flatters every result |
| Costs on **every** trade | Brokerage, taxes and slippage; a wider default for options, where the spread is the real cost |
| Positions **never held past expiry** | See below |

**Every number carries its error bars.** A 60% hit rate on 40 trades has a 95%
confidence interval of **44.6%–73.7%** — it does not exclude a coin flip.
`verdict()` refuses to call such a result meaningful and says "sample too small
to conclude anything" instead. It needs ~400 samples before that interval
tightens usefully.

**Two baselines**, because they answer different questions: buy-and-hold the same
instrument (did the *exits* help?) and the universe average (did the *selection*
help?). Picks returning 2% while the average name returned 2% is a strategy that
has done nothing but take risk.

### The null test

The most important test in the repo runs the whole harness on **pure random-walk
data**, where by construction there is nothing to find, and asserts it reports no
edge. A backtest that finds an edge in noise will confidently endorse anything.

**It immediately caught a real bug.** The option evaluator clamped the exit
*date* to expiry (so the option priced at `t=0`, i.e. intrinsic) while still
taking the exit *spot* from a later bar. Because intrinsic value is
`max(0, S−K)` — convex and non-negative — feeding it a higher-variance future
price inflated the payoff systematically. It manufactured **+19% mean return out
of pure noise**, with 5 of 8 random seeds showing a "significant" edge. After the
fix: mean of means +0.62%, and 0 of 8 seeds show any edge. That test is now
permanent.

### What the backtest cannot tell you

- **Historical NSE option chains are not freely available.** The chain at each
  decision date is synthesised from the underlying's trailing realised vol.
  Prices track the real index path; **open interest is invented**. So the
  OI factors (PCR, max pain, OI levels, OI buildup) are **excluded from scoring
  by default** — the run tests the price-based factors, which is the part that
  can honestly be tested.
- **Option P&L assumes constant IV**, which is optimistic — a real IV crush makes
  outcomes worse, never better. Read **directional accuracy** as the honest
  number and modelled return as an upper bound.
- **Survivorship bias**: the universe is today's constituents applied to past
  dates. Names that were dropped are missing, and the survivors did better.
- No corporate-action adjustment; intraday bars aren't replayed, so the VWAP
  factor is inactive throughout.

### Running it on generated data tells you nothing

In fixture mode the harness **overrides its own verdict** with
"Not a real result — generated data" and refuses to present the numbers as
findings. The bundled generator builds paths from a drift plus a smooth sine
cycle, and a trend-following engine predicts a sine wave nearly perfectly — a run
like that shows a huge, entirely fake edge. Use `ALPHA_DATA_MODE=live` against
real history before reading any number as evidence.

**Do not tune factor weights on backtest output.** The attribution table is
printed with its sample size and a warning for exactly this reason: fitting
weights to your own history is how a strategy is overfitted and then fails live.

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
  backtest/               replay, evaluate, metrics, runner, report
  api.py  cli.py
  reference/              holidays.json, universe.json  (edit these, not code)
web/                      dashboard (index.html + static/)
tests/                    201 tests
```

**Tuning:** thresholds live in `OptionEngineConfig` and `EquityEngineConfig` —
one place, so they can be argued with rather than hunted for.

---

## Known limitations

- **The backtest has not been run on real data.** The harness exists and is
  tested, but the network this was built on blocks market-data hosts, so every
  run so far has been on generated history — which proves the machinery works and
  nothing else. Until you run it against real NSE history, treat every signal as
  an untested hypothesis.
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
