# Getting the dashboard open

The dashboard is a small web server. Something has to run it — it is not a file
you can open by tapping it. This is the shortest honest path to having it on
your tablet screen.

Everything below starts in **fixture mode**: simulated data, clearly labelled in
the UI. Switch to live NSE data once it is running.

---

## First, the thing that decides your route

**NSE blocks a lot of datacenter IP addresses.** It has no public API; what the
site uses is defended by cookie and header checks, and requests from cloud
providers get refused far more often than requests from an ordinary home
connection in India.

That single fact splits the options:

| Route | Opening it on the tablet | Live NSE data likely to work? |
|---|---|---|
| **A. Cloud host** | Easiest — a URL from anywhere | **Often blocked.** Datacenter IP |
| **B. Termux on the tablet** | Local, always with you | **Best chance.** Home/mobile IP in India |
| **C. A computer on your Wi-Fi** | Only at home | **Best chance.** Home IP |

If you mainly want to *look at it and learn the reasoning*, take **A** — it is by
far the least work. If you want it reading **real NSE data**, take **B** or **C**.

You can also do both: cloud for convenience, Termux when you want live data.

---

## Route A — Cloud host (easiest to open)

Free tier on [Render](https://render.com). The repo already has the two files it
needs (`Dockerfile`, `render.yaml`).

1. Sign in to Render with your GitHub account.
2. **New → Blueprint**, pick the `Alpha-Trading` repo.
3. Choose branch `claude/trading-platform-tablet-df45uw`.
4. Apply. First build takes a few minutes.
5. Open the URL it gives you. Bookmark it on the tablet home screen.

The free tier sleeps after ~15 minutes idle and takes ~30s to wake. Fine for a
dashboard you open a few times a day.

**Then test whether NSE is reachable from it.** In the Render shell:

```bash
python -m alpha.data.nse selftest
```

- Prints spot, expiry and strike count → set `ALPHA_DATA_MODE=live` in Render's
  environment settings and redeploy. Done.
- Prints a 401/403 or "rate-limited or blocked" → the host's IP is blocked.
  Leave it in fixture mode and use Route B for live data.

Railway, Fly.io and any other Docker host work the same way.

---

## Route B — On the tablet itself, with Termux

Fully self-contained: nothing else has to be switched on, and it uses your own
Indian IP, which is the version most likely to get real NSE data.

1. Install **Termux** from [F-Droid](https://f-droid.org/packages/com.termux/).
   *Not* the Play Store version — it is outdated and its packages are broken.
2. Open Termux and run:

```bash
curl -sSL https://raw.githubusercontent.com/Dlsdls121/Alpha-Trading/claude/trading-platform-tablet-df45uw/setup.sh | bash
```

3. When it finishes:

```bash
cd ~/alpha-trading
python -m alpha.cli serve
```

4. Open **http://localhost:8000** in the tablet's browser.

**Expect the install to take a while.** `setup.sh` installs numpy and pandas from
Termux's own package repository specifically to avoid compiling them on-device,
which can take an hour or more on a tablet and often fails. If `pkg` does not
have `python-pandas` for your Android version, the script says so and pip will
try to build it — that is the slow path, and Route A or C is the better answer.

To keep it running when you switch apps, run `termux-wake-lock` first.

*I could not test this route from where I built it — there is no Android device
here. The package choices are the standard ones for Termux, but treat the first
run as the real test.*

---

## Route C — A computer on your Wi-Fi

Any desktop, laptop, Raspberry Pi or old machine at home. It does not need to be
your laptop, and nothing is stored there that you cannot delete with `rm -rf`.

```bash
git clone -b claude/trading-platform-tablet-df45uw \
  https://github.com/Dlsdls121/Alpha-Trading.git
cd Alpha-Trading
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"

ALPHA_HOST=0.0.0.0 ./.venv/bin/python -m alpha.cli serve
```

Find that machine's local IP (`ip addr` on Linux, `ipconfig` on Windows,
`ifconfig | grep inet` on macOS) and open **http://THAT-IP:8000** on the tablet.
Both devices must be on the same Wi-Fi.

`ALPHA_HOST=0.0.0.0` is what makes it reachable from other devices. Without it
the server only answers on the machine itself.

---

## Switching to live NSE data

Wherever it is running:

```bash
export ALPHA_DATA_MODE=live
python -m alpha.data.nse selftest     # confirm NSE answers before trusting it
python -m alpha.cli serve
```

If a provider fails, the app does **not** silently fall back and pretend. It
falls back to fixtures and says so in the dashboard banner, in the CLI footer,
and in every signal's `data_quality`. An orange banner means you are looking at
simulated data.

---

## Using it without any server at all

The CLI gives you the same analysis as text, over SSH or directly in Termux:

```bash
python -m alpha.cli brief --explain      # everything, with full reasoning
python -m alpha.cli options --explain    # NIFTY / BANKNIFTY only
python -m alpha.cli equity --top 5       # positional candidates
python -m alpha.cli backtest --which both
```

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| Page loads, no data, orange banner | Running in fixture mode | `export ALPHA_DATA_MODE=live` |
| `selftest` gives 401/403 | NSE blocks that IP | Use Route B or C |
| `/` returns 404 with `looked_in` | Dashboard assets not found | Set `ALPHA_WEB_DIR` to the folder holding `index.html` |
| Can't reach it from the tablet | Server bound to localhost | Start with `ALPHA_HOST=0.0.0.0` |
| Termux: pandas compiles forever | No prebuilt package | Ctrl-C, use Route A or C |
| Cloud URL slow on first open | Free tier was asleep | Wait ~30s |

---

## One reminder

Everything here is analysis, not advice, and the signals have **not been
validated on real data** yet — the backtest harness exists and is tested, but no
run against real NSE history has happened. Paper-trade it and keep score before
any money is involved.
