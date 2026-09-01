"""Terminal interface. Advisory output only -- this never places an order.

    python -m alpha.cli brief
    python -m alpha.cli options --symbols NIFTY,BANKNIFTY
    python -m alpha.cli equity --top 5 --explain
    python -m alpha.cli sectors
    python -m alpha.cli expiries
    python -m alpha.cli serve
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from datetime import date

from alpha.calendar import EXPIRY_RULES, expiry_context
from alpha.data import build_provider
from alpha.engines import equity_positional as eq
from alpha.engines import index_options as io
from alpha.models import Direction, Signal

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, BLUE = "\033[32m", "\033[31m", "\033[33m", "\033[34m"

DISCLAIMER = (
    "Advisory only. Heuristic signals, not predictions and not investment advice. "
    "No backtested edge is claimed. Verify independently before risking money."
)


def _colour(enabled: bool):
    if enabled and sys.stdout.isatty():
        return BOLD, DIM, RESET, GREEN, RED, YELLOW, BLUE
    return ("",) * 7


def _wrap(text: str, indent: str = "    ", width: int = 92) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def print_signal(sig: Signal, explain: bool, colour: bool = True) -> None:
    b, d, r, g, rd, y, bl = _colour(colour)
    tone = {Direction.LONG: g, Direction.SHORT: rd, Direction.NO_TRADE: y}[sig.direction]

    print(f"\n{b}{tone}{sig.headline}{r}")
    print(_wrap(sig.summary))

    if sig.leg:
        print(f"\n    {b}Contract:{r} {sig.leg.tradingsymbol} @ ~{sig.leg.ltp:,.2f}")
        print(_wrap(sig.leg.rationale, indent="      "))

    bits = []
    if sig.spot is not None:
        bits.append(f"spot {sig.spot:,.2f}")
    if sig.invalidation is not None:
        bits.append(f"invalid at {sig.invalidation:,.2f}")
    if sig.targets:
        bits.append("targets " + " / ".join(f"{t:,.2f}" for t in sig.targets))
    if sig.horizon and sig.horizon != "-":
        bits.append(f"horizon {sig.horizon}")
    if bits:
        print(f"    {d}{' | '.join(bits)}{r}")

    for v in sig.scorecard.vetoes:
        tag = f"{rd}BLOCKED{r}" if v.severity == "block" else f"{y}CAUTION{r}"
        print(f"\n    {tag} {b}{v.label}{r}")
        print(_wrap(v.detail, indent="      "))

    if explain:
        print(f"\n    {b}Why{r} {d}(weighted score {sig.scorecard.raw_score:+.2f}, "
              f"{sig.scorecard.agreement * 100:.0f}% agreement){r}")
        for f in sig.scorecard.factors:
            if f.weight == 0:
                mark = f"{bl}[context]{r}"
            elif f.verdict.sign > 0:
                mark = f"{g}[bullish]{r}"
            elif f.verdict.sign < 0:
                mark = f"{rd}[bearish]{r}"
            else:
                mark = f"{d}[neutral]{r}"
            print(f"      {mark} {b}{f.label}{r} {d}{f.value}{r}")
            print(_wrap(f.detail, indent="        "))


def cmd_options(args, provider) -> None:
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in syms:
        try:
            print_signal(io.build_signal(sym, provider, args.as_of), args.explain, not args.no_color)
        except Exception as exc:                            # noqa: BLE001
            print(f"\n  {sym}: could not build a signal - {exc}", file=sys.stderr)


def cmd_equity(args, provider) -> None:
    cfg = eq.EquityEngineConfig(top_n=args.top)
    sigs = eq.scan(provider, args.as_of, cfg, include_rejected=args.all)
    if not sigs:
        print("\n  Nothing currently clears the bar for a positional long. "
              "That is a result, not an error.")
        return
    for s in sigs:
        print_signal(s, args.explain, not args.no_color)


def cmd_sectors(args, provider) -> None:
    rows = eq.sector_table(provider, args.as_of)
    print(f"\n  {'#':<4}{'SECTOR':<22}{'3M RS':>9}   {'N':>3}  LEADERS")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['rank']:<4}{r['sector']:<22}{r['mean_rs_3m']:>+9.2f}   "
              f"{r['constituents']:>3}  {', '.join(r['leaders'])}")


def cmd_expiries(args, provider) -> None:
    today = args.as_of
    print(f"\n  {'SYMBOL':<12}{'CYCLE':<16}{'NEXT EXPIRY':<14}{'DAYS':>5}{'SESSIONS':>10}")
    print("  " + "-" * 60)
    for sym in EXPIRY_RULES:
        try:
            c = expiry_context(sym, today)
        except Exception:                                   # noqa: BLE001
            continue
        cycle = "weekly" if c.has_weekly else "monthly only"
        flag = "  <- expires today" if c.is_expiry_day else ""
        print(f"  {sym:<12}{cycle:<16}{c.expiry:%d-%b-%Y}{c.calendar_days:>7}"
              f"{c.trading_days:>10}{flag}")
    print(f"\n  {EXPIRY_RULES['BANKNIFTY'].note}")


def cmd_brief(args, provider) -> None:
    print(f"\n{'=' * 78}\n  MARKET BRIEF - {args.as_of:%d %b %Y}\n{'=' * 78}")
    print("\n--- INDEX OPTIONS " + "-" * 58)
    cmd_options(args, provider)
    print("\n\n--- POSITIONAL EQUITY " + "-" * 54)
    cmd_equity(args, provider)
    print("\n\n--- SECTOR ROTATION " + "-" * 56)
    cmd_sectors(args, provider)


def cmd_serve(args, provider) -> None:                      # pragma: no cover
    import uvicorn

    host, port = os.getenv("ALPHA_HOST", "0.0.0.0"), int(os.getenv("ALPHA_PORT", "8000"))
    print(f"\n  Dashboard: http://{host}:{port}/\n  Mode: {args.mode}\n")
    uvicorn.run("alpha.api:app", host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    # Shared flags are attached to the top-level parser *and* to every
    # subcommand, so `alpha brief --explain` and `alpha --explain brief` both
    # work. argparse otherwise demands globals precede the subcommand, which is
    # the opposite of what anyone types.
    # Every shared flag defaults to SUPPRESS. The subparser parses into the same
    # namespace as the top-level parser, so an ordinary default would let the
    # subcommand's default silently overwrite a flag given before it --
    # `alpha --explain options` would quietly lose --explain. With SUPPRESS the
    # attribute is set only when actually passed, and defaults are applied below.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=["fixture", "live"], default=argparse.SUPPRESS,
                        help="fixture = simulated offline data (default); live = NSE + Yahoo")
    common.add_argument("--as-of", type=date.fromisoformat, default=argparse.SUPPRESS,
                        metavar="YYYY-MM-DD")
    common.add_argument("--explain", action="store_true", default=argparse.SUPPRESS,
                        help="print the full factor breakdown")
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    p = argparse.ArgumentParser(
        prog="alpha", parents=[common],
        description="Advisory signal engine for NSE indices and equities. "
                    "Analysis only - it never places orders.")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("brief", parents=[common], help="everything in one view")

    o = sub.add_parser("options", parents=[common],
                       help="NIFTY / BANKNIFTY option-buying signals")
    o.add_argument("--symbols", default="NIFTY,BANKNIFTY")

    e = sub.add_parser("equity", parents=[common], help="positional equity candidates")
    e.add_argument("--top", type=int, default=5)
    e.add_argument("--all", action="store_true", help="include names that were passed over")

    sub.add_parser("sectors", parents=[common], help="sector leadership board")
    sub.add_parser("expiries", parents=[common], help="expiry rules and dates")
    sub.add_parser("serve", parents=[common], help="run the dashboard server")

    args = p.parse_args(argv)

    defaults = {
        "mode": os.getenv("ALPHA_DATA_MODE", "fixture"),
        "as_of": date.today(),
        "explain": False,
        "no_color": False,
        "verbose": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(levelname)s %(message)s")

    os.environ["ALPHA_DATA_MODE"] = args.mode
    provider = build_provider(args.mode)

    for attr in ("top", "all", "symbols"):
        if not hasattr(args, attr):
            setattr(args, attr, {"top": 5, "all": False,
                                 "symbols": "NIFTY,BANKNIFTY"}[attr])

    {"brief": cmd_brief, "options": cmd_options, "equity": cmd_equity,
     "sectors": cmd_sectors, "expiries": cmd_expiries, "serve": cmd_serve}[args.cmd](args, provider)

    if args.cmd != "serve":
        if not provider.is_live:
            print(f"\n  {YELLOW}! Simulated data - these are illustrations, not tradeable "
                  f"calls. Use --mode live for real NSE data.{RESET}")
        print(f"\n  {DIM}{DISCLAIMER}{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
