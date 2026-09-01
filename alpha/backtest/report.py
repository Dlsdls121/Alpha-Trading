"""Backtest reporting.

Ordering is deliberate: the verdict comes first, the caveats come before the
numbers, and the numbers always carry their intervals. A reader who stops after
the first screen should come away with the correct impression, which for most
runs is "this does not yet show anything".
"""

from __future__ import annotations

from alpha.backtest.runner import BacktestResult

RULE = "=" * 78
THIN = "-" * 78


def _fmt_ci(ci: tuple[float, float], unit: str = "%") -> str:
    return f"{ci[0]:+.2f}{unit} to {ci[1]:+.2f}{unit}"


def render(result: BacktestResult, show_trades: int = 0) -> str:
    st, base, uni, v = (result.stats, result.baseline_hold,
                        result.baseline_universe, result.verdict)
    out: list[str] = []
    a = out.append

    title = ("INDEX OPTION BUYING" if result.kind == "index_option"
             else "POSITIONAL EQUITY")
    a(RULE)
    a(f"  BACKTEST - {title}")
    a(f"  {result.config_note}")
    a(RULE)

    # -- verdict first
    a("")
    a(f"  VERDICT: {v.conclusion.upper()}")
    a("")
    for line in _wrap(v.detail, 74):
        a(f"  {line}")

    # -- caveats before numbers
    if result.caveats:
        a("")
        a("  READ BEFORE THE NUMBERS")
        a(THIN)
        for c in result.caveats:
            wrapped = _wrap(c, 72)
            a(f"  * {wrapped[0]}")
            for cont in wrapped[1:]:
                a(f"    {cont}")

    # -- coverage
    a("")
    a("  COVERAGE")
    a(THIN)
    a(f"  Signals generated        {result.signals_generated}")
    a(f"  Stood aside (no trade)   {result.no_trade_count}")
    a(f"  Evaluated outcomes       {st.n_evaluable}")
    if result.signals_generated:
        rate = result.no_trade_count / result.signals_generated * 100
        a(f"  Stand-aside rate         {rate:.1f}%")

    if st.n_evaluable == 0:
        a("")
        a("  No evaluable outcomes. Nothing further to report.")
        a(RULE)
        return "\n".join(out)

    # -- results
    a("")
    a("  RESULTS (net of costs)")
    a(THIN)
    a(f"  Hit rate                 {st.hit_rate:.1f}%   95% CI "
      f"{st.hit_rate_ci[0]:.1f}% - {st.hit_rate_ci[1]:.1f}%")
    a(f"  Mean return / signal     {st.mean_return:+.2f}%   95% CI {_fmt_ci(st.mean_return_ci)}")
    a(f"  Median return            {st.median_return:+.2f}%")
    a(f"  Average win / loss       {st.avg_win:+.2f}% / {st.avg_loss:+.2f}%")
    a(f"  Profit factor            {st.profit_factor:.2f}")
    a(f"  Best / worst             {st.best:+.1f}% / {st.worst:+.1f}%")
    a(f"  Max drawdown (seq.)      {st.max_drawdown:+.1f}%")
    a(f"  Avg bars held            {st.avg_bars_held:.1f}")
    if st.directional_accuracy is not None:
        a(f"  Directional accuracy     {st.directional_accuracy:.1f}%   95% CI "
          f"{st.directional_ci[0]:.1f}% - {st.directional_ci[1]:.1f}%")
        if result.kind == "index_option":
            a("                           (this is the honest number - modelled option")
            a("                            P&L above assumes constant IV and flatters)")

    if st.exit_breakdown:
        a("")
        a("  HOW POSITIONS ENDED")
        a(THIN)
        total = sum(st.exit_breakdown.values())
        for reason, count in sorted(st.exit_breakdown.items(), key=lambda x: -x[1]):
            a(f"  {reason:<12} {count:>5}  ({count / total * 100:.1f}%)")

    # -- baselines
    a("")
    a("  BASELINES")
    a(THIN)
    if base.n_evaluable:
        a(f"  Buy & hold same instrument   {base.mean_return:+.2f}% per instance "
          f"(n={base.n_evaluable})")
    if uni and uni.n_evaluable:
        a(f"  Universe average             {uni.mean_return:+.2f}% per instance "
          f"(n={uni.n_evaluable})")
        edge = st.mean_return - uni.mean_return
        a(f"  Selection edge               {edge:+.2f}% per signal")
        a("  (Beating the universe average is the only evidence that picking helped.)")

    # -- attribution
    if result.factor_attribution:
        a("")
        a("  FACTOR ATTRIBUTION - correlation of factor score with realised return")
        a(THIN)
        a(f"  {'FACTOR':<18}{'N':>6}{'CORR':>9}   {'MEAN SCORE':>10}")
        for row in result.factor_attribution[:12]:
            a(f"  {row['factor']:<18}{row['n']:>6}{row['correlation']:>+9.3f}   "
              f"{row['mean_score']:>+10.3f}")
        a("")
        a("  These are extremely noisy at this sample size. Do NOT tune weights from")
        a("  them - fitting factor weights to a backtest is how a strategy is overfitted")
        a("  to its own history and then fails live.")

    if show_trades:
        a("")
        a(f"  SAMPLE TRADES (first {show_trades})")
        a(THIN)
        a(f"  {'DATE':<12}{'SYMBOL':<12}{'DIR':<7}{'RET':>9}{'EXIT':>10}{'BARS':>6}")
        for o in result.outcomes[:show_trades]:
            a(f"  {str(o.signal_date):<12}{o.symbol:<12}{o.direction:<7}"
              f"{o.net_return_pct:>+9.2f}{o.exit_reason:>10}{o.bars_held:>6}")

    a("")
    a(RULE)
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines
