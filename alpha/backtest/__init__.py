"""Backtesting: does any of this reasoning actually work?

Read ``BacktestResult.caveats`` before reading any number. The limitations there
are not boilerplate -- survivorship bias, synthetic option chains and constant-IV
modelling each move results in a knowable direction, and all three flatter.
"""

from alpha.backtest.evaluate import CostModel, Outcome, evaluate
from alpha.backtest.metrics import Stats, Verdict, summarise, verdict, wilson_interval
from alpha.backtest.replay import (
    SYNTHETIC_CHAIN_FACTORS, HistoryStore, PointInTimeProvider,
)
from alpha.backtest.runner import BacktestConfig, BacktestResult, run_equity, run_options

__all__ = [
    "CostModel", "Outcome", "evaluate", "Stats", "Verdict", "summarise", "verdict",
    "wilson_interval", "HistoryStore", "PointInTimeProvider", "SYNTHETIC_CHAIN_FACTORS",
    "BacktestConfig", "BacktestResult", "run_equity", "run_options",
]
