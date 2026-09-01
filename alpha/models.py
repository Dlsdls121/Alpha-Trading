"""Core signal vocabulary.

The design rule for this whole project: a signal is never a bare number or a
bare direction. It is a *scorecard* -- an ordered list of independent factors,
each of which was computed from real data, states its own verdict, and explains
itself in plain language using the actual values it saw.

The dashboard renders those factors directly. Nothing in the explanation is
written after the fact to justify a conclusion that was reached some other way;
the conclusion is the arithmetic of the factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Which way a single piece of evidence points."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return {"bullish": 1, "bearish": -1, "neutral": 0}[self.value]


class Direction(str, Enum):
    """The final call."""

    LONG = "long"           # buy calls / buy the stock
    SHORT = "short"         # buy puts
    NO_TRADE = "no_trade"   # conditions do not justify a position


class Category(str, Enum):
    """Groups factors on the dashboard so the reasoning reads in sections."""

    TREND = "trend"
    MOMENTUM = "momentum"
    POSITIONING = "positioning"   # open interest, PCR, max pain
    VOLATILITY = "volatility"     # IV, VIX, IV percentile
    COST = "cost"                 # theta, days to expiry, spread
    LIQUIDITY = "liquidity"
    RELATIVE = "relative"         # relative strength, sector rotation
    STRUCTURE = "structure"       # 52w position, base, support/resistance


@dataclass
class Factor:
    """One piece of evidence.

    ``score`` is the factor's own opinion in [-1, +1] where +1 is maximally
    bullish. ``weight`` is how much that opinion counts. ``detail`` must quote
    the numbers the factor actually saw -- it is the audit trail.
    """

    key: str
    label: str
    category: Category
    verdict: Verdict
    score: float
    weight: float
    detail: str
    value: str = ""            # short display value, e.g. "RSI 62.4"
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(-1.0, min(1.0, float(self.score)))
        if self.weight < 0:
            raise ValueError(f"factor {self.key!r} has negative weight")

    @property
    def contribution(self) -> float:
        """Signed points this factor pushes into the total."""
        return self.score * self.weight


@dataclass
class Veto:
    """A hard block. Vetoes are not outvoted by a strong score.

    Option buying has genuine kill conditions -- paying a rich premium into an
    IV crush, or buying decay on expiry day -- where being directionally right
    still loses money. Those are vetoes, not negative points.
    """

    key: str
    label: str
    detail: str
    severity: str = "block"    # "block" kills the signal, "warn" is advisory


@dataclass
class Scorecard:
    """Accumulates factors, then aggregates them into a direction + conviction."""

    factors: list[Factor] = field(default_factory=list)
    vetoes: list[Veto] = field(default_factory=list)

    def add(self, factor: Factor) -> Factor:
        self.factors.append(factor)
        return factor

    def veto(self, key: str, label: str, detail: str, severity: str = "block") -> None:
        self.vetoes.append(Veto(key=key, label=label, detail=detail, severity=severity))

    # -- aggregation -----------------------------------------------------

    @property
    def blocking_vetoes(self) -> list[Veto]:
        return [v for v in self.vetoes if v.severity == "block"]

    @property
    def warnings(self) -> list[Veto]:
        return [v for v in self.vetoes if v.severity == "warn"]

    @property
    def total_weight(self) -> float:
        return sum(f.weight for f in self.factors)

    @property
    def raw_score(self) -> float:
        """Weighted mean of factor scores, in [-1, +1]."""
        tw = self.total_weight
        if tw == 0:
            return 0.0
        return sum(f.contribution for f in self.factors) / tw

    @property
    def agreement(self) -> float:
        """Fraction of directional weight that agrees with the majority side.

        Ten factors all mildly bullish is a better setup than five screaming
        bullish and five screaming bearish, even when the mean is identical.
        Conviction is discounted when the factors disagree.
        """
        bull = sum(f.weight for f in self.factors if f.verdict is Verdict.BULLISH)
        bear = sum(f.weight for f in self.factors if f.verdict is Verdict.BEARISH)
        directional = bull + bear
        if directional == 0:
            return 0.0
        return max(bull, bear) / directional

    def direction(self, threshold: float = 0.15) -> Direction:
        if self.blocking_vetoes:
            return Direction.NO_TRADE
        score = self.raw_score
        if score >= threshold:
            return Direction.LONG
        if score <= -threshold:
            return Direction.SHORT
        return Direction.NO_TRADE

    def conviction(self) -> int:
        """0-100. Magnitude of the call, discounted by internal disagreement.

        This is deliberately NOT a probability. It has no calibration behind it
        and must not be read as "72% chance of being right".
        """
        if self.blocking_vetoes:
            return 0
        base = abs(self.raw_score)                       # 0..1
        conv = base * (0.55 + 0.45 * self.agreement)     # disagreement discount
        conv *= max(0.0, 1.0 - 0.10 * len(self.warnings))
        return int(round(max(0.0, min(1.0, conv)) * 100))

    def by_category(self) -> dict[str, list[Factor]]:
        out: dict[str, list[Factor]] = {}
        for f in self.factors:
            out.setdefault(f.category.value, []).append(f)
        return out

    def top_reasons(self, n: int = 3) -> list[Factor]:
        """The factors that moved the needle most, for the summary line."""
        ranked = sorted(self.factors, key=lambda f: abs(f.contribution), reverse=True)
        return [f for f in ranked if abs(f.contribution) > 1e-9][:n]


@dataclass
class OptionLeg:
    """A concrete instrument the signal points at. Advisory only -- describing
    a contract is not the same as trading it, and nothing in this codebase
    can send an order."""

    symbol: str
    expiry: date
    strike: float
    option_type: str          # "CE" or "PE"
    ltp: float | None = None
    delta: float | None = None
    iv: float | None = None
    oi: int | None = None
    volume: int | None = None
    rationale: str = ""

    @property
    def tradingsymbol(self) -> str:
        return f"{self.symbol} {int(self.strike)} {self.option_type} {self.expiry:%d-%b-%Y}"


@dataclass
class Signal:
    """What the dashboard shows for one instrument."""

    signal_id: str
    kind: str                       # "index_option" | "equity_positional"
    symbol: str
    generated_at: datetime
    direction: Direction
    conviction: int
    headline: str
    summary: str
    scorecard: Scorecard

    spot: float | None = None
    horizon: str = ""               # "intraday", "3-5 sessions", ...
    entry_zone: tuple[float, float] | None = None
    invalidation: float | None = None
    targets: list[float] = field(default_factory=list)
    leg: OptionLeg | None = None
    invalidated_by: list[str] = field(default_factory=list)
    data_quality: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generated_at"] = self.generated_at.isoformat()
        d["direction"] = self.direction.value
        if self.leg is not None:
            d["leg"]["expiry"] = self.leg.expiry.isoformat()
            d["leg"]["tradingsymbol"] = self.leg.tradingsymbol
        d["scorecard"]["raw_score"] = round(self.scorecard.raw_score, 4)
        d["scorecard"]["agreement"] = round(self.scorecard.agreement, 4)
        d["scorecard"]["top_reasons"] = [f.key for f in self.scorecard.top_reasons()]
        for fd, f in zip(d["scorecard"]["factors"], self.scorecard.factors):
            fd["category"] = f.category.value
            fd["verdict"] = f.verdict.value
            fd["contribution"] = round(f.contribution, 4)
        return d
