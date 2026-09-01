"""Scan universe and sector grouping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_UNIVERSE_FILE = Path(__file__).parent / "reference" / "universe.json"


@dataclass(frozen=True)
class Constituent:
    symbol: str
    name: str
    sector: str


@dataclass
class Universe:
    constituents: list[Constituent]
    benchmark: str = "NIFTY"
    verified_through: date | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Universe":
        raw = json.loads((path or _UNIVERSE_FILE).read_text())
        vt = raw.get("verified_through")
        return cls(
            constituents=[Constituent(c["symbol"], c["name"], c["sector"])
                          for c in raw["constituents"]],
            benchmark=raw.get("benchmark", "NIFTY"),
            verified_through=date.fromisoformat(vt) if vt else None,
        )

    @property
    def symbols(self) -> list[str]:
        return [c.symbol for c in self.constituents]

    def sector_of(self, symbol: str) -> str:
        for c in self.constituents:
            if c.symbol == symbol.upper():
                return c.sector
        return "Unknown"

    def name_of(self, symbol: str) -> str:
        for c in self.constituents:
            if c.symbol == symbol.upper():
                return c.name
        return symbol.upper()

    def by_sector(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c in self.constituents:
            out.setdefault(c.sector, []).append(c.symbol)
        return out
