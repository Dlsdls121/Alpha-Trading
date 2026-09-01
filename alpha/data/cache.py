"""Small disk cache with TTL.

NSE rate-limits and will start refusing a client that hammers it. Caching is
therefore not an optimisation here, it is what keeps the provider working.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".cache")


class DiskCache:
    def __init__(self, directory: Path | str = DEFAULT_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.dir / f"{safe}.json"

    def get(self, key: str, ttl_seconds: float) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - payload.get("_ts", 0) > ttl_seconds:
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(json.dumps({"_ts": time.time(), "value": value}))
        except (OSError, TypeError):
            pass          # a cache write failure must never break a signal

    def age_seconds(self, key: str) -> float | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return time.time() - json.loads(p.read_text()).get("_ts", 0)
        except (json.JSONDecodeError, OSError):
            return None
