from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .candidates.base import CallRecord
from .candidates.llm import Call


class CostCapReached(RuntimeError):
    """The spend limit was hit; completed work is saved and the run can be resumed."""


class CostCap:
    def __init__(self, limit: float | None):
        self.limit = limit
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, amount: float | None) -> None:
        with self._lock:
            self.spent += amount or 0.0

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.spent >= self.limit


class PredictionStore:
    """Append-only JSONL of every provider call; the single source for caching and resume."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[tuple, CallRecord] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = CallRecord(**json.loads(line))
                    self._records[(rec.candidate, rec.example_id, rec.repeat)] = rec

    def get(self, key) -> CallRecord | None:
        rec = self._records.get(key)
        return rec if rec is not None and rec.error is None else None

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self._records[(rec.candidate, rec.example_id, rec.repeat)] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")

    def for_candidate(self, name: str) -> list[CallRecord]:
        return [r for k, r in self._records.items() if k[0] == name]


_SECRET = re.compile(
    r"(?:(?:api[_-]?key|authorization)\s*[:=]\s*(?:(?:Bearer|Basic)\s+)?|(?:Bearer|Basic)\s+)\S+", re.I
)


def _scrub(text: str) -> str:
    """Never let credentials reach the prediction store through an exception message."""
    return _SECRET.sub("[redacted]", text)


def _price_cost(price, tokens_in: int, tokens_out: int) -> float | None:
    if price is None:
        return None
    return tokens_in / 1000 * price[0] + tokens_out / 1000 * price[1]


def execute_calls(
    calls: list[Call], completer, store: PredictionStore, *,
    price=None, cap: CostCap | None = None, concurrency: int = 4,
    retries: int = 3, backoff: float = 1.0, sleep=time.sleep,
) -> list[CallRecord]:
    cap = cap or CostCap(None)
    results: dict[int, CallRecord] = {}
    todo = []
    for i, c in enumerate(calls):
        rec = store.get(c.key)
        if rec is not None:
            results[i] = rec
        else:
            todo.append((i, c))

    def work(item):
        i, c = item
        if cap.exceeded:
            return i, None
        last = ""
        for attempt in range(retries + 1):
            try:
                comp = completer.complete(c.prompt)
                cost = comp.cost_usd if comp.cost_usd is not None else _price_cost(price, comp.tokens_in, comp.tokens_out)
                rec = CallRecord(c.key[0], c.key[1], c.key[2], comp.text, comp.tokens_in,
                                 comp.tokens_out, cost, comp.latency_s, None)
                cap.add(cost)
                store.add(rec)
                return i, rec
            except Exception as e:  # provider errors are data, not crashes
                last = _scrub(f"{type(e).__name__}: {e}")
                if attempt < retries:
                    sleep(backoff * 2**attempt)
        rec = CallRecord(c.key[0], c.key[1], c.key[2], "", 0, 0, None, 0.0, last)
        store.add(rec)
        return i, rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for i, rec in pool.map(work, todo):
            if rec is not None:
                results[i] = rec
    if len(results) < len(calls):
        raise CostCapReached(f"cost cap reached after spending about ${cap.spent:.4f}; resume to continue")
    return [results[i] for i in range(len(calls))]
