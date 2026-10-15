from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

from .candidates.base import CallRecord
from .candidates.llm import Call, hash_text

log = logging.getLogger("tiebreak.runner")

_RECORD_FIELDS = {f.name for f in fields(CallRecord)}
MAX_ERROR_CHARS = 300


class CostCapReached(RuntimeError):
    """The spend limit was hit; completed work is saved and the run can be resumed."""


class CostUnknownError(RuntimeError):
    """A cost cap is set but a call's cost cannot be determined, so the cap could never trigger."""


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
        self._totals: Counter = Counter()  # spend per candidate over EVERY line ever seen, superseded ones included
        self._needs_newline = False
        self.skipped_lines = 0
        if self.path.exists():
            raw = self.path.read_bytes()
            self._needs_newline = bool(raw) and not raw.endswith(b"\n")
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    rec = CallRecord(**{k: v for k, v in data.items() if k in _RECORD_FIELDS})
                except (ValueError, TypeError, AttributeError):
                    self.skipped_lines += 1
                    continue
                self._records[(rec.candidate, rec.example_id, rec.repeat)] = rec
                self._totals[rec.candidate] += rec.cost_usd or 0.0
            if self.skipped_lines:
                log.warning("%s: skipped %d unreadable line(s) (e.g. a run killed mid-write); those calls will be re-run",
                            self.path, self.skipped_lines)

    def lookup(self, key) -> CallRecord | None:
        """Latest record for key, successful or not."""
        with self._lock:
            return self._records.get(key)

    def get(self, key) -> CallRecord | None:
        rec = self.lookup(key)
        return rec if rec is not None and rec.error is None else None

    def add(self, rec: CallRecord) -> None:
        line = json.dumps(asdict(rec)) + "\n"
        with self._lock:
            with open(self.path, "ab") as f:
                if self._needs_newline:
                    f.write(b"\n")  # never fuse a new record onto a partial line
                    self._needs_newline = False
                f.write(line.encode("utf-8"))
                f.flush()
            self._records[(rec.candidate, rec.example_id, rec.repeat)] = rec
            self._totals[rec.candidate] += rec.cost_usd or 0.0

    def total_cost(self, name: str) -> float:
        """Everything ever spent on a candidate in this store, including records a config change superseded."""
        with self._lock:
            return float(self._totals[name])

    def valid_for(self, calls: list[Call]) -> list[CallRecord]:
        """Stored records that answer exactly these calls: same key AND same prompt/config hash.
        Anything else (old repeats, old prompt, old config) must never be scored."""
        out = []
        for c in calls:
            rec = self.lookup(c.key)
            if rec is not None and rec.prompt_hash == hash_text(c.prompt) and rec.config_hash == c.config_hash:
                out.append(rec)
        return out

    def for_candidate(self, name: str) -> list[CallRecord]:
        with self._lock:
            return [r for k, r in self._records.items() if k[0] == name]


_SECRET_PATTERNS = [
    re.compile(r"(?:api[_-]?key|authorization)\s*[:=]\s*(?:(?:Bearer|Basic)\s+)?\S+", re.I),
    re.compile(r"""["'](?:api[_-]?key|authorization|token)["']\s*:\s*["'][^"']*["']""", re.I),
    re.compile(r"Incorrect API key provided:\s*\S+", re.I),
    re.compile(r"(?:Bearer|Basic)\s+\S+", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"""(?<=[?&])(?:api_)?key=[^&\s"']+""", re.I),
]


def _scrub(text: str, secrets: list[str] | None = None) -> str:
    """Never let credentials reach the prediction store through an exception message."""
    for s in sorted((s for s in (secrets or []) if s), key=len, reverse=True):
        text = text.replace(s, "[redacted]")
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text[:MAX_ERROR_CHARS]


def _retryable(e: Exception) -> bool:
    if getattr(e, "retryable", True) is False:
        return False
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is None:
        status = getattr(e, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return False
    return True


def _is_rate_limit(e: Exception) -> bool:
    status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
    return status == 429 or "RateLimit" in type(e).__name__


RATE_LIMIT_RETRIES = 6      # rate limits clear with time, so they get more attempts than other errors
RATE_LIMIT_WAIT = 8.0       # seconds before the first rate-limit retry; doubles each time, capped below
RATE_LIMIT_MAX_WAIT = 60.0


def _price_cost(price, tokens_in: int, tokens_out: int) -> float | None:
    if price is None:
        return None
    return tokens_in / 1000 * price[0] + tokens_out / 1000 * price[1]


def execute_calls(
    calls: list[Call], completer, store: PredictionStore, *,
    price=None, cap: CostCap | None = None, concurrency: int = 4,
    retries: int = 3, backoff: float = 1.0, sleep=time.sleep,
    secrets: list[str] | None = None, seed_from_store: bool = True,
    min_interval: float = 0.0, clock=time.monotonic,
) -> list[CallRecord]:
    cap = cap or CostCap(None)
    stop = threading.Event()
    results: dict[int, CallRecord] = {}
    todo = []
    stale: Counter = Counter()
    for i, c in enumerate(calls):
        rec = store.get(c.key)
        if rec is not None and rec.prompt_hash == hash_text(c.prompt) and rec.config_hash == c.config_hash:
            results[i] = rec
        else:
            if rec is not None:
                stale[c.key[0]] += 1
            todo.append((i, c))
    for name, n in stale.items():
        log.warning("%d cached answers for '%s' were produced with a different prompt/config and were re-run", n, name)

    if cap.limit is not None and seed_from_store:  # the cap is cumulative across resumes
        cap.add(sum(store.total_cost(name) for name in {c.key[0] for c in calls}))

    pace_lock, next_ok = threading.Lock(), [0.0]

    def pace():
        """Space call starts by min_interval seconds (a per-minute request limit), across all threads."""
        if not min_interval:
            return
        with pace_lock:
            now = clock()
            wait = max(0.0, next_ok[0] - now)
            next_ok[0] = max(now, next_ok[0]) + min_interval
        if wait:
            sleep(wait)

    def attempt_call(c: Call):
        last = ""
        attempt = 0
        while True:
            pace()
            try:
                return completer.complete(c.prompt), ""
            except Exception as e:  # provider errors are data, not crashes
                last = _scrub(f"{type(e).__name__}: {e}", secrets)
                limited = _is_rate_limit(e)
                if attempt >= (max(retries, RATE_LIMIT_RETRIES) if limited else retries) or not _retryable(e):
                    break
                sleep(min(RATE_LIMIT_MAX_WAIT, max(backoff, RATE_LIMIT_WAIT) * 2**attempt) if limited
                      else backoff * 2**attempt)
                attempt += 1
        return None, last

    def work(item):
        i, c = item
        if stop.is_set() or cap.exceeded:
            return i, None
        try:
            comp, err = attempt_call(c)
            ph = hash_text(c.prompt)
            if comp is None:
                rec = CallRecord(c.key[0], c.key[1], c.key[2], "", 0, 0, None, 0.0, err, ph, c.config_hash)
                store.add(rec)
                return i, rec
            # the user's pricing.yaml wins; the provider-reported cost is the fallback
            cost = _price_cost(price, comp.tokens_in, comp.tokens_out) if price is not None else comp.cost_usd
            rec = CallRecord(c.key[0], c.key[1], c.key[2], comp.text, comp.tokens_in,
                             comp.tokens_out, cost, comp.latency_s, None, ph, c.config_hash)
            cap.add(cost)
            store.add(rec)  # a failure here propagates; the paid call is never repeated
            if cap.limit is not None and cost is None:
                raise CostUnknownError(
                    f"--max-cost is set but the cost of model {c.key[0]} is unknown; "
                    "add it to pricing.yaml or remove --max-cost"
                )
            return i, rec
        except BaseException:
            stop.set()  # stops queued calls from spending, including on Ctrl-C
            raise

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        for i, rec in pool.map(work, todo):
            if rec is not None:
                results[i] = rec
    except BaseException:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if len(results) < len(calls):
        raise CostCapReached(f"cost cap reached after spending about ${cap.spent:.4f}; resume to continue")
    return [results[i] for i in range(len(calls))]
