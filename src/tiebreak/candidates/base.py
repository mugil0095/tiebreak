from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CallRecord:
    candidate: str
    example_id: str
    repeat: int
    raw: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    latency_s: float
    error: str | None = None
    prompt_hash: str = ""
    config_hash: str = ""


@dataclass
class CandidateResult:
    name: str
    kind: str  # "llm" | "ml"
    predictions: list
    calls: list[CallRecord] = field(default_factory=list)
    train_seconds: float = 0.0
    infer_seconds_per_pred: float = 0.0
    latencies: list[float] = field(default_factory=list)
    invalid_rate: float = 0.0
    error_rate: float = 0.0
    repeat_disagreement: float = 0.0
    eliminated: bool = False
    mask: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    n_unanswered: int = 0  # examples whose every call failed (no successful answer)
    local_compute: bool = False  # local model: costed from latency x compute price, not per call


def aggregate(votes: list, classification: bool):
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    if classification:
        counts = Counter(votes)
        top = max(counts.values())
        return next(v for v in votes if counts[v] == top)  # ties -> first seen
    return float(np.mean(votes))
