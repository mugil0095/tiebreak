from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates.base import CandidateResult
from .config import TaskConfig


@dataclass
class CostSummary:
    per_prediction: float
    train_cost: float
    total_at_volume: float
    per_1k: float
    price_known: bool


def summarize_cost(result: CandidateResult, task: TaskConfig) -> CostSummary:
    if result.kind == "llm":
        known = [c.cost_usd for c in result.calls if c.cost_usd is not None]
        price_known = bool(result.calls) and len(known) == len(result.calls)
        per = float(np.mean(known)) if known else 0.0
        train = 0.0
    else:
        per = result.infer_seconds_per_pred * task.compute_price_per_hour / 3600
        train = result.train_seconds * task.compute_price_per_hour / 3600
        price_known = True
    return CostSummary(per, train, train + per * task.volume, per * 1000, price_known)


def latency_stats(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))
