from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .candidates.base import CandidateResult
from .config import ModelSpec, TaskConfig

PRICE_SOURCES = ("pricing.yaml", "provider", "unknown", "compute time")


@dataclass
class CostSummary:
    per_prediction: float
    train_cost: float
    total_at_volume: float
    per_1k: float
    price_known: bool
    price_source: str = "unknown"


def summarize_cost(result: CandidateResult, task: TaskConfig, price: tuple[float, float] | None = None) -> CostSummary:
    """Cost of one candidate. `price` is the user's (input, output) price per 1k tokens and, if given, wins over
    the provider-reported cost; local LLMs are costed from latency x compute price like ML candidates."""
    train = 0.0
    if result.kind == "llm":
        ok = [c for c in result.calls if c.error is None]  # failed calls carry no cost and say nothing about price
        if result.local_compute:
            lat = [c.latency_s for c in ok]
            per = float(np.mean(lat)) * task.compute_price_per_hour / 3600 if lat else 0.0
            price_known, source = bool(ok), "compute time"
        elif price is not None:
            costs = [c.tokens_in / 1000 * price[0] + c.tokens_out / 1000 * price[1] for c in ok]
            per, price_known, source = (float(np.mean(costs)) if costs else 0.0), bool(ok), "pricing.yaml"
        else:
            known = [c.cost_usd for c in ok if c.cost_usd is not None]
            price_known = bool(ok) and len(known) == len(ok)
            per = float(np.mean(known)) if known else 0.0
            source = "provider" if price_known else "unknown"
    else:
        per = result.infer_seconds_per_pred * task.compute_price_per_hour / 3600
        train = result.train_seconds * task.compute_price_per_hour / 3600
        price_known, source = True, "compute time"
    return CostSummary(per, train, train + per * task.volume, per * 1000, price_known, source)


def litellm_price(spec: ModelSpec) -> tuple[float, float] | None:
    """(input, output) price per 1k tokens from LiteLLM's table if litellm is importable, else None.
    Used by `estimate` only; a run uses pricing.yaml or the cost the provider reports."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # bundled table: no network access
    try:
        import litellm
        table = litellm.model_cost
    except Exception:
        return None
    for key in (f"{spec.provider}/{spec.model}", spec.model):
        entry = table.get(key) if isinstance(table, dict) else None
        if isinstance(entry, dict):
            tin, tout = entry.get("input_cost_per_token"), entry.get("output_cost_per_token")
            if isinstance(tin, (int, float)) and isinstance(tout, (int, float)):
                return float(tin) * 1000, float(tout) * 1000
    return None


def latency_stats(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))
