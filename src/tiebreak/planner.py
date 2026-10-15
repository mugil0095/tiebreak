from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates.llm import LLMCandidate
from .config import ModelSpec, TaskConfig
from .protocol import Split
from .stats import clearly_worse_p, paired_bootstrap

OUTPUT_TOKENS = 8


@dataclass
class Estimate:
    calls: dict[str, int]
    cost: dict[str, float | None]
    total: float
    notes: list[str] = field(default_factory=list)


class _Null:
    def complete(self, prompt):  # never called while estimating
        raise RuntimeError("estimate does not call models")


def estimate(task: TaskConfig, models: list[ModelSpec], split: Split, pricing: dict) -> Estimate:
    calls, cost, notes = {}, {}, []
    for spec in models:
        if spec.kind != "llm":
            continue
        cand = LLMCandidate(spec, task, _Null())
        built = cand.build_calls(split.scoring, split.shots)
        calls[spec.name] = len(built)
        price = pricing.get(spec.model)
        if price is None:
            cost[spec.name] = None
            if spec.provider != "mock":
                notes.append(f"{spec.name}: no price known for model '{spec.model}'; add it to pricing.yaml for a cost estimate")
            continue
        tokens_in = sum(len(c.prompt) // 4 for c in built)
        cost[spec.name] = tokens_in / 1000 * price[0] + len(built) * OUTPUT_TOKENS / 1000 * price[1]
    total = float(sum(v for v in cost.values() if v is not None))
    return Estimate(calls, cost, total, notes)


def clearly_worse(y_enc, preds_enc: dict, metric_fn, margin: float, eligible: set[str],
                  n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> set[str]:
    if len(preds_enc) < 2:
        return set()
    boot = paired_bootstrap(y_enc, preds_enc, metric_fn, n_boot=n_boot, seed=seed)
    return {
        name for name in eligible
        if name in preds_enc and name != boot.best and clearly_worse_p(boot, name, margin) < alpha
    }
