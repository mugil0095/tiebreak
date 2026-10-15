from __future__ import annotations

from dataclasses import dataclass, field

from .candidates.llm import LLMCandidate
from .config import ModelSpec, TaskConfig
from .protocol import Split
from .stats import clearly_worse_p, discordance_counts, paired_bootstrap

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
    calls, cost, notes, unpriced = {}, {}, [], False
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
                unpriced = True
                notes.append(f"{spec.name}: no price known for model '{spec.model}'; add it to pricing.yaml for a cost estimate")
            continue
        tokens_in = sum(len(c.prompt) // 4 for c in built)
        cost[spec.name] = tokens_in / 1000 * price[0] + len(built) * OUTPUT_TOKENS / 1000 * price[1]
    if unpriced:
        notes.append("Total covers priced models only.")
    total = float(sum(v for v in cost.values() if v is not None))
    return Estimate(calls, cost, total, notes)


def clearly_worse(y_enc, preds_enc: dict, metric_fn, margin: float, eligible: set[str],
                  n_boot: int = 1000, alpha: float = 0.01, seed: int = 0,
                  min_n: int = 50, min_discordant: int = 10, loss_factor: float = 2.0) -> set[str]:
    """Names in `eligible` that are clearly worse than the best. Deliberately strict: eliminating wrongly
    hides a model's full result, so this needs a large sample, a Bonferroni-corrected p, a loss of at
    least `loss_factor` * margin, and enough discordant examples (a 3-0 split is not evidence)."""
    n = len(y_enc)
    tested = [m for m in eligible if m in preds_enc]
    if n < min_n or len(preds_enc) < 2 or not tested:
        return set()
    boot = paired_bootstrap(y_enc, preds_enc, metric_fn, n_boot=n_boot, seed=seed)
    tested = [m for m in tested if m != boot.best]
    if not tested:
        return set()
    level = alpha / max(1, len(tested))
    out = set()
    for name in tested:
        if clearly_worse_p(boot, name, margin) >= level:
            continue
        if boot.point[boot.best] - boot.point[name] < max(loss_factor * margin, 5 / n):
            continue
        b, c, _ = discordance_counts(y_enc, preds_enc[boot.best], preds_enc[name])
        if b + c >= min_discordant:
            out.add(name)
    return out
