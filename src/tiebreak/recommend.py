from __future__ import annotations

import math
from dataclasses import dataclass, field

from .stats import BootstrapResult, discordance_counts, extra_examples_needed, guard_extra_examples, upper_loss


@dataclass
class Recommendation:
    status: str  # "recommended" | "inconclusive"
    winner: str
    best: str
    equivalent: list[str]
    inconclusive: list[str] = field(default_factory=list)
    extra_examples: dict[str, int | None] = field(default_factory=dict)
    reason: str = ""


def recommend(
    boot: BootstrapResult,
    equiv: dict[str, bool],
    objective: dict[str, float],
    margin: float,
    alpha: float = 0.05,
    guard: bool = True,
) -> Recommendation:
    best = boot.best
    k = len(boot.names)
    level = alpha / max(k - 1, 1)
    q = 1 - level  # same Holm-style adjustment as the equivalence test

    def cost(m: str) -> float:
        v = objective.get(m)
        return math.inf if v is None or math.isnan(v) else float(v)

    unknown = [m for m in boot.names if math.isinf(cost(m))]
    members = [m for m in boot.names if equiv[m]]
    winner = min(members, key=lambda m: (cost(m), -boot.point[m], m))

    inconclusive: list[str] = []
    extra: dict[str, int | None] = {}
    for m in boot.names:
        if equiv[m] or cost(m) >= cost(winner):
            continue
        point_loss = boot.point[best] - boot.point[m]
        if point_loss <= margin:
            inconclusive.append(m)
            need = extra_examples_needed(boot.n, point_loss, upper_loss(boot, m, q), margin)
            if need is not None and guard and boot.y is not None and boot.preds is not None:
                b, c, e = discordance_counts(boot.y, boot.preds[best], boot.preds[m])
                g = guard_extra_examples(b, c, e, margin, level, seed=boot.seed)
                need = None if g is None else max(need, g)
            extra[m] = None if need is None else max(need, 1)

    if winner == best:
        base = f"{winner} has the highest score and no cheaper candidate is confirmed within {margin:g} of it."
    else:
        base = (
            f"{winner} is confirmed within {margin:g} of the best ({best}) and is the cheapest "
            f"candidate that is."
        )
    if unknown:
        base += f" No usable cost for {', '.join(unknown)}; treated as most expensive."
    if inconclusive:
        hints = []
        for m in inconclusive:
            need = extra[m]
            hints.append(
                f"{m} (approximately {need} more labelled examples)"
                if need is not None
                else f"{m} (margin too tight for this data)"
            )
        base += " The data cannot separate these cheaper candidates from the best: " + ", ".join(hints) + "."
    return Recommendation(
        status="inconclusive" if inconclusive else "recommended",
        winner=winner,
        best=best,
        equivalent=members,
        inconclusive=inconclusive,
        extra_examples=extra,
        reason=base,
    )
