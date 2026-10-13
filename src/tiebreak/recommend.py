from __future__ import annotations

from dataclasses import dataclass, field

from .stats import BootstrapResult, extra_examples_needed, upper_loss


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
) -> Recommendation:
    best = boot.best
    members = [m for m in boot.names if equiv[m]]
    winner = min(members, key=lambda m: (objective[m], m != best, m))

    inconclusive: list[str] = []
    extra: dict[str, int | None] = {}
    for m in boot.names:
        if equiv[m] or objective[m] >= objective[winner]:
            continue
        point_loss = boot.point[best] - boot.point[m]
        if point_loss <= margin:
            inconclusive.append(m)
            extra[m] = extra_examples_needed(boot.n, point_loss, upper_loss(boot, m), margin)

    if winner == best:
        base = f"{winner} has the highest score and no cheaper candidate is confirmed within {margin:g} of it."
    else:
        base = (
            f"{winner} is confirmed within {margin:g} of the best ({best}) and is the cheapest "
            f"candidate that is."
        )
    if inconclusive:
        hints = []
        for m in inconclusive:
            need = extra[m]
            hints.append(f"{m} (about {need} more labelled examples)" if need else f"{m} (margin too tight for this data)")
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
