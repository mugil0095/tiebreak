from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binomtest


@dataclass
class BootstrapResult:
    names: list[str]
    n: int
    n_boot: int
    point: dict[str, float]
    ci: dict[str, tuple[float, float]]
    loss: np.ndarray  # (n_boot, k): resample-best score minus candidate score
    best: str


def paired_bootstrap(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 10_000,
    seed: int = 0,
    chunk: int = 500,
) -> BootstrapResult:
    names = list(preds)
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 examples to bootstrap")
    point = {m: float(metric_fn(y, preds[m])) for m in names}
    rng = np.random.default_rng(seed)
    scores = np.empty((n_boot, len(names)))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        for r in range(b):
            ii = idx[r]
            yy = y[ii]
            for j, m in enumerate(names):
                scores[done + r, j] = metric_fn(yy, preds[m][ii])
        done += b
    loss = scores.max(axis=1, keepdims=True) - scores
    ci = {
        m: (float(np.percentile(scores[:, j], 2.5)), float(np.percentile(scores[:, j], 97.5)))
        for j, m in enumerate(names)
    }
    best = max(names, key=lambda m: point[m])
    return BootstrapResult(names, n, n_boot, point, ci, loss, best)


def holm(pvals, alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(np.argsort(pvals)):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def equivalent_set(boot: BootstrapResult, margin: float, alpha: float = 0.05) -> dict[str, bool]:
    """A candidate is equivalent to the best when, with Holm-adjusted confidence,
    its loss relative to the best is no more than `margin`."""
    out = {boot.best: True}
    others = [j for j, m in enumerate(boot.names) if m != boot.best]
    if not others:
        return out
    B = boot.loss.shape[0]
    pvals = [(1 + int(np.sum(boot.loss[:, j] > margin))) / (B + 1) for j in others]
    for j, rej in zip(others, holm(pvals, alpha)):
        out[boot.names[j]] = bool(rej)
    return out


def upper_loss(boot: BootstrapResult, name: str, q: float = 0.95) -> float:
    j = boot.names.index(name)
    return float(np.quantile(boot.loss[:, j], q))


def clearly_worse_p(boot: BootstrapResult, name: str, margin: float) -> float:
    """Share of resamples in which `name` is within `margin` of the best (small = clearly worse)."""
    j = boot.names.index(name)
    return float(np.mean(boot.loss[:, j] <= margin))


def extra_examples_needed(n: int, point_loss: float, upper: float, margin: float) -> int | None:
    """Rough number of additional examples to confirm equivalence (interval width ~ 1/sqrt(n))."""
    slack = margin - point_loss
    if slack <= 0:
        return None
    width = max(upper - point_loss, 1e-12)
    return max(math.ceil(n * (width / slack) ** 2) - n, 0)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)
