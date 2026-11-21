from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binomtest, t as student_t


GUARD_PRIOR = 2.0  # Dirichlet pseudo-count per cell; chosen by the false-tie experiment (0.5 gave 11% at n=100)


@dataclass
class BootstrapResult:
    names: list[str]
    n: int
    n_boot: int
    point: dict[str, float]
    ci: dict[str, tuple[float, float]]
    loss: np.ndarray  # (n_boot, k): resample-best score minus candidate score
    best: str
    y: np.ndarray | None = None  # encoded labels, kept for the discordance guard
    preds: dict[str, np.ndarray] | None = None
    seed: int = 0  # bootstrap seed, reused by the discordance guard


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
    return BootstrapResult(names, n, n_boot, point, ci, loss, best, np.asarray(y), {m: np.asarray(preds[m]) for m in names}, seed)


def holm(pvals, alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(np.argsort(pvals)):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def discordance_counts(y: np.ndarray, best_pred: np.ndarray, cand_pred: np.ndarray) -> tuple[int, int, int]:
    """(b, c, e): b = best right & cand wrong, c = best wrong & cand right, e = the rest."""
    y = np.asarray(y)
    a = np.asarray(best_pred) == y
    c_ok = np.asarray(cand_pred) == y
    b = int(np.sum(a & ~c_ok))
    c = int(np.sum(~a & c_ok))
    return b, c, len(y) - b - c


def _guard_p_counts(
    b: float, c: float, e: float, margin: float, n_draws: int, prior: float | None, seed: int
) -> float:
    prior = GUARD_PRIOR if prior is None else prior
    rng = np.random.default_rng(seed)
    d = rng.dirichlet([b + prior, c + prior, e + prior], size=n_draws)
    loss = d[:, 0] - d[:, 1]
    return (1 + int(np.sum(loss > margin))) / (n_draws + 1)


def guard_p(
    y: np.ndarray,
    best_pred: np.ndarray,
    cand_pred: np.ndarray,
    margin: float,
    n_draws: int = 20_000,
    prior: float | None = None,
    seed: int = 0,
) -> float:
    """Bayesian discordance guard: P(loss > margin) for a classification candidate.

    The paired bootstrap only resamples observed discordant pairs, so it has no
    variance when two candidates rarely disagree. Here b = #(best right, cand wrong),
    c = #(best wrong, cand right), e = the rest; (p_b, p_c, p_e) ~ Dirichlet(b+k, c+k, e+k), k = GUARD_PRIOR
    and loss = p_b - p_c. Returns (1 + #(loss > margin)) / (n_draws + 1).
    """
    b, c, e = discordance_counts(y, best_pred, cand_pred)
    return _guard_p_counts(b, c, e, margin, n_draws, prior, seed)


GUARD_MAX_MULT = 100.0  # the extra-examples search stops at this multiple of the current data


def guard_extra_examples(
    b: int, c: int, e: int, margin: float, level: float, seed: int = 0, max_mult: float = GUARD_MAX_MULT
) -> int | None:
    """Extra examples (same discordance rates, counts scaled by m = n'/n) until the guard p <= level.

    None when the observed loss already exceeds the margin (no slack); 0 when the guard already
    passes; if the search cap (100x the data) is hit, returns the cap's extra examples.
    """
    n = b + c + e
    if (b - c) / n > margin:
        return None
    if _guard_p_counts(b, c, e, margin, 20_000, None, seed) <= level:
        return 0
    m = 1.25
    while True:
        m = min(m, max_mult)
        if _guard_p_counts(b * m, c * m, e * m, margin, 20_000, None, seed) <= level:
            return math.ceil(n * m) - n
        if m >= max_mult:
            return math.ceil(n * max_mult) - n
        m *= 1.25


GUARD_DF = 2.0  # pseudo-degrees of freedom of the variance prior in mean_guard_p (cf. GUARD_PRIOR above)


def _mean_guard_p_from(dbar: float, s2: float, s0: float, n: int, margin: float, prior_df: float) -> float:
    df = prior_df + n - 1
    var = (prior_df * s0 + (n - 1) * s2) / df
    se = math.sqrt(var / n)
    if se == 0:  # no variation at all (e.g. a judge that scores everything the same): nothing to measure
        return 0.0 if dbar < margin else 1.0
    return float(student_t.cdf((dbar - margin) / se, df))


def mean_guard_p(
    best_scores: np.ndarray, cand_scores: np.ndarray, margin: float, prior_df: float | None = None
) -> float:
    """t-test guard for metrics that are the mean of a per-example score (neg_mae, hence judge scores).

    Like the discordance guard for classification, it covers a weakness of the bootstrap: when two
    candidates score almost the same on a small sample, the resampled loss has almost no variance.
    One-sided paired t-test of H0: mean(best - candidate) >= margin, where the variance of the
    differences is blended with a prior of GUARD_DF pseudo-degrees of freedom at the pooled variance
    of the scores themselves, so a handful of identical scores is not evidence of a tie.
    Small p = confirmed within the margin. Constant scores (no variation anywhere) are left tied.
    """
    prior_df = GUARD_DF if prior_df is None else prior_df
    best = np.asarray(best_scores, dtype=float)
    cand = np.asarray(cand_scores, dtype=float)
    n = len(best)
    if n < 2:
        return 1.0
    d = best - cand
    s0 = float(np.var(np.concatenate([best, cand]), ddof=1))
    return _mean_guard_p_from(float(d.mean()), float(d.var(ddof=1)), s0, n, margin, prior_df)


def mean_guard_extra_examples(
    best_scores: np.ndarray, cand_scores: np.ndarray, margin: float, level: float, max_mult: float = GUARD_MAX_MULT
) -> int | None:
    """Extra examples (same mean and variances, n scaled by m) until mean_guard_p <= level.

    None when the observed loss already reaches the margin (no slack); 0 when the guard already
    passes; if the search cap (100x the data) is hit, returns the cap's extra examples.
    """
    best = np.asarray(best_scores, dtype=float)
    cand = np.asarray(cand_scores, dtype=float)
    n = len(best)
    d = best - cand
    if n < 2 or float(d.mean()) >= margin:
        return None
    dbar, s2 = float(d.mean()), float(d.var(ddof=1))
    s0 = float(np.var(np.concatenate([best, cand]), ddof=1))

    def p_at(n_new: int) -> float:
        return _mean_guard_p_from(dbar, s2, s0, n_new, margin, GUARD_DF)

    if p_at(n) <= level:
        return 0
    m = 1.25
    while True:
        m = min(m, max_mult)
        n_new = math.ceil(n * m)
        if p_at(n_new) <= level:
            return n_new - n
        if m >= max_mult:
            return math.ceil(n * max_mult) - n
        m *= 1.25


def equivalent_set(
    boot: BootstrapResult, margin: float, alpha: float = 0.05, guard: bool = True,
    row_score: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, bool]:
    """A candidate is equivalent to the best when, with Holm-adjusted confidence,
    its loss relative to the best is no more than `margin`.

    guard=True (classification metrics) takes p = max(p_bootstrap, p_guard) so that
    candidates which almost never disagree with the best are not tied by default.
    guard=False with a row_score (the per-example score whose mean is the metric: neg_mae,
    hence judge scores) takes p = max(p_bootstrap, p_mean_guard) in the same way.
    guard=False without a row_score uses the bootstrap alone (neg_rmse), with the
    limitation that near-identical predictions then have no resampling variance and
    are declared equivalent regardless of sample size. The engine passes
    guard=metric.classification and row_score=metric.row_score.
    """
    out = {boot.best: True}
    others = [j for j, m in enumerate(boot.names) if m != boot.best]
    if not others:
        return out
    B = boot.loss.shape[0]
    pvals = [(1 + int(np.sum(boot.loss[:, j] > margin))) / (B + 1) for j in others]
    have_data = boot.y is not None and boot.preds is not None
    if guard and have_data:
        pvals = [
            max(p, guard_p(boot.y, boot.preds[boot.best], boot.preds[boot.names[j]], margin, seed=boot.seed))
            for p, j in zip(pvals, others)
        ]
    elif row_score is not None and have_data:
        best_s = row_score(boot.y, boot.preds[boot.best])
        pvals = [
            max(p, mean_guard_p(best_s, row_score(boot.y, boot.preds[boot.names[j]]), margin))
            for p, j in zip(pvals, others)
        ]
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
