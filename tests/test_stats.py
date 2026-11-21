import numpy as np
import pytest

from tiebreak.metrics import accuracy, neg_mae, neg_mae_rows
from tiebreak.stats import (
    equivalent_set, extra_examples_needed, guard_p, holm, mcnemar_exact, mean_guard_extra_examples, mean_guard_p,
    paired_bootstrap, upper_loss,
)


def preds_with_accuracy(y, acc, rng):
    wrong = rng.random(len(y)) > acc
    return np.where(wrong, (y + 1) % 3, y)


def test_bootstrap_is_deterministic_and_ci_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    preds = {"a": preds_with_accuracy(y, 0.9, rng), "b": preds_with_accuracy(y, 0.7, rng)}
    r1 = paired_bootstrap(y, preds, accuracy, n_boot=500, seed=3)
    r2 = paired_bootstrap(y, preds, accuracy, n_boot=500, seed=3)
    assert np.array_equal(r1.loss, r2.loss) and r1.best == "a"
    lo, hi = r1.ci["a"]
    assert lo <= r1.point["a"] <= hi


def test_clearly_worse_candidate_is_not_equivalent():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    preds = {"good": preds_with_accuracy(y, 0.95, rng), "bad": preds_with_accuracy(y, 0.6, rng)}
    boot = paired_bootstrap(y, preds, accuracy, n_boot=1000)
    eq = equivalent_set(boot, margin=0.02)
    assert eq == {"good": True, "bad": False}


def test_identical_predictions_are_equivalent_and_do_not_crash():
    y = np.array([0, 1, 2, 0, 1, 2] * 100)
    p = y.copy()
    boot = paired_bootstrap(y, {"a": p, "b": p.copy()}, accuracy, n_boot=300)
    assert equivalent_set(boot, margin=0.02) == {"a": True, "b": True}
    assert upper_loss(boot, "b") == 0.0


def test_single_candidate_is_trivially_equivalent():
    y = np.array([0, 1] * 10)
    boot = paired_bootstrap(y, {"only": y}, accuracy, n_boot=100)
    assert equivalent_set(boot, 0.02) == {"only": True}


def test_holm_step_down():
    assert holm([0.01, 0.04, 0.03], alpha=0.05) == [True, False, False]
    assert holm([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]


def test_mcnemar_exact():
    a = np.ones(10, dtype=bool)
    b = np.zeros(10, dtype=bool)
    assert mcnemar_exact(a, b) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(a, a) == 1.0


def test_extra_examples_needed():
    assert extra_examples_needed(100, point_loss=0.03, upper=0.05, margin=0.02) is None  # no slack
    assert extra_examples_needed(100, point_loss=0.0, upper=0.04, margin=0.02) == 300  # (0.04/0.02)^2*100 - 100
    assert extra_examples_needed(100, point_loss=0.0, upper=0.01, margin=0.02) == 0


def test_too_few_examples_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1]), {"a": np.array([1])}, accuracy)


def test_identical_predictions_are_not_tied_on_tiny_data():
    y = np.array([0, 1, 2, 0, 1] * 4)  # n=20
    boot = paired_bootstrap(y, {"a": y, "b": y.copy()}, accuracy, n_boot=300)
    assert equivalent_set(boot, margin=0.02) == {"a": True, "b": False}
    assert equivalent_set(boot, margin=0.02, guard=False) == {"a": True, "b": True}


def test_guard_p_shrinks_with_more_data_and_flags_real_gaps():
    def p(n):
        y = np.zeros(n, dtype=int)
        return guard_p(y, y, y, 0.02)
    assert p(600) < p(100) < p(20)
    y = np.zeros(300, dtype=int)
    bad = y.copy()
    bad[:90] = 1
    assert guard_p(y, y, bad, 0.02) > 0.99


def tile(n):
    """n judge-style scores (1 to 5) in a repeating pattern."""
    return np.tile(np.array([5.0, 4.0, 3.0, 4.0, 5.0]), n // 5)


def test_mean_guard_does_not_tie_identical_scores_on_a_tiny_sample():
    a = tile(5)
    y = np.full(5, 5.0)
    boot = paired_bootstrap(y, {"a": a, "b": a.copy()}, neg_mae, n_boot=300)
    assert mean_guard_p(a, a.copy(), 0.25) > 0.05
    assert equivalent_set(boot, 0.25, guard=False) == {"a": True, "b": True}  # the bootstrap alone cannot see the risk
    assert equivalent_set(boot, 0.25, guard=False, row_score=neg_mae_rows) == {"a": True, "b": False}


def test_mean_guard_ties_identical_scores_once_there_is_enough_data():
    a = tile(60)
    assert mean_guard_p(a, a.copy(), 0.25) <= 0.05
    boot = paired_bootstrap(np.full(60, 5.0), {"a": a, "b": a.copy()}, neg_mae, n_boot=300)
    assert equivalent_set(boot, 0.25, guard=False, row_score=neg_mae_rows) == {"a": True, "b": True}


def test_mean_guard_p_shrinks_with_more_data_and_flags_real_gaps():
    def p(n):
        a = tile(n)
        return mean_guard_p(a, a.copy(), 0.25)
    assert p(5) > p(10) > p(25) > p(60)
    a = tile(200)
    assert mean_guard_p(a, a - 0.5, 0.25) > 0.99  # the candidate is 0.5 worse on every prompt, beyond the margin


def test_mean_guard_extra_examples_is_none_zero_or_enough_to_pass():
    a = tile(40)
    assert mean_guard_extra_examples(a, a - 0.5, 0.25, 0.05) is None  # no slack: loss 0.5 exceeds the margin
    assert mean_guard_extra_examples(a, a.copy(), 0.25, 0.05) == 0  # the guard already passes
    small = tile(5)
    need = mean_guard_extra_examples(small, small.copy(), 0.25, 0.05)
    assert need >= 1
    big = tile(5 * -(-(5 + need) // 5))  # the same pattern repeated until there are 5 + need prompts
    assert mean_guard_p(big, big.copy(), 0.25) <= 0.06


def test_mean_guard_leaves_constant_scores_to_the_existing_warning():
    """A judge that scores everything 5 gives no variation to measure: the tie stays (the report warns about it)."""
    a = np.full(10, 5.0)
    assert mean_guard_p(a, a.copy(), 0.25) == 0.0


def test_guard_false_reproduces_plain_bootstrap_p_values():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    preds = {"good": preds_with_accuracy(y, 0.95, rng), "ok": preds_with_accuracy(y, 0.93, rng)}
    boot = paired_bootstrap(y, preds, accuracy, n_boot=500)
    j = boot.names.index("ok" if boot.best == "good" else "good")
    p = (1 + int(np.sum(boot.loss[:, j] > 0.02))) / (500 + 1)
    assert equivalent_set(boot, 0.02, guard=False)[boot.names[j]] == (p <= 0.05)
