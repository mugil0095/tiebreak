import numpy as np
import pytest

from tiebreak.metrics import accuracy
from tiebreak.stats import (
    equivalent_set, extra_examples_needed, holm, mcnemar_exact, paired_bootstrap, upper_loss,
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
    y = np.array([0, 1, 2, 0, 1, 2] * 20)
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
