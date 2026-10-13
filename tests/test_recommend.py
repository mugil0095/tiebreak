import numpy as np

from tiebreak.metrics import accuracy
from tiebreak.recommend import recommend
from tiebreak.stats import equivalent_set, paired_bootstrap


def run(preds, objective, y, margin=0.02, n_boot=600):
    boot = paired_bootstrap(y, preds, accuracy, n_boot=n_boot)
    return recommend(boot, equivalent_set(boot, margin), objective, margin), boot


def noisy(y, acc, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(len(y)) > acc, (y + 1) % 3, y)


def test_cheaper_tied_model_wins():
    y = np.random.default_rng(0).integers(0, 3, 400)
    p = noisy(y, 0.9, 1)
    rec, _ = run({"big": p, "small": p.copy()}, {"big": 10.0, "small": 1.0}, y)
    assert rec.status == "recommended" and rec.winner == "small" and rec.best in ("big", "small")
    assert "small" in rec.reason


def test_best_wins_when_cheaper_model_is_clearly_worse():
    y = np.random.default_rng(0).integers(0, 3, 400)
    rec, _ = run({"big": noisy(y, 0.95, 1), "small": noisy(y, 0.5, 2)}, {"big": 10.0, "small": 1.0}, y)
    assert rec.winner == "big" and rec.status == "recommended" and rec.inconclusive == []


def test_inconclusive_when_cheaper_model_is_close_but_data_is_small():
    y = np.random.default_rng(0).integers(0, 3, 60)
    big = noisy(y, 0.92, 1)
    small = big.copy()
    small[:2] = (small[:2] + 1) % 3  # ~3 points worse on 60 examples
    rec, _ = run({"big": big, "small": small}, {"big": 10.0, "small": 1.0}, y)
    assert rec.status == "inconclusive" and rec.winner == "big"
    assert rec.inconclusive == ["small"] and rec.extra_examples["small"] is None or rec.extra_examples["small"] >= 0
    assert "cannot separate" in rec.reason


def test_identical_predictions_pick_cheapest_without_crashing():
    y = np.array([0, 1, 2] * 30)
    rec, _ = run({"a": y.copy(), "b": y.copy(), "c": y.copy()}, {"a": 3.0, "b": 1.0, "c": 2.0}, y)
    assert rec.winner == "b" and rec.status == "recommended"


def test_ties_in_cost_prefer_the_best_scoring_model():
    y = np.array([0, 1, 2] * 30)
    good, worse = y.copy(), y.copy()
    worse[0] = 1
    rec, boot = run({"worse": worse, "good": good}, {"worse": 1.0, "good": 1.0}, y)
    assert rec.winner == "good"
