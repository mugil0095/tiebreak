import numpy as np

from tiebreak.stats import BootstrapResult

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
    assert rec.inconclusive == ["small"]
    need = rec.extra_examples["small"]
    assert need is None or need >= 1
    assert "cannot separate" in rec.reason


def test_identical_predictions_pick_cheapest_without_crashing():
    y = np.array([0, 1, 2] * 200)
    rec, _ = run({"a": y.copy(), "b": y.copy(), "c": y.copy()}, {"a": 3.0, "b": 1.0, "c": 2.0}, y)
    assert rec.winner == "b" and rec.status == "recommended"


def test_ties_in_cost_prefer_the_best_scoring_model():
    y = np.array([0, 1, 2] * 200)
    good, worse = y.copy(), y.copy()
    worse[0] = 1
    rec, boot = run({"worse": worse, "good": good}, {"worse": 1.0, "good": 1.0}, y)
    assert rec.winner == "good"


def test_holm_blocked_candidate_gets_close_message_not_margin_too_tight():
    # k=4: the unadjusted 95% bound (0.01) is within the margin, the Holm-level bound (0.03) is not.
    col = np.array([0.01] * 980 + [0.03] * 20)
    loss = np.zeros((1000, 4))
    loss[:, 3] = col
    boot = BootstrapResult(
        names=["best", "a", "b", "c"], n=100, n_boot=1000,
        point={"best": 0.9, "a": 0.9, "b": 0.9, "c": 0.89},
        ci={}, loss=loss, best="best",
    )
    equiv = {"best": True, "a": True, "b": True, "c": False}
    rec = recommend(boot, equiv, {"best": 5.0, "a": 4.0, "b": 4.0, "c": 1.0}, 0.02)
    assert rec.status == "inconclusive" and rec.inconclusive == ["c"]
    need = rec.extra_examples["c"]
    assert 299 <= need <= 301  # (0.02/0.01)^2 * 100 - 100, up to float rounding
    assert f"approximately {need}" in rec.reason and "margin too tight" not in rec.reason


def test_missing_or_nan_objective_is_most_expensive():
    y = np.array([0, 1, 2] * 200)
    rec, _ = run({"a": y.copy(), "b": y.copy(), "c": y.copy()}, {"a": float("nan"), "b": 2.0}, y)
    assert rec.winner == "b" and "No usable cost for a" in rec.reason


def test_guard_blocked_tie_reports_realistic_extra_examples():
    y = np.array([0, 1, 2, 0] * 25)  # n=100, identical predictions
    rec, boot = run({"big": y.copy(), "small": y.copy()}, {"big": 10.0, "small": 1.0}, y)
    assert rec.status == "inconclusive" and rec.inconclusive == ["small"]
    need = rec.extra_examples["small"]
    assert need >= 80
    assert f"approximately {need}" in rec.reason
    # adding that many identical examples lets the guard pass
    reps = -(-(100 + need) // 4)
    y2 = np.array([0, 1, 2, 0] * reps)
    rec2, _ = run({"big": y2.copy(), "small": y2.copy()}, {"big": 10.0, "small": 1.0}, y2)
    assert rec2.status == "recommended" and rec2.winner == "small"


def test_extra_examples_beyond_100x_the_data_are_not_reported_as_a_count():
    y = np.array([0, 1, 2] * 100)
    small = y.copy()
    small[:6] = (small[:6] + 1) % 3  # loss 0.02 against a margin of 0.0201: no realistic amount of data separates them
    rec, _ = run({"big": y.copy(), "small": small}, {"big": 10.0, "small": 1.0}, y, margin=0.0201)
    assert rec.status == "inconclusive" and rec.extra_examples["small"] >= 29700
    assert "more than 100x the current data" in rec.reason and "approximately" not in rec.reason
