import numpy as np
import pytest

from tiebreak.candidates.ml import ESTIMATORS, MLCandidate
from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import DataError, Dataset, load_dataset
from tiebreak.demo import make_reviews
from tiebreak.protocol import make_folds, make_split

import pandas as pd


def reviews_dataset(n=150, seed=0):
    df = make_reviews(n, seed)
    return Dataset(df["id"].tolist(), df[["review"]].reset_index(drop=True), df["sentiment"].to_numpy(dtype=object), df["review"].tolist())


TASK = TaskConfig(type="classify", label_column="sentiment", text_column="review",
                  labels=["negative", "neutral", "positive"], cv_folds=3, cv_repeats=2)


def test_logistic_regression_learns_text_task():
    ds = reviews_dataset()
    split = make_split(ds, TASK)
    res = MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), TASK).run(
        split, make_folds(split.scoring.y, TASK))
    acc = np.mean([p == y for p, y in zip(res.predictions, split.scoring.y)])
    assert acc > 0.9 and res.kind == "ml" and len(res.predictions) == len(split.scoring)
    assert res.train_seconds > 0 and res.infer_seconds_per_pred > 0 and res.latencies


@pytest.mark.parametrize("est", ["random_forest", "hist_gradient_boosting"])
def test_other_estimators_run(est):
    ds = reviews_dataset(90)
    split = make_split(ds, TASK)
    res = MLCandidate(ModelSpec(name=est, kind="ml", estimator=est), TASK).run(
        split, make_folds(split.scoring.y, TASK))
    assert all(p in TASK.labels for p in res.predictions)


def test_numeric_and_categorical_features():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    color = rng.choice(["r", "g"], n)
    y = np.where(x + (color == "r") > 0.5, "yes", "no").astype(object)
    ds = Dataset([f"r{i}" for i in range(n)], pd.DataFrame({"x": x, "color": color}), y, None)
    task = TaskConfig(type="binary", label_column="y", labels=["no", "yes"], cv_folds=3, cv_repeats=1)
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="rf", kind="ml", estimator="random_forest"), task).run(
        split, make_folds(split.scoring.y, task))
    assert np.mean([p == t for p, t in zip(res.predictions, split.scoring.y)]) > 0.8


def test_regression_ridge():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    y = x[:, 0] * 2 + x[:, 1]
    ds = Dataset([f"r{i}" for i in range(100)], pd.DataFrame(x, columns=["a", "b"]), y, None)
    task = TaskConfig(type="regress", label_column="y", cv_folds=3, cv_repeats=1)
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="ridge", kind="ml", estimator="ridge"), task).run(
        split, make_folds(split.scoring.y, task))
    assert np.sqrt(np.mean((np.array(res.predictions) - y) ** 2)) < 0.2


def test_unknown_estimator_lists_choices():
    with pytest.raises(ValueError, match="unknown estimator.*logistic_regression"):
        MLCandidate(ModelSpec(name="x", kind="ml", estimator="nope"), TASK)


def test_extract_exact_tasks_reject_ml_candidates():
    task = TaskConfig(type="extract_exact", label_column="y")
    with pytest.raises(ValueError, match="extract_exact"):
        MLCandidate(ModelSpec(name="x", kind="ml", estimator="logistic_regression"), task)


def test_no_usable_feature_columns_is_a_clear_error():
    ds = Dataset(["a", "b", "c", "d"], pd.DataFrame(index=range(4)), np.array(["x", "y", "x", "y"], dtype=object), None)
    task = TaskConfig(type="binary", label_column="y", labels=["x", "y"], cv_folds=2, cv_repeats=1)
    split = make_split(ds, task)
    with pytest.raises(DataError, match="no feature columns"):
        MLCandidate(ModelSpec(name="x", kind="ml", estimator="logistic_regression"), task).run(
            split, make_folds(split.scoring.y, task))


def test_csv_numeric_column_trains_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    x = rng.normal(size=n)
    lines = ["id,x,y"] + [f"r{i},{x[i]:.4f},{'yes' if x[i] > 0 else 'no'}" for i in range(n)]
    p = tmp_path / "d.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    task = TaskConfig(type="binary", label_column="y", labels=["no", "yes"], cv_folds=3, cv_repeats=1)
    ds = load_dataset(p, task)
    assert pd.api.types.is_float_dtype(ds.frame["x"])
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), task).run(
        split, make_folds(split.scoring.y, task))
    assert np.mean([p_ == t for p_, t in zip(res.predictions, split.scoring.y)]) > 0.9
