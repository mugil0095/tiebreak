import numpy as np
import pandas as pd
import pytest

from tiebreak.candidates.ml import MLCandidate, to_numeric_features
from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import DataError, Dataset, load_dataset
from tiebreak.demo import make_reviews
from tiebreak.protocol import make_folds, make_split


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
    assert ds.frame["x"].iloc[0] == f"{x[0]:.4f}"  # frame stays the original strings
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), task).run(
        split, make_folds(split.scoring.y, task))
    assert np.mean([p_ == t for p_, t in zip(res.predictions, split.scoring.y)]) > 0.9


def test_zip_codes_stay_categorical_and_frame_is_lossless(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("\n".join(["id,zip,age,y", "r0,01234,34,a", "r1,98765,40,b", "r2,01234,35,a", ""]), encoding="utf-8")
    task = TaskConfig(type="binary", label_column="y", labels=["a", "b"])
    ds = load_dataset(p, task)
    assert ds.frame["zip"].tolist() == ["01234", "98765", "01234"] and ds.frame["age"].tolist() == ["34", "40", "35"]
    out = to_numeric_features(ds.frame, None)
    assert out["zip"].tolist() == ["01234", "98765", "01234"]  # leading zero -> categorical
    assert out["age"].dtype.kind == "f"


@pytest.mark.parametrize("bad", ["inf", "nan", "-inf", "+5", "007", "1234567890123456"])
def test_non_plain_numbers_keep_column_categorical(bad):
    out = to_numeric_features(pd.DataFrame({"c": ["1", bad, "2"]}), None)
    assert out["c"].tolist() == ["1", bad, "2"]


def test_scientific_notation_negatives_blanks_parse_numeric():
    out = to_numeric_features(pd.DataFrame({"c": ["1e3", "-2.5", " ", None, "0.5", "0"]}), None)
    assert out["c"].dtype.kind == "f"
    assert out["c"].iloc[0] == 1000.0 and out["c"].iloc[1] == -2.5 and np.isnan(out["c"].iloc[2]) and np.isnan(out["c"].iloc[3])


def test_all_blank_column_dropped_text_untouched_mixed_stays_string():
    df = pd.DataFrame({"t": ["1", "2"], "blank": ["", " "], "mix": ["1", "two"], "n": ["3", "4"]})
    out = to_numeric_features(df, "t")
    assert list(out.columns) == ["t", "mix", "n"] and out["t"].tolist() == ["1", "2"]
    assert out["mix"].tolist() == ["1", "two"] and out["n"].dtype.kind == "f"


def test_only_blank_columns_gives_no_feature_columns_error():
    ds = Dataset(["a", "b", "c", "d"], pd.DataFrame({"z": ["", "", "", ""]}), np.array(["x", "y", "x", "y"], dtype=object), None)
    task = TaskConfig(type="binary", label_column="y", labels=["x", "y"], cv_folds=2, cv_repeats=1)
    split = make_split(ds, task)
    with pytest.raises(DataError, match="no feature columns"):
        MLCandidate(ModelSpec(name="x", kind="ml", estimator="logistic_regression"), task).run(
            split, make_folds(split.scoring.y, task))


def test_run_input_errors_are_clear():
    ds = reviews_dataset(30)
    split = make_split(ds, TASK)
    cand = MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), TASK)
    with pytest.raises(ValueError, match="no cross-validation folds"):
        cand.run(split, [])
    folds = make_folds(split.scoring.y, TASK)
    narrow = TaskConfig(type="classify", label_column="sentiment", text_column="review", labels=["negative", "positive"])
    with pytest.raises(ValueError, match="not in task.labels"):
        MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), narrow).run(split, folds)
    no_text = TaskConfig(type="classify", label_column="sentiment", text_column="nope", labels=TASK.labels)
    with pytest.raises(DataError, match="text column"):
        MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), no_text).run(split, folds)


def test_regression_repeat_disagreement_is_undefined():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    ds = Dataset([f"r{i}" for i in range(60)], pd.DataFrame(x, columns=["a", "b"]), x[:, 0], None)
    task = TaskConfig(type="regress", label_column="y", cv_folds=3, cv_repeats=2)
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="ridge", kind="ml", estimator="ridge"), task).run(split, make_folds(split.scoring.y, task))
    assert res.repeat_disagreement is None


@pytest.mark.parametrize("bad", ["two", "007"])
def test_column_type_decided_once_not_per_fold(bad):
    n = 60
    vals = [str(i % 7 + 1) for i in range(n)]
    vals[n - 1] = bad  # lands in only one held-out fold per repeat
    y = np.array(["a", "b"] * (n // 2), dtype=object)
    ds = Dataset([f"r{i}" for i in range(n)], pd.DataFrame({"c": vals}), y, None)
    task = TaskConfig(type="binary", label_column="y", labels=["a", "b"], cv_folds=3, cv_repeats=1)
    split = make_split(ds, task)
    res = MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), task).run(
        split, make_folds(split.scoring.y, task))
    assert len(res.predictions) == n
    from tiebreak.candidates import ml  # noqa
    seen = []
    orig = ml._preprocessor

    def spy(frame, text_col):
        seen.append(pd.api.types.is_numeric_dtype(frame["c"]))
        return orig(frame, text_col)

    ml._preprocessor = spy
    try:
        MLCandidate(ModelSpec(name="lr", kind="ml", estimator="logistic_regression"), task).run(
            split, make_folds(split.scoring.y, task))
    finally:
        ml._preprocessor = orig
    assert seen and not any(seen)  # categorical in every fold


def _boost_run(est, task, ds):
    pytest.importorskip(est)
    split = make_split(ds, task)
    return split, MLCandidate(ModelSpec(name=est, kind="ml", estimator=est), task).run(
        split, make_folds(split.scoring.y, task))


@pytest.mark.parametrize("est", ["xgboost", "lightgbm"])
def test_boosted_estimators_classify_text(est):
    split, res = _boost_run(est, TASK, reviews_dataset(300))
    assert all(p in TASK.labels for p in res.predictions)
    assert np.mean([p == y for p, y in zip(res.predictions, split.scoring.y)]) > 0.9


@pytest.mark.parametrize("est", ["xgboost", "lightgbm"])
def test_boosted_estimators_use_numeric_and_categorical_features(est):
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    color = rng.choice(["r", "g"], n)
    y = np.where(x + (color == "r") > 0.5, "yes", "no").astype(object)
    ds = Dataset([f"r{i}" for i in range(n)], pd.DataFrame({"x": x, "color": color}), y, None)
    task = TaskConfig(type="binary", label_column="y", labels=["no", "yes"], cv_folds=3, cv_repeats=1)
    split, res = _boost_run(est, task, ds)
    assert np.mean([p == t for p, t in zip(res.predictions, split.scoring.y)]) > 0.8


@pytest.mark.parametrize("est", ["xgboost", "lightgbm"])
def test_boosted_estimators_regress(est):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    y = x[:, 0] * 2 + x[:, 1]
    ds = Dataset([f"r{i}" for i in range(100)], pd.DataFrame(x, columns=["a", "b"]), y, None)
    task = TaskConfig(type="regress", label_column="y", cv_folds=3, cv_repeats=1)
    split, res = _boost_run(est, task, ds)
    assert all(isinstance(p, float) for p in res.predictions)
    assert np.corrcoef(res.predictions, split.scoring.y.astype(float))[0, 1] > 0.9


def _without_neutral():
    df = make_reviews(90, 0)
    df = df[df["sentiment"] != "neutral"]  # 'neutral' stays in task.labels but never appears in the data
    return Dataset(df["id"].tolist(), df[["review"]].reset_index(drop=True),
                   df["sentiment"].to_numpy(dtype=object), df["review"].tolist())


def test_xgboost_label_missing_from_training_folds_gives_a_clear_error():
    with pytest.raises(ValueError, match=r"no training example for label\(s\) \['neutral'\]") as e:
        _boost_run("xgboost", TASK, _without_neutral())
    assert "Expected" not in str(e.value)  # not the raw xgboost message


def test_lightgbm_trains_when_a_label_is_missing_from_training_folds():
    _, res = _boost_run("lightgbm", TASK, _without_neutral())
    assert set(res.predictions) <= {"negative", "positive"}
