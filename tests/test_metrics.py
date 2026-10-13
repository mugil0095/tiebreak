import numpy as np
import pytest
from sklearn.metrics import f1_score

from tiebreak.config import TaskConfig
from tiebreak.metrics import encode_labels, get_metric
from tiebreak.textnorm import normalize


def test_normalize():
    assert normalize("  **Positive.** ") == "positive"
    assert normalize('"Hello   World"') == "hello world"


def test_accuracy():
    m = get_metric("accuracy")
    assert m.fn(np.array([0, 1, 1]), np.array([0, 1, 0])) == pytest.approx(2 / 3)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 4, 200))
    assert get_metric("macro_f1").fn(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_macro_f1_invalid_predictions_count_as_wrong():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, -1, 1, -1])
    # class 0: tp=1 pred=1 true=2 -> 2/3 ; class 1 likewise -> 2/3
    assert get_metric("macro_f1").fn(y, p) == pytest.approx(2 / 3)


def test_regression_metrics_are_higher_is_better():
    y = np.array([1.0, 2.0, 3.0])
    assert get_metric("neg_rmse").fn(y, y) == 0.0
    assert get_metric("neg_rmse").fn(y, y + 1) == pytest.approx(-1.0)
    assert get_metric("neg_mae").fn(y, y + 2) == pytest.approx(-2.0)


def test_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        get_metric("nope")


def test_encode_classification_marks_invalid_as_minus_one():
    task = TaskConfig(type="classify", label_column="y", labels=["a", "b"])
    y, enc = encode_labels(task, np.array(["a", "b"], dtype=object), {"m": ["a", None], "n": ["zzz", "b"]})
    assert list(y) == [0, 1] and list(enc["m"]) == [0, -1] and list(enc["n"]) == [-1, 1]


def test_encode_extract_exact_normalises_text():
    task = TaskConfig(type="extract_exact", label_column="y")
    y, enc = encode_labels(task, np.array(["Paris", "Rome"], dtype=object), {"m": [" paris. ", "Berlin"]})
    assert list(enc["m"]) == [list(y)[0], -1]


def test_encode_regression_fills_missing_with_mean():
    task = TaskConfig(type="regress", label_column="y")
    y, enc = encode_labels(task, np.array([1.0, 3.0]), {"m": [2.0, None]})
    assert list(enc["m"]) == [2.0, 2.0]
