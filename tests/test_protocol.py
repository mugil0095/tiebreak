import numpy as np
import pytest

from tiebreak.config import TaskConfig
from tiebreak.data import DataError, Dataset
from tiebreak.protocol import make_folds, make_split
import pandas as pd


def ds(n, labels=("a", "b")):
    y = np.array([labels[i % len(labels)] for i in range(n)], dtype=object)
    return Dataset([f"r{i}" for i in range(n)], pd.DataFrame({"x": range(n)}), y, None)


def task(**kw):
    return TaskConfig(type="classify", label_column="y", labels=["a", "b"], **kw)


def test_no_shots_pool_returns_everything_as_scoring():
    s = make_split(ds(40), task())
    assert len(s.scoring) == 40 and s.shots is None


def test_shots_pool_is_disjoint_from_scoring():
    s = make_split(ds(40), task(shots_pool=8))
    assert len(s.shots) == 8 and len(s.scoring) == 32
    assert not set(s.shots.ids) & set(s.scoring.ids)


def test_shots_pool_too_large_rejected():
    with pytest.raises(DataError, match="at least 10"):
        make_split(ds(15), task(shots_pool=10))


def test_folds_cover_each_example_once_per_repeat():
    d = ds(60)
    folds = make_folds(d.y, task(cv_folds=5, cv_repeats=2))
    assert len(folds) == 10
    for r in (0, 1):
        tests = np.concatenate([f.test for f in folds if f.repeat == r])
        assert sorted(tests) == list(range(60))


def test_folds_reduce_k_when_a_class_is_small():
    y = np.array(["a"] * 20 + ["b"] * 3, dtype=object)
    folds = make_folds(y, task(cv_folds=5, cv_repeats=1))
    assert len(folds) == 3  # k reduced to the smallest class count


def test_single_example_class_gives_clear_error():
    y = np.array(["a"] * 20 + ["b"], dtype=object)
    with pytest.raises(DataError, match="only 1 example"):
        make_folds(y, task())


def test_regression_folds_need_enough_rows():
    t = TaskConfig(type="regress", label_column="y", cv_folds=5)
    with pytest.raises(DataError, match="at least"):
        make_folds(np.arange(3, dtype=float), t)
