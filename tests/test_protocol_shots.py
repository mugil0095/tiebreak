import numpy as np
import pandas as pd

from tiebreak.config import TaskConfig
from tiebreak.data import Dataset
from tiebreak.protocol import make_split


def _ds(y):
    n = len(y)
    return Dataset([f"r{i}" for i in range(n)], pd.DataFrame({"t": [f"x{i}" for i in range(n)]}),
                   np.asarray(y, dtype=object), [f"x{i}" for i in range(n)])


def _task(**kw):
    base = dict(type="classify", label_column="y", text_column="t", labels=["a", "b", "c"], shots_pool=6, seed=0)
    base.update(kw)
    return TaskConfig(**base)


def test_shots_pool_covers_every_class():
    y = ["a"] * 80 + ["b"] * 15 + ["c"] * 5
    for seed in range(5):
        split = make_split(_ds(y), _task(seed=seed))
        assert set(split.shots.y) == {"a", "b", "c"}
        assert len(split.shots) == 6 and len(split.scoring) == 94


def test_shots_pool_keeps_two_of_each_class_for_scoring():
    y = ["a"] * 30 + ["b"] * 3 + ["c"] * 3
    split = make_split(_ds(y), _task(shots_pool=9))
    counts = pd.Series(split.scoring.y).value_counts()
    assert counts.min() >= 2 and len(split.shots) == 9


def test_shots_pool_is_deterministic_and_disjoint_from_scoring():
    y = ["a"] * 40 + ["b"] * 40 + ["c"] * 20
    one, two = make_split(_ds(y), _task()), make_split(_ds(y), _task())
    assert one.shots.ids == two.shots.ids
    assert not set(one.shots.ids) & set(one.scoring.ids)


def test_regression_shots_pool_is_still_random():
    y = list(np.arange(50, dtype=float))
    task = _task(type="regress", labels=None, shots_pool=5)
    split = make_split(_ds(y), task)
    assert len(split.shots) == 5 and len(split.scoring) == 45
