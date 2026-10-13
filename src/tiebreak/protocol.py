from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from .config import TaskConfig
from .data import DataError, Dataset


@dataclass
class Split:
    scoring: Dataset
    shots: Dataset | None


@dataclass(frozen=True)
class Fold:
    repeat: int
    train: np.ndarray
    test: np.ndarray


def make_split(ds: Dataset, task: TaskConfig) -> Split:
    p = task.shots_pool
    if p == 0:
        return Split(ds, None)
    if len(ds) - p < 10:
        raise DataError(f"shots_pool={p} leaves {len(ds) - p} scoring examples; need at least 10")
    perm = np.random.default_rng(task.seed).permutation(len(ds))
    return Split(ds.take(np.sort(perm[p:])), ds.take(np.sort(perm[:p])))


def make_folds(y: np.ndarray, task: TaskConfig) -> list[Fold]:
    n = len(y)
    if task.type in ("classify", "binary"):
        counts = pd.Series(y).value_counts()
        if counts.min() < 2:
            bad = counts.idxmin()
            raise DataError(
                f"class '{bad}' has only 1 example; need at least 2 per class for cross-validation"
            )
        k = int(min(task.cv_folds, counts.min()))
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=task.cv_repeats, random_state=task.seed)
        pairs = list(splitter.split(np.zeros(n), y))
    else:
        if n < task.cv_folds:
            raise DataError(f"need at least {task.cv_folds} examples for {task.cv_folds}-fold cross-validation")
        k = task.cv_folds
        splitter = RepeatedKFold(n_splits=k, n_repeats=task.cv_repeats, random_state=task.seed)
        pairs = list(splitter.split(np.zeros(n)))
    return [Fold(i // k, tr, te) for i, (tr, te) in enumerate(pairs)]
