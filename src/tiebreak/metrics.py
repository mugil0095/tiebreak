from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import TaskConfig
from .textnorm import normalize


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    classification: bool


def accuracy(y, p) -> float:
    return float(np.mean(y == p))


def macro_f1(y, p) -> float:
    k = int(max(y.max(initial=-1), p.max(initial=-1))) + 1
    if k <= 0:
        return 0.0
    tp = np.bincount(y[y == p], minlength=k)
    pred = np.bincount(p[p >= 0], minlength=k)
    true = np.bincount(y[y >= 0], minlength=k)
    denom = pred + true
    present = denom > 0
    if not present.any():
        return 0.0
    f1 = 2 * tp / np.maximum(denom, 1)
    return float(f1[present].mean())


def neg_rmse(y, p) -> float:
    return -float(np.sqrt(np.mean((y - p) ** 2)))


def neg_mae(y, p) -> float:
    return -float(np.mean(np.abs(y - p)))


METRICS = {
    "accuracy": Metric("accuracy", accuracy, True),
    "macro_f1": Metric("macro_f1", macro_f1, True),
    "neg_rmse": Metric("neg_rmse", neg_rmse, False),
    "neg_mae": Metric("neg_mae", neg_mae, False),
}


def get_metric(name: str) -> Metric:
    if name not in METRICS:
        raise ValueError(f"unknown metric '{name}'; choose from {sorted(METRICS)}")
    return METRICS[name]


def encode_labels(task: TaskConfig, y, preds_by_name: dict):
    """Return (y_enc, {name: enc}) ready for the metric functions.

    Classification: int64 codes, -1 for invalid or missing predictions.
    Regression: floats; missing predictions are replaced by the mean of y.
    """
    if task.type == "regress":
        yf = np.asarray(y, dtype=float)
        fill = float(yf.mean())

        def enc_r(values):
            out = []
            for v in values:
                bad = v is None or (isinstance(v, float) and np.isnan(v))
                out.append(fill if bad else float(v))
            return np.array(out, dtype=float)

        return yf, {n: enc_r(p) for n, p in preds_by_name.items()}

    norm = normalize if task.type == "extract_exact" else str
    labels = task.labels or sorted({norm(v) for v in y})
    vocab = {norm(label): i for i, label in enumerate(labels)}

    def enc(values):
        return np.array(
            [-1 if v is None else vocab.get(norm(v), -1) for v in values], dtype=np.int64
        )

    return enc(y), {n: enc(p) for n, p in preds_by_name.items()}
