from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import ModelSpec, TaskConfig
from ..data import DataError, concat
from ..protocol import Fold, Split
from .base import CandidateResult, aggregate

ESTIMATORS = sorted(
    ["logistic_regression", "random_forest", "hist_gradient_boosting", "ridge", "xgboost", "lightgbm"]
)


def _make_estimator(name: str, classification: bool, seed: int):
    if name == "logistic_regression" and classification:
        return LogisticRegression(max_iter=1000)
    if name == "ridge" and not classification:
        return Ridge()
    if name == "random_forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(n_estimators=200, random_state=seed)
    if name == "hist_gradient_boosting":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        return cls(random_state=seed)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is not installed; run: pip install tiebreak[boost]") from e
        cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
        return cls(n_estimators=200, random_state=seed, verbosity=0)
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm is not installed; run: pip install tiebreak[boost]") from e
        cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        return cls(n_estimators=200, random_state=seed, verbose=-1)
    kind = "classification" if classification else "regression"
    raise ValueError(f"estimator '{name}' is not available for {kind}; choose from {ESTIMATORS}")


def _preprocessor(frame: pd.DataFrame, text_col: str | None) -> ColumnTransformer:
    parts = []
    if text_col and text_col in frame.columns:
        parts.append(("text", TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), text_col))
    others = [c for c in frame.columns if c != text_col]
    num = [c for c in others if pd.api.types.is_numeric_dtype(frame[c])]
    cat = [c for c in others if c not in num]
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    if not parts:
        raise DataError("no feature columns for ML candidates; add a text column or other columns")
    return ColumnTransformer(parts, sparse_threshold=0)


class MLCandidate:
    kind = "ml"

    def __init__(self, spec: ModelSpec, task: TaskConfig):
        if task.type == "extract_exact":
            raise ValueError("ML candidates do not support extract_exact tasks; use LLM candidates")
        if spec.estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator '{spec.estimator}'; choose from {ESTIMATORS}")
        self.spec, self.task, self.name = spec, task, spec.name
        self.classification = task.type in ("classify", "binary")
        _make_estimator(spec.estimator, self.classification, task.seed)  # fail early on bad combos

    def run(self, split: Split, folds: list[Fold]) -> CandidateResult:
        scoring, task = split.scoring, self.task
        n = len(scoring)
        n_rep = max(f.repeat for f in folds) + 1
        votes: list[list] = [[None] * n_rep for _ in range(n)]
        vocab = {label: i for i, label in enumerate(task.labels)} if self.classification else None
        inverse = {i: label for label, i in vocab.items()} if vocab else None
        fit_times, pred_time, pred_count, latencies = [], 0.0, 0, []

        for f in folds:
            train = concat(scoring.take(f.train), split.shots)
            test = scoring.take(f.test)
            y_train = (
                np.array([vocab[v] for v in train.y]) if vocab else train.y.astype(float)
            )
            pipe = Pipeline([
                ("pre", _preprocessor(train.frame, task.text_column)),
                ("est", _make_estimator(self.spec.estimator, self.classification, task.seed)),
            ])
            t0 = time.perf_counter()
            pipe.fit(train.frame, y_train)
            fit_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pred = pipe.predict(test.frame)
            dt = time.perf_counter() - t0
            pred_time += dt
            pred_count += len(test)
            latencies.append(dt / max(len(test), 1))
            for pos, p in zip(f.test, pred):
                votes[pos][f.repeat] = inverse[int(p)] if inverse else float(p)

        predictions = [aggregate(v, self.classification) for v in votes]
        disagree = float(np.mean([len({x for x in v if x is not None}) > 1 for v in votes]))
        return CandidateResult(
            name=self.name, kind="ml", predictions=predictions,
            train_seconds=float(np.mean(fit_times)),
            infer_seconds_per_pred=pred_time / max(pred_count, 1),
            latencies=latencies, repeat_disagreement=disagree,
        )
