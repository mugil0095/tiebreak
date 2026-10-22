from __future__ import annotations

import re
import time
from dataclasses import replace

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
from ..install import install_hint
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
            raise ImportError(f"xgboost is not installed; run: {install_hint('boost')}") from e
        cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
        return cls(n_estimators=200, random_state=seed, verbosity=0)
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(f"lightgbm is not installed; run: {install_hint('boost')}") from e
        cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        return cls(n_estimators=200, random_state=seed, verbose=-1)
    kind = "classification" if classification else "regression"
    raise ValueError(f"estimator '{name}' is not available for {kind}; choose from {ESTIMATORS}")


_LEADING_ZERO = re.compile(r"^-?0\d")


def _parses_as_number(values: pd.Series) -> bool:
    """True when every value is a plain finite number that survives a float round trip."""
    nums = pd.to_numeric(values, errors="coerce")
    if nums.isna().any() or not np.isfinite(nums.to_numpy(dtype=float)).all():
        return False
    for v in values:
        if v.startswith("+") or _LEADING_ZERO.match(v):  # "+1", "01234", "007": identifiers, not numbers
            return False
        digits = re.sub(r"\D", "", re.split(r"[eE]", v)[0]).lstrip("0")
        if len(digits) > 15:  # would lose precision as a float
            return False
    return True


def to_numeric_features(frame: pd.DataFrame, text_col: str | None) -> pd.DataFrame:
    """Return a copy of ``frame`` for ML use: numeric-looking string columns become float.

    Only used on the ML path (Dataset.frame itself stays lossless for LLM prompts). A column
    becomes numeric only if every non-blank value is a plain finite number (no leading zeros,
    no leading "+", at most 15 significant digits); blank -> NaN. Otherwise it stays a string.
    Entirely blank columns are dropped; the text column is never touched. The decision uses
    feature values only (no labels), so computing it over all rows is leak-free.
    """
    out = {}
    for col in frame.columns:
        ser = frame[col]
        if col == text_col:
            out[col] = ser
        elif pd.api.types.is_numeric_dtype(ser):
            if ser.notna().any():
                out[col] = ser
        else:
            txt = ser.astype("string").str.strip()
            txt = txt.mask(txt == "")
            nonblank = txt.dropna()
            if nonblank.empty:
                continue
            if _parses_as_number(nonblank.astype(str)):
                out[col] = pd.to_numeric(txt, errors="coerce").astype(float)
            else:
                out[col] = ser
    return pd.DataFrame(out, index=frame.index)


def _preprocessor(frame: pd.DataFrame, text_col: str | None) -> ColumnTransformer:
    # Column types were decided once in MLCandidate._numeric_split; classify by dtype only.
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

    def _numeric_split(self, split: Split):
        """Convert numeric-looking columns once over scoring+shots rows so train and test agree."""
        sc, sh = split.scoring, split.shots
        conv = to_numeric_features(concat(sc, sh).frame, self.task.text_column)
        n = len(sc)
        sc2 = replace(sc, frame=conv.iloc[:n].reset_index(drop=True))
        sh2 = None if sh is None else replace(sh, frame=conv.iloc[n:].reset_index(drop=True))
        return sc2, sh2

    def run(self, split: Split, folds: list[Fold]) -> CandidateResult:
        task = self.task
        if not folds:
            raise ValueError("no cross-validation folds were given")
        if task.text_column and task.text_column not in split.scoring.frame.columns:
            raise DataError(f"text column '{task.text_column}' is missing from the data frame")
        bad = sorted({str(v) for v in split.scoring.y} - set(task.labels)) if self.classification else []
        if bad:
            raise ValueError(f"labels not in task.labels: {bad[:5]}")
        scoring, shots = self._numeric_split(split)
        n = len(scoring)
        n_rep = max(f.repeat for f in folds) + 1
        votes: list[list] = [[None] * n_rep for _ in range(n)]
        vocab = {label: i for i, label in enumerate(task.labels)} if self.classification else None
        inverse = {i: label for label, i in vocab.items()} if vocab else None
        fit_times, pred_time, pred_count, latencies = [], 0.0, 0, []

        for f in folds:
            train = concat(scoring.take(f.train), shots)
            test = scoring.take(f.test)
            y_train = (
                np.array([vocab[v] for v in train.y]) if vocab else train.y.astype(float)
            )
            pipe = Pipeline([
                ("pre", _preprocessor(train.frame, task.text_column)),
                ("est", _make_estimator(self.spec.estimator, self.classification, task.seed)),
            ])
            t0 = time.perf_counter()
            try:
                pipe.fit(train.frame, y_train)
            except ValueError as e:
                missing = [label for label, i in vocab.items() if i not in y_train] if vocab else []
                if not missing:
                    raise
                raise ValueError(
                    f"{self.name}: no training example for label(s) {missing} in a cross-validation fold, and "
                    f"'{self.spec.estimator}' needs every label in task.labels present. Add examples of them or "
                    "remove them from task.labels") from e
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
        # Regression predictions are floats that never compare equal, so disagreement is not defined there (None, shown as n/a).
        disagree = (
            float(np.mean([len({x for x in v if x is not None}) > 1 for v in votes])) if self.classification else None
        )
        return CandidateResult(
            name=self.name, kind="ml", predictions=predictions,
            train_seconds=float(np.mean(fit_times)),
            infer_seconds_per_pred=pred_time / max(pred_count, 1),
            latencies=latencies, repeat_disagreement=disagree,
        )
