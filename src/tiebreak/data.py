from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import TaskConfig


class DataError(ValueError):
    """The input data cannot be used; the message says why and where."""


@dataclass
class Dataset:
    ids: list[str]
    frame: pd.DataFrame
    y: np.ndarray
    text: list[str] | None

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx: Sequence[int]) -> "Dataset":
        idx = np.asarray(idx, dtype=int)
        return Dataset(
            [self.ids[i] for i in idx],
            self.frame.iloc[idx].reset_index(drop=True),
            self.y[idx],
            None if self.text is None else [self.text[i] for i in idx],
        )


def concat(a: Dataset, b: Dataset | None) -> Dataset:
    if b is None:
        return a
    text = None if a.text is None or b.text is None else a.text + b.text
    return Dataset(
        a.ids + b.ids,
        pd.concat([a.frame, b.frame], ignore_index=True),
        np.concatenate([a.y, b.y]),
        text,
    )


def load_dataset(path, task: TaskConfig) -> Dataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".jsonl", ".ndjson"):
        raise DataError(f"unsupported file type '{suffix}'; use .csv or .jsonl")
    try:
        if suffix == ".csv":
            # Read everything as text so "None", "NA", "007" survive; only truly blank cells are NaN.
            df = pd.read_csv(
                path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""]
            )
        else:
            df = pd.read_json(path, lines=True, dtype=False)
    except (OSError, ValueError) as e:  # EmptyDataError, ParserError, bad JSON, bad encoding are ValueErrors
        raise DataError(f"could not read {path.name}: {e}") from e

    lc, tc = task.label_column, task.text_column
    if lc not in df.columns:
        raise DataError(f"label column '{lc}' not found; columns are {list(df.columns)}")
    if tc and tc not in df.columns:
        raise DataError(f"text column '{tc}' not found; columns are {list(df.columns)}")

    missing = df.index[df[lc].isna() | (df[lc].astype(str).str.strip() == "")].tolist()
    if missing:
        rows = [i + 1 for i in missing[:5]]
        raise DataError(f"{len(missing)} rows have no label (first data rows, 1-based: {rows})")

    text = None
    if tc:
        empty = df.index[df[tc].isna() | (df[tc].astype(str).str.strip() == "")].tolist()
        if empty:
            raise DataError(
                f"{len(empty)} rows have empty text (first data rows, 1-based: {[i + 1 for i in empty[:5]]})"
            )
        text = df[tc].astype(str).tolist()

    if "id" in df.columns:
        blank = df.index[df["id"].isna() | (df["id"].astype(str).str.strip() == "")].tolist()
        if blank:
            rows = [i + 1 for i in blank[:5]]
            raise DataError(f"{len(blank)} rows have a blank id (first data rows, 1-based: {rows})")
        ids = df["id"].astype(str).tolist()
        seen = pd.Series(ids)
        dups = list(dict.fromkeys(seen[seen.duplicated()]))
        if dups:
            raise DataError(f"duplicate values in the 'id' column, e.g. {dups[:5]}")
    else:
        ids = [f"row-{i}" for i in range(len(df))]

    if task.type == "regress":
        try:
            y = pd.to_numeric(df[lc], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as e:
            raise DataError(f"label column '{lc}' has non-numeric values: {e}") from e
        bad = np.flatnonzero(~np.isfinite(y))
        if bad.size:
            rows = [int(i) + 1 for i in bad[:5]]
            raise DataError(
                f"label column '{lc}' has {bad.size} non-finite values (inf, -inf or nan); "
                f"first data rows, 1-based: {rows}"
            )
    else:
        y = df[lc].astype(str).str.strip().to_numpy(dtype=object)
        if task.type in ("classify", "binary"):
            unknown = sorted(set(y) - set(task.labels))
            if unknown:
                raise DataError(f"labels not in task.labels: {unknown[:5]}")
            if len(set(y)) < 2:
                raise DataError("the data needs at least two distinct labels")

    frame = df.drop(columns=[c for c in (lc, "id") if c in df.columns]).reset_index(drop=True)
    return Dataset(ids, frame, y, text)
