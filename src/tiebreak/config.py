from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

TaskType = Literal["classify", "binary", "extract_exact", "regress"]
DEFAULT_METRIC = {
    "classify": "macro_f1",
    "binary": "accuracy",
    "extract_exact": "accuracy",
    "regress": "neg_rmse",
}


class TaskConfig(BaseModel):
    type: TaskType
    label_column: str
    text_column: str | None = None
    labels: list[str] | None = None
    metric: str | None = None
    margin: float = Field(0.02, gt=0)
    volume: int = Field(10_000, gt=0)
    shots_pool: int = Field(0, ge=0)
    repeats: int = Field(3, ge=1)
    cv_folds: int = Field(5, ge=2)
    cv_repeats: int = Field(3, ge=1)
    seed: int = 0
    compute_price_per_hour: float = Field(0.10, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if self.type in ("classify", "binary") and not self.labels:
            raise ValueError("labels are required for classify/binary tasks")
        if self.type == "binary" and len(self.labels) != 2:
            raise ValueError("binary tasks need exactly two labels")
        return self

    @property
    def metric_name(self) -> str:
        return self.metric or DEFAULT_METRIC[self.type]


class ModelSpec(BaseModel):
    name: str
    kind: Literal["llm", "ml"]
    provider: str | None = None
    model: str | None = None
    estimator: str | None = None
    prompt: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "llm" and not (self.provider and self.model):
            raise ValueError("llm candidates need provider and model")
        if self.kind == "ml" and not self.estimator:
            raise ValueError("ml candidates need an estimator")
        return self


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task(path) -> TaskConfig:
    return TaskConfig(**_read_yaml(path))


def load_models(path) -> list[ModelSpec]:
    specs = [ModelSpec(**m) for m in _read_yaml(path)]
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate candidate names: {dupes}")
    return specs


def load_pricing(path) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    raw = _read_yaml(path) or {}
    return {k: (float(v["input_per_1k"]), float(v["output_per_1k"])) for k, v in raw.items()}
