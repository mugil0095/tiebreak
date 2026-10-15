from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

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


class ConfigError(ValueError):
    """A config file is empty, malformed, or missing something; the message names the file."""


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            detail = " ".join(str(e).split())
            raise ConfigError(f"{path}: not valid YAML ({detail})") from e


def _read_required(path):
    raw = _read_yaml(path)
    if raw is None:
        raise ConfigError(f"{path}: the file is empty")
    return raw


def _fmt_validation(path, e: ValidationError, prefix: str = "") -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{prefix}{loc}: {msg}" if err["loc"] else f"{prefix}{msg}")
    return f"{path}: " + "; ".join(parts)


def load_task(path) -> TaskConfig:
    raw = _read_required(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected key: value pairs (type, label_column, ...)")
    try:
        return TaskConfig(**raw)
    except ValidationError as e:
        raise ConfigError(_fmt_validation(path, e)) from e


def load_models(path) -> list[ModelSpec]:
    raw = _read_required(path)
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: expected a list of candidates, each starting with '- '")
    specs = []
    for i, m in enumerate(raw, 1):
        if not isinstance(m, dict):
            raise ConfigError(f"{path}: candidate #{i} must be a mapping like {{name: x, kind: ml, ...}}")
        try:
            specs.append(ModelSpec(**m))
        except ValidationError as e:
            raise ConfigError(_fmt_validation(path, e, f"candidate #{i} ({m.get('name', '?')}): ")) from e
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"{path}: duplicate candidate names: {dupes}")
    return specs


def load_pricing(path) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    raw = _read_yaml(path) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected 'model-id: {{input_per_1k: x, output_per_1k: y}}' entries")
    out = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            raise ConfigError(f"{path}: price for '{k}' must be {{input_per_1k: x, output_per_1k: y}}")
        for key in ("input_per_1k", "output_per_1k"):
            if key not in v:
                raise ConfigError(f"{path}: price for '{k}' is missing {key}")
        try:
            out[str(k)] = (float(v["input_per_1k"]), float(v["output_per_1k"]))
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{path}: price for '{k}' must be numbers") from e
    return out
