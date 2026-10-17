"""Notebook helper: run a comparison from Python and see the report inline."""
from __future__ import annotations

import html
import tempfile
from pathlib import Path

import pandas as pd

from .config import ModelSpec, TaskConfig, load_models, load_pricing, load_task, parse_pricing
from .data import load_dataset
from .engine import new_run_dir, run_experiment
from .report import render_report

_COLUMNS = ["name", "kind", "status", "score", "ci_lo", "ci_hi", "cost_per_1k",
            "total_cost_at_volume", "p50_latency_s", "invalid_rate", "error_rate"]


def _task(t) -> TaskConfig:
    if isinstance(t, TaskConfig):
        return t
    return TaskConfig(**t) if isinstance(t, dict) else load_task(t)


def _models(m) -> list[ModelSpec]:
    if isinstance(m, (str, Path)):
        return load_models(m)
    return [x if isinstance(x, ModelSpec) else ModelSpec(**x) for x in m]


def _pricing(p) -> dict:
    if p is None:
        return {}
    if not isinstance(p, dict):
        return load_pricing(p)
    # accept [input_per_1k, output_per_1k] pairs as well as the pricing.yaml mapping shape
    p = {k: {"input_per_1k": v[0], "output_per_1k": v[1]} if isinstance(v, (list, tuple)) and len(v) == 2 else v
         for k, v in p.items()}
    return parse_pricing(p, "pricing")


def _dataset(data, task: TaskConfig):
    if isinstance(data, pd.DataFrame):  # same validation as a CSV file: round-trip through one
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            data.to_csv(path, index=False)
            return load_dataset(path, task)
    return load_dataset(data, task)


class Comparison:
    def __init__(self, run_dir, data: dict):
        self.run_dir, self.data = Path(run_dir), data

    @property
    def recommendation(self):
        return self.data["recommendation"]

    @property
    def warnings(self) -> list[str]:
        return self.data["warnings"]

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.data["candidates"])[_COLUMNS].set_index("name")

    def report_html(self, redact: bool = False) -> str:
        return render_report(self.data, redact)

    def save_report(self, path=None, redact: bool = False) -> Path:
        path = Path(path) if path else self.run_dir / "report.html"
        path.write_text(self.report_html(redact), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        # an iframe keeps the report's CSS from leaking into the notebook page
        return ('<iframe sandbox srcdoc="' + html.escape(self.report_html(), quote=True)
                + '" style="width:100%;height:900px;border:0"></iframe>')


def compare(task, models, data, *, pricing=None, run_dir=None, **engine_kwargs) -> Comparison:
    """Run a comparison. engine_kwargs go to run_experiment: allow_hosted, local_only, max_cost, concurrency, retries, n_boot.

    A DataFrame's index is not used as the id (include an `id` column); an existing run_dir is resumed."""
    t = _task(task)
    result = run_experiment(t, _models(models), _dataset(data, t),
                            Path(run_dir) if run_dir else new_run_dir("runs"),
                            pricing=_pricing(pricing), **engine_kwargs)
    return Comparison(result.run_dir, result.data)
