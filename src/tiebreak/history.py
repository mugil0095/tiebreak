from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HistoryRow:
    run: str
    date: str
    task_type: str
    metric: str
    model: str
    kind: str
    status: str
    score: float | None
    cost_at_volume: float | None
    p50_latency_s: float | None


def _number(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _rows_of(run: str, data: dict) -> list[HistoryRow]:
    task = data["task"]
    return [HistoryRow(run, str(data.get("run_date") or ""), str(task.get("type") or ""), str(task.get("metric") or ""),
                       str(c["name"]), str(c.get("kind") or ""), str(c.get("status") or ""), _number(c.get("score")),
                       _number(c.get("total_cost_at_volume")), _number(c.get("p50_latency_s")))
            for c in data["candidates"]]


def collect_history(runs_dir: Path) -> tuple[list[HistoryRow], int]:
    """One row per (run, candidate) from every finished run in `runs_dir`, oldest first.

    The second value counts run folders that were skipped because results.json is missing or unreadable."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return [], 0
    rows: list[HistoryRow] = []
    skipped = 0
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            rows += _rows_of(d.name, json.loads((d / "results.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            skipped += 1
    return sorted(rows, key=lambda r: (r.date, r.run)), skipped
