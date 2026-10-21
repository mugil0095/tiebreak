from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .candidates.base import CandidateResult
from .candidates.llm import is_local
from .config import ModelSpec, TaskConfig, validate_models_for_task
from .cost import summarize_cost
from .data import Dataset
from .engine import (
    ExperimentResult,
    _candidate_rows,
    _check_safety,
    _clean,
    _collect_secrets,
    _compare,
    _decide_status,
    _diagnostics,
    _prices_used,
    _results_data,
    _write_outputs,
    _write_snapshot,
)
from .judge_run import JudgeScores, run_judging
from .metrics import get_metric
from .runner import CostCap

METRIC_LABEL = "mean_judge_score"


def spearman_or_none(a, b) -> float | None:
    """Spearman rank correlation, or None when it is undefined (fewer than 3 pairs or a constant side)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    rho = spearmanr(a, b).statistic
    return None if not np.isfinite(rho) else float(rho)


def _candidate_result(spec: ModelSpec, out: JudgeScores, n: int) -> CandidateResult:
    scores = out.by_candidate[spec.name]
    calls = out.answer_calls[spec.name]
    ok = [c for c in calls if c.error is None]
    lat = [c.latency_s for c in ok]
    empties = sum(1 for a in out.answers[spec.name] if a is not None and not a.strip())
    mask = ~np.isnan(scores)
    return CandidateResult(
        name=spec.name, kind="llm", predictions=[None if np.isnan(s) else float(s) for s in scores], calls=calls,
        infer_seconds_per_pred=float(np.mean(lat)) if lat else 0.0, latencies=lat,
        invalid_rate=empties / len(calls) if calls else 0.0,
        error_rate=(len(calls) - len(ok)) / len(calls) if calls else 0.0,
        repeat_disagreement=0.0, mask=mask, n_unanswered=int((~mask).sum()), local_compute=is_local(spec))


def _judge_block(task: TaskConfig, dataset: Dataset, candidates: list[ModelSpec], judge: ModelSpec,
                 out: JudgeScores, self_judging: bool) -> dict:
    judged = [j for js in out.judgments.values() for j in js if j is not None]
    unusable = sum(1 for j in judged if j.value is None)
    confident = [j.confidence for j in judged if j.value is not None and j.confidence is not None]
    length = {}
    for spec in candidates:
        words = np.array([len(a.split()) if a is not None else np.nan for a in out.answers[spec.name]])
        scores = out.by_candidate[spec.name]
        both = ~np.isnan(words) & ~np.isnan(scores)
        length[spec.name] = {
            "mean_words": float(np.nanmean(words)) if (~np.isnan(words)).any() else None,
            "score_length_spearman": spearman_or_none(words[both], scores[both])}
    human = None
    col = task.human_score_column
    if col and col in dataset.frame.columns:
        first = candidates[0].name
        h = pd.to_numeric(dataset.frame[col], errors="coerce").to_numpy(dtype=float)
        s = out.by_candidate[first]
        ok = np.isfinite(h) & np.isfinite(s)
        human = {"candidate": first, "n": int(ok.sum()), "spearman": spearman_or_none(h[ok], s[ok])}
    return {
        "name": judge.name, "model": judge.model, "backend": judge.provider, "self_judging": self_judging,
        "evaluation_cost": out.evaluation_cost,
        "invalid_rate": unusable / len(judged) if judged else 0.0,
        "low_confidence_rate": (sum(1 for c in confident if c < task.judge_confidence_min) / len(confident)
                                if confident else 0.0),
        "length": length, "human_agreement": human,
    }


def _judge_price_entry(judge: ModelSpec, pricing: dict) -> dict:
    if judge.model in pricing:
        return {"candidate": judge.name, "model": judge.model, "source": "pricing.yaml",
                "input_per_1k": pricing[judge.model][0], "output_per_1k": pricing[judge.model][1]}
    source = "typesafe docs" if judge.provider == "typesafe" else "compute time" if is_local(judge) else "provider"
    return {"candidate": judge.name, "model": judge.model, "source": source}


def run_judge_experiment(
    task: TaskConfig, models: list[ModelSpec], dataset: Dataset, run_dir, *,
    pricing: dict | None = None, max_cost: float | None = None, concurrency: int = 4, retries: int = 3,
    allow_hosted: bool = False, local_only: bool = False, completer_factory=None, n_boot: int = 10_000,
    sleep=time.sleep,
) -> ExperimentResult:
    """Candidates answer prompts, a judge scores the answers, and the scores are compared like a regression metric.

    The scores go through the existing comparison code with a synthetic target (the top of the scale) and
    `neg_mae`, so a candidate's score is its mean judge score minus the top of the scale; the rows convert it back."""
    pricing = pricing or {}
    validate_models_for_task(task, models)
    _check_safety(task, models, allow_hosted, local_only)
    run_dir = Path(run_dir)
    _write_snapshot(run_dir, task, models, pricing)
    n = len(dataset)
    if n < 2:
        raise ValueError(f"need at least 2 prompts to compare candidates; got {n}")
    judge = next(m for m in models if m.role == "judge")
    candidates = [m for m in models if m.role == "candidate"]
    prices = {m.name: (0.0, 0.0) if is_local(m) else pricing.get(m.model) for m in models}
    exec_kw = dict(concurrency=concurrency, retries=retries, sleep=sleep, secrets=_collect_secrets(models))
    out = run_judging(task, dataset, candidates, judge, run_dir, cap=CostCap(max_cost), prices=prices, exec_kw=exec_kw)

    reg = task.model_copy(update={"type": "regress", "label_column": "score", "metric": "neg_mae"})
    metric = get_metric("neg_mae")
    results = {spec.name: _candidate_result(spec, out, n) for spec in candidates}
    warnings: list[str] = []
    notes: dict[str, str] = {}
    status = _decide_status(results, n, warnings, notes)
    for name, text in out.notes.items():
        if status[name] == "excluded":
            warnings.append(f"{name}: {text}")
            notes[name] = text
        else:
            notes.setdefault(name, text)
    costs = {name: summarize_cost(res, task, price=None if res.local_compute else prices.get(name))
             for name, res in results.items()}
    y_enc, enc, boot, equiv, rec = _compare(reg, dataset, metric, results, status, costs, out.incomplete,
                                            n_boot, warnings, notes)
    included = [name for name in results if status[name] == "included"]
    diagnostics, pairwise = _diagnostics(reg, dataset, included, y_enc, enc, boot, warnings)

    scored = np.concatenate([s[~np.isnan(s)] for s in out.by_candidate.values()])
    all_same = bool(len(scored) >= 2 and np.ptp(scored) == 0)
    if all_same:
        warnings.append(f"The judge gave every answer the same score ({scored[0]:g}), so it cannot tell the candidates "
                        "apart: a tie here says nothing about quality. Try a stronger judge or a sharper rubric.")
    self_judging = any(c.provider == judge.provider and c.model == judge.model for c in candidates)
    if self_judging:
        warnings.append(f"The judge model ({judge.model}) is also a candidate: judges tend to favour their own "
                        "answers, so treat its score with caution.")
    if any(m.provider == "mock" for m in models):
        warnings.insert(0, "SIMULATED: 'mock' candidates and judges are fake models. Do not use these results "
                           "to choose a real model.")
    warnings.append("Scores come from a judge model, not from ground truth; they inherit the judge's biases.")

    prices_used = _prices_used(results, costs, candidates, prices, task) + [_judge_price_entry(judge, pricing)]
    rows = _candidate_rows(reg, dataset, metric, results, status, costs, boot, equiv, notes)
    top = float(task.scale[1])
    for r in rows:  # back from "distance below the top of the scale" to the mean judge score
        for key in ("score", "ci_lo", "ci_hi"):
            if r[key] is not None and np.isfinite(r[key]):
                r[key] = r[key] + top
    data = _results_data(task, n, rows, rec, diagnostics, pairwise, warnings, out.incomplete,
                         any(m.provider == "mock" for m in models), [], prices_used)
    data["task"]["metric"] = METRIC_LABEL
    data["task"]["repeats"] = 1
    data["warnings"] = [w.replace("labelled examples", "prompts") for w in data["warnings"]]
    data["task"]["scale"] = list(task.scale)
    data["judge"] = _clean({**_judge_block(task, dataset, candidates, judge, out, self_judging),
                            "all_same_score": all_same})
    _write_outputs(run_dir, data, task, models, dataset, n_boot, prices_used)
    return ExperimentResult(run_dir, data)
