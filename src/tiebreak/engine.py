from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from . import __version__
from .candidates.base import CandidateResult
from .candidates.llm import LLMCandidate, build_completer, check_task_support, is_hosted, is_local
from .candidates.ml import MLCandidate
from .config import ModelSpec, TaskConfig
from .cost import latency_stats, summarize_cost
from .data import Dataset
from .diagnostics import class_balance, disagreements, sample_warnings, suspected_mislabels
from .metrics import encode_labels, get_metric
from .planner import clearly_worse
from .protocol import make_folds, make_split
from .recommend import recommend
from .runner import CostCap, CostCapReached, CostUnknownError, PredictionStore, execute_calls
from .stats import equivalent_set, mcnemar_exact, paired_bootstrap

HIGH_ERROR_RATE = 0.5
ELIMINATION_MIN_N = 120  # below this, one round and no early elimination
ELIMINATION_FIRST_ROUND_MIN = 50
ALPHA = 0.05
INVALID_WARN_RATE = 0.05


class SafetyError(RuntimeError):
    """A run was refused to protect the user's data or budget."""


@dataclass
class ExperimentResult:
    run_dir: Path
    data: dict


def _hash_dataset(ds: Dataset) -> str:
    h = hashlib.sha256()
    for i, eid in enumerate(ds.ids):
        h.update(f"{eid}|{ds.y[i]}|".encode())
    return h.hexdigest()[:16]


def _clean(obj):
    """Make a structure strict-JSON safe: numpy scalars -> Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# Benign keys such as max_tokens / num_tokens must survive; a bare "token" key is a secret.
_SECRET_KEY = re.compile(
    r"api[_-]?key|secret|password|authorization|access[_-]?token|auth[_-]?token|bearer|^token$", re.I)


def _redact_params(obj):
    if isinstance(obj, dict):
        return {k: "[redacted]" if _SECRET_KEY.search(str(k)) else _redact_params(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_params(v) for v in obj]
    return obj


def _safe_spec(m: ModelSpec) -> dict:
    d = m.model_dump()
    d["params"] = _redact_params(d["params"])
    if d.get("base_url"):
        u = urlparse(d["base_url"])
        if u.username or u.password:
            host = u.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            d["base_url"] = u._replace(netloc=host + (f":{u.port}" if u.port else "")).geturl()
    return d


_SECRET_ENV = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD|APIKEY)$", re.I)
MIN_SECRET_LEN = 8


def _collect_secrets(models: list[ModelSpec]) -> list[str]:
    """Values to scrub from stored errors: the configured api_key_env vars and every credential-looking env var."""
    out = [os.environ[m.api_key_env] for m in models
           if m.kind == "llm" and m.api_key_env and os.environ.get(m.api_key_env)]
    out += [v for k, v in os.environ.items() if _SECRET_ENV.search(k) and len(v) >= MIN_SECRET_LEN]
    return sorted(set(out))


def _check_safety(task: TaskConfig, models: list[ModelSpec], allow_hosted: bool, local_only: bool) -> None:
    """Refuse unsafe or invalid setups before anything is written to disk."""
    hosted = [m for m in models if is_hosted(m)]
    if hosted and local_only:
        raise SafetyError(f"--local-only is set but these candidates are hosted: {[m.name for m in hosted]}")
    if hosted and not allow_hosted:
        raise SafetyError(
            "hosted candidates send example text to third-party APIs: "
            f"{sorted({m.provider for m in hosted})}. Re-run with --yes-send-data to allow it."
        )
    if task.type == "extract_exact" and any(m.kind == "ml" for m in models):
        raise ValueError("ML candidates do not support extract_exact tasks; use LLM candidates")

    for m in models:
        check_task_support(m, task)
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate candidate names: {sorted({n for n in names if names.count(n) > 1})}")


def _write_snapshot(run_dir: Path, task: TaskConfig, models: list[ModelSpec], pricing: dict) -> None:
    (run_dir / "config.snapshot").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.snapshot" / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "config.snapshot" / "models.json").write_text(
        json.dumps([_safe_spec(m) for m in models], indent=2), encoding="utf-8")
    (run_dir / "config.snapshot" / "pricing.json").write_text(
        json.dumps({k: list(v) for k, v in pricing.items()}, indent=2), encoding="utf-8")


def _run_ml(models: list[ModelSpec], task: TaskConfig, split) -> dict[str, CandidateResult]:
    """Cheap ML candidates first (their predictions also help eliminate bad LLMs)."""
    ml_specs = [m for m in models if m.kind == "ml"]
    if not ml_specs:
        return {}
    folds = make_folds(split.scoring.y, task)
    return {spec.name: MLCandidate(spec, task).run(split, folds) for spec in ml_specs}


def _plan_rounds(n: int, seed: int) -> list[list]:
    if n >= ELIMINATION_MIN_N:
        first = max(ELIMINATION_FIRST_ROUND_MIN, n // 3)
    else:
        first = n  # too few examples to eliminate anyone safely: a single round
    order = np.random.default_rng(seed).permutation(n)
    return [list(order[:first]), list(order[first:])] if first < n else [list(order)]


def _drop_clearly_worse(task, scoring, shots, positions, active, cands, results, store, metric,
                        notes, elimination_log) -> None:
    """Judge the first round and remove clearly worse LLMs from `active` (in place)."""
    n = len(scoring)
    sub = np.array(sorted(positions))
    round_calls = {name: cands[name].build_calls(scoring, shots, positions) for name in active}
    recs = {name: store.valid_for(round_calls[name]) for name in active}
    finals = {name: cands[name].finalize(scoring, recs[name]) for name in active}
    # a candidate with ANY failed call in this round is judged at the end, never dropped here
    failed = {name for name in active if any(r.error is not None for r in recs[name])}
    y_all, enc = encode_labels(task, scoring.y, {
        **{k: v.predictions for k, v in results.items()},
        **{name: f.predictions for name, f in finals.items()},
    })
    valid = {k: v for k, v in enc.items() if k in results or (k not in failed and bool(finals[k].mask[sub].all()))}
    if len(valid) >= 2:
        drop = clearly_worse(y_all[sub], {k: v[sub] for k, v in valid.items()}, metric.fn,
                             task.margin, eligible=set(active), seed=task.seed)
        for name in sorted(drop):
            elimination_log.append(
                f"{name}: dropped after {len(sub)} examples; clearly worse than the best "
                f"by more than {task.margin:g}")
            notes[name] = f"eliminated early after {len(sub)} of {n} examples; score covers those only"
            active.remove(name)


def _run_llm_rounds(task, scoring, shots, cands, results, prices, store, metric, exec_kw, warnings, notes):
    """Run the LLM candidates in rounds, dropping clearly worse ones after the first.
    Returns (active names, elimination log, incomplete)."""
    rounds = _plan_rounds(len(scoring), task.seed)
    active = list(cands)
    elimination_log: list[str] = []
    incomplete = False
    try:
        for rnd, positions in enumerate(rounds):
            for name in active:
                cand = cands[name]
                round_calls = cand.build_calls(scoring, shots, positions)
                execute_calls(round_calls, cand.completer, store, price=prices[name], **exec_kw)
            if rnd == 0 and len(rounds) == 2 and len(active) + len(results) >= 2:
                _drop_clearly_worse(task, scoring, shots, positions, active, cands, results, store, metric,
                                    notes, elimination_log)
    except CostUnknownError as e:
        raise SafetyError(str(e)) from e
    except CostCapReached:
        incomplete = True
        warnings.append("The cost cap was reached; the run is incomplete. Re-run with --resume and a higher cap.")
    return active, elimination_log, incomplete


def _decide_status(results: dict[str, CandidateResult], n: int, warnings: list[str], notes: dict[str, str]) -> dict[str, str]:
    """included / excluded / eliminated per candidate; appends to `warnings` and `notes` in place."""
    status: dict[str, str] = {}
    for name, res in results.items():
        if res.kind == "llm" and res.error_rate >= HIGH_ERROR_RATE:
            status[name] = "excluded"
            warnings.append(f"{name}: {res.error_rate:.0%} of calls failed; excluded from the comparison")
            notes[name] = f"{res.error_rate:.0%} of calls failed"
        elif res.eliminated:
            status[name] = "eliminated"
        elif res.kind == "llm" and (res.mask is None or not res.mask.all()):
            status[name] = "excluded"
            if res.n_unanswered:
                missing = n - int(res.mask.sum()) if res.mask is not None else n
                warnings.append(f"{name}: {missing} of {n} examples have no successful answer (provider errors); "
                                "re-run with --resume to retry them")
                notes[name] = "incomplete: provider errors"
            else:
                warnings.append(f"{name}: not all examples were evaluated; excluded from the comparison")
                notes[name] = "incomplete evaluation"
        else:
            status[name] = "included"
    for name, res in results.items():
        if res.invalid_rate > INVALID_WARN_RATE:
            warnings.append(f"{name}: {res.invalid_rate:.0%} of answers could not be parsed and are scored as maximally wrong")
    return status


def _compare(task, scoring, metric, results, status, costs, incomplete, n_boot, warnings, notes):
    """Bootstrap the included candidates and recommend. Returns (y_enc, enc, boot, equiv, rec)."""
    included = [name for name in results if status[name] == "included"]
    y_enc, enc = encode_labels(task, scoring.y, {k: results[k].predictions for k in included})
    boot = rec = None
    equiv: dict[str, bool] = {}
    if included:
        boot = paired_bootstrap(y_enc, enc, metric.fn, n_boot=n_boot, seed=task.seed)
        equiv = equivalent_set(boot, task.margin, alpha=ALPHA, guard=metric.classification)
        rec = recommend(boot, equiv, {k: costs[k].total_at_volume if costs[k].price_known else None for k in included},
                        task.margin, alpha=ALPHA, guard=metric.classification)
    else:
        warnings.append("No candidate completed the evaluation; there is nothing to recommend.")
    left_out = [(k, notes.get(k, "excluded")) for k in results if status[k] == "excluded"]
    if rec is not None and (incomplete or left_out):
        why = []
        if incomplete:
            why.append("the run is incomplete (cost cap reached)")
        if left_out:
            why.append("left out of the comparison: " + ", ".join(f"{k} ({v})" for k, v in left_out))
        rec = dataclasses.replace(rec, status="provisional", reason="Provisional: " + "; ".join(why) + ". " + rec.reason)
    return y_enc, enc, boot, equiv, rec


def _diagnostics(task, scoring, included, y_enc, enc, boot, warnings):
    """Returns (diagnostics dict, pairwise list); appends sample-size warnings."""
    if task.type in ("classify", "binary"):
        labels = task.labels
        mis = [{"id": m["id"], "label": labels[m["label_code"]], "disagree_fraction": m["disagree_fraction"]}
               for m in suspected_mislabels(scoring.ids, y_enc, enc)] if len(included) >= 2 else []
    else:
        mis = []
    dis = disagreements(scoring.ids, enc) if len(included) >= 2 and task.type != "regress" else []
    balance = class_balance(scoring.y) if task.type in ("classify", "binary") else {}
    widths = {k: boot.ci[k][1] - boot.ci[k][0] for k in included} if boot else {}
    warnings += sample_warnings(len(scoring), task.margin, balance, widths)

    pairwise = []
    if task.type in ("classify", "binary", "extract_exact") and task.metric_name == "accuracy" and len(included) >= 2:
        for i, a in enumerate(included):
            for b in included[i + 1:]:
                pairwise.append({"a": a, "b": b, "mcnemar_p": mcnemar_exact(enc[a] == y_enc, enc[b] == y_enc)})
    return {"mislabels": mis, "disagreement_ids": dis, "class_balance": balance}, pairwise


def _prices_used(results, costs, models, prices, task) -> list[dict]:
    spec_by_name = {m.name: m for m in models}
    used = []
    for name, res in results.items():
        src = costs[name].price_source
        if res.kind != "llm" or src == "unknown":
            continue
        entry = {"candidate": name, "model": spec_by_name[name].model, "source": src}
        if src == "pricing.yaml":
            entry.update(input_per_1k=prices[name][0], output_per_1k=prices[name][1])
        elif src == "compute time":
            entry.update(compute_price_per_hour=task.compute_price_per_hour)
        used.append(entry)
    return used


def _candidate_rows(task, scoring, metric, results, status, costs, boot, equiv, notes) -> list[dict]:
    n = len(scoring)
    rows = []
    for name, res in results.items():
        p50, p95 = latency_stats(res.latencies)
        c = costs[name]
        in_boot = boot is not None and name in boot.point
        # eliminated/excluded candidates still get a score on whatever they completed
        if in_boot:
            score, lo, hi = boot.point[name], boot.ci[name][0], boot.ci[name][1]
        else:
            ok = np.array([p is not None for p in res.predictions])
            if res.mask is not None:
                ok &= res.mask
            ye, ee = encode_labels(task, scoring.y, {name: res.predictions})
            score = float(metric.fn(ye[ok], ee[name][ok])) if ok.any() else float("nan")
            lo = hi = None
            if status[name] == "excluded" and ok.any() and ok.sum() < n:
                notes[name] = (notes.get(name, "") + f"; score covers {int(ok.sum())} of {n} examples").lstrip("; ")
        rows.append({
            "name": name, "kind": res.kind, "status": status[name], "note": notes.get(name, ""),
            "score": score, "ci_lo": lo, "ci_hi": hi,
            "cost_per_1k": c.per_1k, "total_cost_at_volume": c.total_at_volume, "price_known": c.price_known,
            "price_source": c.price_source,
            "p50_latency_s": p50, "p95_latency_s": p95, "invalid_rate": res.invalid_rate,
            "error_rate": res.error_rate, "repeat_disagreement": res.repeat_disagreement,
            "equivalent": bool(equiv.get(name, False)),
        })
    return rows


def _write_outputs(run_dir: Path, data: dict, task: TaskConfig, models: list[ModelSpec], scoring: Dataset,
                   n_boot: int, prices_used: list[dict]) -> None:
    (run_dir / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "tiebreak_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "data_hash": _hash_dataset(scoring),
        "seed": task.seed, "n_boot": n_boot,
        "models": [{"name": m.name, "kind": m.kind, "provider": m.provider, "model": m.model} for m in models],
        "prices_used": {u["model"]: [u["input_per_1k"], u["output_per_1k"]] for u in prices_used if u["source"] == "pricing.yaml"},
    }, indent=2), encoding="utf-8")


def _run_llm(task, scoring, shots, llm_specs, factory, pricing, results, store, metric, exec_kw, warnings, notes):
    """Run and finalize all LLM candidates into `results`. Returns (prices, elimination log, incomplete)."""
    cands = {s.name: LLMCandidate(s, task, factory(s)) for s in llm_specs}
    local_names = {s.name for s in llm_specs if is_local(s)}
    # user pricing.yaml wins (never for local models, which are costed by compute time); local calls cost 0 per call
    prices = {s.name: (0.0, 0.0) if s.name in local_names else pricing.get(s.model) for s in llm_specs}
    expected = {name: cand.build_calls(scoring, shots, list(range(len(scoring)))) for name, cand in cands.items()}
    active, elimination_log, incomplete = _run_llm_rounds(
        task, scoring, shots, cands, results, prices, store, metric, exec_kw, warnings, notes)
    for name, cand in cands.items():
        res = cand.finalize(scoring, store.valid_for(expected[name]))
        res.eliminated = name not in active
        res.local_compute = name in local_names
        results[name] = res
    return prices, elimination_log, incomplete


def _results_data(task, n, rows, rec, diagnostics, pairwise, warnings, incomplete, simulated, elimination_log,
                  prices_used) -> dict:
    return _clean({
        "task": {"type": task.type, "metric": task.metric_name, "margin": task.margin, "volume": task.volume,
                 "n_scoring": n, "shots_pool": task.shots_pool, "repeats": task.repeats,
                 "cv_folds": task.cv_folds, "cv_repeats": task.cv_repeats},
        "candidates": rows,
        "recommendation": None if rec is None else {
            "status": rec.status, "winner": rec.winner, "best": rec.best, "equivalent": rec.equivalent,
            "inconclusive": rec.inconclusive, "extra_examples": rec.extra_examples, "reason": rec.reason},
        "diagnostics": diagnostics,
        "pairwise": pairwise, "warnings": warnings, "incomplete": incomplete,
        "simulated": simulated, "elimination_log": elimination_log,
        "run_date": datetime.now(timezone.utc).date().isoformat(), "prices_used": prices_used,
    })


def run_experiment(
    task: TaskConfig, models: list[ModelSpec], dataset: Dataset, run_dir, *,
    pricing: dict | None = None, max_cost: float | None = None, concurrency: int = 4,
    retries: int = 3, allow_hosted: bool = False, local_only: bool = False,
    completer_factory=None, n_boot: int = 10_000, sleep=time.sleep,
) -> ExperimentResult:
    pricing = pricing or {}
    _check_safety(task, models, allow_hosted, local_only)
    run_dir = Path(run_dir)
    _write_snapshot(run_dir, task, models, pricing)
    store = PredictionStore(run_dir / "predictions.jsonl")
    llm_specs = [m for m in models if m.kind == "llm"]
    cap = CostCap(max_cost)
    if max_cost is not None:  # seed ONCE (the cap is cumulative across resumes); calls must not re-seed it
        cap.add(sum(store.total_cost(m.name) for m in llm_specs))
    exec_kw = dict(cap=cap, concurrency=concurrency, retries=retries, sleep=sleep,
                   secrets=_collect_secrets(models), seed_from_store=False)
    factory = completer_factory or (lambda spec: build_completer(spec, task))

    split = make_split(dataset, task)
    scoring, n = split.scoring, len(split.scoring)
    if n < 2:
        raise ValueError(f"need at least 2 scoring examples to compare candidates; got {n}")
    metric = get_metric(task.metric_name)
    warnings: list[str] = []
    notes: dict[str, str] = {}
    results = _run_ml(models, task, split)

    prices, elimination_log, incomplete = _run_llm(
        task, scoring, split.shots, llm_specs, factory, pricing, results, store, metric, exec_kw, warnings, notes)

    status = _decide_status(results, n, warnings, notes)
    costs = {name: summarize_cost(res, task, price=None if res.local_compute else prices.get(name))
             for name, res in results.items()}
    y_enc, enc, boot, equiv, rec = _compare(task, scoring, metric, results, status, costs, incomplete,
                                            n_boot, warnings, notes)
    included = [name for name in results if status[name] == "included"]
    diagnostics, pairwise = _diagnostics(task, scoring, included, y_enc, enc, boot, warnings)
    simulated = any(m.provider == "mock" for m in llm_specs)
    if simulated:
        warnings.insert(0, "SIMULATED: 'mock' candidates are fake models with preset accuracy. Do not use these results to choose a real model.")
    prices_used = _prices_used(results, costs, models, prices, task)

    rows = _candidate_rows(task, scoring, metric, results, status, costs, boot, equiv, notes)
    data = _results_data(task, n, rows, rec, diagnostics, pairwise, warnings, incomplete, simulated,
                         elimination_log, prices_used)
    _write_outputs(run_dir, data, task, models, scoring, n_boot, prices_used)
    return ExperimentResult(run_dir, data)
