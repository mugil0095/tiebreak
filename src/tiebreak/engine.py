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
from .candidates.llm import LLMCandidate, build_completer
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

LOCAL_PROVIDERS = {"mock", "ollama"}
HIGH_ERROR_RATE = 0.5
ELIMINATION_MIN_N = 120  # below this, one round and no early elimination
ELIMINATION_FIRST_ROUND_MIN = 50
ALPHA = 0.05


class SafetyError(RuntimeError):
    """A run was refused to protect the user's data or budget."""


def is_hosted(spec: ModelSpec) -> bool:
    if spec.kind != "llm" or spec.provider in LOCAL_PROVIDERS:
        return False
    if spec.provider == "openai_compatible" and spec.base_url:
        return urlparse(spec.base_url).hostname not in ("localhost", "127.0.0.1", "::1")
    return True


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


def run_experiment(
    task: TaskConfig, models: list[ModelSpec], dataset: Dataset, run_dir, *,
    pricing: dict | None = None, max_cost: float | None = None, concurrency: int = 4,
    retries: int = 3, allow_hosted: bool = False, local_only: bool = False,
    completer_factory=None, n_boot: int = 10_000, sleep=time.sleep,
) -> ExperimentResult:
    pricing = pricing or {}
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

    names_ = [m.name for m in models]
    if len(set(names_)) != len(names_):
        raise ValueError(f"duplicate candidate names: {sorted({n for n in names_ if names_.count(n) > 1})}")

    run_dir = Path(run_dir)
    (run_dir / "config.snapshot").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.snapshot" / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "config.snapshot" / "models.json").write_text(
        json.dumps([_safe_spec(m) for m in models], indent=2), encoding="utf-8")
    store = PredictionStore(run_dir / "predictions.jsonl")
    llm_names = [m.name for m in models if m.kind == "llm"]
    cap = CostCap(max_cost)
    if max_cost is not None:  # seed ONCE (the cap is cumulative across resumes); calls must not re-seed it
        cap.add(sum(r.cost_usd or 0.0 for name in llm_names for r in store.for_candidate(name)))
    secrets = [os.environ[m.api_key_env] for m in models
               if m.kind == "llm" and m.api_key_env and os.environ.get(m.api_key_env)]
    factory = completer_factory or (lambda spec: build_completer(spec, task))

    split = make_split(dataset, task)
    scoring, n = split.scoring, len(split.scoring)
    if n < 2:
        raise ValueError(f"need at least 2 scoring examples to compare candidates; got {n}")
    metric = get_metric(task.metric_name)
    warnings: list[str] = []
    elimination_log: list[str] = []
    incomplete = False
    results: dict[str, CandidateResult] = {}
    notes: dict[str, str] = {}

    # --- cheap ML candidates first (their predictions also help eliminate bad LLMs) ---
    ml_specs = [m for m in models if m.kind == "ml"]
    if ml_specs:
        folds = make_folds(scoring.y, task)
        for spec in ml_specs:
            results[spec.name] = MLCandidate(spec, task).run(split, folds)

    # --- LLM candidates, in two rounds so clearly worse ones can be dropped early ---
    llm_specs = [m for m in models if m.kind == "llm"]
    cands = {s.name: LLMCandidate(s, task, factory(s)) for s in llm_specs}
    prices = {s.name: pricing.get(s.model) for s in llm_specs}
    if n >= ELIMINATION_MIN_N:
        first = max(ELIMINATION_FIRST_ROUND_MIN, n // 3)
    else:
        first = n  # too few examples to eliminate anyone safely: a single round
    order = np.random.default_rng(task.seed).permutation(n)
    rounds = [list(order[:first]), list(order[first:])] if first < n else [list(order)]
    active = list(cands)
    all_positions = list(range(n))
    expected = {name: cand.build_calls(scoring, split.shots, all_positions) for name, cand in cands.items()}

    def valid_records(name, calls=None):
        return store.valid_for(expected[name] if calls is None else calls)

    try:
        for r, positions in enumerate(rounds):
            for name in active:
                cand = cands[name]
                round_calls = cand.build_calls(scoring, split.shots, positions)
                execute_calls(round_calls, cand.completer, store,
                              price=prices[name], cap=cap, concurrency=concurrency, retries=retries,
                              sleep=sleep, secrets=secrets, seed_from_store=False)
            if r == 0 and len(rounds) == 2 and len(active) + len(ml_specs) >= 2:
                sub = np.array(sorted(positions))
                round_calls = {name: cands[name].build_calls(scoring, split.shots, positions) for name in active}
                recs = {name: valid_records(name, round_calls[name]) for name in active}
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
    except CostUnknownError as e:
        raise SafetyError(str(e)) from e
    except CostCapReached:
        incomplete = True
        warnings.append("The cost cap was reached; the run is incomplete. Re-run with --resume and a higher cap.")

    for name, cand in cands.items():
        res = cand.finalize(scoring, valid_records(name))
        res.eliminated = name not in active
        results[name] = res

    # --- decide which candidates take part in the comparison ---
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
    included = [n_ for n_ in results if status[n_] == "included"]

    y_enc, enc = encode_labels(task, scoring.y, {k: results[k].predictions for k in included})
    boot = rec = None
    equiv: dict[str, bool] = {}
    costs = {name: summarize_cost(res, task) for name, res in results.items()}
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

    # --- diagnostics ---
    if task.type in ("classify", "binary"):
        labels = task.labels
        mis = [{"id": m["id"], "label": labels[m["label_code"]], "disagree_fraction": m["disagree_fraction"]}
               for m in suspected_mislabels(scoring.ids, y_enc, enc)] if len(included) >= 2 else []
    else:
        mis = []
    dis = disagreements(scoring.ids, enc) if len(included) >= 2 and task.type != "regress" else []
    balance = class_balance(scoring.y) if task.type in ("classify", "binary") else {}
    widths = {k: boot.ci[k][1] - boot.ci[k][0] for k in included} if boot else {}
    warnings += sample_warnings(n, task.margin, balance, widths)

    pairwise = []
    if task.type in ("classify", "binary", "extract_exact") and task.metric_name == "accuracy" and len(included) >= 2:
        names = included
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairwise.append({"a": a, "b": b, "mcnemar_p": mcnemar_exact(enc[a] == y_enc, enc[b] == y_enc)})

    simulated = any(m.provider == "mock" for m in llm_specs)
    if simulated:
        warnings.insert(0, "SIMULATED: 'mock' candidates are fake models with preset accuracy. Do not use these results to choose a real model.")

    cand_rows = []
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
        cand_rows.append({
            "name": name, "kind": res.kind, "status": status[name], "note": notes.get(name, ""),
            "score": score, "ci_lo": lo, "ci_hi": hi,
            "cost_per_1k": c.per_1k, "total_cost_at_volume": c.total_at_volume, "price_known": c.price_known,
            "p50_latency_s": p50, "p95_latency_s": p95, "invalid_rate": res.invalid_rate,
            "error_rate": res.error_rate, "repeat_disagreement": res.repeat_disagreement,
            "equivalent": bool(equiv.get(name, False)),
        })

    data = {
        "task": {"type": task.type, "metric": task.metric_name, "margin": task.margin, "volume": task.volume,
                 "n_scoring": n, "shots_pool": task.shots_pool, "repeats": task.repeats,
                 "cv_folds": task.cv_folds, "cv_repeats": task.cv_repeats},
        "candidates": cand_rows,
        "recommendation": None if rec is None else {
            "status": rec.status, "winner": rec.winner, "best": rec.best, "equivalent": rec.equivalent,
            "inconclusive": rec.inconclusive, "extra_examples": rec.extra_examples, "reason": rec.reason},
        "diagnostics": {"mislabels": mis, "disagreement_ids": dis, "class_balance": balance},
        "pairwise": pairwise, "warnings": warnings, "incomplete": incomplete,
        "simulated": simulated, "elimination_log": elimination_log,
    }
    data = _clean(data)
    (run_dir / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "tiebreak_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "data_hash": _hash_dataset(scoring),
        "seed": task.seed, "n_boot": n_boot,
        "models": [{"name": m.name, "kind": m.kind, "provider": m.provider, "model": m.model} for m in models],
        "prices_used": {k: list(v) for k, v in pricing.items()},
    }, indent=2), encoding="utf-8")
    return ExperimentResult(run_dir, data)
