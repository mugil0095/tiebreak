from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .candidates.base import CallRecord
from .candidates.llm import Call, Completion, build_completer, config_hash, hash_text, is_local, row_text
from .config import ModelSpec, TaskConfig
from .data import Dataset
from .judge import JudgeScore, build_judge
from .runner import CostCap, CostCapReached, PredictionStore, execute_calls

GENERATION_MAX_TOKENS = 512  # answers are free text, so the label-task default of 64 would cut them off


@dataclass
class JudgeScores:
    by_candidate: dict[str, np.ndarray] = field(default_factory=dict)  # score per example, nan = not scored
    answers: dict[str, list[str | None]] = field(default_factory=dict)  # None = no answer (every call failed)
    judgments: dict[str, list[JudgeScore | None]] = field(default_factory=dict)
    answer_calls: dict[str, list[CallRecord]] = field(default_factory=dict)  # generation calls: cost and latency
    judge_calls: list[CallRecord] = field(default_factory=list)  # judge calls: the evaluation cost
    notes: dict[str, str] = field(default_factory=dict)  # candidate -> why it is excluded or what to know
    incomplete: bool = False  # the cost cap stopped the run
    evaluation_cost: float = 0.0


class MockAnswerCompleter:
    """SIMULATED candidate for judge tasks: a deterministic answer of `answer_words` words, no model is called."""

    def __init__(self, spec: ModelSpec):
        p = spec.params
        self.words = int(p.get("answer_words", 40))
        self.cost = float(p.get("cost_per_call", 0.0005))
        self.latency = float(p.get("latency_s", 0.2))

    def complete(self, prompt: str) -> Completion:
        return Completion(" ".join(["word"] * self.words), len(prompt.split()), self.words, self.cost, self.latency)


class _JudgeCompleter:
    """Lets the judge run through execute_calls (cache, retries, cost cap). The call's prompt is the JSON
    pair [question, answer]; a failed judge call raises so it is recorded as an error, an unreadable reply
    is a successful call whose stored value is null."""

    def __init__(self, judge):
        self.judge = judge

    def complete(self, composite: str) -> Completion:
        question, answer = json.loads(composite)
        t0 = time.perf_counter()
        s = self.judge.score(question, answer)
        if s.failed:
            raise RuntimeError(s.error)
        text = json.dumps({"value": s.value, "confidence": s.confidence, "error": s.error, "raw": s.raw})
        return Completion(text, s.tokens_in, 0, s.cost_usd, time.perf_counter() - t0)


class _NoCalls:
    """Stands in when every judgment is already cached."""

    def complete(self, prompt: str):
        raise RuntimeError("a cached judgment was expected but is missing")


def _effective(spec: ModelSpec) -> ModelSpec:
    if spec.provider == "mock":
        return spec
    return spec.model_copy(update={"params": {"max_tokens": GENERATION_MAX_TOKENS, **spec.params}})


def build_answer_completer(spec: ModelSpec, task: TaskConfig):
    return MockAnswerCompleter(spec) if spec.provider == "mock" else build_completer(spec, task)


def _execute(calls, completer, store, price, cap, exec_kw) -> tuple[dict, bool]:
    """Run calls through the shared machinery. Returns ({key: record}, cap_reached)."""
    try:
        records = execute_calls(calls, completer, store, price=price, cap=cap, **exec_kw)
        return {c.key: r for c, r in zip(calls, records)}, False
    except CostCapReached:
        return {(r.candidate, r.example_id, r.repeat): r for r in store.valid_for(calls)}, True


def run_judging(task: TaskConfig, dataset: Dataset, candidates: list[ModelSpec], judge_spec: ModelSpec,
                run_dir, *, cap: CostCap | None = None, prices: dict | None = None,
                exec_kw: dict | None = None) -> JudgeScores:
    """Every candidate answers every prompt, then the judge scores each answer.

    Answers are cached in predictions.jsonl and judgments in judgments.jsonl, so a resumed run repeats nothing."""
    run_dir, exec_kw, cap = Path(run_dir), dict(exec_kw or {}), cap or CostCap(None)
    prices = prices or {}
    answer_store = PredictionStore(run_dir / "predictions.jsonl")
    judgment_store = PredictionStore(run_dir / "judgments.jsonl")
    judge_completer = None  # built when a judgment is actually missing: a fully cached run needs no judge or key
    judge_cfg = hash_text(json.dumps([config_hash(judge_spec), task.rubric, list(task.scale)]))
    judge_price = (0.0, 0.0) if is_local(judge_spec) else prices.get(judge_spec.name)
    lo = float(task.scale[0])
    n = len(dataset)
    out = JudgeScores()

    for spec in candidates:
        eff = _effective(spec)
        cfg = config_hash(eff)
        price = (0.0, 0.0) if is_local(spec) else prices.get(spec.name)
        calls = [Call((spec.name, dataset.ids[i], 0), row_text(dataset, i), cfg) for i in range(n)]
        recs, capped = _execute(calls, build_answer_completer(eff, task), answer_store, price, cap, exec_kw)
        out.incomplete |= capped
        answers = [None if (r := recs.get(c.key)) is None or r.error is not None else r.raw for c in calls]
        out.answers[spec.name] = answers
        out.answer_calls[spec.name] = [r for r in recs.values()]

        scores = np.full(n, np.nan)
        judgments: list[JudgeScore | None] = [None] * n
        to_judge = []
        for i, a in enumerate(answers):
            if a is not None and not a.strip():
                scores[i] = lo  # an empty answer is the candidate's failure: the worst score, no judge call
            elif a is not None:
                to_judge.append(i)
        empties = sum(1 for a in answers if a is not None and not a.strip())

        if to_judge and not cap.exceeded:
            jcalls = [Call((f"judge:{spec.name}", dataset.ids[i], 0), json.dumps([row_text(dataset, i), answers[i]]), judge_cfg)
                      for i in to_judge]
            if len(judgment_store.valid_for(jcalls)) < len(jcalls):
                if judge_completer is None:
                    judge_completer = _JudgeCompleter(build_judge(judge_spec, task))  # fails fast: no key, no SDK
                completer = judge_completer
            else:
                completer = _NoCalls()
            jrecs, capped = _execute(jcalls, completer, judgment_store, judge_price, cap, exec_kw)
            out.incomplete |= capped
            out.judge_calls += list(jrecs.values())
            for i, c in zip(to_judge, jcalls):
                r = jrecs.get(c.key)
                if r is None:
                    continue
                if r.error is not None:
                    judgments[i] = JudgeScore(None, None, r.tokens_in, r.cost_usd, r.error, "", failed=True)
                    continue
                data = json.loads(r.raw)
                judgments[i] = JudgeScore(data["value"], data["confidence"], r.tokens_in, r.cost_usd, data["error"], data["raw"])
                if data["value"] is not None:
                    scores[i] = float(data["value"])
        elif to_judge:
            out.incomplete = True

        out.by_candidate[spec.name] = scores
        out.judgments[spec.name] = judgments
        if empties:
            out.notes[spec.name] = f"{empties} empty answer(s) were scored as the worst score"
        if np.all(np.isnan(scores)) and not out.incomplete:
            out.notes[spec.name] = "excluded: the judge returned no usable score for any of its answers"

    out.evaluation_cost = float(sum(r.cost_usd or 0.0 for r in out.judge_calls))
    return out
