from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import numpy as np

from ..config import ModelSpec, TaskConfig
from ..data import Dataset
from ..parser import parse_answer
from .base import CallRecord, CandidateResult, aggregate


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    latency_s: float


class Completer(Protocol):
    def complete(self, prompt: str) -> Completion: ...


@dataclass(frozen=True)
class Call:
    key: tuple[str, str, int]  # (candidate, example_id, repeat)
    prompt: str
    config_hash: str = ""  # hash of provider+model+params; a cached answer is only valid for the same config


class NonRetryableError(ValueError):
    """A failure that retrying cannot fix (bad configuration, missing credentials)."""

    retryable = False


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def is_hosted(spec: ModelSpec) -> bool:
    """True if running this candidate sends example text to a third party. Any base_url / api_base that is
    not a loopback address means hosted, whatever the provider; mock and an ollama without api_base are local."""
    if spec.kind != "llm" or spec.provider == "mock":
        return False
    url = spec.base_url or spec.params.get("api_base")
    if url:
        url = str(url)
        host = urlparse(url if "://" in url else "//" + url).hostname
        return host not in LOCAL_HOSTS
    return spec.provider != "ollama"


def is_local(spec: ModelSpec) -> bool:
    """A real model running on this machine: free per call, but it costs compute time."""
    return spec.kind == "llm" and spec.provider != "mock" and not is_hosted(spec)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_hash(spec: ModelSpec) -> str:
    payload = json.dumps([spec.provider, spec.model, spec.base_url, spec.params], sort_keys=True, default=str)
    return hash_text(payload)


_INSTRUCTIONS = {
    "classify": "Classify the input into exactly one of these labels: {labels}. Answer with only the label.",
    "binary": "Classify the input into exactly one of these labels: {labels}. Answer with only the label.",
    "extract_exact": "Extract the requested answer from the input. Answer with only the extracted text.",
    "regress": "Predict the numeric value for the input. Answer with only a number.",
}


def row_text(ds: Dataset, i: int) -> str:
    if ds.text is not None:
        return ds.text[i]
    row = ds.frame.iloc[i]
    return "; ".join(f"{c}: {row[c]}" for c in ds.frame.columns)


def render_prompt(task: TaskConfig, input_text: str, shots_text: str, template: str | None) -> str:
    labels = ", ".join(task.labels or [])
    if template:
        return (
            template.replace("{input}", input_text)
            .replace("{labels}", labels)
            .replace("{examples}", shots_text)
        )
    parts = [_INSTRUCTIONS[task.type].replace("{labels}", labels)]
    if shots_text:
        parts.append("Examples:\n" + shots_text)
    parts.append(f"Input: {input_text}\nAnswer:")
    return "\n\n".join(parts)


class OpenAICompatCompleter:
    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, client=None):
        try:
            import httpx
        except ImportError as e:
            raise ImportError("httpx is not installed; run: pip install tiebreak[llm]") from e
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key_env = model, api_key_env
        self.client = client or httpx.Client(timeout=60)

    def complete(self, prompt: str) -> Completion:
        headers = {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise NonRetryableError(f"environment variable {self.api_key_env} is not set or empty")
            headers["Authorization"] = f"Bearer {key}"
        t0 = time.perf_counter()
        r = self.client.post(self.url, headers=headers, json={
            "model": self.model, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage") or {}
        return Completion(
            body["choices"][0]["message"]["content"] or "",
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
            None, time.perf_counter() - t0,
        )


class LiteLLMCompleter:
    def __init__(self, provider: str, model: str, params: dict | None = None,
                 base_url: str | None = None, api_key_env: str | None = None):
        params = dict(params or {})
        self.timeout = params.pop("timeout", 60)
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm is not installed; run: pip install tiebreak[llm]") from e
        self.litellm, self.params = litellm, params
        self.model_id = f"{provider}/{model}"
        self.base_url, self.api_key_env = base_url, api_key_env

    def complete(self, prompt: str) -> Completion:
        extra = dict(self.params)
        if self.base_url:
            extra["api_base"] = self.base_url
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:  # name the variable, never a value
                raise NonRetryableError(f"environment variable {self.api_key_env} is not set or empty")
            extra["api_key"] = key
        t0 = time.perf_counter()
        # defaults suit short labels; a reasoning model may need params: {max_tokens: ...} or {think: false}
        call = {"temperature": 0, "max_tokens": 64, "timeout": self.timeout, **extra}
        resp = self.litellm.completion(model=self.model_id, messages=[{"role": "user", "content": prompt}], **call)
        latency = time.perf_counter() - t0
        try:
            cost = float(self.litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = None
        usage = getattr(resp, "usage", None)
        return Completion(
            resp.choices[0].message.content or "",
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            cost, latency,
        )


def check_task_support(spec: ModelSpec, task: TaskConfig) -> None:
    if spec.provider == "mock" and spec.kind == "llm" and task.type not in ("classify", "binary"):
        raise ValueError(
            f"candidate '{spec.name}': the mock provider supports classification tasks only (classify, binary), "
            f"not '{task.type}'")


def build_completer(spec: ModelSpec, task: TaskConfig) -> Completer:
    check_task_support(spec, task)
    if spec.provider == "mock":
        from ..simulate import SimulatedCompleter  # simulation only; imported lazily

        p = spec.params
        return SimulatedCompleter(
            accuracy=float(p.get("accuracy", 0.8)), labels=task.labels or [], seed=int(p.get("seed", 0)),
            cost_per_call=float(p.get("cost_per_call", 0.0005)), latency_s=float(p.get("latency_s", 0.2)),
        )
    if spec.provider == "openai_compatible":
        if not spec.base_url:
            raise ValueError(f"candidate '{spec.name}': openai_compatible needs base_url")
        return OpenAICompatCompleter(spec.base_url, spec.model, spec.api_key_env)
    return LiteLLMCompleter(spec.provider, spec.model, spec.params, spec.base_url, spec.api_key_env)


class LLMCandidate:
    kind = "llm"

    def __init__(self, spec: ModelSpec, task: TaskConfig, completer: Completer):
        self.spec, self.task, self.completer, self.name = spec, task, completer, spec.name

    @staticmethod
    def _shots_text(shots: Dataset | None) -> str:
        if shots is None or len(shots) == 0:
            return ""
        return "\n\n".join(f"Input: {row_text(shots, i)}\nAnswer: {shots.y[i]}" for i in range(len(shots)))

    def build_calls(self, scoring: Dataset, shots: Dataset | None, positions=None) -> list[Call]:
        positions = range(len(scoring)) if positions is None else positions
        shots_text = self._shots_text(shots)
        calls, truth = [], {}
        cfg = config_hash(self.spec)
        for i in positions:
            prompt = render_prompt(self.task, row_text(scoring, i), shots_text, self.spec.prompt)
            truth[prompt] = str(scoring.y[i])
            for r in range(self.task.repeats):
                calls.append(Call((self.name, scoring.ids[i], r), prompt, cfg))
        prime = getattr(self.completer, "prime", None)  # simulation hook: simulated completers are handed the true answers
        if prime:
            prime(truth)
        return calls

    def finalize(self, scoring: Dataset, records: list[CallRecord]) -> CandidateResult:
        index = {eid: i for i, eid in enumerate(scoring.ids)}
        votes: dict[int, list] = defaultdict(list)
        seen: set[int] = set()
        answered: set[int] = set()  # examples with at least one call that did not fail
        n_calls = n_err = n_invalid = 0
        used: list[CallRecord] = []
        for rec in sorted(records, key=lambda r: (r.example_id, r.repeat)):
            if rec.example_id not in index:
                continue
            used.append(rec)
            n_calls += 1
            i = index[rec.example_id]
            seen.add(i)
            if rec.error is not None:
                n_err += 1
                continue
            answered.add(i)
            value = parse_answer(rec.raw, self.task)
            if value is None:
                n_invalid += 1
                continue
            votes[i].append(value)

        classification = self.task.type != "regress"
        preds: list = [None] * len(scoring)
        mask = np.zeros(len(scoring), dtype=bool)
        disagree = 0
        for i in answered:  # an example with only provider errors is NOT evaluated (never scored as wrong)
            mask[i] = True
            preds[i] = aggregate(votes[i], classification)
            if len(set(votes[i])) > 1:
                disagree += 1
        lat = [r.latency_s for r in used if r.error is None]
        return CandidateResult(
            name=self.name, kind="llm", predictions=preds, calls=used,
            infer_seconds_per_pred=float(np.mean(lat)) if lat else 0.0, latencies=lat,
            invalid_rate=n_invalid / n_calls if n_calls else 0.0,
            error_rate=n_err / n_calls if n_calls else 0.0,
            repeat_disagreement=disagree / len(answered) if answered else 0.0, mask=mask,
            n_unanswered=len(seen - answered),
        )
