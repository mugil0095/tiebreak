
import json

import numpy as np
import pytest

from tiebreak.candidates.llm import Completion
from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.demo import make_reviews
from tiebreak.engine import SafetyError, is_hosted, run_experiment
import pandas as pd

TASK = TaskConfig(type="classify", label_column="sentiment", text_column="review",
                  labels=["negative", "neutral", "positive"], repeats=1, cv_folds=3, cv_repeats=1)
NS = dict(sleep=lambda s: None, n_boot=300)


def dataset(n=120):
    df = make_reviews(n, 0)
    return Dataset(df["id"].tolist(), df[["review"]], df["sentiment"].to_numpy(dtype=object), df["review"].tolist())


def mock(name, acc, cost=0.01):
    return ModelSpec(name=name, kind="llm", provider="mock", model=name, params={"accuracy": acc, "cost_per_call": cost})


LR = ModelSpec(name="logreg", kind="ml", estimator="logistic_regression")


def test_full_run_writes_folder_and_recommends(tmp_path):
    res = run_experiment(TASK, [LR, mock("big", 0.97, 0.05), mock("small", 0.96, 0.001)], dataset(), tmp_path / "run", **NS)
    d = res.data
    for f in ("manifest.json", "results.json", "predictions.jsonl", "config.snapshot/task.json", "config.snapshot/models.json"):
        assert (tmp_path / "run" / f).exists(), f
    rec = d["recommendation"]
    assert rec["winner"] in {"logreg", "small"}  # both cheap and tied with the best
    assert d["simulated"] is True and not d["incomplete"]
    assert d["warnings"][0].startswith("SIMULATED")
    assert {c["name"] for c in d["candidates"]} == {"logreg", "big", "small"}
    assert json.loads((tmp_path / "run" / "results.json").read_text())["task"]["n_scoring"] == 120


def test_resume_makes_no_repeat_calls(tmp_path):
    counts = {"n": 0}

    class Counting:
        def __init__(self, inner): self.inner = inner
        def prime(self, m): self.inner.prime(m)
        def complete(self, p):
            counts["n"] += 1
            return self.inner.complete(p)

    from tiebreak.candidates.llm import build_completer
    factory = lambda spec: Counting(build_completer(spec, TASK))
    models = [LR, mock("m1", 0.9)]
    run_experiment(TASK, models, dataset(60), tmp_path / "r", completer_factory=factory, **NS)
    first = counts["n"]
    run_experiment(TASK, models, dataset(60), tmp_path / "r", completer_factory=factory, **NS)
    assert first == 60 and counts["n"] == 60


def test_hosted_provider_needs_explicit_consent_and_local_only_blocks_it(tmp_path):
    hosted = ModelSpec(name="h", kind="llm", provider="anthropic", model="x")
    assert is_hosted(hosted) and not is_hosted(mock("m", 0.9))
    assert not is_hosted(ModelSpec(name="o", kind="llm", provider="ollama", model="x"))
    assert not is_hosted(ModelSpec(name="c", kind="llm", provider="openai_compatible", model="x", base_url="http://localhost:8000/v1"))
    assert is_hosted(ModelSpec(name="c", kind="llm", provider="openai_compatible", model="x", base_url="https://api.example.com/v1"))
    with pytest.raises(SafetyError, match="yes-send-data"):
        run_experiment(TASK, [hosted], dataset(40), tmp_path / "a", **NS)
    with pytest.raises(SafetyError, match="local-only"):
        run_experiment(TASK, [hosted], dataset(40), tmp_path / "b", local_only=True, allow_hosted=True, **NS)


def test_candidate_whose_every_call_errors_is_excluded_not_fatal(tmp_path):
    class Broken:
        def complete(self, p): raise TimeoutError("down")

    factory = lambda spec: Broken()
    res = run_experiment(TASK, [LR, mock("broken", 0.9)], dataset(60), tmp_path / "r",
                         completer_factory=factory, retries=0, **NS)
    broken = next(c for c in res.data["candidates"] if c["name"] == "broken")
    assert broken["status"] == "excluded" and broken["error_rate"] == 1.0
    assert res.data["recommendation"]["winner"] == "logreg"
    assert any("broken" in w for w in res.data["warnings"])


def test_clearly_worse_llm_is_eliminated_early_and_saves_calls(tmp_path):
    counts = {"bad": 0}

    from tiebreak.candidates.llm import SimulatedCompleter

    class Bad(SimulatedCompleter):
        def complete(self, p):
            counts["bad"] += 1
            return super().complete(p)

    def factory(spec):
        if spec.name == "bad":
            return Bad(0.34, TASK.labels)
        return SimulatedCompleter(0.97, TASK.labels, cost_per_call=0.001)

    res = run_experiment(TASK, [LR, mock("good", 0.97), mock("bad", 0.34)], dataset(200), tmp_path / "r",
                         completer_factory=factory, **NS)
    bad = next(c for c in res.data["candidates"] if c["name"] == "bad")
    assert bad["status"] == "eliminated" and counts["bad"] < 200
    assert res.data["elimination_log"]


def test_cost_cap_marks_run_incomplete_and_does_not_recommend_partial_llms(tmp_path):
    res = run_experiment(TASK, [LR, mock("big", 0.97, 0.05)], dataset(60), tmp_path / "r", max_cost=0.5, **NS)
    assert res.data["incomplete"] is True
    assert any("cost cap" in w.lower() for w in res.data["warnings"])
    big = next(c for c in res.data["candidates"] if c["name"] == "big")
    assert big["status"] == "excluded"
    assert res.data["recommendation"]["winner"] == "logreg"


def test_ml_candidates_cannot_run_on_extract_tasks(tmp_path):
    task = TaskConfig(type="extract_exact", label_column="y")
    ds = Dataset(["a", "b"], pd.DataFrame({"x": [1, 2]}), np.array(["u", "v"], dtype=object), None)
    with pytest.raises(ValueError, match="extract_exact"):
        run_experiment(task, [LR], ds, tmp_path / "r", **NS)


def test_results_json_is_strict_json_with_null_for_nan_scores(tmp_path):
    class Broken:
        def complete(self, p): raise TimeoutError("down")

    run_experiment(TASK, [LR, mock("broken", 0.9)], dataset(60), tmp_path / "r",
                   completer_factory=lambda spec: Broken(), retries=0, **NS)
    text = (tmp_path / "r" / "results.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    broken = next(c for c in json.loads(text)["candidates"] if c["name"] == "broken")
    assert broken["score"] is None and broken["ci_lo"] is None


def test_unknown_cost_with_a_cap_is_a_safety_error(tmp_path):
    from tiebreak.candidates.llm import SimulatedCompleter

    class NoCost(SimulatedCompleter):
        def complete(self, p):
            c = super().complete(p)
            c.cost_usd = None
            return c

    with pytest.raises(SafetyError, match="unknown"):
        run_experiment(TASK, [mock("m", 0.9)], dataset(30), tmp_path / "r", max_cost=1.0,
                       completer_factory=lambda spec: NoCost(0.9, TASK.labels), concurrency=1, **NS)


def test_cost_cap_is_shared_across_candidates_and_not_double_counted_on_resume(tmp_path):
    models = [mock("a", 0.9, 0.01), mock("b", 0.9, 0.01)]
    # 2 x 20 calls x $0.01 = $0.40 in total; a $0.45 cap must not trigger, even after a resume
    first = run_experiment(TASK, models, dataset(20), tmp_path / "r", max_cost=0.45, **NS)
    assert not first.data["incomplete"]
    again = run_experiment(TASK, models, dataset(20), tmp_path / "r", max_cost=0.45, **NS)
    assert not again.data["incomplete"]


def test_api_key_values_are_scrubbed_from_stored_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "supersecretvalue123")

    class Leaky:
        def complete(self, p): raise ValueError("bad request for key supersecretvalue123")

    spec = ModelSpec(name="k", kind="llm", provider="mock", model="k", api_key_env="MY_KEY")
    run_experiment(TASK, [LR, spec], dataset(20), tmp_path / "r", completer_factory=lambda s: Leaky(), retries=0, **NS)
    assert "supersecretvalue123" not in (tmp_path / "r" / "predictions.jsonl").read_text()


def _scores(res):
    return {c["name"]: (c["status"], c["score"]) for c in res.data["candidates"]}


def test_reducing_repeats_in_same_folder_matches_a_fresh_run(tmp_path):
    from collections import Counter

    class Drifting:
        """Right on the first call for a prompt, wrong on later ones (so repeats 1-2 would flip the vote)."""
        def __init__(self): self.seen, self.truth = Counter(), {}
        def prime(self, m): self.truth.update(m)
        def complete(self, p):
            self.seen[p] += 1
            t = self.truth[p]
            return Completion(t if self.seen[p] == 1 else next(l for l in TASK.labels if l != t), 10, 2, 0.001, 0.1)

    t3 = TASK.model_copy(update={"repeats": 3})
    models = [LR, mock("m", 0.8)]
    f = lambda spec: Drifting()
    run_experiment(t3, models, dataset(60), tmp_path / "r", completer_factory=f, concurrency=1, **NS)
    again = run_experiment(TASK, models, dataset(60), tmp_path / "r", completer_factory=f, concurrency=1, **NS)
    fresh = run_experiment(TASK, models, dataset(60), tmp_path / "fresh", completer_factory=f, concurrency=1, **NS)
    assert _scores(again) == _scores(fresh)
    assert dict(_scores(fresh))["m"][1] > 0.9


def test_old_config_answers_never_fill_a_partially_run_candidate(tmp_path):
    from tiebreak.candidates.llm import SimulatedCompleter

    def factory(acc):
        return lambda spec: SimulatedCompleter(acc, TASK.labels, cost_per_call=0.001) if spec.name != "logreg" else None

    old = ModelSpec(name="m", kind="llm", provider="mock", model="m", params={"accuracy": 0.97})
    new = ModelSpec(name="m", kind="llm", provider="mock", model="m", params={"accuracy": 0.30})
    run_experiment(TASK, [LR, old], dataset(200), tmp_path / "r", completer_factory=factory(0.97), **NS)
    res = run_experiment(TASK, [LR, new], dataset(200), tmp_path / "r", completer_factory=factory(0.30), **NS)
    m = next(c for c in res.data["candidates"] if c["name"] == "m")
    assert m["status"] == "eliminated"
    assert m["score"] < 0.6  # only the new low-accuracy answers; old answers would give ~0.75


def test_two_round_path_flags_failing_candidate_instead_of_eliminating(tmp_path):
    class Broken:
        def complete(self, p): raise TimeoutError("down")

    res = run_experiment(TASK, [LR, mock("broken", 0.9)], dataset(200), tmp_path / "r",
                         completer_factory=lambda s: Broken(), retries=0, **NS)
    b = next(c for c in res.data["candidates"] if c["name"] == "broken")
    assert b["status"] == "excluded" and b["error_rate"] == 1.0
    assert any("broken" in w and "failed" in w for w in res.data["warnings"])
    assert not res.data["elimination_log"]


def test_partial_round_one_errors_do_not_cause_elimination(tmp_path):
    from tiebreak.candidates.llm import SimulatedCompleter

    class Flaky(SimulatedCompleter):
        def complete(self, p):
            c = super().complete(p)
            if sum(map(ord, p)) % 7 == 0:
                raise TimeoutError("flaky")
            return c

    def factory(spec):
        return Flaky(0.34, TASK.labels) if spec.name == "bad" else SimulatedCompleter(0.97, TASK.labels)

    res = run_experiment(TASK, [LR, mock("good", 0.97), mock("bad", 0.34)], dataset(200), tmp_path / "r",
                         completer_factory=factory, retries=0, **NS)
    bad = next(c for c in res.data["candidates"] if c["name"] == "bad")
    assert 0 < bad["error_rate"] < 0.5
    assert bad["status"] != "eliminated" and not res.data["elimination_log"]


def test_snapshot_never_contains_secrets(tmp_path):
    spec = ModelSpec(name="m", kind="llm", provider="mock", model="m", base_url="http://user:pw123@localhost:8000/v1",
                     params={"accuracy": 0.9, "api_key": "sk-abc123456", "max_tokens": 256, "num_tokens": 7, "token": "tok-777",
                             "access_token": "at-888", "nested": {"Authorization": "Bearer zzz"}})
    run_experiment(TASK, [spec], dataset(30), tmp_path / "r", **NS)
    text = (tmp_path / "r" / "config.snapshot" / "models.json").read_text()
    for secret in ("sk-abc123456", "pw123", "Bearer zzz", "tok-777", "at-888"):
        assert secret not in text
    assert "localhost:8000" in text and '"accuracy": 0.9' in text
    assert '"max_tokens": 256' in text and '"num_tokens": 7' in text  # benign keys are kept


def test_fewer_than_two_scoring_examples_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        run_experiment(TASK, [mock("m", 0.9)], dataset(1), tmp_path / "r", **NS)


class _Flaky:
    """Wraps a simulated model; a deterministic share of prompts always fails (transient provider errors)."""

    def __init__(self, inner, rate, fail=True):
        self.inner, self.rate, self.fail = inner, rate, fail

    def prime(self, m): self.inner.prime(m)

    def complete(self, p):
        h = int.from_bytes(__import__("hashlib").sha256(("flaky" + p).encode()).digest()[:4], "big") / 2**32
        if self.fail and h < self.rate:
            raise TimeoutError("503 service unavailable")
        return self.inner.complete(p)


def _flaky_factory(rate, fail=True):
    from tiebreak.candidates.llm import SimulatedCompleter

    def factory(spec):
        acc = {"best": 0.97, "ok": 0.75}[spec.name]
        inner = SimulatedCompleter(acc, TASK.labels, cost_per_call=0.01 if spec.name == "best" else 0.001)
        return _Flaky(inner, rate, fail) if spec.name == "best" else inner
    return factory


def test_transient_errors_below_half_exclude_the_candidate_instead_of_scoring_them_wrong(tmp_path):
    """C1: 0.97 model with ~31% provider errors must not be scored as a 0.67 model and silently lose."""
    res = run_experiment(TASK, [mock("best", 0.97), mock("ok", 0.75)], dataset(100), tmp_path / "r",
                         completer_factory=_flaky_factory(0.31), retries=0, **NS)
    best = next(c for c in res.data["candidates"] if c["name"] == "best")
    assert 0.15 < best["error_rate"] < 0.5
    assert best["status"] == "excluded" and best["note"].startswith("incomplete: provider errors")
    assert any(w.startswith("best: ") and "have no successful answer (provider errors)" in w and "--resume" in w
               for w in res.data["warnings"])
    rec = res.data["recommendation"]
    assert rec["status"] == "provisional" and "best" in rec["reason"] and "provider errors" in rec["reason"]


def test_no_errors_is_unchanged_and_resume_completes_an_incomplete_candidate(tmp_path):
    clean = run_experiment(TASK, [mock("best", 0.97), mock("ok", 0.75)], dataset(100), tmp_path / "c",
                           completer_factory=_flaky_factory(0.0), retries=0, **NS)
    assert {c["status"] for c in clean.data["candidates"]} <= {"included", "eliminated"}
    assert clean.data["recommendation"]["status"] != "provisional"
    run_experiment(TASK, [mock("best", 0.97), mock("ok", 0.75)], dataset(100), tmp_path / "r",
                   completer_factory=_flaky_factory(0.31), retries=0, **NS)
    again = run_experiment(TASK, [mock("best", 0.97), mock("ok", 0.75)], dataset(100), tmp_path / "r",
                           completer_factory=_flaky_factory(0.31, fail=False), retries=0, **NS)
    best = next(c for c in again.data["candidates"] if c["name"] == "best")
    assert best["status"] == "included" and best["score"] > 0.9
    assert again.data["recommendation"]["status"] != "provisional"


def test_cost_cap_run_is_provisional_and_a_clean_run_is_recommended(tmp_path):
    capped = run_experiment(TASK, [LR, mock("big", 0.97, 0.05)], dataset(60), tmp_path / "a", max_cost=0.5, **NS)
    rec = capped.data["recommendation"]
    assert rec["status"] == "provisional" and rec["reason"].startswith("Provisional: ")
    assert "big" in rec["reason"] and "cost cap" in rec["reason"].lower()
    clean = run_experiment(TASK, [LR, mock("big", 0.97, 0.05)], dataset(60), tmp_path / "b", **NS)
    assert clean.data["recommendation"]["status"] in {"recommended", "inconclusive"}
    assert not clean.data["recommendation"]["reason"].startswith("Provisional")


def test_early_elimination_alone_does_not_make_the_run_provisional(tmp_path):
    from tiebreak.candidates.llm import SimulatedCompleter

    def factory(spec):
        return SimulatedCompleter(0.34 if spec.name == "bad" else 0.97, TASK.labels, cost_per_call=0.001)

    res = run_experiment(TASK, [LR, mock("good", 0.97), mock("bad", 0.34)], dataset(200), tmp_path / "r",
                         completer_factory=factory, **NS)
    assert next(c for c in res.data["candidates"] if c["name"] == "bad")["status"] == "eliminated"
    assert res.data["recommendation"]["status"] != "provisional"
