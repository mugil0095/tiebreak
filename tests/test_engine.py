
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

    from tiebreak.simulate import SimulatedCompleter

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
    from tiebreak.simulate import SimulatedCompleter

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
    from tiebreak.simulate import SimulatedCompleter

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
    from tiebreak.simulate import SimulatedCompleter

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
    from tiebreak.simulate import SimulatedCompleter

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
    from tiebreak.simulate import SimulatedCompleter

    def factory(spec):
        return SimulatedCompleter(0.34 if spec.name == "bad" else 0.97, TASK.labels, cost_per_call=0.001)

    res = run_experiment(TASK, [LR, mock("good", 0.97), mock("bad", 0.34)], dataset(200), tmp_path / "r",
                         completer_factory=factory, **NS)
    assert next(c for c in res.data["candidates"] if c["name"] == "bad")["status"] == "eliminated"
    assert res.data["recommendation"]["status"] != "provisional"


def _ollama(name):
    return ModelSpec(name=name, kind="llm", provider="ollama", model=name)


def _regress_ds(n=60):
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, n)
    y = np.round(x * 2 + rng.normal(0, 1, n), 2)
    return Dataset([f"e{i}" for i in range(n)], pd.DataFrame({"t": [f"x={v:.3f}" for v in x]}), y.astype(object), [f"x={v:.3f}" for v in x])


class _Numeric:
    def __init__(self, text=None, noise=0.0, seed=0):
        self.text, self.noise, self.truth, self.rng = text, noise, {}, np.random.default_rng(seed)

    def prime(self, m): self.truth.update(m)

    def complete(self, p):
        out = self.text if self.text is not None else str(float(self.truth[p]) + self.rng.normal(0, self.noise))
        return Completion(out, 10, 2, 0.001, 0.1)


def test_regression_abstainer_cannot_beat_a_real_model(tmp_path):
    """C2: invalid answers are scored as maximally wrong, not filled with the gold mean."""
    rtask = TaskConfig(type="regress", label_column="y", text_column="t", repeats=1)
    factory = lambda spec: _Numeric("I cannot answer") if spec.name == "abstain" else _Numeric(noise=8.0)
    res = run_experiment(rtask, [_ollama("abstain"), _ollama("real")], _regress_ds(), tmp_path / "r",
                         completer_factory=factory, **NS)
    rows = {c["name"]: c for c in res.data["candidates"]}
    assert rows["abstain"]["invalid_rate"] == 1.0
    assert rows["abstain"]["score"] < rows["real"]["score"]
    assert res.data["recommendation"]["winner"] == "real"
    assert any(w.startswith("abstain: 100% of answers could not be parsed and are scored as maximally wrong")
               for w in res.data["warnings"])
    assert not any(w.startswith("real:") and "parsed" in w for w in res.data["warnings"])


def test_classification_invalid_answers_above_five_percent_warn(tmp_path):
    from tiebreak.simulate import SimulatedCompleter

    class Chatty(SimulatedCompleter):
        def complete(self, p):
            c = super().complete(p)
            if sum(map(ord, p)) % 4 == 0:
                c.text = "hmm hard to say"
            return c

    res = run_experiment(TASK, [mock("chatty", 0.9), mock("fine", 0.9)], dataset(80), tmp_path / "r",
                         completer_factory=lambda s: Chatty(0.9, TASK.labels) if s.name == "chatty" else SimulatedCompleter(0.9, TASK.labels), **NS)
    ws = [w for w in res.data["warnings"] if "could not be parsed" in w]
    assert len(ws) == 1 and ws[0].startswith("chatty: ") and "scored as maximally wrong" in ws[0]


def _fixed_latency(latency, tin=100, tout=2, cost=None):
    from tiebreak.simulate import SimulatedCompleter

    class Fixed(SimulatedCompleter):
        def complete(self, p):
            c = super().complete(p)
            return Completion(c.text, tin, tout, cost, latency)
    return lambda spec: Fixed(0.9, TASK.labels)


def test_price_source_is_recorded_and_pricing_yaml_wins_over_provider_cost(tmp_path):
    models = [mock("priced", 0.9, 0.5), mock("provider", 0.9, 0.5)]
    pricing = {"priced": (0.1, 0.2)}
    res = run_experiment(TASK, models + [LR], dataset(40), tmp_path / "r", pricing=pricing, **NS)
    rows = {c["name"]: c for c in res.data["candidates"]}
    assert rows["priced"]["price_source"] == "pricing.yaml" and rows["provider"]["price_source"] == "provider"
    assert rows["logreg"]["price_source"] == "compute time"
    # priced: tokens x user price, not the 0.5/call the (mock) provider reported
    assert rows["priced"]["cost_per_1k"] < rows["provider"]["cost_per_1k"] / 100
    assert json.loads((tmp_path / "r" / "results.json").read_text())["candidates"][0]["price_source"]


def test_pricing_snapshot_and_manifest_list_only_prices_actually_used(tmp_path):
    pricing = {"priced": (0.1, 0.2), "never-run-model": (9.0, 9.0)}
    res = run_experiment(TASK, [mock("priced", 0.9), mock("provider", 0.9)], dataset(40), tmp_path / "r", pricing=pricing, **NS)
    snap = json.loads((tmp_path / "r" / "config.snapshot" / "pricing.json").read_text())
    assert snap == {"priced": [0.1, 0.2], "never-run-model": [9.0, 9.0]}
    man = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert man["prices_used"] == {"priced": [0.1, 0.2]}
    used = res.data["prices_used"]
    assert [u["candidate"] for u in used if u["source"] == "pricing.yaml"] == ["priced"]
    assert used[0]["input_per_1k"] == 0.1 and used[0]["output_per_1k"] == 0.2 and res.data["run_date"]
    assert {u["candidate"]: u["source"] for u in used}["provider"] == "provider"


def test_local_llm_is_costed_by_compute_time_not_zero(tmp_path):
    local = ModelSpec(name="loc", kind="llm", provider="ollama", model="tag")
    task = TASK.model_copy(update={"compute_price_per_hour": 3.6})  # 0.001 $/s
    res = run_experiment(task, [local, mock("m", 0.9)], dataset(40), tmp_path / "r",
                         completer_factory=lambda s: _fixed_latency(2.0, cost=0.001)(s), max_cost=5.0, **NS)
    row = next(c for c in res.data["candidates"] if c["name"] == "loc")
    assert row["price_source"] == "compute time" and row["price_known"] is True
    assert row["cost_per_1k"] == pytest.approx(2.0)  # 2 s x 0.001 $/s x 1000
    assert row["total_cost_at_volume"] == pytest.approx(20.0)


def test_mock_provider_rejects_regression_with_a_clear_error(tmp_path):
    rtask = TaskConfig(type="regress", label_column="y", text_column="t", repeats=1)
    with pytest.raises(ValueError, match="mock provider supports classification tasks only"):
        run_experiment(rtask, [mock("m", 0.9)], _regress_ds(), tmp_path / "r", **NS)


def test_is_hosted_considers_base_url_and_api_base_for_every_provider():
    spec = lambda **kw: ModelSpec(name="x", kind="llm", model="m", **kw)
    assert not is_hosted(spec(provider="openai", base_url="http://localhost:8000/v1"))
    assert not is_hosted(spec(provider="openai", base_url="http://127.0.0.1:8000/v1"))
    assert not is_hosted(spec(provider="openai", base_url="http://[::1]:8000/v1"))
    assert is_hosted(spec(provider="openai", base_url="https://proxy.example.com/v1"))
    assert is_hosted(spec(provider="ollama", params={"api_base": "https://remote.example.com"}))
    assert not is_hosted(spec(provider="ollama", params={"api_base": "http://localhost:11434"}))
    assert not is_hosted(spec(provider="ollama"))
    assert not is_hosted(spec(provider="mock", base_url="https://example.com"))
    assert is_hosted(spec(provider="anthropic")) and is_hosted(spec(provider="openai"))
    assert is_hosted(spec(provider="openai_compatible", base_url="https://api.example.com/v1"))


def test_remote_ollama_needs_consent(tmp_path):
    remote = ModelSpec(name="r", kind="llm", provider="ollama", model="x", params={"api_base": "https://remote.example.com"})
    with pytest.raises(SafetyError, match="yes-send-data"):
        run_experiment(TASK, [remote], dataset(40), tmp_path / "a", **NS)


def test_secrets_from_the_environment_are_scrubbed_even_when_not_in_api_key_env(tmp_path, monkeypatch):
    """I5: every set *_API_KEY / *_TOKEN / *_SECRET / *_PASSWORD env var is redacted from stored errors."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-envvalue-98765")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tokvalue-abcdef123")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short to be worth scrubbing

    class Leaky:
        def complete(self, p):
            raise ValueError("denied: sk-ant-envvalue-98765 and tokvalue-abcdef123 and abc")

    spec = ModelSpec(name="k", kind="llm", provider="mock", model="k")  # no api_key_env configured
    run_experiment(TASK, [LR, spec], dataset(20), tmp_path / "r", completer_factory=lambda s: Leaky(), retries=0, **NS)
    text = (tmp_path / "r" / "predictions.jsonl").read_text()
    assert "sk-ant-envvalue-98765" not in text and "tokvalue-abcdef123" not in text
    assert "and abc" in text  # short values are not scrubbed


def test_cap_counts_spend_from_before_a_prompt_change(tmp_path):
    """I6: edit the prompt, resume with a cap: the earlier (superseded) spend still counts."""
    first = run_experiment(TASK, [mock("a", 0.9, 0.01)], dataset(30), tmp_path / "r", **NS)  # $0.30 spent
    assert not first.data["incomplete"]
    edited = ModelSpec(name="a", kind="llm", provider="mock", model="a", prompt="Label {input} as one of {labels}.",
                       params={"accuracy": 0.9, "cost_per_call": 0.01})
    res = run_experiment(TASK, [edited], dataset(30), tmp_path / "r", max_cost=0.35, concurrency=1, **NS)
    assert res.data["incomplete"] is True  # 0.30 old + new calls hits 0.35 long before 30 new calls ($0.30 more)


def test_empty_answers_get_a_reasoning_model_hint(tmp_path):
    class Empty:
        def complete(self, p):
            return Completion("", 10, 64, 0.0, 0.1)

    res = run_experiment(TASK, [LR, mock("thinker", 0.9)], dataset(60), tmp_path / "r",
                         completer_factory=lambda spec: Empty(), **NS)
    text = " ".join(res.data["warnings"])
    assert "thinker" in text and "Most answers were empty" in text and "think: false" in text


def test_requests_per_minute_paces_the_calls(tmp_path):
    waits = []
    paced = mock("paced", 0.9).model_copy(update={"requests_per_minute": 600})
    run_experiment(TASK, [LR, paced], dataset(60), tmp_path / "r", concurrency=1,
                   n_boot=300, sleep=waits.append)
    assert len(waits) >= 50  # one wait before almost every call; the stub does not really sleep
