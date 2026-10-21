import numpy as np
import pandas as pd
import pytest

from tiebreak import judge_run as jr
from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.judge import JudgeScore
from tiebreak.runner import CostCap

TASK = TaskConfig(type="judge", prompt_column="prompt")
N = 12


def _ds(n=N):
    prompts = [f"question number {i}" for i in range(n)]
    return Dataset([f"q{i}" for i in range(n)], pd.DataFrame({"x": prompts}), np.full(n, 5.0), prompts)


def _cand(name, words, cost=0.002, **params):
    return ModelSpec(name=name, kind="llm", provider="mock", model=name,
                     params={"answer_words": words, "cost_per_call": cost, **params})


JUDGE = ModelSpec(name="judge", kind="llm", provider="mock", model="j", role="judge",
                  params={"bias_by_length": True, "cost_per_call": 0.001})
QUIET = {"retries": 0, "sleep": lambda s: None, "concurrency": 1}


def _run(tmp_path, cands, judge=JUDGE, ds=None, **kw):
    return jr.run_judging(TASK, ds or _ds(), cands, judge, tmp_path, exec_kw=QUIET, **kw)


def test_scores_per_candidate_and_example(tmp_path):
    res = _run(tmp_path, [_cand("long", 200), _cand("short", 1)])
    assert res.by_candidate["long"].shape == (N,)
    assert np.all(res.by_candidate["long"] == 5.0) and np.all(res.by_candidate["short"] == 1.0)
    assert all(a for a in res.answers["long"])
    assert len(res.answer_calls["long"]) == N and len(res.judge_calls) == 2 * N
    assert res.evaluation_cost == pytest.approx(2 * N * 0.001)
    assert res.incomplete is False and res.notes == {}


def test_resume_reuses_cached_answers_and_judgments(tmp_path, monkeypatch):
    counts = {"answer": 0, "judge": 0}
    real_answerer, real_judge = jr.build_answer_completer, jr.build_judge

    def counting_answerer(spec, task):
        inner = real_answerer(spec, task)

        class C:
            def complete(self, prompt):
                counts["answer"] += 1
                return inner.complete(prompt)
        return C()

    def counting_judge(spec, task):
        inner = real_judge(spec, task)

        class J:
            name = inner.name

            def score(self, prompt, answer):
                counts["judge"] += 1
                return inner.score(prompt, answer)
        return J()

    monkeypatch.setattr(jr, "build_answer_completer", counting_answerer)
    monkeypatch.setattr(jr, "build_judge", counting_judge)
    cands = [_cand("long", 200), _cand("short", 1)]
    first = _run(tmp_path, cands)
    assert counts == {"answer": 2 * N, "judge": 2 * N}
    second = _run(tmp_path, cands)
    assert counts == {"answer": 2 * N, "judge": 2 * N}  # nothing was called again
    assert np.array_equal(first.by_candidate["long"], second.by_candidate["long"])


def test_answer_cache_invalidates_when_params_change(tmp_path):
    first = _run(tmp_path, [_cand("c", 200)])
    second = _run(tmp_path, [_cand("c", 1)])
    assert np.all(first.by_candidate["c"] == 5.0) and np.all(second.by_candidate["c"] == 1.0)


class StubJudge:
    name = "stub"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def score(self, prompt, answer):
        self.calls += 1
        return self.reply(prompt, answer)


def test_unreadable_judge_replies_exclude_the_candidate_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(jr, "build_judge", lambda spec, task: StubJudge(
        lambda p, a: JudgeScore(None, None, 5, 0.0, "unparseable judge reply", "hmm")))
    res = _run(tmp_path, [_cand("c", 50)])
    assert np.all(np.isnan(res.by_candidate["c"]))
    assert "judge" in res.notes["c"].lower()


def test_failed_judge_calls_are_not_scored_as_worst(tmp_path, monkeypatch):
    def reply(p, a):
        if "number 3" in p:
            return JudgeScore(None, None, 0, None, "503 unavailable", "", failed=True)
        return JudgeScore(4.0, 0.9, 5, 0.0, None, "4")

    monkeypatch.setattr(jr, "build_judge", lambda spec, task: StubJudge(reply))
    res = _run(tmp_path, [_cand("c", 50)])
    scores = res.by_candidate["c"]
    assert np.isnan(scores[3]) and np.count_nonzero(~np.isnan(scores)) == N - 1
    assert np.all(scores[~np.isnan(scores)] == 4.0)
    assert "c" not in res.notes  # partly judged is fine; only "nothing judged" excludes


def test_an_empty_answer_is_scored_worst_without_calling_the_judge(tmp_path, monkeypatch):
    stub = StubJudge(lambda p, a: JudgeScore(4.0, 0.9, 5, 0.0, None, "4"))
    monkeypatch.setattr(jr, "build_judge", lambda spec, task: stub)
    res = _run(tmp_path, [_cand("blank", 0)])
    assert np.all(res.by_candidate["blank"] == 1.0)  # the bottom of the scale
    assert stub.calls == 0
    assert "empty" in res.notes["blank"].lower()


def test_cost_cap_stops_spending_and_flags_incomplete(tmp_path):
    res = _run(tmp_path, [_cand("c", 50, cost=0.002)], cap=CostCap(0.005))
    assert res.incomplete is True
    assert np.count_nonzero(~np.isnan(res.by_candidate["c"])) < N


def test_a_fully_cached_run_does_not_need_the_judge(tmp_path, monkeypatch):
    first = _run(tmp_path, [_cand("a", 50)])

    def no_key(spec, task):
        raise ValueError("environment variable TYPESAFE_API_KEY is not set or empty")

    monkeypatch.setattr(jr, "build_judge", no_key)
    second = _run(tmp_path, [_cand("a", 50)])
    assert np.array_equal(first.by_candidate["a"], second.by_candidate["a"])


def test_a_missing_judge_fails_fast_when_judging_is_pending(tmp_path, monkeypatch):
    def no_key(spec, task):
        raise ValueError("environment variable TYPESAFE_API_KEY is not set or empty")

    monkeypatch.setattr(jr, "build_judge", no_key)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        _run(tmp_path, [_cand("a", 50)])
