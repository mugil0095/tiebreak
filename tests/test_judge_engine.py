import numpy as np
import pandas as pd
import pytest

from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.engine import SafetyError, run_experiment
from tiebreak.judge_engine import spearman_or_none

N = 30
QUIET = dict(n_boot=300, concurrency=1, retries=0, sleep=lambda s: None)


def _task(**kw):
    return TaskConfig(type="judge", prompt_column="prompt", **kw)


def _ds(human=None):
    prompts = [f"question number {i}" for i in range(N)]
    frame = pd.DataFrame({"x": prompts})
    if human is not None:
        frame["human"] = human
    return Dataset([f"q{i}" for i in range(N)], frame, np.full(N, 5.0), prompts)


def _cand(name, words, cost, model=None):
    return ModelSpec(name=name, kind="llm", provider="mock", model=model or name,
                     params={"answer_words": words, "cost_per_call": cost})


def _judge(model="judge-model", **params):
    return ModelSpec(name="judge", kind="llm", provider="mock", model=model, role="judge",
                     params={"bias_by_length": True, **params})


def _run(tmp_path, task, models, ds=None):
    return run_experiment(task, models, ds or _ds(), tmp_path / "run", **QUIET).data


def test_recommends_the_cheapest_tied_candidate(tmp_path):
    data = _run(tmp_path, _task(), [_cand("pricey", 200, 0.1), _cand("cheap", 200, 0.001),
                                    _cand("middle", 200, 0.01), _judge()])
    rec = data["recommendation"]
    assert rec["status"] == "recommended" and rec["winner"] == "cheap"
    assert set(rec["equivalent"]) == {"pricey", "cheap", "middle"}


def test_a_clear_winner_is_not_tied_with_a_much_worse_candidate(tmp_path):
    data = _run(tmp_path, _task(), [_cand("good", 200, 0.1), _cand("bad", 1, 0.001), _judge()])
    rec = data["recommendation"]
    assert rec["winner"] == "good" and rec["equivalent"] == ["good"]
    rows = {r["name"]: r for r in data["candidates"]}
    assert rows["good"]["score"] == pytest.approx(5.0) and rows["bad"]["score"] == pytest.approx(1.0)


def test_results_describe_a_judge_run(tmp_path):
    data = _run(tmp_path, _task(), [_cand("a", 200, 0.01), _judge(cost_per_call=0.002)])
    assert data["task"]["type"] == "judge" and data["task"]["metric"] == "mean_judge_score"
    j = data["judge"]
    assert j["name"] == "judge" and j["backend"] == "mock" and j["self_judging"] is False
    assert j["evaluation_cost"] == pytest.approx(N * 0.002)
    assert j["invalid_rate"] == 0.0 and j["low_confidence_rate"] == 0.0
    assert j["length"]["a"]["mean_words"] == 200.0
    assert j["human_agreement"] is None
    assert data["task"]["repeats"] == 1  # one answer per prompt, whatever the label-task default is
    assert not any("labelled" in w for w in data["warnings"])  # judge tasks have prompts, not labels
    assert any("SIMULATED" in w for w in data["warnings"])
    cand_cost = next(r for r in data["candidates"] if r["name"] == "a")["total_cost_at_volume"]
    assert cand_cost == pytest.approx(0.01 * 10_000)  # generation only, never the judge's cost


def test_self_judging_is_flagged_and_warned(tmp_path):
    data = _run(tmp_path, _task(), [_cand("a", 200, 0.01, model="shared"), _judge(model="shared")])
    assert data["judge"]["self_judging"] is True
    assert any("judge" in w.lower() and "also a candidate" in w.lower() for w in data["warnings"])


def test_human_agreement_uses_only_valid_rows(tmp_path):
    human = [str(1 + i % 5) for i in range(N)]
    human[0], human[1], human[2] = "", "n/a", "oops"
    data = _run(tmp_path, _task(human_score_column="human"),
                [_cand("a", 5, 0.01), _judge(bias_by_length=False)], _ds(human))
    h = data["judge"]["human_agreement"]
    assert h["n"] == N - 3 and h["candidate"] == "a" and -1.0 <= h["spearman"] <= 1.0


def test_a_hosted_judge_needs_consent(tmp_path):
    jev = ModelSpec(name="judge", kind="llm", provider="typesafe", model="jev-1.14.0", role="judge")
    with pytest.raises(SafetyError):
        _run(tmp_path, _task(), [_cand("a", 5, 0.01), jev])


def test_models_are_validated_for_judge_tasks(tmp_path):
    with pytest.raises(ValueError, match="judge"):
        _run(tmp_path, _task(), [_cand("a", 5, 0.01)])


def test_spearman_helper():
    assert spearman_or_none([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_or_none([1, 2, 3, 4], [5, 5, 5, 5]) is None
    assert spearman_or_none([1, 2], [1, 2]) is None


def test_estimate_counts_answer_and_judge_calls_without_calling_anything():
    from tiebreak.planner import estimate
    from tiebreak.protocol import Split

    task = _task()
    models = [_cand("a", 5, 0.002), _cand("b", 5, 0.004), _judge(cost_per_call=0.001)]
    est = estimate(task, models, Split(_ds(), None), {})
    assert est.calls == {"a": N, "b": N, "judge (judge)": 2 * N}
    assert est.cost["a"] == pytest.approx(N * 0.002) and est.cost["judge (judge)"] == pytest.approx(2 * N * 0.001)
    assert est.total == pytest.approx(N * 0.006 + 2 * N * 0.001)
    assert any("answer" in n.lower() and "assum" in n.lower() for n in est.notes)
