import threading

import pytest

from tiebreak.candidates.llm import Call, Completion
from tiebreak.runner import CostCap, CostCapReached, PredictionStore, execute_calls

NO_SLEEP = dict(sleep=lambda s: None)


class Counting:
    def __init__(self, fail_first=0, always_fail=False, cost=0.01):
        self.calls, self.fail_first, self.always_fail, self.cost = 0, fail_first, always_fail, cost
        self.lock = threading.Lock()

    def complete(self, prompt):
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.always_fail or n <= self.fail_first:
            raise TimeoutError("boom")
        return Completion("pos", 100, 2, self.cost, 0.1)


def calls(n):
    return [Call(("m", f"e{i}", 0), f"p{i}") for i in range(n)]


def test_results_follow_call_order_and_are_stored(tmp_path):
    store = PredictionStore(tmp_path / "p.jsonl")
    recs = execute_calls(calls(5), Counting(), store, concurrency=3, **NO_SLEEP)
    assert [r.example_id for r in recs] == [f"e{i}" for i in range(5)]
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 5


def test_retries_then_succeeds(tmp_path):
    comp = Counting(fail_first=2)
    recs = execute_calls(calls(1), comp, PredictionStore(tmp_path / "p.jsonl"), retries=3, **NO_SLEEP)
    assert recs[0].error is None and comp.calls == 3


def test_exhausted_retries_record_the_error_and_continue(tmp_path):
    store = PredictionStore(tmp_path / "p.jsonl")
    recs = execute_calls(calls(2), Counting(always_fail=True), store, retries=1, **NO_SLEEP)
    assert all(r.error and "TimeoutError" in r.error for r in recs)


def test_resume_does_not_repeat_successful_calls(tmp_path):
    path = tmp_path / "p.jsonl"
    first = Counting()
    execute_calls(calls(4), first, PredictionStore(path), **NO_SLEEP)
    second = Counting()
    execute_calls(calls(4), second, PredictionStore(path), **NO_SLEEP)
    assert first.calls == 4 and second.calls == 0


def test_resume_retries_previously_failed_calls(tmp_path):
    path = tmp_path / "p.jsonl"
    execute_calls(calls(2), Counting(always_fail=True), PredictionStore(path), retries=0, **NO_SLEEP)
    again = Counting()
    recs = execute_calls(calls(2), again, PredictionStore(path), **NO_SLEEP)
    assert again.calls == 2 and all(r.error is None for r in recs)
    assert len(PredictionStore(path).for_candidate("m")) == 2


def test_cost_cap_stops_cleanly_and_stays_resumable(tmp_path):
    path = tmp_path / "p.jsonl"
    cap = CostCap(0.025)
    with pytest.raises(CostCapReached):
        execute_calls(calls(10), Counting(cost=0.01), PredictionStore(path), cap=cap, concurrency=1, **NO_SLEEP)
    done = len(PredictionStore(path).for_candidate("m"))
    assert 3 <= done < 10
    resumed = Counting()
    execute_calls(calls(10), resumed, PredictionStore(path), **NO_SLEEP)
    assert resumed.calls == 10 - done


def test_price_table_fills_unknown_cost(tmp_path):
    class NoCost(Counting):
        def complete(self, prompt):
            return Completion("pos", 1000, 500, None, 0.1)

    recs = execute_calls(calls(1), NoCost(), PredictionStore(tmp_path / "p.jsonl"), price=(1.0, 2.0), **NO_SLEEP)
    assert recs[0].cost_usd == pytest.approx(1.0 + 1.0)


def test_unknown_price_stays_none(tmp_path):
    class NoCost(Counting):
        def complete(self, prompt):
            return Completion("pos", 10, 1, None, 0.1)

    recs = execute_calls(calls(1), NoCost(), PredictionStore(tmp_path / "p.jsonl"), **NO_SLEEP)
    assert recs[0].cost_usd is None


def test_error_text_scrubs_authorization_secrets(tmp_path):
    class Leaky:
        def complete(self, prompt):
            raise RuntimeError("failed: Authorization: Bearer sk-abc123 rest")

    path = tmp_path / "p.jsonl"
    recs = execute_calls(calls(1), Leaky(), PredictionStore(path), retries=0, **NO_SLEEP)
    assert "sk-abc123" not in recs[0].error and "sk-abc123" not in path.read_text()
