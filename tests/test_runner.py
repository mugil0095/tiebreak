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


# ---------------- fix round 1 ----------------
import json as _json
import logging

import httpx

from tiebreak.candidates.llm import NonRetryableError
from tiebreak.runner import CostUnknownError, _scrub


def test_cache_misses_when_prompt_or_config_changes(tmp_path, caplog):
    path = tmp_path / "p.jsonl"
    c1 = Counting()
    execute_calls([Call(("m", "e0", 0), "old prompt", "cfgA")], c1, PredictionStore(path), **NO_SLEEP)
    same = Counting()
    execute_calls([Call(("m", "e0", 0), "old prompt", "cfgA")], same, PredictionStore(path), **NO_SLEEP)
    assert same.calls == 0
    new_prompt = Counting()
    with caplog.at_level(logging.WARNING, logger="tiebreak.runner"):
        recs = execute_calls([Call(("m", "e0", 0), "new prompt", "cfgA")], new_prompt, PredictionStore(path), **NO_SLEEP)
    assert new_prompt.calls == 1 and recs[0].prompt_hash
    assert "1 cached answers for 'm'" in caplog.text
    new_cfg = Counting()
    execute_calls([Call(("m", "e0", 0), "new prompt", "cfgB")], new_cfg, PredictionStore(path), **NO_SLEEP)
    assert new_cfg.calls == 1
    assert len(PredictionStore(path).for_candidate("m")) == 1


def test_store_tolerates_truncated_line_and_next_append_is_valid(tmp_path, caplog):
    path = tmp_path / "p.jsonl"
    execute_calls(calls(3), Counting(), PredictionStore(path), **NO_SLEEP)
    with open(path, "ab") as f:
        f.write(b'{"candidate": "m", "example_id": "e9", "rep')  # killed mid-write
    with caplog.at_level(logging.WARNING, logger="tiebreak.runner"):
        store = PredictionStore(path)
    assert "skipped 1 unreadable line" in caplog.text
    resumed = Counting()
    execute_calls(calls(5), resumed, store, **NO_SLEEP)
    assert resumed.calls == 2
    lines = path.read_text().splitlines()
    parsed = []
    for ln in lines:
        try:
            parsed.append(_json.loads(ln))
        except ValueError:
            pass
    assert len(parsed) == len(lines) - 1  # only the partial one is bad
    assert len(PredictionStore(path).for_candidate("m")) == 5


def test_store_ignores_unknown_keys_and_non_dict_lines(tmp_path):
    path = tmp_path / "p.jsonl"
    good = {"candidate": "m", "example_id": "e0", "repeat": 0, "raw": "pos", "tokens_in": 1, "tokens_out": 1,
            "cost_usd": 0.0, "latency_s": 0.1, "error": None, "future_field": 7}
    path.write_text(_json.dumps(good) + '\n[1, 2]\n{"candidate": 1}\n')
    assert len(PredictionStore(path).for_candidate("m")) == 1


def test_keyboard_interrupt_stops_run_and_keeps_records_valid(tmp_path):
    class Boom:
        def __init__(self):
            self.calls, self.lock = 0, threading.Lock()

        def complete(self, prompt):
            with self.lock:
                self.calls += 1
                n = self.calls
            if n == 3:
                raise KeyboardInterrupt
            return Completion("pos", 1, 1, 0.0, 0.1)

    path = tmp_path / "p.jsonl"
    comp = Boom()
    with pytest.raises(KeyboardInterrupt):
        execute_calls(calls(50), comp, PredictionStore(path), concurrency=1, **NO_SLEEP)
    assert comp.calls == 3
    assert len(PredictionStore(path).for_candidate("m")) == 2
    assert all(_json.loads(ln) for ln in path.read_text().splitlines())


def test_cost_cap_with_unknown_cost_raises_clear_error(tmp_path):
    class NoCost(Counting):
        def complete(self, prompt):
            return Completion("pos", 10, 1, None, 0.1)

    with pytest.raises(CostUnknownError, match="--max-cost is set but the cost of model m is unknown"):
        execute_calls(calls(3), NoCost(), PredictionStore(tmp_path / "p.jsonl"), cap=CostCap(1.0),
                      concurrency=1, **NO_SLEEP)
    recs = execute_calls(calls(1), NoCost(), PredictionStore(tmp_path / "q.jsonl"), cap=CostCap(1.0),
                         price=(1.0, 1.0), **NO_SLEEP)
    assert recs[0].cost_usd is not None


def test_cost_cap_is_cumulative_across_resumes(tmp_path):
    path = tmp_path / "p.jsonl"
    with pytest.raises(CostCapReached):
        execute_calls(calls(10), Counting(cost=0.01), PredictionStore(path), cap=CostCap(0.025),
                      concurrency=1, **NO_SLEEP)  # 3 calls, 0.03 spent
    comp = Counting(cost=0.01)
    with pytest.raises(CostCapReached):
        execute_calls(calls(10), comp, PredictionStore(path), cap=CostCap(0.05), concurrency=1, **NO_SLEEP)
    assert comp.calls == 2  # 0.03 already spent + 2 x 0.01 reaches 0.05


def test_storage_failure_after_paid_call_is_not_retried(tmp_path):
    class BadStore(PredictionStore):
        def add(self, rec):
            raise OSError("disk full")

    comp = Counting()
    with pytest.raises(OSError):
        execute_calls(calls(1), comp, BadStore(tmp_path / "p.jsonl"), retries=3, **NO_SLEEP)
    assert comp.calls == 1


@pytest.mark.parametrize("raw", [
    "Incorrect API key provided: sk-abcdef1234567890. You can find...",
    '{"error": {"api_key": "hunter2hunter2"}}',
    "request failed with token sk-proj-AbCdEf_123456",
    "Authorization: Bearer abc.def.ghi",
    "key=exactsecretvalue leaked",
])
def test_scrub_shapes(raw):
    out = _scrub(raw, ["exactsecretvalue"])
    for leaked in ("sk-abcdef", "hunter2", "sk-proj", "abc.def.ghi", "exactsecretvalue"):
        assert leaked not in out
    assert "[redacted]" in out


def test_scrub_truncates_stored_error(tmp_path):
    class Long:
        def complete(self, prompt):
            raise RuntimeError("x" * 5000)

    recs = execute_calls(calls(1), Long(), PredictionStore(tmp_path / "p.jsonl"), retries=0, **NO_SLEEP)
    assert len(recs[0].error) <= 300


def test_secrets_param_redacts_exact_values(tmp_path):
    class Leaky:
        def complete(self, prompt):
            raise RuntimeError("upstream said: my-weird-token-value is invalid")

    recs = execute_calls(calls(1), Leaky(), PredictionStore(tmp_path / "p.jsonl"), retries=0,
                         secrets=["my-weird-token-value"], **NO_SLEEP)
    assert "my-weird-token-value" not in recs[0].error


def _status_error(code):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize("code,expected_calls", [(401, 1), (400, 1), (404, 1), (429, 3), (408, 3), (500, 3)])
def test_retry_only_transient_http_errors(tmp_path, code, expected_calls):
    sleeps = []

    class Http:
        n = 0

        def complete(self, prompt):
            self.n += 1
            raise _status_error(code)

    comp = Http()
    execute_calls(calls(1), comp, PredictionStore(tmp_path / "p.jsonl"), retries=2, sleep=sleeps.append)
    assert comp.n == expected_calls and len(sleeps) == expected_calls - 1


def test_non_retryable_error_is_not_retried(tmp_path):
    class NoKey:
        n = 0

        def complete(self, prompt):
            self.n += 1
            raise NonRetryableError("environment variable K is not set or empty")

    comp = NoKey()
    recs = execute_calls(calls(1), comp, PredictionStore(tmp_path / "p.jsonl"), retries=3, **NO_SLEEP)
    assert comp.n == 1 and "K" in recs[0].error


def test_many_threads_every_line_parses(tmp_path):
    path = tmp_path / "p.jsonl"
    recs = execute_calls(calls(400), Counting(), PredictionStore(path), concurrency=16, **NO_SLEEP)
    lines = path.read_text().splitlines()
    assert len(lines) == 400 and len(recs) == 400
    assert {_json.loads(ln)["example_id"] for ln in lines} == {f"e{i}" for i in range(400)}


def test_seed_from_store_false_leaves_a_shared_cap_untouched(tmp_path):
    path = tmp_path / "p.jsonl"
    execute_calls(calls(4), Counting(cost=0.01), PredictionStore(path), **NO_SLEEP)
    seeded, shared = CostCap(1.0), CostCap(1.0)
    execute_calls(calls(4), Counting(), PredictionStore(path), cap=seeded, **NO_SLEEP)
    execute_calls(calls(4), Counting(), PredictionStore(path), cap=shared, seed_from_store=False, **NO_SLEEP)
    assert seeded.spent == pytest.approx(0.04)
    assert shared.spent == 0.0


def test_valid_for_keeps_only_records_matching_key_and_hashes(tmp_path):
    from tiebreak.candidates.llm import hash_text
    store = PredictionStore(tmp_path / "p.jsonl")
    cs = [Call(("m", f"e{i}", 0), f"p{i}", "cfg") for i in range(3)]
    execute_calls(cs, Counting(), store, **NO_SLEEP)
    execute_calls([Call(("m", "e0", 1), "p0", "cfg")], Counting(), store, **NO_SLEEP)  # extra repeat
    wanted = [cs[0], Call(("m", "e1", 0), "changed prompt", "cfg"), Call(("m", "e2", 0), "p2", "other")]
    got = store.valid_for(wanted)
    assert [r.example_id for r in got] == ["e0"] and got[0].repeat == 0
    assert hash_text("p0") == got[0].prompt_hash


def test_user_price_table_wins_over_provider_reported_cost(tmp_path):
    recs = execute_calls(calls(1), Counting(cost=99.0), PredictionStore(tmp_path / "p.jsonl"), price=(1.0, 2.0), **NO_SLEEP)
    assert recs[0].cost_usd == pytest.approx(100 * 1 / 1000 + 2 * 2 / 1000)  # 100 in, 2 out tokens


def test_query_string_keys_are_redacted():
    out = _scrub("HTTPError 403 for url: https://host/v1/models/x:generate?alt=json&key=AIzaSyDUMMYKEY123456&other=1")
    assert "AIzaSyDUMMYKEY123456" not in out and "other=1" in out
    assert "AIzaSy" not in _scrub("GET https://h/p?api_key=AIzaSy-xyz")
    assert "https://host/path?page=2" in _scrub("see https://host/path?page=2")


def test_cost_cap_counts_superseded_spend_after_a_config_change(tmp_path):
    """I6: spend recorded under an old prompt still counts toward the cap after the prompt changes."""
    path = tmp_path / "p.jsonl"
    execute_calls([Call(("m", f"e{i}", 0), f"old{i}", "cfg") for i in range(5)], Counting(cost=0.1), PredictionStore(path), **NO_SLEEP)
    store = PredictionStore(path)
    assert store.total_cost("m") == pytest.approx(0.5)
    new = [Call(("m", f"e{i}", 0), f"new{i}", "cfg") for i in range(5)]  # same keys, new prompt: old records superseded
    cap = CostCap(0.62)
    with pytest.raises(CostCapReached):
        execute_calls(new, Counting(cost=0.1), store, cap=cap, concurrency=1, **NO_SLEEP)
    assert store.total_cost("m") == pytest.approx(0.5 + 0.2)  # 0.5 old + two new calls before the cap bit
    assert PredictionStore(path).total_cost("m") == pytest.approx(0.7)
