import json

import httpx
import numpy as np
import pandas as pd
import pytest

from tiebreak.candidates.base import CallRecord
from tiebreak.candidates.llm import (
    LLMCandidate, OpenAICompatCompleter, SimulatedCompleter,
    build_completer, render_prompt, row_text,
)
from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset

TASK = TaskConfig(type="classify", label_column="s", text_column="t", labels=["neg", "pos"], repeats=3)


def ds(texts, labels):
    return Dataset([f"e{i}" for i in range(len(texts))], pd.DataFrame({"t": texts}), np.array(labels, dtype=object), list(texts))


def test_render_prompt_default_and_few_shot_and_custom_template():
    p = render_prompt(TASK, "great", "", None)
    assert "neg, pos" in p and p.rstrip().endswith("Answer:") and "Input: great" in p
    p2 = render_prompt(TASK, "great", "Input: bad\nAnswer: neg", None)
    assert "Examples:" in p2 and "Answer: neg" in p2
    p3 = render_prompt(TASK, "great", "", "Label {input} as one of {labels}. {examples}")
    assert p3 == "Label great as one of neg, pos. "


def test_row_text_falls_back_to_columns():
    d = Dataset(["a"], pd.DataFrame({"x": [1], "y": ["q"]}), np.array(["pos"], dtype=object), None)
    assert row_text(d, 0) == "x: 1; y: q"


def test_candidate_builds_one_call_per_example_and_repeat():
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())
    calls = cand.build_calls(ds(["a", "b"], ["pos", "neg"]), None)
    assert len(calls) == 6 and calls[0].key == ("m", "e0", 0) and calls[5].key == ("m", "e1", 2)


def test_finalize_votes_counts_invalid_and_errors():
    scoring = ds(["a", "b"], ["pos", "neg"])
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())

    def rec(eid, rep, raw, err=None):
        return CallRecord("m", eid, rep, raw, 10, 1, 0.001, 0.1, err)

    records = [
        rec("e0", 0, "pos"), rec("e0", 1, "Pos."), rec("e0", 2, "neg"),   # majority pos, disagreement
        rec("e1", 0, "no idea"), rec("e1", 1, "", err="Timeout: x"), rec("e1", 2, "neg"),
    ]
    res = cand.finalize(scoring, records)
    assert res.predictions == ["pos", "neg"]
    assert res.invalid_rate == pytest.approx(1 / 6) and res.error_rate == pytest.approx(1 / 6)
    assert res.repeat_disagreement == pytest.approx(0.5)
    assert res.mask.all() and len(res.calls) == 6


def test_finalize_marks_unrun_examples_in_mask():
    scoring = ds(["a", "b"], ["pos", "neg"])
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())
    res = cand.finalize(scoring, [CallRecord("m", "e0", 0, "pos", 1, 1, 0.0, 0.1, None)])
    assert list(res.mask) == [True, False] and res.predictions[1] is None


def test_simulated_completer_hits_target_accuracy():
    comp = SimulatedCompleter(accuracy=0.8, labels=["neg", "pos"], seed=1)
    truth = {f"prompt-{i}": ("pos" if i % 2 else "neg") for i in range(2000)}
    comp.prime(truth)
    hits = np.mean([comp.complete(p).text == t for p, t in truth.items()])
    assert abs(hits - 0.8) < 0.03
    assert comp.complete("prompt-1").text == comp.complete("prompt-1").text  # deterministic


def test_openai_compatible_completer_with_mock_transport(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "pos"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    comp = OpenAICompatCompleter("http://localhost:1234/v1", "tiny", "MY_KEY", client=client)
    out = comp.complete("hello")
    assert out.text == "pos" and out.tokens_in == 12 and out.tokens_out == 1 and out.cost_usd is None
    assert seen["auth"] == "Bearer secret" and seen["body"]["model"] == "tiny" and seen["body"]["temperature"] == 0


def test_build_completer_mock():
    spec = ModelSpec(name="m", kind="llm", provider="mock", model="sim", params={"accuracy": 0.7})
    assert isinstance(build_completer(spec, TASK), SimulatedCompleter)


def test_http_error_message_never_contains_the_api_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-super-secret")
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad"})))
    comp = OpenAICompatCompleter("http://localhost:1234/v1", "tiny", "MY_KEY", client=client)
    with pytest.raises(httpx.HTTPStatusError) as ei:
        comp.complete("hi")
    assert "sk-super-secret" not in str(ei.value) and "Bearer" not in str(ei.value)


def test_build_calls_carry_config_hash_that_tracks_config():
    a = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x", params={"accuracy": 0.5}), TASK, object())
    b = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x", params={"accuracy": 0.9}), TASK, object())
    d = ds(["a"], ["pos"])
    ha, hb = a.build_calls(d, None)[0].config_hash, b.build_calls(d, None)[0].config_hash
    assert ha and hb and ha != hb and ha == a.build_calls(d, None)[0].config_hash


def test_finalize_with_only_error_records():
    scoring = ds(["a"], ["pos"])
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())
    recs = [CallRecord("m", "e0", r, "", 0, 0, None, 0.0, "Timeout: x") for r in range(3)]
    res = cand.finalize(scoring, recs)
    assert res.predictions == [None] and res.error_rate == 1.0 and not res.mask.any() and res.n_unanswered == 1
    assert res.infer_seconds_per_pred == 0.0 and res.latencies == []


def test_missing_api_key_env_is_a_clear_non_retryable_error(monkeypatch):
    from tiebreak.candidates.llm import NonRetryableError
    monkeypatch.delenv("NOPE_KEY", raising=False)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    comp = OpenAICompatCompleter("http://localhost:1/v1", "tiny", "NOPE_KEY", client=client)
    with pytest.raises(NonRetryableError, match="NOPE_KEY"):
        comp.complete("hi")


def test_litellm_completer_timeout_default_override_and_collisions(monkeypatch):
    import sys
    import types
    seen = {}

    def completion(**kw):
        seen.update(kw)
        msg = types.SimpleNamespace(content="pos")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    fake = types.SimpleNamespace(completion=completion, completion_cost=lambda completion_response: 0.1)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    from tiebreak.candidates.llm import LiteLLMCompleter
    LiteLLMCompleter("openai", "gpt", {}).complete("hi")
    assert seen["timeout"] == 60 and seen["temperature"] == 0
    LiteLLMCompleter("openai", "gpt", {"timeout": 5, "top_p": 0.5}).complete("hi")
    assert seen["timeout"] == 5 and seen["top_p"] == 0.5
    with pytest.raises(ValueError, match="temperature"):
        LiteLLMCompleter("openai", "gpt", {"temperature": 1})


def test_example_with_only_error_calls_is_not_evaluated_and_counted_unanswered():
    """C1: an example whose every repeat failed has no answer; it must not be scored as a wrong answer."""
    scoring = ds(["a", "b", "c"], ["pos", "neg", "pos"])
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())
    recs = [
        CallRecord("m", "e0", 0, "pos", 1, 1, 0.0, 0.1, None),
        CallRecord("m", "e1", 0, "", 0, 0, None, 0.0, "Timeout: x"),
        CallRecord("m", "e2", 0, "garbage", 1, 1, 0.0, 0.1, None),  # answered, but unparseable: still evaluated
    ]
    res = cand.finalize(scoring, recs)
    assert list(res.mask) == [True, False, True] and res.n_unanswered == 1
    assert res.predictions[1] is None and res.predictions[2] is None


def test_one_successful_repeat_is_enough_to_evaluate_an_example():
    scoring = ds(["a"], ["pos"])
    cand = LLMCandidate(ModelSpec(name="m", kind="llm", provider="mock", model="x"), TASK, completer=object())
    recs = [CallRecord("m", "e0", 0, "", 0, 0, None, 0.0, "Timeout: x"),
            CallRecord("m", "e0", 1, "pos", 1, 1, 0.0, 0.1, None),
            CallRecord("m", "e0", 2, "", 0, 0, None, 0.0, "Timeout: x")]
    res = cand.finalize(scoring, recs)
    assert res.mask.all() and res.predictions == ["pos"] and res.n_unanswered == 0


def _fake_litellm(monkeypatch):
    import sys
    import types
    seen = {}

    def completion(**kw):
        seen.clear()
        seen.update(kw)
        msg = types.SimpleNamespace(content="pos")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion, completion_cost=lambda completion_response: 0.1))
    return seen


def test_litellm_completer_gets_base_url_as_api_base_and_the_key_from_the_env(monkeypatch):
    """I4: base_url and api_key_env must reach litellm for a non-openai_compatible provider."""
    seen = _fake_litellm(monkeypatch)
    monkeypatch.setenv("MY_PROXY_KEY", "proxy-secret-123")
    spec = ModelSpec(name="m", kind="llm", provider="openai", model="gpt", base_url="http://localhost:8000/v1", api_key_env="MY_PROXY_KEY")
    build_completer(spec, TASK).complete("hi")
    assert seen["api_base"] == "http://localhost:8000/v1" and seen["api_key"] == "proxy-secret-123"
    plain = ModelSpec(name="p", kind="llm", provider="openai", model="gpt")
    build_completer(plain, TASK).complete("hi")
    assert "api_base" not in seen and "api_key" not in seen


def test_litellm_completer_unset_key_env_is_a_non_retryable_error_naming_the_variable_only(monkeypatch):
    from tiebreak.candidates.llm import NonRetryableError
    _fake_litellm(monkeypatch)
    monkeypatch.delenv("MISSING_KEY_ENV", raising=False)
    spec = ModelSpec(name="m", kind="llm", provider="openai", model="gpt", api_key_env="MISSING_KEY_ENV")
    with pytest.raises(NonRetryableError, match="MISSING_KEY_ENV") as ei:
        build_completer(spec, TASK).complete("hi")
    assert ei.value.retryable is False
