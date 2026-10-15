import pytest

from tiebreak.candidates.base import CallRecord, CandidateResult, aggregate
from tiebreak.config import TaskConfig
from tiebreak.cost import latency_stats, summarize_cost

TASK = TaskConfig(type="classify", label_column="y", labels=["a", "b"], volume=1000, compute_price_per_hour=3.6)


def call(cost):
    return CallRecord("m", "e", 0, "a", 10, 2, cost, 0.1, None)


def test_aggregate_majority_mean_and_none():
    assert aggregate(["a", "b", "a"], True) == "a"
    assert aggregate(["b", "a"], True) == "b"  # tie -> first seen
    assert aggregate([1.0, 3.0], False) == 2.0
    assert aggregate([], True) is None


def test_llm_cost_per_prediction_and_volume():
    r = CandidateResult("m", "llm", [], calls=[call(0.01), call(0.03)])
    s = summarize_cost(r, TASK)
    assert s.per_prediction == pytest.approx(0.02) and s.total_at_volume == pytest.approx(20.0)
    assert s.per_1k == pytest.approx(20.0) and s.price_known


def test_unknown_llm_price_is_flagged():
    r = CandidateResult("m", "llm", [], calls=[call(None), call(None)])
    assert not summarize_cost(r, TASK).price_known


def test_ml_cost_uses_compute_price_and_amortises_training():
    r = CandidateResult("m", "ml", [], train_seconds=10.0, infer_seconds_per_pred=0.001)
    s = summarize_cost(r, TASK)  # 3.6 $/h = 0.001 $/s
    assert s.train_cost == pytest.approx(0.01)
    assert s.per_prediction == pytest.approx(1e-6)
    assert s.total_at_volume == pytest.approx(0.01 + 1e-6 * 1000)


def test_latency_stats():
    assert latency_stats([0.1, 0.2, 0.3, 0.4, 1.0]) == (pytest.approx(0.3), pytest.approx(0.88))
    assert latency_stats([]) == (0.0, 0.0)


def test_price_known_ignores_failed_calls():
    """C1: failed calls carry no cost; only successful calls decide whether the price is known."""
    err = CallRecord("m", "e", 1, "", 0, 0, None, 0.0, "Timeout: x")
    r = CandidateResult("m", "llm", [], calls=[call(0.01), err, call(0.03)])
    s = summarize_cost(r, TASK)
    assert s.price_known and s.per_prediction == pytest.approx(0.02)
    assert not summarize_cost(CandidateResult("m", "llm", [], calls=[err]), TASK).price_known


def tok_call(cost, tin=1000, tout=500, lat=0.1):
    return CallRecord("m", "e", 0, "a", tin, tout, cost, lat, None)


def test_user_price_wins_over_provider_cost_and_is_recomputed_from_tokens():
    r = CandidateResult("m", "llm", [], calls=[tok_call(99.0), tok_call(99.0)])
    s = summarize_cost(r, TASK, price=(1.0, 2.0))  # 1000 in * 1/1k + 500 out * 2/1k = 2.0
    assert s.per_prediction == pytest.approx(2.0) and s.price_known and s.price_source == "pricing.yaml"


def test_price_source_provider_and_unknown():
    assert summarize_cost(CandidateResult("m", "llm", [], calls=[call(0.01)]), TASK).price_source == "provider"
    assert summarize_cost(CandidateResult("m", "llm", [], calls=[call(None)]), TASK).price_source == "unknown"
    ml = CandidateResult("m", "ml", [], train_seconds=1.0, infer_seconds_per_pred=0.001)
    assert summarize_cost(ml, TASK).price_source == "compute time"


def test_local_llm_is_costed_from_latency_and_compute_price():
    """I3: a local model is not free: latency x compute_price_per_hour / 3600 per call."""
    calls_ = [tok_call(0.0, lat=2.0), tok_call(0.0, lat=4.0)]
    r = CandidateResult("m", "llm", [], calls=calls_, local_compute=True)
    s = summarize_cost(r, TASK)  # 3.6 $/h = 0.001 $/s; mean latency 3 s
    assert s.per_prediction == pytest.approx(0.003) and s.total_at_volume == pytest.approx(3.0)
    assert s.price_known and s.price_source == "compute time"


def test_litellm_price_lookup_uses_the_package_table_when_importable(monkeypatch):
    import sys
    import types
    from tiebreak.config import ModelSpec
    from tiebreak.cost import litellm_price
    table = {"anthropic/some-model": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6}}
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(model_cost=table))
    spec = ModelSpec(name="a", kind="llm", provider="anthropic", model="some-model")
    assert litellm_price(spec) == pytest.approx((3e-3, 15e-3))
    assert litellm_price(ModelSpec(name="a", kind="llm", provider="anthropic", model="other")) is None
    monkeypatch.setitem(sys.modules, "litellm", None)  # not importable
    assert litellm_price(spec) is None
