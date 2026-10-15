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
