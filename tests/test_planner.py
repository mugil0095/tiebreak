import numpy as np
import pandas as pd

from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.metrics import accuracy
from tiebreak.planner import clearly_worse, estimate
from tiebreak.protocol import Split


def test_clearly_worse_only_flags_eligible_and_clearly_worse():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    good = np.where(rng.random(200) < 0.95, y, (y + 1) % 3)
    bad = np.where(rng.random(200) < 0.5, y, (y + 1) % 3)
    close = np.where(rng.random(200) < 0.94, y, (y + 1) % 3)
    out = clearly_worse(y, {"good": good, "bad": bad, "close": close}, accuracy, 0.02, eligible={"bad", "close"})
    assert out == {"bad"}


def test_clearly_worse_never_flags_the_best_or_ineligible():
    y = np.array([0, 1] * 50)
    assert clearly_worse(y, {"a": y, "b": 1 - y}, accuracy, 0.02, eligible={"a"}) == set()


def test_estimate_counts_calls_and_prices():
    task = TaskConfig(type="classify", label_column="s", text_column="t", labels=["a", "b"], repeats=2)
    ds = Dataset(["e0", "e1", "e2"], pd.DataFrame({"t": ["x" * 40] * 3}), np.array(["a", "b", "a"], dtype=object), ["x" * 40] * 3)
    models = [
        ModelSpec(name="priced", kind="llm", provider="p", model="m1"),
        ModelSpec(name="unpriced", kind="llm", provider="p", model="m2"),
        ModelSpec(name="ml", kind="ml", estimator="logistic_regression"),
    ]
    est = estimate(task, models, Split(ds, None), {"m1": (1.0, 2.0)})
    assert est.calls == {"priced": 6, "unpriced": 6}
    assert est.cost["priced"] is not None and est.cost["priced"] > 0 and est.cost["unpriced"] is None
    assert est.total == est.cost["priced"]
    assert any("unpriced" in n for n in est.notes)


def test_clearly_worse_needs_min_n():
    y = np.array([0, 1] * 15)
    assert clearly_worse(y, {"a": y, "b": 1 - y}, accuracy, 0.02, eligible={"b"}) == set()


def test_clearly_worse_sparse_discordance_does_not_eliminate():
    y = np.array([0, 1] * 50)
    cand = y.copy()
    cand[:3] = 1 - cand[:3]  # 3-0 discordant split
    assert clearly_worse(y, {"a": y, "b": cand}, accuracy, 0.02, eligible={"b"}) == set()


def test_clearly_worse_tied_pair_is_not_eliminated_in_simulation():
    rng = np.random.default_rng(1)
    dropped = 0
    for _ in range(100):
        y = rng.integers(0, 3, 100)
        a = np.where(rng.random(100) < 0.8, y, (y + 1) % 3)
        b = np.where(rng.random(100) < 0.8, y, (y + 1) % 3)
        dropped += len(clearly_worse(y, {"a": a, "b": b}, accuracy, 0.02, eligible={"a", "b"}, n_boot=300))
    assert dropped <= 1


def test_clearly_worse_drops_a_clearly_bad_model():
    rng = np.random.default_rng(2)
    for n in (100, 200):
        y = rng.integers(0, 3, n)
        good = np.where(rng.random(n) < 0.95, y, (y + 1) % 3)
        bad = np.where(rng.random(n) < 0.5, y, (y + 1) % 3)
        assert clearly_worse(y, {"good": good, "bad": bad}, accuracy, 0.02, eligible={"bad"}) == {"bad"}


def test_estimate_notes_partial_total():
    task = TaskConfig(type="classify", label_column="s", text_column="t", labels=["a", "b"], repeats=1)
    ds = Dataset(["e0"], pd.DataFrame({"t": ["x"]}), np.array(["a"], dtype=object), ["x"])
    est = estimate(task, [ModelSpec(name="u", kind="llm", provider="p", model="m2")], Split(ds, None), {})
    assert any("priced models only" in n for n in est.notes)
