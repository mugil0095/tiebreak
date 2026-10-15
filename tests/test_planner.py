import numpy as np
import pandas as pd

from tiebreak.candidates.llm import SimulatedCompleter
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
