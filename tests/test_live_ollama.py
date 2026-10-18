"""Live test against a real local model. Never runs in CI or in the default suite.

Run it with:  TIEBREAK_LIVE_OLLAMA_MODEL=qwen3.5:0.8b pytest -m live
It skips itself unless that variable is set and Ollama answers on localhost:11434.
"""
import os
import urllib.request

import pytest

from tiebreak.demo import make_reviews
from tiebreak.notebook import compare

MODEL = os.environ.get("TIEBREAK_LIVE_OLLAMA_MODEL")

pytestmark = pytest.mark.live


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not MODEL, reason="set TIEBREAK_LIVE_OLLAMA_MODEL to run live tests")
def test_a_real_local_model_runs_end_to_end(tmp_path):
    if not _ollama_up():
        pytest.skip("Ollama is not running on localhost:11434")
    task = {"type": "classify", "label_column": "sentiment", "text_column": "review",
            "labels": ["negative", "neutral", "positive"], "repeats": 1, "cv_folds": 3, "cv_repeats": 1}
    models = [
        {"name": "logreg", "kind": "ml", "estimator": "logistic_regression"},
        {"name": "local", "kind": "llm", "provider": "ollama", "model": MODEL, "params": {"think": False}},
    ]
    c = compare(task, models, make_reviews(30, 0), run_dir=tmp_path / "r", n_boot=200)
    row = c.table.loc["local"]
    assert row["error_rate"] == 0.0, "calls to the local model failed"
    assert row["invalid_rate"] < 0.5, "the model's answers could not be parsed"
    assert (tmp_path / "r" / "predictions.jsonl").stat().st_size > 0
