"""Live judge tests. Never run in CI or in the default suite.

Ollama:  TIEBREAK_LIVE_OLLAMA_MODEL=qwen3.5:0.8b pytest -m live tests/test_live_judge.py
Jev:     TYPESAFE_API_KEY=...  pytest -m live tests/test_live_judge.py   (needs: pip install "tiebreak[judge] @ git+https://github.com/mugil0095/tiebreak.git"; spends a few cents at most)
"""
import os
import urllib.request

import numpy as np
import pandas as pd
import pytest

from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.engine import run_experiment

pytestmark = pytest.mark.live
MODEL = os.environ.get("TIEBREAK_LIVE_OLLAMA_MODEL")
N = 6
TASK = TaskConfig(type="judge", prompt_column="prompt", scale=(1, 5))


def _ds():
    prompts = ["Explain what a hash table is in one sentence.", "Why is the sky blue? Answer briefly.",
               "Give one tip for learning a language.", "What does an API do?",
               "Name a safe place to keep a spare key.", "Why reuse of passwords is risky? One sentence."]
    return Dataset([f"q{i}" for i in range(N)], pd.DataFrame({"x": prompts}), np.full(N, 5.0), prompts)


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


def _mock_candidate(name, words):
    return ModelSpec(name=name, kind="llm", provider="mock", model=name,
                     params={"answer_words": words, "cost_per_call": 0.001})


@pytest.mark.skipif(not MODEL, reason="set TIEBREAK_LIVE_OLLAMA_MODEL to run live tests")
def test_a_local_model_can_judge_end_to_end(tmp_path):
    if not _ollama_up():
        pytest.skip("Ollama is not running on localhost:11434")
    judge = ModelSpec(name="judge", kind="llm", provider="ollama", model=MODEL, role="judge",
                      params={"think": False, "max_tokens": 8})
    data = run_experiment(TASK, [_mock_candidate("long", 80), _mock_candidate("short", 3), judge], _ds(),
                          tmp_path / "r", n_boot=200, local_only=False).data
    assert data["judge"]["invalid_rate"] < 1.0, "the local judge never returned a readable score"
    assert (tmp_path / "r" / "judgments.jsonl").stat().st_size > 0


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="set TYPESAFE_API_KEY to run the Jev test")
def test_jev_can_judge_end_to_end(tmp_path):
    pytest.importorskip("typesafe_sdk")
    judge = ModelSpec(name="judge", kind="llm", provider="typesafe", model="jev-1.14.0", role="judge")
    data = run_experiment(TASK, [_mock_candidate("long", 80), _mock_candidate("short", 3), judge], _ds(),
                          tmp_path / "r", n_boot=200, allow_hosted=True).data
    j = data["judge"]
    assert j["backend"] == "typesafe" and j["invalid_rate"] == 0.0
    assert 0 < j["evaluation_cost"] < 0.05, "Jev is priced per input token; six short judgments cost well under 5 cents"
