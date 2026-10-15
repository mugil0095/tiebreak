from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_WORDS = {
    "positive": ["great", "excellent", "love", "wonderful", "fantastic"],
    "negative": ["terrible", "awful", "hate", "horrible", "poor"],
    "neutral": ["okay", "average", "fine", "ordinary", "decent"],
}
_FILLER = ["the", "a", "this", "really", "very", "quite"]


def make_reviews(n: int = 300, seed: int = 0, noise: float = 0.0) -> pd.DataFrame:
    """Synthetic reviews. `noise` = fraction whose wording comes from a different class (seeded)."""
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1_000_003)
    rows = []
    for i in range(n):
        label = str(rng.choice(["negative", "neutral", "positive"]))
        wording = label
        if noise > 0 and noise_rng.random() < noise:
            wording = str(noise_rng.choice([k for k in _WORDS if k != label]))
        words = list(rng.choice(_WORDS[wording], 3)) + ["product"] + list(rng.choice(_FILLER, 2))
        rng.shuffle(words)
        rows.append({"id": f"r{i}", "review": " ".join(words), "sentiment": label})
    return pd.DataFrame(rows)


_DEMO_TASK = """type: classify
label_column: sentiment
text_column: review
labels: [negative, neutral, positive]
margin: 0.05
volume: 10000
repeats: 1
"""

_DEMO_MODELS = """# 'mock' candidates are SIMULATED models with a preset accuracy so the demo needs no API keys.
# Replace them with real ones, for example:
#   - {name: my-hosted, kind: llm, provider: <provider>, model: "<model id from your account>"}
#   - {name: local, kind: llm, provider: ollama, model: "<local model tag>"}
- {name: sim-large, kind: llm, provider: mock, model: sim-large, params: {accuracy: 0.93, cost_per_call: 0.01}}
- {name: sim-small, kind: llm, provider: mock, model: sim-small, params: {accuracy: 0.92, cost_per_call: 0.0008}}
- {name: sim-weak, kind: llm, provider: mock, model: sim-weak, params: {accuracy: 0.55, cost_per_call: 0.0002}}
- {name: logreg, kind: ml, estimator: logistic_regression}
- {name: forest, kind: ml, estimator: random_forest}
"""

_DEMO_README = """tiebreak demo (SIMULATED)

data.csv is synthetic (about 12% of reviews use misleading wording, so no model is perfect). The 'sim-*' candidates in models.yaml are SIMULATED models with a preset
accuracy; they make no API calls. The results only show how tiebreak works. Do not use them to
choose a real model. The story it shows: a cheap model (sim-small) can tie an expensive one
(sim-large), so the cheaper one is recommended.
The margin in task.yaml is 0.05 (not the default 0.02) because 300 examples cannot confirm a tighter margin.

Try:
  tiebreak estimate --task task.yaml --models models.yaml --data data.csv
  tiebreak run --task task.yaml --models models.yaml --data data.csv
"""

DEMO_FILES = ("data.csv", "task.yaml", "models.yaml", "README.txt")


def write_demo(directory) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    make_reviews(300, 0, noise=0.12).to_csv(d / "data.csv", index=False)
    (d / "task.yaml").write_text(_DEMO_TASK, encoding="utf-8")
    (d / "models.yaml").write_text(_DEMO_MODELS, encoding="utf-8")
    (d / "README.txt").write_text(_DEMO_README, encoding="utf-8")
