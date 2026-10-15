from __future__ import annotations

import numpy as np
import pandas as pd

_WORDS = {
    "positive": ["great", "excellent", "love", "wonderful", "fantastic"],
    "negative": ["terrible", "awful", "hate", "horrible", "poor"],
    "neutral": ["okay", "average", "fine", "ordinary", "decent"],
}
_FILLER = ["the", "a", "this", "really", "very", "quite"]


def make_reviews(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = str(rng.choice(["negative", "neutral", "positive"]))
        words = list(rng.choice(_WORDS[label], 3)) + ["product"] + list(rng.choice(_FILLER, 2))
        rng.shuffle(words)
        rows.append({"id": f"r{i}", "review": " ".join(words), "sentiment": label})
    return pd.DataFrame(rows)
