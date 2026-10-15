from __future__ import annotations

import numpy as np

IMBALANCE_RATIO = 0.2


def suspected_mislabels(ids, y_enc, preds_enc: dict[str, np.ndarray], min_fraction: float = 0.8) -> list[dict]:
    if len(preds_enc) < 2:
        return []
    out = []
    for i, eid in enumerate(ids):
        votes = [p[i] for p in preds_enc.values() if p[i] >= 0]
        if len(votes) < 2:
            continue
        frac = float(np.mean([v != y_enc[i] for v in votes]))
        if frac >= min_fraction:
            out.append({"id": eid, "label_code": int(y_enc[i]), "disagree_fraction": frac})
    return out


def disagreements(ids, preds_enc: dict[str, np.ndarray]) -> list[str]:
    """Ids where candidates differ; an invalid prediction (-1) counts as its own distinct value."""
    if not preds_enc:
        return []
    stacked = np.vstack(list(preds_enc.values()))
    return [ids[i] for i in range(stacked.shape[1]) if len(set(stacked[:, i])) > 1]


def class_balance(y) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y).astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def sample_warnings(n: int, margin: float, class_counts: dict[str, int], ci_widths: dict[str, float]) -> list[str]:
    out = []
    if n < 50:
        out.append(f"Only {n} examples: fewer than 50 labelled examples rarely separate models reliably.")
    if len(class_counts) > 1 and max(class_counts.values()) > 0 and min(class_counts.values()) / max(class_counts.values()) < IMBALANCE_RATIO:
        out.append("Classes are imbalanced; plain accuracy can mislead on imbalanced classes; consider macro_f1 as the metric.")
    wide = [m for m, w in ci_widths.items() if w > margin]
    if wide:
        out.append(
            "Score intervals are wider than the margin for: " + ", ".join(sorted(wide))
            + ". Collect more labelled examples before trusting small differences."
        )
    out.append("Make sure this sample resembles the data you will see in production; results only hold for similar data.")
    return out
