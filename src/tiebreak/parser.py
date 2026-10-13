from __future__ import annotations

import json
import re

from .config import TaskConfig
from .textnorm import normalize

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_JSON_KEYS = ("label", "answer", "class", "prediction")
_NEGATORS = {"not", "no", "non", "never", "neither", "nor"}


def parse_label(text: str, labels: list[str]) -> str | None:
    t = normalize(text)
    if not t:
        return None
    by_norm = {normalize(label): label for label in labels}
    first_line = normalize(text.strip().splitlines()[0]) if text.strip() else ""
    for cand in (t, first_line):
        if cand in by_norm:
            return by_norm[cand]
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        for key in _JSON_KEYS:
            v = obj.get(key)
            if v is not None and normalize(v) in by_norm:
                return by_norm[normalize(v)]
    hits: dict[str, list[int]] = {}
    for n, label in by_norm.items():
        for m in re.finditer(rf"(?<!\w){re.escape(n)}(?!\w)", t):
            hits.setdefault(label, []).append(m.start())
    if len(hits) != 1:
        return None
    ((label, starts),) = hits.items()
    if any(_negated(t, s) for s in starts):
        return None
    return label


def _negated(t: str, start: int) -> bool:
    """True if the match at `start` is negated: prefixed by non-/un-/anti-, or
    preceded within 3 tokens by a negator (not, no, never, ..., or an n't form)."""
    prefix = t[:start]
    if re.search(r"(?:non|un|anti)-$", prefix):
        return True
    window = re.findall(r"\w+(?:['’]\w+)?", prefix)[-3:]
    return any(w in _NEGATORS or w.replace("’", "'").endswith("n't") for w in window)


def parse_number(text: str) -> float | None:
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    m = _NUM.search(cleaned)
    return float(m.group()) if m else None


def parse_text(text: str) -> str | None:
    for line in text.strip().splitlines():
        line = line.strip().strip("\"'`")
        if line:
            return line
    return None


def parse_answer(text: str, task: TaskConfig):
    if task.type in ("classify", "binary"):
        return parse_label(text, task.labels)
    if task.type == "regress":
        return parse_number(text)
    return parse_text(text)
