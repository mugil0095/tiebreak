"""Simulation only: it is handed the true answers via prime(); never use it for real comparisons."""
from __future__ import annotations

import hashlib

from .candidates.llm import Completion


class SimulatedCompleter:
    """Deterministic fake model with a target accuracy. Used by the demo; reports are labelled SIMULATED.

    Simulation only: it is handed the true answers via prime(); never use it for real comparisons.
    """

    simulated = True

    def __init__(self, accuracy: float, labels: list[str], seed: int = 0,
                 cost_per_call: float = 0.0005, latency_s: float = 0.2):
        self.accuracy, self.labels, self.seed = accuracy, labels, seed
        self.cost_per_call, self.latency_s = cost_per_call, latency_s
        self.truth: dict[str, str] = {}

    def prime(self, mapping: dict[str, str]) -> None:
        self.truth.update(mapping)

    def complete(self, prompt: str) -> Completion:
        h = int.from_bytes(hashlib.sha256(f"{self.seed}|{prompt}".encode()).digest()[:8], "big") / 2**64
        truth = self.truth.get(prompt)
        wrong = [label for label in self.labels if label != truth] or self.labels
        answer = truth if (truth is not None and h < self.accuracy) else wrong[int(h * 1e6) % len(wrong)]
        return Completion(answer, len(prompt) // 4, 2, self.cost_per_call, self.latency_s)
