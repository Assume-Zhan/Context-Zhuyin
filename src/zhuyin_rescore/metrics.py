"""Accuracy metrics: character error rate, sentence accuracy, Oracle@k."""

from __future__ import annotations

from collections.abc import Sequence


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (character level)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def oracle_errors(ref: str, hyps: Sequence[str]) -> int:
    """Fewest character errors among hyps; the rescoring upper bound."""
    return min(edit_distance(ref, h) for h in hyps)


class Accumulator:
    """Sums character errors and exact matches over a corpus."""

    def __init__(self) -> None:
        self.errors = 0
        self.chars = 0
        self.correct = 0
        self.sentences = 0

    def add(self, ref: str, hyp: str) -> None:
        self.add_errors(ref, edit_distance(ref, hyp))

    def add_errors(self, ref: str, errors: int) -> None:
        self.errors += errors
        self.chars += len(ref)
        self.correct += errors == 0
        self.sentences += 1

    @property
    def cer(self) -> float:
        return self.errors / max(self.chars, 1)

    @property
    def sent_acc(self) -> float:
        return self.correct / max(self.sentences, 1)

    def summary(self) -> dict:
        return {
            "cer": self.cer,
            "sent_acc": self.sent_acc,
            "errors": self.errors,
            "chars": self.chars,
            "sentences": self.sentences,
        }


def recovered_fraction(cer_base: float, cer_sys: float, cer_oracle: float) -> float:
    """Share of the recoverable errors fixed: (base - sys) / (base - oracle)."""
    gap = cer_base - cer_oracle
    return (cer_base - cer_sys) / gap if gap > 0 else 0.0
