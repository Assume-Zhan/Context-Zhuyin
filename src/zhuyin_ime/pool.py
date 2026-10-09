"""Candidate pool for the IME reranker and the fusion weights that go with it.

Evaluation on the five test domains (docs/results-cpu-reranker.md) showed:
- for input typed with tones, adding libchewing 0.14's whole sentence
  n-best (a word bigram decoder, so it makes different mistakes) after the
  decoder's 10-best gives the best reranked CER;
- for toneless input, libchewing's fuzzy n-best hurts, while a deeper
  decoder list (30) helps a little.

Space finishes a syllable without a tone mark, which the decoder treats as
any tone, so tone 1 never carries a mark. Input counts as typed with tones
when at least half of its syllables carry a mark; libchewing then gets the
same keys, where Space is tone 1, exactly like the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from zhuyin_rescore.zhuyin import has_tone

TONED_WITH_CHEWING = "toned+chewing"
TONED = "toned"
OPEN = "open"


@dataclass(frozen=True)
class Fusion:
    """score = lm + a * ngram - beta * [not first] - mu * rank"""

    a: float
    beta: float
    mu: float


# Dev tuned weights per pool type (outputs/reports/widen/*.json).
FUSION_PRESETS: dict[str, dict[str, Fusion]] = {
    "qwen": {
        TONED_WITH_CHEWING: Fusion(0.2, 2.0, 0.05),
        TONED: Fusion(0.0, 2.0, 0.1),
        OPEN: Fusion(0.1, 2.0, 0.05),
    },
    "charlm": {
        TONED_WITH_CHEWING: Fusion(0.3, 1.0, 0.1),
        TONED: Fusion(0.3, 1.0, 0.1),
        OPEN: Fusion(0.75, 0.5, 0.05),
    },
}


def typed_with_tones(syllables: list[str]) -> bool:
    return bool(syllables) and sum(has_tone(s) for s in syllables) * 2 >= len(syllables)


class PoolBuilder:
    def __init__(self, chewing_lib: str | None = None, chewing_syspath: str | None = None, k: int = 30):
        """chewing_lib: path to a libchewing 0.14 shared library (optional)."""
        self.k = k
        self.chewing = None
        if chewing_lib:
            from zhuyin_rescore.chewing import CHEWING_CONVERSION_ENGINE, Chewing

            self.chewing = Chewing(CHEWING_CONVERSION_ENGINE, syspath=chewing_syspath, lib_path=chewing_lib)

    def chewing_nbest(self, syllables: list[str]) -> list[str]:
        from zhuyin_rescore.candidates import in_charset, is_han, tab_nbest

        onebest = self.chewing.convert(syllables).text
        n = len(syllables)
        nbest = tab_nbest(self.chewing, onebest)
        return [t for t in nbest if len(t) == n and is_han(t) and in_charset(t, "cp950")]

    def build(self, decoder_texts: list[str], syllables: list[str]) -> tuple[str, list[str]]:
        """Pool type and texts; the decoder's 1-best stays first."""
        if not typed_with_tones(syllables):
            return OPEN, decoder_texts[: self.k]
        if self.chewing is None:
            return TONED, decoder_texts[: self.k]
        pool, seen = [], set()
        for t in decoder_texts[:10] + self.chewing_nbest(syllables):
            if t not in seen:
                seen.add(t)
                pool.append(t)
        return TONED_WITH_CHEWING, pool


def fuse(lm: list[float], ngram: list[float], weights: Fusion) -> int:
    """Index of the best candidate under the fusion score."""
    best, best_s = 0, float("-inf")
    for i, (x, g) in enumerate(zip(lm, ngram, strict=True)):
        s = x + weights.a * g - weights.beta * (i > 0) - weights.mu * i
        if s > best_s:
            best, best_s = i, s
    return best
