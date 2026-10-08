"""Lattice beam search over dictionary phrases scored by a char n-gram.

The lattice edges are libchewing dictionary phrases whose reading matches a
span of the typed syllables (under the lexicon's input condition), so every
multi character word on a path has a valid reading. Hypotheses are kept per
end position; each extension is scored with

    score = ngram + w_prior * log(1 + freq) + w_edge + w_mismatch * mismatches

where a mismatch is a window of 2 to 4 characters that crosses an edge
boundary, is itself a dictionary phrase, and has no reading that matches the
typed syllables (for example 長大 assembled from single characters for an
input that is not read zhang da).

The search is pull based: the beam at end position j gathers extensions of
the beams at j - 1 .. j - max_len. A beam only depends on the syllables
before it, so when the user types one more syllable, decode() reuses the
cached beams of the unchanged prefix and computes one new position
(incremental decoding). Locked spans (phrases the user picked) restrict the
lattice to edges consistent with them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from zhuyin_rescore.lexicon import Lexicon
from zhuyin_rescore.ngram import EOS, CharNgram

SINGLE_LIMIT = {"full": None, "notone": 40, "initial": 40, "mixed": 40}
LOCK_FREQ = 10**7


@dataclass
class Weights:
    prior: float = 0.0
    edge: float = 0.0
    mismatch: float = -3.0


@dataclass
class Hyp:
    text: str
    ctx: tuple[int, ...]
    score: float
    feats: np.ndarray = field(default_factory=lambda: np.zeros(4))  # ngram, prior, edges, mismatch


@dataclass
class Decoded:
    text: str
    score: float
    feats: list[float]


Lock = tuple[int, int, str]


class BeamDecoder:
    def __init__(
        self,
        lexicon: Lexicon,
        ngram: CharNgram,
        beam: int = 32,
        weights: Weights | None = None,
        single_limit: int | None = -1,
        expand_factor: int = 4,
    ):
        self.lexicon = lexicon
        self.ngram = ngram
        self.beam = beam
        self.weights = weights or Weights()
        self.single_limit = SINGLE_LIMIT[lexicon.condition] if single_limit == -1 else single_limit
        self.expand_factor = expand_factor
        self.last_ms = 0.0
        self.last_new_positions = 0
        # Incremental state: the inputs the cached beams were built from.
        self._cache_key: tuple | None = None
        self._cache_syllables: list[str] = []
        self._cache_beams: list[list[Hyp]] = []

    # ------------------------------------------------------------------ lattice
    def _edges_ending_at(self, syllables: list[str], j: int, locks: tuple[Lock, ...]):
        """Edges (start, phrase, freq) that end at position j."""
        out = []
        for length in range(1, min(self.lexicon.max_len, j) + 1):
            i = j - length
            limit = self.single_limit if length == 1 else None
            for phrase, freq in self.lexicon.lookup_span(tuple(syllables[i:j]), limit):
                out.append((i, phrase, freq))
        for s, e, text in locks:
            # Locked span: only the locked phrase may cover it, and nothing may
            # straddle its boundaries.
            out = [(i, p, f) for i, p, f in out if not (i < e and j > s) or (i == s and j == e and p == text)]
            if e == j and not any(i == s and p == text for i, p, _ in out):
                out.append((s, text, LOCK_FREQ))
        return out

    def _mismatches(self, syllables: list[str], text: str, start: int) -> int:
        """Windows that end in the new edge, start before it, and misread."""
        n_bad = 0
        end = len(text)
        for e in range(start + 1, end + 1):
            for length in (2, 3, 4):
                s = e - length
                if s < 0 or s >= start:
                    continue
                if self.lexicon.reading_matches(text[s:e], tuple(syllables[s:e])) is False:
                    n_bad += 1
        return n_bad

    # ------------------------------------------------------------------ search
    def _extend(
        self, beams: list[list[Hyp]], syllables: list[str], j: int, locks: tuple[Lock, ...]
    ) -> list[Hyp]:
        w = self.weights
        per_group = self.beam * self.expand_factor
        by_start: dict[int, list[tuple[str, int]]] = {}
        for i, phrase, freq in self._edges_ending_at(syllables, j, locks):
            by_start.setdefault(i, []).append((phrase, freq))
        cand: list[tuple[float, Hyp, str, int, float, tuple[int, ...]]] = []
        for i, group in by_start.items():
            # Positions keep every surviving hypothesis (the final one feeds
            # the k-best list); only the top `beam` are extended further.
            hyps = beams[i][: self.beam]
            if not hyps:
                continue
            length = j - i
            m = len(hyps) * len(group)
            hyp_idx = np.repeat(np.arange(len(hyps)), len(group))
            edge_idx = np.tile(np.arange(len(group)), len(hyps))
            hyp_ctx = np.array([h.ctx for h in hyps], dtype=np.int64)
            edge_ids = np.array([self.ngram.ids(p) for p, _ in group], dtype=np.int64)
            ctx = hyp_ctx[hyp_idx]
            chars = edge_ids[edge_idx]
            lp = np.zeros(m)
            for t in range(length):
                lp += self.ngram.logp(ctx, chars[:, t])
                ctx = np.concatenate([ctx[:, 1:], chars[:, t : t + 1]], axis=1)
            freqs = np.array([f for _, f in group])[edge_idx]
            base = np.array([h.score for h in hyps])[hyp_idx]
            total = base + lp + w.prior * np.log1p(freqs) + w.edge
            top = np.argpartition(-total, per_group)[:per_group] if m > per_group else np.arange(m)
            for r in top:
                phrase, freq = group[edge_idx[r]]
                cand.append((total[r], hyps[hyp_idx[r]], phrase, freq, lp[r], tuple(ctx[r])))
        cand.sort(key=lambda c: -c[0])
        out = []
        # Mismatch checks are Python level; only run them on the best few.
        for total, h, phrase, freq, lp_r, new_ctx in cand[: per_group * 2]:
            text = h.text + phrase
            bad = self._mismatches(syllables, text, len(h.text)) if h.text else 0
            feats = h.feats + np.array([lp_r, Lexicon.log_prior(freq), 1.0, bad])
            out.append(Hyp(text, new_ctx, total + w.mismatch * bad, feats))
        out.sort(key=lambda h: -h.score)
        return out

    def decode(
        self,
        syllables: list[str],
        history: str = "",
        k: int = 30,
        locks: list[Lock] | None = None,
        incremental: bool = True,
    ) -> list[Decoded]:
        t0 = time.perf_counter()
        n = len(syllables)
        locks_t = tuple(sorted(locks or []))
        key = (history, locks_t)
        reuse = 0
        if incremental and key == self._cache_key:
            for a, b in zip(self._cache_syllables, syllables, strict=False):
                if a != b:
                    break
                reuse += 1
        if reuse:
            beams = self._cache_beams[: reuse + 1]
        else:
            beams = [[Hyp("", self.ngram.start_context(history), 0.0)]]
        for j in range(len(beams), n + 1):
            beams.append(self._extend(beams, syllables, j, locks_t))
        self._cache_key, self._cache_syllables, self._cache_beams = key, list(syllables), beams
        self.last_new_positions = n - reuse

        finals = beams[n] if n else []
        if not finals:
            self.last_ms = (time.perf_counter() - t0) * 1000
            return []
        ctx = np.array([h.ctx for h in finals], dtype=np.int64)
        end_lp = self.ngram.logp(ctx, np.full(len(finals), EOS))
        best: dict[str, Decoded] = {}
        for h, e in zip(finals, end_lp, strict=True):
            feats = h.feats.copy()
            feats[0] += e
            d = Decoded(h.text, h.score + e, feats.tolist())
            if h.text not in best or d.score > best[h.text].score:
                best[h.text] = d
        out = sorted(best.values(), key=lambda d: -d.score)[:k]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out
