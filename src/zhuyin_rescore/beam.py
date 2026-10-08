"""Lattice beam search over dictionary phrases scored by a char n-gram.

The lattice edges are libchewing dictionary phrases whose reading matches a
span of the typed syllables (under the input condition), so every multi
character word on a path has a valid reading. Hypotheses are kept per end
position; each extension is scored with

    score = ngram + w_prior * log(1 + freq) + w_edge + w_mismatch * mismatches

where a mismatch is a window of 2 to 4 characters that crosses an edge
boundary, is itself a dictionary phrase, and has no reading that matches the
typed syllables (for example 長大 assembled from single characters for an
input that is not read zhang da). All n-gram lookups of one search step are
done in a single vectorized call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from zhuyin_rescore.lexicon import Lexicon
from zhuyin_rescore.ngram import EOS, CharNgram

SINGLE_LIMIT = {"full": None, "notone": 40, "initial": 40}


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

    def _mismatches(self, syllables: list[str], text: str, start: int) -> int:
        """Windows that end in the new edge, start before it, and misread."""
        n_bad = 0
        end = len(text)
        keys = self.lexicon.phrase_keys
        for e in range(start + 1, end + 1):
            for length in (2, 3, 4):
                s = e - length
                if s < 0 or s >= start:
                    continue
                w = text[s:e]
                readings = keys.get(w)
                if readings and tuple(syllables[s:e]) not in readings:
                    n_bad += 1
        return n_bad

    def decode(self, syllables: list[str], history: str = "", k: int = 30) -> list[Decoded]:
        t0 = time.perf_counter()
        n = len(syllables)
        w = self.weights
        by_start: dict[int, list[tuple[int, str, int]]] = {}
        for i, j, phrase, freq in self.lexicon.edges(syllables, self.single_limit):
            by_start.setdefault(i, []).append((j, phrase, freq))
        beams: list[list[Hyp]] = [[] for _ in range(n + 1)]
        beams[0] = [Hyp("", self.ngram.start_context(history), 0.0)]

        for i in range(n):
            if not beams[i] or i not in by_start:
                continue
            hyps = sorted(beams[i], key=lambda h: -h.score)[: self.beam]
            edges = by_start[i]
            # Group edges by length so each group is one rectangular array.
            by_len: dict[int, list[tuple[int, str, int]]] = {}
            for edge in edges:
                by_len.setdefault(edge[0] - i, []).append(edge)
            per_group = self.beam * self.expand_factor
            cand: list[tuple[float, Hyp, int, str, int, float, tuple[int, ...]]] = []
            for length, group in by_len.items():
                m = len(hyps) * len(group)
                hyp_idx = np.repeat(np.arange(len(hyps)), len(group))
                edge_idx = np.tile(np.arange(len(group)), len(hyps))
                hyp_ctx = np.array([h.ctx for h in hyps], dtype=np.int64)
                edge_ids = np.array([self.ngram.ids(phrase) for _, phrase, _ in group], dtype=np.int64)
                ctx = hyp_ctx[hyp_idx]
                chars = edge_ids[edge_idx]
                lp = np.zeros(m)
                for t in range(length):
                    lp += self.ngram.logp(ctx, chars[:, t])
                    ctx = np.concatenate([ctx[:, 1:], chars[:, t : t + 1]], axis=1)
                freqs = np.array([f for _, _, f in group])[edge_idx]
                base = np.array([h.score for h in hyps])[hyp_idx]
                total = base + lp + w.prior * np.log1p(freqs) + w.edge
                top = np.argpartition(-total, per_group)[:per_group] if m > per_group else np.arange(m)
                for r in top:
                    j, phrase, freq = group[edge_idx[r]]
                    cand.append((total[r], hyps[hyp_idx[r]], j, phrase, freq, lp[r], tuple(ctx[r])))
            # Mismatch checks are Python level; only run them on the best few.
            cand.sort(key=lambda c: -c[0])
            for total, h, j, phrase, freq, lp_r, new_ctx in cand[: per_group * 2]:
                text = h.text + phrase
                bad = self._mismatches(syllables, text, len(h.text)) if i > 0 else 0
                feats = h.feats + np.array([lp_r, Lexicon.log_prior(freq), 1.0, bad])
                beams[j].append(Hyp(text, new_ctx, total + w.mismatch * bad, feats))

        finals = beams[n]
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
