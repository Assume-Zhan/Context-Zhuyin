"""Character n-gram LM with stupid backoff, stored as sorted numpy arrays.

Built for a CPU first pass: every lookup is a vectorized searchsorted over
sorted uint64 keys, so scoring thousands of (context, next char) pairs in a
beam step is a few numpy calls.

Token ids: 0 sentence start, 1 unknown, 2 sentence end, 3 padding (never
seen in training, so lookups that include it back off), then characters.
Ids fit in 16 bits, so an n-gram of order <= 4 packs into one uint64 key.

Two storage forms answer the same queries:
- counts (from train): int counts per n-gram; a score needs the n-gram's
  and its context's count;
- compact (from compact): the stupid backoff score log(c(ngram) / c(context))
  precomputed per n-gram and quantized through a codebook per order (16 or
  8 bit codes), optionally with rare n-grams pruned, and keys of order <= 2
  in uint32. Smaller, and one lookup per order instead of two.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

BOS, UNK, EOS, PAD = 0, 1, 2, 3
N_SPECIAL = 4
BITS = 16


def pack(ids: np.ndarray) -> np.ndarray:
    """Pack an (M, k) id array into (M,) uint64 keys, oldest id in the high bits."""
    ids = ids.astype(np.uint64)
    key = np.zeros(ids.shape[0], dtype=np.uint64)
    for j in range(ids.shape[1]):
        key = (key << np.uint64(BITS)) | ids[:, j]
    return key


def quantize(values: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Codebook (bin means over quantile bins) and one code per value."""
    n = 1 << bits
    edges = np.quantile(values, np.linspace(0, 1, n + 1)[1:-1])
    codes = np.searchsorted(edges, values).astype(np.uint8 if bits <= 8 else np.uint16)
    sums = np.bincount(codes, weights=values, minlength=n)
    sizes = np.bincount(codes, minlength=n)
    book = np.where(sizes > 0, sums / np.maximum(sizes, 1), 0.0).astype(np.float32)
    return book, codes


class CharNgram:
    def __init__(
        self,
        order: int,
        chars: list[str],
        keys: list[np.ndarray],
        counts: list[np.ndarray] | None = None,
        alpha: float = 0.4,
        compact: dict | None = None,
    ):
        """counts: counts per order; or compact: {"total", "logp1", "codes",
        "books"} with codes[k - 2] / books[k - 2] for order k >= 2."""
        if order > 4:
            raise ValueError("order > 4 does not fit 16 bit ids into a uint64 key")
        self.order = order
        self.chars = chars
        self.char_to_id = {c: i + N_SPECIAL for i, c in enumerate(chars)}
        self.keys = keys  # keys[k - 1] holds sorted keys of order k
        self.counts = counts
        self.compact_data = compact
        self.total = float(compact["total"]) if compact else float(counts[0].sum())
        self.log_alpha = math.log(alpha)
        self.unk_logp = (order - 1) * self.log_alpha + math.log(0.5 / self.total)
        self.unseen_logp = math.log(0.5 / self.total)

    # ---------------------------------------------------------------- build
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        order: int = 4,
        min_char_count: int = 2,
        min_count: dict[int, int] | None = None,
        chunk: int = 50_000_000,
    ) -> CharNgram:
        """Count n-grams over pure Han runs; each run gets BOS and EOS.

        Texts must not contain newlines; they are joined with newlines and
        mapped to ids through a code point lookup table in one pass.
        """
        cps = np.frombuffer("\n".join(texts).encode("utf-32-le"), dtype=np.uint32)
        sep = cps == ord("\n")
        uniq, ucount = np.unique(cps[~sep], return_counts=True)
        chars = [chr(int(c)) for c, n in zip(uniq, ucount, strict=True) if n >= min_char_count]
        if len(chars) + N_SPECIAL >= 1 << BITS:
            raise ValueError("vocabulary does not fit 16 bit ids")
        lut = np.full(0x110000, UNK, dtype=np.uint16)
        lut[[ord(c) for c in chars]] = np.arange(N_SPECIAL, N_SPECIAL + len(chars), dtype=np.uint16)
        ids = lut[cps]
        ids[sep] = EOS
        del cps, lut
        # Every separator becomes EOS followed by BOS; add the outer BOS and EOS.
        seq = np.insert(ids, np.flatnonzero(sep) + 1, BOS)
        seq = np.concatenate([[BOS], seq, [EOS]]).astype(np.uint16)
        runs = np.cumsum(seq == BOS, dtype=np.uint32)
        del ids, sep

        min_count = min_count or {3: 2, 4: 2}
        keys, counts = [], []
        for k in range(1, order + 1):
            uk, uc = [], []
            # Chunks overlap by k - 1 tokens; every window ends inside its chunk.
            for start in range(0, len(seq), chunk):
                lo = max(0, start - (k - 1))
                s_ = seq[lo : start + chunk]
                r_ = runs[lo : start + chunk]
                if len(s_) < k:
                    continue
                win = np.lib.stride_tricks.sliding_window_view(s_, k)
                rwin = np.lib.stride_tricks.sliding_window_view(r_, k)
                u, c = np.unique(pack(win[rwin[:, 0] == rwin[:, -1]]), return_counts=True)
                uk.append(u)
                uc.append(c)
            allk = np.concatenate(uk)
            allc = np.concatenate(uc)
            order_idx = np.argsort(allk, kind="stable")
            allk, allc = allk[order_idx], allc[order_idx]
            starts = np.concatenate([[0], np.flatnonzero(np.diff(allk)) + 1])
            fk = allk[starts]
            fc = np.add.reduceat(allc, starts).astype(np.uint32)
            mc = min_count.get(k, 1)
            if mc > 1:
                keep = fc >= mc
                fk, fc = fk[keep], fc[keep]
            keys.append(fk)
            counts.append(fc)
        return cls(order, chars, keys, counts)

    def compact(self, min_count: dict[int, int] | None = None, bits: int = 8) -> CharNgram:
        """Compact form: precomputed, quantized scores; n-grams of order k
        seen fewer than min_count[k] times dropped (contexts are counted
        before pruning, so kept scores do not change)."""
        if self.counts is None:
            raise ValueError("already compact")
        min_count = min_count or {}
        keys = [self.keys[0].astype(np.uint32)]
        logp1 = np.log(self.counts[0] / self.total).astype(np.float32)
        codes, books = [], []
        for k in range(2, self.order + 1):
            kk, cc = np.asarray(self.keys[k - 1]), np.asarray(self.counts[k - 1])
            c_ctx = self._lookup(k - 1, kk >> np.uint64(BITS))
            # An n-gram whose context was pruned never scores (backoff), as in logp.
            keep = (cc >= min_count.get(k, 1)) & (c_ctx > 0)
            kk, cc, c_ctx = kk[keep], cc[keep], c_ctx[keep]
            book, code = quantize(np.log(cc / c_ctx), bits)
            keys.append(kk.astype(np.uint32) if k <= 2 else kk)
            codes.append(code)
            books.append(book)
        data = {"total": self.total, "logp1": logp1, "codes": codes, "books": books}
        return CharNgram(self.order, self.chars, keys, alpha=math.exp(self.log_alpha), compact=data)

    def save(self, path: str | Path) -> None:
        """Save as a directory of .npy files so loads can memory map them."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "chars.npy", np.array(self.chars))
        for k in range(self.order):
            np.save(path / f"keys{k + 1}.npy", self.keys[k])
            if self.compact_data is None:
                np.save(path / f"counts{k + 1}.npy", self.counts[k])
        if self.compact_data is not None:
            np.save(path / "logp1.npy", self.compact_data["logp1"])
            np.save(path / "total.npy", np.array(self.total))
            for k in range(2, self.order + 1):
                np.save(path / f"codes{k}.npy", self.compact_data["codes"][k - 2])
                np.save(path / f"book{k}.npy", self.compact_data["books"][k - 2])

    @classmethod
    def load(cls, path: str | Path, order: int | None = None, mmap: bool = True) -> CharNgram:
        """Load a saved model (either form); order below the saved order truncates it."""
        path = Path(path)
        saved = len(list(path.glob("keys*.npy")))
        order = order or saved
        if order > saved:
            raise ValueError(f"model at {path} has order {saved}, asked for {order}")
        mode = "r" if mmap else None
        keys = [np.load(path / f"keys{k + 1}.npy", mmap_mode=mode) for k in range(order)]
        chars = [str(c) for c in np.load(path / "chars.npy")]
        if (path / "logp1.npy").exists():
            data = {
                "total": float(np.load(path / "total.npy")),
                "logp1": np.load(path / "logp1.npy"),
                "codes": [np.load(path / f"codes{k}.npy", mmap_mode=mode) for k in range(2, order + 1)],
                "books": [np.load(path / f"book{k}.npy") for k in range(2, order + 1)],
            }
            return cls(order, chars, keys, compact=data)
        counts = [np.load(path / f"counts{k + 1}.npy", mmap_mode=mode) for k in range(order)]
        return cls(order, chars, keys, counts)

    # ---------------------------------------------------------------- query
    def ids(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, UNK) for c in text]

    def start_context(self, history: str = "") -> tuple[int, ...]:
        """Context ids for a new clause; continues history if it ends in Han text."""
        width = self.order - 1
        tail = []
        for c in reversed(history):
            if c not in self.char_to_id or len(tail) == width:
                break
            tail.append(self.char_to_id[c])
        if tail:
            ctx = list(reversed(tail))
        else:
            ctx = [BOS]
        return tuple([PAD] * (width - len(ctx)) + ctx[-width:])

    def _find(self, k: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whether each key is present and its index (clipped when absent)."""
        keys = self.keys[k - 1]
        if keys.dtype == np.uint32:
            q = q.astype(np.uint32)  # else searchsorted would upcast the whole array
        idx = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return keys[idx] == q, idx

    def _lookup(self, k: int, q: np.ndarray) -> np.ndarray:
        found, idx = self._find(k, q)
        return np.where(found, self.counts[k - 1][idx], 0).astype(np.float64)

    def logp(self, ctx: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Stupid backoff log score of w given ctx; ctx is (M, order - 1)."""
        m = len(w)
        out = np.empty(m, dtype=np.float64)
        done = np.zeros(m, dtype=bool)
        penalty = 0.0
        data = self.compact_data
        for k in range(self.order, 1, -1):
            h = ctx[:, ctx.shape[1] - (k - 1) :]
            if data is not None:
                found, idx = self._find(k, pack(np.column_stack([h, w])))
                hit = ~done & found
                out[hit] = penalty + data["books"][k - 2][data["codes"][k - 2][idx[hit]]]
            else:
                c_ng = self._lookup(k, pack(np.column_stack([h, w])))
                c_h = self._lookup(k - 1, pack(h))
                hit = ~done & (c_ng > 0) & (c_h > 0)
                out[hit] = penalty + np.log(c_ng[hit] / c_h[hit])
            done |= hit
            penalty += self.log_alpha
        rest = ~done
        if data is not None:
            found, idx = self._find(1, pack(w[rest].reshape(-1, 1)))
            out[rest] = penalty + np.where(found, data["logp1"][idx], self.unseen_logp)
        else:
            c1 = self._lookup(1, pack(w[rest].reshape(-1, 1)))
            out[rest] = penalty + np.log(np.maximum(c1, 0.5) / self.total)
        # UNK pools every out of vocabulary char, so its counts are not a
        # probability of any single one of them; give it a floor instead.
        out[w == UNK] = self.unk_logp
        return out

    def score_batch(self, texts: list[str], history: str = "", end: bool = True) -> np.ndarray:
        """Scores of equal length texts in one vectorized pass per position."""
        if not texts:
            return np.zeros(0)
        n = len(texts[0])
        if any(len(t) != n for t in texts):
            return np.array([self.score_text(t, history, end) for t in texts])
        ids = np.array([self.ids(t) for t in texts], dtype=np.int64).reshape(len(texts), n)
        if end:
            ids = np.concatenate([ids, np.full((len(texts), 1), EOS)], axis=1)
        ctx = np.tile(np.array(self.start_context(history), dtype=np.int64), (len(texts), 1))
        total = np.zeros(len(texts))
        for t in range(ids.shape[1]):
            total += self.logp(ctx, ids[:, t])
            ctx = np.concatenate([ctx[:, 1:], ids[:, t : t + 1]], axis=1)
        return total

    def score_text(self, text: str, history: str = "", end: bool = True) -> float:
        ctx = np.array([self.start_context(history)], dtype=np.int64)
        total = 0.0
        for c in self.ids(text) + ([EOS] if end else []):
            total += float(self.logp(ctx, np.array([c]))[0])
            ctx = np.concatenate([ctx[:, 1:], [[c]]], axis=1)
        return total
