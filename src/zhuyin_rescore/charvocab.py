"""Character vocabulary, context encoding and candidate prefix tries.

Shared by the torch model (charlm.py) and the numpy scorer (charlm_np.py);
torch free so the numpy scorer can run without importing torch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PAD, BOS, UNK, SEP = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<unk>", "<sep>"]
PUNCT = list("，。！？；：、,.!?;:「」『』（）()《》〈〉【】[]\"'“”‘’…—～~|/／")
WS_RE = re.compile(r"\s+")


class CharVocab:
    def __init__(self, chars: list[str]):
        self.tokens = SPECIALS + PUNCT + [c for c in chars if c not in PUNCT]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        lut = np.full(0x110000, UNK, dtype=np.int32)
        for t, i in self.index.items():
            if len(t) == 1:
                lut[ord(t)] = i
        self._lut = lut

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Characters to ids; whitespace runs become one SEP, unknown runs one UNK."""
        if not text:
            return []
        text = WS_RE.sub("\n", text)
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        ids = self._lut[cps]
        ids[cps == 10] = SEP
        if len(ids) > 1:
            keep = np.ones(len(ids), dtype=bool)
            keep[1:] = ~(((ids[1:] == UNK) & (ids[:-1] == UNK)) | ((ids[1:] == SEP) & (ids[:-1] == SEP)))
            ids = ids[keep]
        return ids.tolist()

    def encode_chars(self, text: str) -> list[int]:
        """One id per character, no collapsing (candidates must keep their length)."""
        if not text:
            return []
        return self._lut[np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)].tolist()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens[len(SPECIALS) + len(PUNCT) :], ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        return cls(json.loads(path.read_text()))


def context_ids(vocab: CharVocab, context: str, context_chars: int) -> list[int]:
    """BOS plus the last context_chars ids of the context; the scorer and the
    reranker training both build the context this way."""
    return [BOS] + vocab.encode(context)[-context_chars:]


def prefix_trie(vocab: CharVocab, candidates: list[str]):
    """Unique prefixes of the candidates: char id, parent node, depth, and
    each candidate's path of nodes."""
    index: dict[tuple[int, ...], int] = {}
    chars, parent, depth, paths = [], [], [], []
    for cand in candidates:
        prefix: tuple[int, ...] = ()
        prev, path = -1, []
        for t, tok in enumerate(vocab.encode_chars(cand)):
            prefix += (tok,)
            j = index.get(prefix)
            if j is None:
                j = index[prefix] = len(chars)
                chars.append(tok)
                parent.append(prev)
                depth.append(t)
            path.append(j)
            prev = j
        paths.append(path)
    return chars, parent, depth, paths
