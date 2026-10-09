"""Small character level LM for CPU reranking.

A pre-norm causal transformer over characters: the Han characters of the
n-gram vocabulary, the clause punctuation, and a few special tokens. Every
candidate covers the same syllables with one character each, so a character
model scores candidates position by position without any tokenization
mismatch.

Two normalizations are available at scoring time:
- full: log softmax over the whole vocabulary, P(chars), the same quantity
  the n-gram and the Qwen reranker score;
- homophone: log softmax over the characters whose reading matches the typed
  syllable at that position, P(chars | readings). It needs only a few dozen
  rows of the output matrix instead of 14k, which is most of the CPU saving.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, UNK, SEP = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<unk>", "<sep>"]
PUNCT = list("，。！？；：、,.!?;:「」『』（）()《》〈〉【】[]\"'“”‘’…—～~|/／")
WS_RE = re.compile(r"\s+")
HOMO_FLOOR = -30.0  # log prob for a candidate character outside its homophone set


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


@dataclass
class CharLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_len: int = 128


class Block(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.ff1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.ff2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, past=None, mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        if past is not None:
            pk, pv = past
            k = torch.cat([pk.expand(b, -1, -1, -1), k], dim=2)
            v = torch.cat([pv.expand(b, -1, -1, -1), v], dim=2)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(o.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x, (k, v)


class CharLM(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids, pos_offset: int = 0, past=None, mask=None, positions=None):
        t = ids.shape[1]
        pos = positions
        if pos is None:
            pos = torch.arange(pos_offset, pos_offset + t, device=ids.device)
        x = self.emb(ids) + self.pos(pos)
        new_past = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if past is None else past[i], mask)
            new_past.append(kv)
        return self.ln_f(x), new_past

    def forward(self, ids):
        """Training: logits for next character at every position."""
        h, _ = self.hidden(ids)
        return h @ self.emb.weight.T

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_charlm(model: CharLM, vocab: CharVocab, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps(asdict(model.cfg), indent=2) + "\n")
    vocab.save(path / "vocab.json")


def load_charlm(path: str | Path, device: str = "cpu") -> tuple[CharLM, CharVocab]:
    path = Path(path)
    cfg = CharLMConfig(**json.loads((path / "config.json").read_text()))
    model = CharLM(cfg)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), CharVocab.load(path / "vocab.json")


class CharLMScorer:
    """CPU friendly candidate scoring with a resident context cache.

    Same interface as LMScorer (context_cache, score_cached). With
    homophones set (a function from a typed syllable to candidate characters),
    score_both also returns the homophone normalized scores.
    """

    def __init__(self, path: str | Path, device: str = "cpu", context_chars: int = 64, homophones=None):
        self.device = torch.device(device)
        self.model, self.vocab = load_charlm(path, device)
        self.context_chars = min(context_chars, self.model.cfg.max_len - 32)
        self.homophones = homophones
        self._homo_cache: dict[str, np.ndarray] = {}
        self._ctx_key: str | None = None
        self._ctx_cache = None

    def quantize_dynamic_int8(self) -> None:
        """int8 dynamic quantization of the transformer's Linear layers (CPU).

        The tied embedding and output projection stay in fp32.
        """
        quantize = torch.ao.quantization.quantize_dynamic
        with warnings.catch_warnings():
            # torch.ao quantized tensor creation is deprecated upstream but works.
            warnings.simplefilter("ignore", UserWarning)
            self.model = quantize(self.model, {nn.Linear}, dtype=torch.qint8)
        self._ctx_key, self._ctx_cache = None, None

    def context_ids(self, context: str) -> list[int]:
        return [BOS] + self.vocab.encode(context)[-self.context_chars :]

    @torch.inference_mode()
    def context_cache(self, context: str):
        if context == self._ctx_key and self._ctx_cache is not None:
            return self._ctx_cache
        ids = torch.tensor([self.context_ids(context)], device=self.device)
        h, past = self.model.hidden(ids)
        cache = (past, h[:, -1:], ids.shape[1])
        self._ctx_key, self._ctx_cache = context, cache
        return cache

    @torch.inference_mode()
    def _hidden(self, context: str, candidates: list[str], cache=None):
        past, last_h, p = cache if cache is not None else self.context_cache(context)
        ids = torch.tensor([self.vocab.encode_chars(c) for c in candidates], device=self.device)
        h, _ = self.model.hidden(ids, pos_offset=p, past=past, mask=self._mask(p, ids.shape[1]))
        # Character t is predicted by the state before it; the first by the context.
        h = torch.cat([last_h.expand(len(candidates), 1, -1), h[:, :-1]], dim=1)
        return h, ids

    def _mask(self, p: int, w: int) -> torch.Tensor:
        causal = torch.ones(w, w, dtype=torch.bool, device=self.device).tril()
        return torch.cat([torch.ones(w, p, dtype=torch.bool, device=self.device), causal], dim=1)

    def _trie(self, candidates: list[str]):
        """Unique prefixes of the candidates: char id, parent node, depth, and
        each candidate's path of nodes."""
        index: dict[tuple[int, ...], int] = {}
        chars, parent, depth, paths = [], [], [], []
        for cand in candidates:
            prefix: tuple[int, ...] = ()
            prev, path = -1, []
            for t, tok in enumerate(self.vocab.encode_chars(cand)):
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

    @torch.inference_mode()
    def _tree_states(self, context: str, candidates: list[str], cache=None):
        """One forward over the prefix trie (tree attention).

        Every unique prefix is one token; it attends to the context and to its
        own ancestors only, so shared prefixes are computed once. Returns the
        predicting state for each node (its parent's output, or the context's
        last state for depth 0) plus the trie.
        """
        past, last_h, p = cache if cache is not None else self.context_cache(context)
        chars, parent, depth, paths = self._trie(candidates)
        n = len(chars)
        anc = np.zeros((n, n), dtype=bool)
        for i in range(n):
            j = i
            while j >= 0:
                anc[i, j] = True
                j = parent[j]
        mask = torch.cat([torch.ones(n, p, dtype=torch.bool), torch.from_numpy(anc)], dim=1).to(self.device)
        ids = torch.tensor([chars], device=self.device)
        positions = torch.tensor(depth, device=self.device) + p
        h, _ = self.model.hidden(ids, past=past, mask=mask, positions=positions)
        states = torch.cat([last_h[0], h[0]])  # row 0: context state, row i + 1: node i
        pred = np.asarray(parent) + 1
        uniq, row = np.unique(pred, return_inverse=True)
        return states[torch.from_numpy(uniq).to(self.device)], row, chars, depth, paths

    @staticmethod
    def _sum_paths(node_lp: np.ndarray, paths: list[list[int]]) -> list[float]:
        return [float(node_lp[path].sum()) for path in paths]

    @torch.inference_mode()
    def score_cached(self, context: str, candidates: list[str], cache=None) -> list[float]:
        """Full softmax log P(candidate | context), prefix sharing via a trie."""
        if not candidates:
            return []
        pred_states, row, chars, _, paths = self._tree_states(context, candidates, cache)
        lsm = torch.log_softmax(pred_states @ self.model.emb.weight.T, dim=-1)
        node_lp = lsm[torch.from_numpy(row), torch.tensor(chars)].cpu().numpy()
        return self._sum_paths(node_lp, paths)

    @torch.inference_mode()
    def score_cached_flat(self, context: str, candidates: list[str], cache=None) -> list[float]:
        """Reference: full softmax without prefix sharing (equal length candidates)."""
        if not candidates:
            return []
        h, ids = self._hidden(context, candidates, cache)
        lp = torch.log_softmax(h @ self.model.emb.weight.T, dim=-1)
        return lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1).tolist()

    def _homophone_ids(self, syllable: str) -> np.ndarray:
        ids = self._homo_cache.get(syllable)
        if ids is None:
            chars = self.homophones(syllable) if self.homophones else []
            ids = np.unique(np.array([self.vocab.index.get(c, UNK) for c in chars], dtype=np.int64))
            self._homo_cache[syllable] = ids
        return ids

    @torch.inference_mode()
    def score_homophone(
        self, context: str, candidates: list[str], syllables: list[str], cache=None
    ) -> list[float]:
        """log P(candidate | context, readings): softmax over each position's homophones."""
        if not candidates:
            return []
        pred_states, row, chars, depth, paths = self._tree_states(context, candidates, cache)
        n_pos = max(depth) + 1
        chars_a, depth_a = np.asarray(chars), np.asarray(depth)
        sets = [
            np.union1d(
                self._homophone_ids(syllables[t]) if t < len(syllables) else chars_a[:0],
                chars_a[depth_a == t],
            )
            for t in range(n_pos)
        ]
        union = np.unique(np.concatenate(sets))
        member = np.zeros((n_pos, len(union)), dtype=bool)
        for t, st in enumerate(sets):
            member[t, np.searchsorted(union, st)] = True
        logits = pred_states @ self.model.emb.weight[torch.from_numpy(union).to(self.device)].T
        node_logits = logits[torch.from_numpy(row)]  # (nodes, |union|)
        node_member = torch.from_numpy(member[depth_a]).to(self.device)
        z = torch.logsumexp(node_logits.masked_fill(~node_member, -math.inf), dim=-1)
        col = torch.from_numpy(np.searchsorted(union, chars_a)).to(self.device)
        target = node_logits.gather(-1, col.unsqueeze(-1)).squeeze(-1)
        node_lp = torch.clamp(target - z, min=HOMO_FLOOR).cpu().numpy()
        return self._sum_paths(node_lp, paths)
