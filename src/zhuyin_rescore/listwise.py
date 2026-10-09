"""Discriminative training of the character LM reranker over candidate pools.

At inference the reranker picks from a pool with

    score_i = lm_i + a * ngram_i - beta * [i > 0] - mu * i

where lm_i is the full softmax log P(candidate | context) that
CharLMScorer.score_cached computes from a cached context. Training computes
the same lm_i as one batched forward over rows [BOS, context, candidate]:
the positions, the context truncation and the character encoding are the
scorer's own, so the trained model is scored at inference on exactly the
input it was trained on (tests/test_listwise.py checks this).

The loss per pool, with p = softmax(score / T) over the pool:
- expected error (MWER): sum_i p_i * (err_i - mean(err)), errors normalized
  by the clause length;
- cross entropy on the correct sentence, when the pool contains it;
- an LM anchor, the NLL per character of the correct sentence given the
  context, so the model stays a language model and not only a pool sorter.
The fusion weights (a, beta, mu) and a softmax scale are learned per pool
kind.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhuyin_rescore.charlm import PAD, CharLM, CharVocab, context_ids

POOL_KINDS = ("toned+chewing", "open")


@dataclass
class PoolExample:
    context: str
    gold: str
    texts: list[str]
    ngram: list[float]
    errors: list[int]
    kind: int  # index into POOL_KINDS


@dataclass
class Batch:
    ids: torch.Tensor  # (rows, length) [BOS, context, candidate], right padded
    pred_pos: torch.Tensor  # (rows, width) position whose output predicts each character
    targets: torch.Tensor  # (rows, width) candidate character ids
    mask: torch.Tensor  # (rows, width) 1.0 on real characters
    pools: list[tuple[int, int]]  # per example: first row and number of pool rows
    gold_rows: list[int]  # per example: row holding the correct sentence
    gold_in_pool: list[int]  # per example: its index in the pool, or -1
    ngram: list[torch.Tensor]
    errors: list[torch.Tensor]  # normalized by clause length
    kinds: list[int]


def collate(vocab: CharVocab, examples: list[PoolExample], context_chars: int, device) -> Batch:
    """Rows for every pool candidate, plus the correct sentence when the pool
    lacks it (used by the LM anchor only)."""
    seqs, ctx_lens, cands = [], [], []
    pools, gold_rows, gold_in_pool = [], [], []
    for ex in examples:
        ctx = context_ids(vocab, ex.context, context_chars)
        start = len(seqs)
        for t in ex.texts:
            ids = vocab.encode_chars(t)
            seqs.append(ctx + ids)
            ctx_lens.append(len(ctx))
            cands.append(ids)
        pools.append((start, len(ex.texts)))
        idx = ex.texts.index(ex.gold) if ex.gold in ex.texts else -1
        gold_in_pool.append(idx)
        if idx >= 0:
            gold_rows.append(start + idx)
        else:
            ids = vocab.encode_chars(ex.gold)
            gold_rows.append(len(seqs))
            seqs.append(ctx + ids)
            ctx_lens.append(len(ctx))
            cands.append(ids)
    length = max(len(s) for s in seqs)
    width = max(len(c) for c in cands)
    rows = len(seqs)
    ids = torch.full((rows, length), PAD, dtype=torch.long)
    pred_pos = torch.zeros((rows, width), dtype=torch.long)
    targets = torch.zeros((rows, width), dtype=torch.long)
    mask = torch.zeros((rows, width))
    for r, (s, p, c) in enumerate(zip(seqs, ctx_lens, cands, strict=True)):
        ids[r, : len(s)] = torch.tensor(s)
        w = len(c)
        # Character t of the candidate sits at position p + t and is predicted
        # by the output at p + t - 1; the first one by the context's last state.
        pred_pos[r, :w] = torch.arange(p - 1, p - 1 + w)
        targets[r, :w] = torch.tensor(c)
        mask[r, :w] = 1.0
    return Batch(
        ids=ids.to(device),
        pred_pos=pred_pos.to(device),
        targets=targets.to(device),
        mask=mask.to(device),
        pools=pools,
        gold_rows=gold_rows,
        gold_in_pool=gold_in_pool,
        ngram=[torch.tensor(ex.ngram, device=device) for ex in examples],
        errors=[torch.tensor(ex.errors, device=device) / max(len(ex.gold), 1) for ex in examples],
        kinds=[ex.kind for ex in examples],
    )


def candidate_logprobs(model: CharLM, batch: Batch) -> torch.Tensor:
    """log P(candidate | context) per row, full softmax, as score_cached."""
    h, _ = model.hidden(batch.ids)
    idx = batch.pred_pos.unsqueeze(-1).expand(-1, -1, h.shape[-1])
    states = h.gather(1, idx)  # (rows, width, d)
    # The output projection and softmax in fp32 even under autocast: log-prob
    # differences between near identical candidates are small.
    with torch.autocast(states.device.type, enabled=False):
        logits = states.float() @ model.emb.weight.float().T
    nll = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), reduction="none")
    return -(nll.view_as(batch.mask) * batch.mask).sum(-1)


class Fusion(nn.Module):
    """Learnable (a, beta, mu) per pool kind, plus a log scale per kind that
    multiplies the scores inside the training softmax only. The argmax
    ignores the scale; it is there to absorb the pressure to sharpen the
    pool distribution, which the network would otherwise meet by lowering
    its own temperature (and raising its NLL) without ranking better."""

    def __init__(self, init: dict[str, tuple[float, float, float]]):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([init[k] for k in POOL_KINDS], dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(len(POOL_KINDS)))

    def forward(self, lm: torch.Tensor, ngram: torch.Tensor, kind: int) -> torch.Tensor:
        a, beta, mu = self.weights[kind]
        rank = torch.arange(len(lm), device=lm.device, dtype=lm.dtype)
        return lm + a * ngram - beta * (rank > 0).to(lm.dtype) - mu * rank

    def as_dict(self) -> dict[str, list[float]]:
        return {k: [round(float(x), 4) for x in self.weights[i].detach()] for i, k in enumerate(POOL_KINDS)}


def pool_losses(lm: torch.Tensor, batch: Batch, fusion: Fusion, temperature: float = 1.0) -> dict:
    """Mean MWER, CE (over pools that contain the correct sentence) and anchor."""
    mwer, ce, anchor, picked_err = [], [], [], []
    for i, (start, n) in enumerate(batch.pools):
        score = fusion(lm[start : start + n], batch.ngram[i], batch.kinds[i])
        logp = torch.log_softmax(score * fusion.log_scale[batch.kinds[i]].exp() / temperature, dim=-1)
        err = batch.errors[i]
        mwer.append((logp.exp() * (err - err.mean())).sum())
        if batch.gold_in_pool[i] >= 0:
            ce.append(-logp[batch.gold_in_pool[i]])
        n_chars = batch.mask[batch.gold_rows[i]].sum()
        anchor.append(-lm[batch.gold_rows[i]] / n_chars)
        picked_err.append(err[score.argmax()])
    zero = lm.new_zeros(())
    return {
        "mwer": torch.stack(mwer).mean(),
        "ce": torch.stack(ce).mean() if ce else zero,
        "anchor": torch.stack(anchor).mean(),
        "picked_err": torch.stack(picked_err).mean().detach(),
    }
