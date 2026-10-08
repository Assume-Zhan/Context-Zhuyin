"""Candidate sentence scoring with a small causal LM.

Score of a candidate = sum of token log-probs of the candidate given the
committed context. Every candidate covers the same syllables with one
character each, so joint log-probs are comparable without length
normalization, whatever the tokenization.

Context and candidate are tokenized separately and their ids concatenated,
so a token never spans the boundary and the context KV cache stays valid.
A document start token is always prepended, so the first candidate token is
scored even with an empty context.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class LMScorer:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_context_tokens: int = 64,
    ):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPES[dtype]).to(self.device).eval()
        )
        self.decoder = self.model.get_decoder()
        self.lm_head = self.model.get_output_embeddings()
        self.max_context_tokens = max_context_tokens
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if start is None or start == self.tokenizer.unk_token_id:
            raise ValueError(f"no document start token for {model_name}")
        self.start_id = int(start)
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.start_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def prefix_ids(self, context: str) -> list[int]:
        ctx = self.encode(context) if context else []
        if self.max_context_tokens >= 0:
            ctx = ctx[len(ctx) - self.max_context_tokens :] if len(ctx) > self.max_context_tokens else ctx
        return [self.start_id] + ctx

    def _token_logprobs(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Log-prob of each target given the hidden state that predicts it."""
        logits = self.lm_head(hidden).float()
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    @torch.inference_mode()
    def score_reference(self, context: str, candidates: list[str]) -> list[float]:
        """Full sequence scoring, one padded batch, no cache. Slow but simple."""
        prefix = self.prefix_ids(context)
        cand_ids = [self.encode(c) for c in candidates]
        seqs = [prefix + ids for ids in cand_ids]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attn[i, : len(s)] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        hidden = self.decoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        p = len(prefix)
        # Position t predicts token t + 1; candidate tokens sit at p .. p + len - 1.
        h = hidden[:, p - 1 : width - 1]
        tgt = input_ids[:, p:width]
        lp = self._token_logprobs(h, tgt)
        mask = attn[:, p:width].to(lp.dtype)
        return (lp * mask).sum(-1).tolist()

    @torch.inference_mode()
    def context_cache(self, context: str) -> tuple[DynamicCache, torch.Tensor, int]:
        """Run the prefix once; return its cache, last hidden state, length."""
        prefix = torch.tensor([self.prefix_ids(context)], device=self.device)
        out = self.decoder(input_ids=prefix, use_cache=True)
        return out.past_key_values, out.last_hidden_state[:, -1:], prefix.shape[1]

    @torch.inference_mode()
    def score_cached(self, context: str, candidates: list[str], cache=None) -> list[float]:
        """Score with the context KV cache; only candidate tokens are computed.

        cache: optional result of context_cache(context) to reuse across
        calls with the same committed context.
        """
        past, last_hidden, p = cache if cache is not None else self.context_cache(context)
        n = len(candidates)
        cand_ids = [self.encode(c) for c in candidates]
        width = max(len(ids) for ids in cand_ids)
        input_ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        cand_mask = torch.zeros((n, width), dtype=torch.long)
        for i, ids in enumerate(cand_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            cand_mask[i, : len(ids)] = 1
        input_ids, cand_mask = input_ids.to(self.device), cand_mask.to(self.device)

        # The cache is extended in place by a forward pass, so work on a copy.
        batch_past = DynamicCache()
        for layer_idx, layer in enumerate(past.layers):
            batch_past.update(
                layer.keys.expand(n, -1, -1, -1).contiguous(),
                layer.values.expand(n, -1, -1, -1).contiguous(),
                layer_idx,
            )
        attn = torch.cat([torch.ones((n, p), dtype=torch.long, device=self.device), cand_mask], dim=1)
        position_ids = torch.arange(p, p + width, device=self.device).unsqueeze(0).expand(n, -1)
        hidden = self.decoder(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            past_key_values=batch_past,
            use_cache=True,
        ).last_hidden_state
        # The first candidate token is predicted by the last prefix position.
        h = torch.cat([last_hidden.expand(n, -1, -1), hidden[:, :-1]], dim=1)
        lp = self._token_logprobs(h, input_ids)
        return (lp * cand_mask.to(lp.dtype)).sum(-1).tolist()
