"""Candidate scoring replayed as a CUDA graph.

The eager scorer spends most of its 20 ms launching the ~550 kernels of a
24 layer forward, not computing. This scorer runs the same Qwen2 forward
(RMSNorm, rotary, grouped query attention against the resident context KV
cache, SwiGLU) as plain tensor code on the model's own weights, with every
shape fixed: candidates padded to k_max x w_max tokens (32 x 24), the context padded
to p_max and masked by its true length held in a device tensor. Fixed shapes
let it be captured once as a CUDA graph and replayed per request.

Requests that do not fit the static shapes fall back to the eager scorer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zhuyin_rescore.scorer import LMScorer


class GraphScorer:
    def __init__(self, scorer: LMScorer, k_max: int = 32, w_max: int = 24, p_max: int = 72):
        if scorer.device.type != "cuda":
            raise ValueError("GraphScorer needs a CUDA device")
        self.scorer = scorer
        model = scorer.model
        cfg = model.config
        self.layers = model.get_decoder().layers
        self.norm = model.get_decoder().norm
        self.embed = model.get_input_embeddings()
        self.lm_head = scorer.lm_head
        self.inv_freq = model.get_decoder().rotary_emb.inv_freq.float()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.k_max, self.w_max, self.p_max = k_max, w_max, p_max
        dev, dt = scorer.device, next(model.parameters()).dtype
        self.dtype = dt
        n_layers = len(self.layers)
        self.ctx_k = torch.zeros(n_layers, 1, self.n_kv, p_max, self.head_dim, dtype=dt, device=dev)
        self.ctx_v = torch.zeros_like(self.ctx_k)
        self.ctx_len = torch.zeros((), dtype=torch.long, device=dev)
        self.last_hidden = torch.zeros(1, 1, cfg.hidden_size, dtype=dt, device=dev)
        self.cand_ids = torch.zeros(k_max, w_max, dtype=torch.long, device=dev)
        self.cand_mask = torch.zeros(k_max, w_max, dtype=torch.float32, device=dev)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: torch.Tensor | None = None
        self.context: str | None = None
        self.context_ok = False
        self.fallbacks = 0

    # ---------------------------------------------------------------- forward
    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return weight * xf.to(self.dtype)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    def _forward(self) -> torch.Tensor:
        k, w = self.cand_ids.shape
        dev = self.cand_ids.device
        x = self.embed(self.cand_ids)
        pos = (self.ctx_len + torch.arange(w, device=dev)).float()
        freqs = pos[:, None] * self.inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(self.dtype)[None, None]
        sin = emb.sin().to(self.dtype)[None, None]
        ctx_ok = torch.arange(self.p_max, device=dev) < self.ctx_len
        causal = torch.ones(w, w, dtype=torch.bool, device=dev).tril()
        mask = torch.cat([ctx_ok[None].expand(w, -1), causal], dim=1)[None, None]
        rep = self.n_heads // self.n_kv
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = self._rms(x, layer.input_layernorm.weight)
            q = attn.q_proj(h).view(k, w, self.n_heads, self.head_dim).transpose(1, 2)
            kk = attn.k_proj(h).view(k, w, self.n_kv, self.head_dim).transpose(1, 2)
            vv = attn.v_proj(h).view(k, w, self.n_kv, self.head_dim).transpose(1, 2)
            q, kk = self._rotate(q, cos, sin), self._rotate(kk, cos, sin)
            kk = torch.cat([self.ctx_k[li].expand(k, -1, -1, -1), kk], dim=2).repeat_interleave(rep, dim=1)
            vv = torch.cat([self.ctx_v[li].expand(k, -1, -1, -1), vv], dim=2).repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            x = x + attn.o_proj(o.transpose(1, 2).reshape(k, w, -1))
            h = self._rms(x, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            x = x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))
        x = self._rms(x, self.norm.weight)
        # Token t is predicted by the state before it; the first by the context.
        h = torch.cat([self.last_hidden.expand(k, 1, -1), x[:, :-1]], dim=1)
        lp = torch.log_softmax(self.lm_head(h).float(), dim=-1)
        lp = lp.gather(-1, self.cand_ids.unsqueeze(-1)).squeeze(-1)
        return (lp * self.cand_mask).sum(-1)

    # -------------------------------------------------------------- interface
    @torch.inference_mode()
    def set_context(self, context: str) -> None:
        """Build the context KV cache (eager) and copy it into the static buffers."""
        if context == self.context:
            return
        past, last_hidden, p = self.scorer.context_cache(context)
        self.context = context
        self.context_ok = p <= self.p_max
        if not self.context_ok:
            return
        for li, layer in enumerate(past.layers):
            self.ctx_k[li, :, :, :p].copy_(layer.keys)
            self.ctx_v[li, :, :, :p].copy_(layer.values)
        self.ctx_len.fill_(p)
        self.last_hidden.copy_(last_hidden)

    def _capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._forward()

    @torch.inference_mode()
    def score(self, context: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        self.set_context(context)
        ids = [self.scorer.encode(c) for c in candidates]
        if not self.context_ok or len(ids) > self.k_max or max(len(t) for t in ids) > self.w_max:
            self.fallbacks += 1
            return self.scorer.score_cached(context, candidates)
        self.cand_ids.zero_()
        self.cand_mask.zero_()
        for i, t in enumerate(ids):
            self.cand_ids[i, : len(t)] = torch.tensor(t)
            self.cand_mask[i, : len(t)] = 1.0
        if self.graph is None:
            self._capture()
        self.graph.replay()
        return self.scorer._to_list(self.out[: len(candidates)].clone())

    # Same name as LMScorer so the IME server can use either.
    def score_cached(self, context: str, candidates: list[str], cache=None) -> list[float]:
        return self.score(context, candidates)
