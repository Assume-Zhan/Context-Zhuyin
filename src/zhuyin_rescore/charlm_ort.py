"""Character LM scoring on ONNX Runtime, for a CPU reranker without torch.

Importing torch costs the IME server about 450 MB of resident memory, ONNX
Runtime about 20 MB. The model is exported once (export_onnx, needs torch and
the onnx package) as two graphs next to model.pt:
- context.onnx: [BOS, context] -> each layer's keys / values and the last
  hidden state (the resident context cache);
- candidates.onnx: the candidates' prefix trie as one sequence with tree
  attention against the cached context, the output projection and the log
  softmax, returning each trie node's log-probability.
int8 dynamic quantization of both graphs (*.int8.onnx) is optional.
Scores of the fp32 graphs match CharLMScorer.score_cached
(tests/test_charlm_ort.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from zhuyin_rescore.charvocab import CharVocab, context_ids, prefix_trie

OPSET = 17


def export_onnx(path: str | Path, int8: bool = True) -> None:
    import torch
    import torch.nn as nn

    from zhuyin_rescore.charlm import load_charlm

    path = Path(path)
    model, _ = load_charlm(path)
    model.requires_grad_(False)
    n_layers = model.cfg.n_layers

    class RMSNorm(nn.Module):
        """nn.RMSNorm spelled out: the exporter has no rms_norm before opset 23."""

        def __init__(self, norm: nn.RMSNorm):
            super().__init__()
            self.weight = norm.weight
            self.eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    for blk in model.blocks:
        blk.ln1, blk.ln2 = RMSNorm(blk.ln1), RMSNorm(blk.ln2)
    model.ln_f = RMSNorm(model.ln_f)

    class Context(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, ids):
            h, past = self.model.hidden(ids)
            k = torch.stack([kv[0][0] for kv in past])
            v = torch.stack([kv[1][0] for kv in past])
            return h[0, -1:], k, v

    class Candidates(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            # A plain weight input, so quantization can find the projection.
            self.register_buffer("emb_t", model.emb.weight.detach().T.contiguous())

        def forward(self, ids, positions, mask, past_k, past_v, last_h, pred_idx, node_pred, node_char):
            past = [(past_k[i][None], past_v[i][None]) for i in range(n_layers)]
            h, _ = self.model.hidden(ids[None], past=past, mask=mask, positions=positions)
            states = torch.cat([last_h, h[0]])[pred_idx]
            logits = states @ self.emb_t
            lse = torch.logsumexp(logits, dim=-1)
            node_logit = logits.reshape(-1)[node_pred * logits.shape[1] + node_char]
            return node_logit - lse[node_pred]

    with torch.no_grad():
        ids = torch.tensor([[1, 5, 6, 7]])
        last_h, k, v = Context()(ids)
        torch.onnx.export(
            Context(),
            (ids,),
            path / "context.onnx",
            input_names=["ids"],
            output_names=["last_h", "past_k", "past_v"],
            dynamic_axes={"ids": {1: "p"}, "past_k": {2: "p"}, "past_v": {2: "p"}},
            opset_version=OPSET,
            dynamo=False,
        )
        n = 3
        args = (
            torch.tensor([8, 9, 10]),
            torch.tensor([4, 5, 5]),
            torch.ones(n, 4 + n, dtype=torch.bool),
            k,
            v,
            last_h,
            torch.tensor([0, 1]),
            torch.tensor([0, 1, 1]),
            torch.tensor([8, 9, 10]),
        )
        names = ["ids", "positions", "mask", "past_k", "past_v", "last_h"]
        names += ["pred_idx", "node_pred", "node_char"]
        torch.onnx.export(
            Candidates(),
            args,
            path / "candidates.onnx",
            input_names=names,
            output_names=["node_lp"],
            dynamic_axes={
                "ids": {0: "n"},
                "positions": {0: "n"},
                "mask": {0: "n", 1: "pn"},
                "past_k": {2: "p"},
                "past_v": {2: "p"},
                "pred_idx": {0: "m"},
                "node_pred": {0: "n"},
                "node_char": {0: "n"},
                "node_lp": {0: "n"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for name in ("context", "candidates"):
            quantize_dynamic(path / f"{name}.onnx", path / f"{name}.int8.onnx", weight_type=QuantType.QInt8)


class OrtCharLMScorer:
    """Drop in for CharLMScorer's full softmax scoring (score_cached)."""

    def __init__(self, path: str | Path, int8: bool = True, context_chars: int = 64, threads: int = 1):
        import onnxruntime as ort

        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.vocab = CharVocab.load(path / "vocab.json")
        suffix = ".int8.onnx" if int8 else ".onnx"
        if not (path / f"candidates{suffix}").exists():
            # In a child process, so this one never imports torch.
            code = f"from zhuyin_rescore.charlm_ort import export_onnx; export_onnx({str(path)!r})"
            subprocess.run([sys.executable, "-c", code], check=True)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Worker threads must sleep, not spin, between requests.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        self.ctx_sess = ort.InferenceSession(str(path / f"context{suffix}"), opts, providers=providers)
        self.cand_sess = ort.InferenceSession(str(path / f"candidates{suffix}"), opts, providers=providers)
        self.context_chars = min(context_chars, cfg["max_len"] - 32)
        self.homophones = None
        self._ctx_key: str | None = None
        self._ctx_cache = None

    def context_ids(self, context: str) -> list[int]:
        return context_ids(self.vocab, context, self.context_chars)

    def context_cache(self, context: str):
        if context == self._ctx_key and self._ctx_cache is not None:
            return self._ctx_cache
        ids = np.array([self.context_ids(context)], dtype=np.int64)
        last_h, k, v = self.ctx_sess.run(None, {"ids": ids})
        cache = (k, v, last_h, ids.shape[1])
        self._ctx_key, self._ctx_cache = context, cache
        return cache

    def score_cached(self, context: str, candidates: list[str], cache=None) -> list[float]:
        """Full softmax log P(candidate | context), prefix sharing via a trie."""
        if not candidates:
            return []
        k, v, last_h, p = cache if cache is not None else self.context_cache(context)
        chars, parent, depth, paths = prefix_trie(self.vocab, candidates)
        n = len(chars)
        anc = np.zeros((n, n), dtype=bool)
        for i in range(n):
            j = i
            while j >= 0:
                anc[i, j] = True
                j = parent[j]
        pred = np.asarray(parent, dtype=np.int64) + 1  # row 0: context state, row i + 1: node i
        uniq, row = np.unique(pred, return_inverse=True)
        feeds = {
            "ids": np.asarray(chars, dtype=np.int64),
            "positions": np.asarray(depth, dtype=np.int64) + p,
            "mask": np.concatenate([np.ones((n, p), dtype=bool), anc], axis=1),
            "past_k": k,
            "past_v": v,
            "last_h": last_h,
            "pred_idx": uniq.astype(np.int64),
            "node_pred": row.astype(np.int64),
            "node_char": np.asarray(chars, dtype=np.int64),
        }
        (node_lp,) = self.cand_sess.run(None, feeds)
        return [float(node_lp[path].sum()) for path in paths]
