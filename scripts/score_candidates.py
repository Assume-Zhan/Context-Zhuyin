"""Score candidate pools with a causal LM, with and without context.

Reads outputs/candidates/<split>.<condition>.jsonl and writes
outputs/scores/<tag>/<split>.<condition>.jsonl with one LM log-prob per pool
candidate ("lm") and the no context ablation ("lm_noctx").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.scorer import LMScorer
from zhuyin_rescore.zhuyin import CONDITIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tag", default=None, help="output subdir, default derived from model and dtype")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-context-tokens", type=int, default=64)
    ap.add_argument("--cand-dir", default="outputs/candidates")
    ap.add_argument("--out-dir", default="outputs/scores")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--k", type=int, default=30, help="score at most the first k pool candidates")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model.split('/')[-1]}-{args.dtype}"

    scorer = LMScorer(args.model, dtype=args.dtype, max_context_tokens=args.max_context_tokens)
    out_dir = Path(args.out_dir) / tag
    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "max_context_tokens": args.max_context_tokens,
        "k": args.k,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    for split in args.splits:
        for cond in args.conditions:
            rows = list(read_jsonl(Path(args.cand_dir) / f"{split}.{cond}.jsonl"))[: args.limit]
            out, wall = [], []
            t0 = time.time()
            for row in rows:
                texts = [c["text"] for c in row["candidates"][: args.k]]
                t = time.perf_counter()
                lm = scorer.score_cached(row["context"], texts)
                wall.append((time.perf_counter() - t) * 1000)
                lm_noctx = scorer.score_cached("", texts)
                out.append({"id": row["id"], "lm": lm, "lm_noctx": lm_noctx})
            write_jsonl(out_dir / f"{split}.{cond}.jsonl", out)
            print(
                f"{tag} {split} {cond}: {len(out)} rows in {time.time() - t0:.1f}s, "
                f"call p50 {np.percentile(wall, 50):.1f} ms p95 {np.percentile(wall, 95):.1f} ms "
                f"(wall clock incl. tokenization, {meta['gpu']})"
            )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
