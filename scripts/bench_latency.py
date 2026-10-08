"""Per request rescoring latency with a resident context KV cache.

The context cache is built once per example outside the timed region, as a
resident server would do when text is committed; the timed region is one
batched candidate scoring call. Reports p50 and p95 of CUDA event time and
wall clock, plus VRAM. Always label results with the GPU they came from.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.scorer import LMScorer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--candidates", default="outputs/candidates/dev.full.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/bench")
    args = ap.parse_args()

    rows = list(read_jsonl(args.candidates))
    scorer = LMScorer(args.model, dtype=args.dtype)
    torch.cuda.reset_peak_memory_stats()

    gpu_ms, wall_ms, ctx_ms = [], [], []
    total = args.warmup + args.requests
    for i in range(total):
        row = rows[i % len(rows)]
        texts = [c["text"] for c in row["candidates"][: args.k]]
        t = time.perf_counter()
        cache = scorer.context_cache(row["context"])
        ctx_done = torch.cuda.Event(blocking=True)
        ctx_done.record()
        ctx_done.synchronize()
        c_ms = (time.perf_counter() - t) * 1000
        start = torch.cuda.Event(enable_timing=True, blocking=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        t = time.perf_counter()
        start.record()
        scorer.score_cached(row["context"], texts, cache=cache)
        end.record()
        end.synchronize()
        w_ms = (time.perf_counter() - t) * 1000
        if i >= args.warmup:
            gpu_ms.append(start.elapsed_time(end))
            wall_ms.append(w_ms)
            ctx_ms.append(c_ms)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = torch.cuda.get_device_name(0)
    summary = {
        "config": f"{args.model.split('/')[-1]}-{args.dtype}-k{args.k}",
        "gpu_name": gpu_name,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "torch": torch.__version__,
        "requests": len(wall_ms),
        "latency_p50_ms": float(np.percentile(wall_ms, 50)),
        "latency_p95_ms": float(np.percentile(wall_ms, 95)),
        "gpu_event_p50_ms": float(np.percentile(gpu_ms, 50)),
        "gpu_event_p95_ms": float(np.percentile(gpu_ms, 95)),
        "context_cache_p50_ms": float(np.percentile(ctx_ms, 50)),
        "vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "nvml_used_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    (out / f"{stamp}-{summary['config']}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
