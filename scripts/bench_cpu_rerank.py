"""CPU only reranking latency: fp32 vs int8 dynamic quantization, by thread count.

Scores the decoder's k-best (default 10) for dev examples with the context
KV cache already built (as the resident server keeps it), and reports p50
and p95 of one scoring call plus the CPU time it used. Also checks that the
quantized scores pick the same best candidate as fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/lm/qwen2.5-0.5b-zhtw")
    ap.add_argument("--candidates", default="outputs/cand/beam-o4/news/dev.full.jsonl")
    ap.add_argument("--data", default="data/news/dev.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--out", default="outputs/bench/cpu_rerank.json")
    args = ap.parse_args()

    import torch

    from zhuyin_rescore.scorer import LMScorer

    rows = [json.loads(line) for line in open(args.candidates)][: args.requests]
    ctx = {json.loads(line)["id"]: json.loads(line)["context"] for line in open(args.data)}
    cpu_name = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")
    report = {"cpu": cpu_name, "runs": []}
    ref_best = None
    for variant in ("fp32", "int8"):
        scorer = LMScorer(args.model, device="cpu", dtype="float32")
        if variant == "int8":
            scorer.quantize_dynamic_int8()
        for threads in args.threads:
            torch.set_num_threads(threads)
            lat, cpu, best = [], [], []
            for i, r in enumerate(rows):
                texts = [b["text"] for b in r["beam"][: args.k]]
                cache = scorer.context_cache(ctx[r["id"]])
                t0, c0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
                scores = scorer.score_cached(ctx[r["id"]], texts, cache=cache)
                t1, c1 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
                if i >= 3:  # warm up
                    lat.append((t1 - t0) * 1000)
                    cpu.append((c1.ru_utime - c0.ru_utime + c1.ru_stime - c0.ru_stime) * 1000)
                best.append(int(np.argmax(scores)))
            if ref_best is None:
                ref_best = best
            agree = float(np.mean([a == b for a, b in zip(best, ref_best, strict=True)]))
            run = {
                "variant": variant,
                "threads": threads,
                "p50_ms": float(np.percentile(lat, 50)),
                "p95_ms": float(np.percentile(lat, 95)),
                "cpu_ms_per_call": float(np.mean(cpu)),
                "argmax_agreement_vs_fp32_1thread": agree,
            }
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
