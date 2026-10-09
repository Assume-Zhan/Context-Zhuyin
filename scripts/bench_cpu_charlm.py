"""CPU latency of the character LM reranker (full and homophone softmax).

Same protocol as bench_cpu_rerank.py: the decoder 10-best of dev clauses,
context cache built before the timed call (the server keeps it resident),
p50 / p95 wall time and CPU time per call, by thread count.
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
    ap.add_argument("--models", nargs="+", default=["outputs/charlm/tiny", "outputs/charlm/small"])
    ap.add_argument("--candidates", default="outputs/cand/beam-o4/news/dev.full.jsonl")
    ap.add_argument("--data", default="data/news/dev.jsonl")
    ap.add_argument("--condition", default="full")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out", default="outputs/bench/cpu_charlm.json")
    ap.add_argument("--int8", action="store_true", help="also time int8 dynamic quantization")
    ap.add_argument("--variants", nargs="+", default=["full", "homophone"])
    args = ap.parse_args()

    import torch

    from zhuyin_rescore.charlm import CharLMScorer
    from zhuyin_rescore.lexicon import Lexicon, load_entries

    lex = Lexicon(load_entries(), args.condition)
    homophones = lambda syl: [p for p, _ in lex.lookup_span((syl,))]  # noqa: E731
    examples = {json.loads(line)["id"]: json.loads(line) for line in open(args.data)}
    rows = [json.loads(line) for line in open(args.candidates)][: args.requests]
    rows = [r for r in rows if len(r["beam"]) >= 2]
    report = {"runs": []}
    jobs = [(path, False) for path in args.models]
    if args.int8:
        jobs += [(path, True) for path in args.models]
    reference: dict[str, list[int]] = {}
    for path, int8 in jobs:
        scorer = CharLMScorer(path, device="cpu", homophones=homophones)
        params = sum(p.numel() for p in scorer.model.parameters())
        if int8:
            scorer.quantize_dynamic_int8()
        for variant in args.variants:
            for threads in args.threads:
                torch.set_num_threads(threads)
                lat, cpu, best = [], [], []
                for i, r in enumerate(rows):
                    ex = examples[r["id"]]
                    texts = [b["text"] for b in r["beam"][: args.k]]
                    cache = scorer.context_cache(ex["context"])
                    syl = ex[f"zhuyin_{args.condition}"]
                    t0, c0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
                    if variant == "full":
                        scores = scorer.score_cached(ex["context"], texts, cache=cache)
                    else:
                        scores = scorer.score_homophone(ex["context"], texts, syl, cache=cache)
                    t1, c1 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
                    if i >= 5:
                        lat.append((t1 - t0) * 1000)
                        cpu.append((c1.ru_utime - c0.ru_utime + c1.ru_stime - c0.ru_stime) * 1000)
                    best.append(int(np.argmax(scores)))
                ref = reference.setdefault(f"{path}|{variant}", best)
                agree = float(np.mean([a == b for a, b in zip(best, ref, strict=True)]))
                run = {
                    "model": path,
                    "params_m": round(params / 1e6, 2),
                    "int8": int8,
                    "variant": variant,
                    "best_agrees_with_fp32": round(agree, 4),
                    "threads": threads,
                    "p50_ms": round(float(np.percentile(lat, 50)), 2),
                    "p95_ms": round(float(np.percentile(lat, 95)), 2),
                    "cpu_ms_per_call": round(float(np.mean(cpu)), 2),
                }
                report["runs"].append(run)
                print(json.dumps(run), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
