"""Score the union of all candidate sources once per example.

For every example, the candidate lists of all sources (libchewing builds,
beam decoders) are merged into one deduplicated list, which is scored with
the causal LM (with and without context) and the char 4-gram. Pools for
evaluation are composed later from the per-source lists, so each sentence is
scored once however many pool policies are compared.

Writes outputs/scores/<tag>/<domain>/<split>.<condition>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from zhuyin_rescore.candidates import is_han
from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.ngram import CharNgram
from zhuyin_rescore.scorer import LMScorer
from zhuyin_rescore.zhuyin import CONDITIONS


def source_texts(row: dict, k: int) -> list[str]:
    if "beam" in row:
        return [b["text"] for b in row["beam"][:k]]
    return [c["text"] for c in row["candidates"][:k]]


def score_chunked(scorer: LMScorer, context: str, texts: list[str], chunk: int, cache=None) -> list[float]:
    """Score in fixed size chunks that share one context cache."""
    if not texts:
        return []
    cache = cache or scorer.context_cache(context)
    out: list[float] = []
    for i in range(0, len(texts), chunk):
        out += scorer.score_cached(context, texts[i : i + chunk], cache=cache)
    return out


def load_union(cand_root: Path, sources: list[str], domain: str, split: str, cond: str, k: int):
    """Return examples with their merged candidate list, keyed by id."""
    merged: dict[str, dict] = {}
    for src in sources:
        path = cand_root / src / domain / f"{split}.{cond}.jsonl"
        for row in read_jsonl(path):
            ex = merged.setdefault(row["id"], {"id": row["id"], "text": row["text"], "texts": []})
            if "context" in row:
                ex["context"] = row["context"]
            n = len(row.get("syllables") or row["text"])
            for t in source_texts(row, k):
                if t not in ex["texts"] and len(t) == n and is_han(t):
                    ex["texts"].append(t)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--cand-root", default="outputs/cand")
    default_sources = ["chewing-0.13", "chewing-0.14", "beam-o4", "beam-o3", "beam-o2"]
    ap.add_argument("--sources", nargs="+", default=default_sources)
    ap.add_argument("--data-root", default="data", help="examples, for the context field")
    ap.add_argument("--domains", nargs="+", default=["news", "web", "ptt", "wiki", "cv"])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--ngram", default="outputs/ngram/zhtw-o4")
    ap.add_argument("--out-root", default="outputs/scores")
    ap.add_argument("--no-lm", action="store_true", help="only compute the n-gram feature")
    ap.add_argument("--chunk", type=int, default=32, help="candidates per forward, bounds GPU memory")
    ap.add_argument("--scorer", default="qwen", choices=["qwen", "charlm", "charlm-onnx"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--int8", action="store_true", help="charlm on the CPU / charlm-onnx: int8 quantization")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = ap.parse_args()
    tag = args.tag or f"{args.model.rstrip('/').split('/')[-1]}-{args.dtype}"

    homophone_fns: dict[str, object] = {}
    if args.no_lm:
        scorer = None
    elif args.scorer == "charlm":
        from zhuyin_rescore.charlm import CharLMScorer
        from zhuyin_rescore.lexicon import Lexicon, load_entries

        torch.set_num_threads(args.threads)
        scorer = CharLMScorer(args.model, device=args.device)
        if args.int8:
            scorer.quantize_dynamic_int8()
        entries = load_entries()
        for cond in args.conditions:
            lex = Lexicon(entries, cond)
            homophone_fns[cond] = lambda syl, lex=lex: [p for p, _ in lex.lookup_span((syl,))]
    elif args.scorer == "charlm-onnx":
        from zhuyin_rescore.charlm_ort import OrtCharLMScorer

        scorer = OrtCharLMScorer(args.model, int8=args.int8, threads=args.threads)
    else:
        scorer = LMScorer(args.model, dtype=args.dtype, device=args.device)
    empty_cache = scorer.context_cache("") if scorer is not None else None
    ngram = CharNgram.load(args.ngram)
    for domain in args.domains:
        data_dir = Path(args.data_root) / ("news" if domain == "news" else f"eval/{domain}")
        for split in args.splits:
            examples = {ex["id"]: ex for ex in read_jsonl(data_dir / f"{split}.jsonl")}
            contexts = {i: ex["context"] for i, ex in examples.items()}
            for cond in args.conditions:
                t0 = time.time()
                union = load_union(Path(args.cand_root), args.sources, domain, split, cond, args.k)
                out = []
                for ex in union.values():
                    ctx = contexts[ex["id"]]
                    row = {"id": ex["id"], "texts": ex["texts"]}
                    row["ngram"] = ngram.score_batch(ex["texts"], history=ctx).tolist()
                    if scorer is not None:
                        row["lm"] = score_chunked(scorer, ctx, ex["texts"], args.chunk)
                        row["lm_noctx"] = score_chunked(scorer, "", ex["texts"], args.chunk, empty_cache)
                    if cond in homophone_fns and ex["texts"]:
                        scorer.homophones = homophone_fns[cond]
                        syl = examples[ex["id"]][f"zhuyin_{cond}"]
                        row["lm_homo"] = []
                        for i in range(0, len(ex["texts"]), args.chunk):
                            part = ex["texts"][i : i + args.chunk]
                            row["lm_homo"] += scorer.score_homophone(ctx, part, syl)
                    elif cond in homophone_fns:
                        row["lm_homo"] = []
                    out.append(row)
                write_jsonl(Path(args.out_root) / tag / domain / f"{split}.{cond}.jsonl", out)
                sizes = np.array([len(r["texts"]) for r in out])
                print(
                    f"{tag} {domain} {split} {cond}: {len(out)} rows, union size mean {sizes.mean():.1f} "
                    f"max {sizes.max()}, {time.time() - t0:.0f}s",
                    flush=True,
                )
    meta = {"model": args.model, "dtype": args.dtype, "sources": args.sources, "k": args.k}
    meta["scorer"] = args.scorer
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    (Path(args.out_root) / tag / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
