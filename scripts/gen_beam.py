"""Decode datasets with the lattice beam decoder (char n-gram first pass).

Writes outputs/cand/beam-o<order>/<domain>/<split>.<condition>.jsonl with
the k best sentences, their decoder scores and features
(ngram, prior, edges, mismatch) and the decode time.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from zhuyin_rescore.beam import BeamDecoder, Weights
from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.lexicon import Lexicon, load_entries
from zhuyin_rescore.ngram import CharNgram
from zhuyin_rescore.zhuyin import CONDITIONS

_DEC: BeamDecoder | None = None
_K = 30


def _init(ngram: str, order: int, condition: str, beam: int, k: int, weights: Weights) -> None:
    global _DEC, _K
    lexicon = Lexicon(load_entries(), condition)
    _DEC = BeamDecoder(lexicon, CharNgram.load(ngram, order=order), beam=beam, weights=weights)
    _K = k


def _work(item: tuple[dict, str]) -> dict:
    ex, condition = item
    out = _DEC.decode(ex[f"zhuyin_{condition}"], history=ex["context"], k=_K)
    return {
        "id": ex["id"],
        "condition": condition,
        "text": ex["text"],
        "beam": [{"text": d.text, "score": d.score, "feats": d.feats} for d in out],
        "gen_ms": _DEC.last_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dirs", nargs="+", default=["data/news", "data/eval/web", "data/eval/ptt"])
    ap.add_argument("--ngram", default="outputs/ngram/zhtw-o4")
    ap.add_argument("--orders", nargs="+", type=int, default=[4])
    ap.add_argument("--out-root", default="outputs/cand")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--w-prior", type=float, default=0.0)
    ap.add_argument("--w-edge", type=float, default=0.0)
    ap.add_argument("--w-mismatch", type=float, default=-3.0)
    ap.add_argument("--tag", default=None, help="output subdir, default beam-o<order>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    weights = Weights(prior=args.w_prior, edge=args.w_edge, mismatch=args.w_mismatch)

    for order in args.orders:
        tag = args.tag or f"beam-o{order}"
        for data_dir in args.data_dirs:
            domain = Path(data_dir).name
            for split in args.splits:
                examples = list(read_jsonl(Path(data_dir) / f"{split}.jsonl"))[: args.limit]
                for cond in args.conditions:
                    t0 = time.time()
                    init_args = (args.ngram, order, cond, args.beam, args.k, weights)
                    with mp.Pool(args.workers, initializer=_init, initargs=init_args) as pool:
                        rows = pool.map(_work, [(ex, cond) for ex in examples], chunksize=8)
                    out = Path(args.out_root) / tag / domain / f"{split}.{cond}.jsonl"
                    write_jsonl(out, rows)
                    ms = np.array([r["gen_ms"] for r in rows])
                    print(
                        f"{tag} {domain} {split} {cond}: {len(rows)} rows, decode p50 "
                        f"{np.percentile(ms, 50):.1f} ms p95 {np.percentile(ms, 95):.1f} ms, "
                        f"{time.time() - t0:.0f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
