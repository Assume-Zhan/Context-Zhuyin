"""Run libchewing over a dataset split and store candidate pools.

Writes outputs/candidates/<split>.<condition>.jsonl, one line per example
with the libchewing 1-best, the merged candidate pool, and the raw Tab and
substitution lists (for comparing candidate sources).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from zhuyin_rescore.candidates import CandidateGenerator
from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.zhuyin import CONDITIONS

_GEN: CandidateGenerator | None = None


def _init(condition: str, pool_size: int, per_list: int, charset: str | None) -> None:
    global _GEN
    _GEN = CandidateGenerator(condition, pool_size=pool_size, per_list=per_list, charset=charset)


def _work(item: tuple[dict, str]) -> dict:
    ex, condition = item
    cset = _GEN.generate(ex[f"zhuyin_{condition}"])
    return {
        "id": ex["id"],
        "condition": condition,
        "text": ex["text"],
        "context": ex["context"],
        **cset.to_json(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/news")
    ap.add_argument("--out-dir", default="outputs/candidates")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--pool-size", type=int, default=30)
    ap.add_argument("--per-list", type=int, default=5)
    ap.add_argument("--charset", default="cp950", help="filter alternatives by charset; 'none' disables")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    charset = None if args.charset == "none" else args.charset

    for split in args.splits:
        examples = list(read_jsonl(Path(args.data_dir) / f"{split}.jsonl"))[: args.limit]
        for cond in args.conditions:
            t0 = time.time()
            with mp.Pool(
                args.workers, initializer=_init, initargs=(cond, args.pool_size, args.per_list, charset)
            ) as pool:
                rows = pool.map(_work, [(ex, cond) for ex in examples], chunksize=16)
            out = Path(args.out_dir) / f"{split}.{cond}.jsonl"
            write_jsonl(out, rows)
            print(f"{split} {cond}: {len(rows)} examples -> {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
