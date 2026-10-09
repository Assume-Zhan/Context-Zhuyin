"""Candidate pools for reranker training, composed as at evaluation time.

- full (typed with tones): the decoder's 10-best, then libchewing 0.14's
  whole sentence n-best, deduplicated, at most 20 (pool b4+c14tab@20);
- notone: the decoder's 30-best (pool beam-o4@30).

Inputs come from gen_beam.py and gen_candidates.py run on data/rerank_train
with the evaluation's decoder weights. Each row carries the pool, the char
4-gram score of every candidate given the same context, and its edit
distance to the correct sentence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zhuyin_rescore.candidates import is_han
from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.metrics import edit_distance
from zhuyin_rescore.ngram import CharNgram

KIND = {"full": "toned+chewing", "notone": "open"}


def compose(seq: list[str], n: int, k: int) -> list[str]:
    out, seen = [], set()
    for t in seq:
        if t not in seen and len(t) == n and is_han(t):
            seen.add(t)
            out.append(t)
        if len(out) == k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/rerank_train")
    ap.add_argument("--cand-root", default="outputs/cand")
    ap.add_argument("--ngram", default="outputs/ngram/zhtw-o4")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--k-toned", type=int, default=20)
    ap.add_argument("--k-open", type=int, default=30)
    args = ap.parse_args()
    ngram = CharNgram.load(args.ngram)
    domain = Path(args.data_dir).name
    cand = Path(args.cand_root)

    for split in args.splits:
        examples = {ex["id"]: ex for ex in read_jsonl(Path(args.data_dir) / f"{split}.jsonl")}
        rows = []
        for cond in ("full", "notone"):
            path = cand / "beam-o4" / domain / f"{split}.{cond}.jsonl"
            beam = {r["id"]: [b["text"] for b in r["beam"]] for r in read_jsonl(path)}
            c14 = {}
            if cond == "full":
                c14 = {
                    r["id"]: [r["onebest"]] + r["tab"][1:]
                    for r in read_jsonl(cand / "chewing-0.14" / domain / f"{split}.{cond}.jsonl")
                }
            for rid, ex in examples.items():
                n = len(ex["text"])
                if cond == "full":
                    texts = compose(beam[rid][:10] + c14.get(rid, []), n, args.k_toned)
                else:
                    texts = compose(beam[rid], n, args.k_open)
                if not texts:
                    continue
                ng = ngram.score_batch(texts, history=ex["context"]).tolist()
                rows.append(
                    {
                        "id": rid,
                        "domain": ex["domain"],
                        "kind": KIND[cond],
                        "context": ex["context"],
                        "text": ex["text"],
                        "texts": texts,
                        "ngram": [round(x, 4) for x in ng],
                        "errors": [edit_distance(ex["text"], t) for t in texts],
                    }
                )
        n = write_jsonl(Path(args.data_dir) / f"pools.{split}.jsonl", rows)
        for kind in KIND.values():
            sel = [r for r in rows if r["kind"] == kind]
            gold = np.mean([r["text"] in r["texts"] for r in sel])
            first = np.mean([r["errors"][0] == 0 for r in sel])
            size = np.mean([len(r["texts"]) for r in sel])
            print(
                f"{split} {kind}: {len(sel)} pools, size {size:.1f}, 1-best correct {first:.3f}, "
                f"correct in pool {gold:.3f}",
                flush=True,
            )
        print(f"{split}: {n} rows", flush=True)


if __name__ == "__main__":
    main()
