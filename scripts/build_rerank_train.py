"""Clauses for discriminative reranker training, from documents no model saw.

The n-gram was trained on the first --skip-han-chars Han characters of each
eval-filtered training file and the LMs on less, so the documents after that
point are unseen by every model. Sampling from them keeps the training pools
realistic: the decoder makes its usual mistakes and the n-gram feature is not
inflated by memorized sentences.

Same format as the evaluation sets (scripts/build_eval_sets.py); a share of
contexts is blanked because the IME also runs without committed text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from zhuyin_rescore.data import write_jsonl
from zhuyin_rescore.textproc import HAN, extract_examples, g2pw_readings
from zhuyin_rescore.zhuyin import CONDITIONS, derive_condition, normalize

HAN_RUN = re.compile(f"[{HAN}]+")


def unseen_docs(path: Path, skip_han_chars: float) -> list[dict]:
    han, out = 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if han < skip_han_chars:
                han += sum(len(r) for r in HAN_RUN.findall(doc["text"]))
                continue
            out.append(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="data/corpus/train")
    ap.add_argument("--sources", nargs="+", default=["news", "wiki", "ptt", "web"])
    ap.add_argument("--skip-han-chars", type=float, default=150e6)
    ap.add_argument("--per-source", type=int, default=15000, help="train clauses per source")
    ap.add_argument("--dev-per-source", type=int, default=750)
    ap.add_argument("--max-per-doc", type=int, default=2)
    ap.add_argument("--blank-context", type=float, default=0.15)
    ap.add_argument("--out-dir", default="data/rerank_train")
    ap.add_argument("--g2pw-dir", default="/cache/huggingface/g2pw/G2PWModel/")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "dev": []}
    seen: set[str] = set()
    for source in args.sources:
        docs = unseen_docs(Path(args.train_dir) / f"{source}.jsonl", args.skip_han_chars)
        rng.shuffle(docs)
        want = {"train": args.per_source, "dev": args.dev_per_source}
        got = {"train": 0, "dev": 0}
        for doc in docs:
            split = "dev" if hashlib.md5(doc["id"].encode()).digest()[0] % 20 == 0 else "train"
            if got[split] >= want[split]:
                if all(got[s] >= want[s] for s in got):
                    break
                continue
            exs = [(c, ctx) for c, ctx in extract_examples(doc["text"], 5, 20, 80) if c not in seen]
            rng.shuffle(exs)
            for k, (clause, ctx) in enumerate(exs[: args.max_per_doc]):
                seen.add(clause)
                if rng.random() < args.blank_context:
                    ctx = ""
                row = {"id": f"{doc['id']}-{k:02d}", "doc": doc["id"], "domain": source}
                splits[split].append({**row, "context": ctx, "text": clause})
                got[split] += 1
        print(f"{source}: {len(docs)} unseen docs, sampled {got}", flush=True)

    for split, rows in splits.items():
        readings = g2pw_readings([r["text"] for r in rows], args.g2pw_dir, workers=args.workers)
        kept = []
        for r, reading in zip(rows, readings, strict=True):
            if len(reading) != len(r["text"]) or any(x is None for x in reading):
                continue
            try:
                full = [normalize(x) for x in reading]
            except ValueError:
                continue
            for cond in CONDITIONS:
                r[f"zhuyin_{cond}"] = derive_condition(full, cond)
            kept.append(r)
        n = write_jsonl(Path(args.out_dir) / f"{split}.jsonl", kept)
        print(f"{split}: {n} clauses", flush=True)


if __name__ == "__main__":
    main()
