"""Build multi-domain dev and test sets from heldout documents.

Domains: web (Taiwan web pages, 2025 crawl), ptt (posts from 2022 on), wiki
(newest zh-TW Wikipedia articles, 2026 dump) and cv (Common Voice zh-TW
sentence list, standalone sentences without context). The phase 0 news sets
in data/news are kept unchanged.

Same example format as scripts/build_dataset.py. Clauses that occur in more
than one heldout document of a source are treated as boilerplate and
dropped; clause text is unique across all domains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from fetch_corpora import SOURCES
from zhuyin_rescore.data import read_jsonl, write_jsonl
from zhuyin_rescore.textproc import PURE_HAN, extract_examples, g2pw_readings
from zhuyin_rescore.zhuyin import CONDITIONS, derive_condition, normalize


def doc_examples(source: str, docs: list[dict], args, rng: random.Random, seen: set[str]) -> list[dict]:
    per_doc = [
        list(extract_examples(d["text"], args.min_len, args.max_len, args.context_chars)) for d in docs
    ]
    freq = Counter(c for exs in per_doc for c in {c for c, _ in exs})
    out = []
    order = list(range(len(docs)))
    rng.shuffle(order)
    for di in order:
        exs = [(c, ctx) for c, ctx in per_doc[di] if freq[c] == 1 and c not in seen]
        rng.shuffle(exs)
        for k, (clause, ctx) in enumerate(exs[: args.max_per_doc]):
            seen.add(clause)
            doc_id = docs[di]["id"]
            out.append({"id": f"{doc_id}-{k:02d}", "doc": doc_id, "context": ctx, "text": clause})
    return out


def cv_examples(args, rng: random.Random, seen: set[str]) -> list[dict]:
    repo, files = SOURCES["cv"]
    path = hf_hub_download(repo, files[0], repo_type="dataset")
    out = []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rng.shuffle(rows)
    for r in rows:
        text = r["sentence"].strip().rstrip("。！？!?.，,、")
        if args.min_len <= len(text) <= args.max_len and PURE_HAN.match(text) and text not in seen:
            seen.add(text)
            sid = r["sentence_id"]
            out.append({"id": f"cv-{sid[:12]}", "doc": sid, "context": "", "text": text})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/corpus/raw")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--domains", nargs="+", default=["web", "ptt", "wiki", "cv"])
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-per-doc", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--context-chars", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g2pw-dir", default="/cache/huggingface/g2pw/G2PWModel/")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Never reuse a phase 0 news clause.
    seen = {ex["text"] for split in ("dev", "test") for ex in read_jsonl(f"data/news/{split}.jsonl")}
    meta = {"seed": args.seed, "g2p": "g2pW", "domains": {}}
    for domain in args.domains:
        if domain == "cv":
            examples = cv_examples(args, rng, seen)
        else:
            docs = list(read_jsonl(Path(args.raw_dir) / f"{domain}.heldout.jsonl"))
            examples = doc_examples(domain, docs, args, rng, seen)
        # Split by document: hash the doc id so both splits draw from all docs.
        need = int((args.n_dev + args.n_test) * 1.1)
        examples = examples[: need * 2]
        t0 = time.time()
        readings = g2pw_readings([ex["text"] for ex in examples], args.g2pw_dir, workers=args.workers)
        splits: dict[str, list[dict]] = {"dev": [], "test": []}
        dropped = 0
        for ex, reading in zip(examples, readings, strict=True):
            if len(reading) != len(ex["text"]) or any(r is None for r in reading):
                dropped += 1
                continue
            try:
                full = [normalize(r) for r in reading]
            except ValueError:
                dropped += 1
                continue
            for cond in CONDITIONS:
                ex[f"zhuyin_{cond}"] = derive_condition(full, cond)
            ex["domain"] = domain
            # Not crc32: the heldout selection used crc32 of the doc id, and
            # crc32 is affine, so any crc32 based split would correlate with it.
            split = "dev" if hashlib.md5(str(ex["doc"]).encode()).digest()[0] % 2 == 0 else "test"
            target = args.n_dev if split == "dev" else args.n_test
            if len(splits[split]) < target:
                splits[split].append(ex)
        for split, rows in splits.items():
            write_jsonl(Path(args.out_dir) / domain / f"{split}.jsonl", rows)
        meta["domains"][domain] = {
            split: {
                "examples": len(rows),
                "docs": len({r["doc"] for r in rows}),
                "chars": sum(len(r["text"]) for r in rows),
                "with_context": sum(bool(r["context"]) for r in rows),
            }
            for split, rows in splits.items()
        } | {"dropped_g2p": dropped}
        print(f"{domain}: {json.dumps(meta['domains'][domain])} g2pW {time.time() - t0:.0f}s", flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
