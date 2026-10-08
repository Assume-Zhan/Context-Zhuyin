"""Build dev and test sets of zhuyin input / Traditional Chinese answer pairs.

Source: Taiwan news articles (HF dataset liswei/news-collection-zhtw, already
zh-TW, so no OpenCC conversion). Each example is one clause of 5 to 20 Han
characters, which is roughly what an IME user types before committing. The
article text right before the clause is kept as the committed context.

Readings come from g2pW. Examples where any character has no reading are
dropped. Dev and test are split by article to avoid context leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from zhuyin_rescore.zhuyin import CONDITIONS, derive_condition, normalize

HAN = "㐀-䶿一-鿿"
CLAUSE_DELIMS = re.compile(r"[，。！？；：、,.!?;:\n\r\t「」『』（）()《》〈〉【】\[\]\"'“”‘’…—～~|/／\s]+")
PURE_HAN = re.compile(f"^[{HAN}]+$")


def iter_clauses(text: str):
    """Yield (start, clause) for every delimiter separated span."""
    pos = 0
    for m in CLAUSE_DELIMS.finditer(text):
        if m.start() > pos:
            yield pos, text[pos : m.start()]
        pos = m.end()
    if pos < len(text):
        yield pos, text[pos:]


def extract_examples(text: str, min_len: int, max_len: int, context_chars: int):
    for start, clause in iter_clauses(text):
        if min_len <= len(clause) <= max_len and PURE_HAN.match(clause):
            context = text[max(0, start - context_chars) : start].lstrip()
            yield clause, context


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="liswei/news-collection-zhtw")
    ap.add_argument("--shard", default="data/train-00002-of-00005.parquet")
    ap.add_argument("--out-dir", default="data/news")
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--max-per-doc", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--context-chars", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g2pw-dir", default="/cache/huggingface/g2pw/G2PWModel/")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    path = hf_hub_download(args.repo, args.shard, repo_type="dataset")
    df = pd.read_parquet(path, columns=["title", "text", "category"])
    print(f"loaded {len(df)} articles from {args.shard}")

    rng = random.Random(args.seed)
    order = list(range(len(df)))
    rng.shuffle(order)

    # Oversample to leave room for examples dropped by g2pW.
    targets = {"dev": int(args.n_dev * 1.1), "test": int(args.n_test * 1.1)}
    picked: dict[str, list[dict]] = {"dev": [], "test": []}
    seen_text: set[str] = set()
    split_cycle = ["dev", "test"]
    turn = 0
    for doc_idx in order:
        if all(len(picked[s]) >= targets[s] for s in picked):
            break
        split = split_cycle[turn % 2]
        if len(picked[split]) >= targets[split]:
            split = split_cycle[(turn + 1) % 2]
        row = df.iloc[doc_idx]
        cands = [
            (clause, ctx)
            for clause, ctx in extract_examples(row["text"], args.min_len, args.max_len, args.context_chars)
            if clause not in seen_text
        ]
        if not cands:
            continue
        rng.shuffle(cands)
        for k, (clause, ctx) in enumerate(cands[: args.max_per_doc]):
            seen_text.add(clause)
            picked[split].append(
                {
                    "id": f"news-{doc_idx:06d}-{k:02d}",
                    "doc": int(doc_idx),
                    "category": row["category"],
                    "context": ctx,
                    "text": clause,
                }
            )
        turn += 1

    from g2pw import G2PWConverter

    conv = G2PWConverter(
        model_dir=args.g2pw_dir,
        style="bopomofo",
        enable_non_tradional_chinese=False,
        batch_size=args.batch_size,
        turnoff_tqdm=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"repo": args.repo, "shard": args.shard, "seed": args.seed, "g2p": "g2pW", "splits": {}}
    for split, n_target in (("dev", args.n_dev), ("test", args.n_test)):
        examples = picked[split]
        t0 = time.time()
        readings = conv([ex["text"] for ex in examples])
        kept, dropped = [], 0
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
            kept.append(ex)
        kept = kept[:n_target]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in kept:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        meta["splits"][split] = {
            "examples": len(kept),
            "docs": len({ex["doc"] for ex in kept}),
            "dropped_g2p": dropped,
            "chars": sum(len(ex["text"]) for ex in kept),
        }
        print(f"{split}: kept {len(kept)} dropped {dropped} g2pW {time.time() - t0:.1f}s")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
