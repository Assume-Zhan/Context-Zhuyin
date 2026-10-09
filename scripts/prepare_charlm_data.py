"""Encode the eval-filtered zh-TW training documents for the character LM.

Same documents as the Qwen continued pretraining (data/corpus/train), so the
dedupe against every evaluation clause still holds. Vocabulary: the n-gram
characters plus clause punctuation and special tokens (CharVocab). Each
document starts with BOS. Output: train.bin and val.bin (uint16 ids),
vocab.json and meta.json under --out-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zhuyin_rescore.charlm import BOS, CharVocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="data/corpus/train")
    ap.add_argument("--sources", nargs="+", default=["news", "wiki", "ptt", "web"])
    ap.add_argument("--ngram", default="outputs/ngram/zhtw-o4", help="vocabulary source")
    ap.add_argument("--tokens-per-source", type=float, default=75e6)
    ap.add_argument("--val-tokens", type=float, default=5e5, help="per source")
    ap.add_argument("--out-dir", default="data/charlm/zhtw-v1")
    args = ap.parse_args()

    vocab = CharVocab([str(c) for c in np.load(Path(args.ngram) / "chars.npy")])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_parts, val_parts, stats = [], [], {}
    for source in args.sources:
        n_val = n_train = docs = unk = 0
        with open(Path(args.train_dir) / f"{source}.jsonl", encoding="utf-8") as f:
            for line in f:
                ids = np.asarray([BOS] + vocab.encode(json.loads(line)["text"]), dtype=np.uint16)
                docs += 1
                unk += int((ids == 2).sum())
                if n_val < args.val_tokens:
                    val_parts.append(ids)
                    n_val += len(ids)
                else:
                    train_parts.append(ids)
                    n_train += len(ids)
                if n_train >= args.tokens_per_source:
                    break
        stats[source] = {"docs": docs, "train_tokens": n_train, "val_tokens": n_val}
        stats[source]["unk_rate"] = unk / max(1, n_val + n_train)
        print(source, stats[source], flush=True)
    np.concatenate(train_parts).tofile(out / "train.bin")
    np.concatenate(val_parts).tofile(out / "val.bin")
    vocab.save(out / "vocab.json")
    meta = {"vocab_size": len(vocab), "sources": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
