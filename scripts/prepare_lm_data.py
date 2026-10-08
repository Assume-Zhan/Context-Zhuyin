"""Tokenize the eval-filtered zh-TW corpus for continued LM pretraining.

Documents from each source are tokenized with the model tokenizer and
joined with <|endoftext|>, the same document start token the scorer
prepends. The first documents of each source (up to --val-tokens) form a
validation slice. Output: train.bin and val.bin (uint32 token ids) plus
meta.json under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

_TOK = None


def _init(model: str) -> None:
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(model)


def _encode(line: str) -> list[int]:
    return _TOK(json.loads(line)["text"], add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="data/corpus/train")
    ap.add_argument("--sources", nargs="+", default=["news", "wiki", "ptt", "web"])
    ap.add_argument("--tokens-per-source", type=float, default=75e6)
    ap.add_argument("--val-tokens", type=float, default=5e5, help="per source")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out-dir", default="data/lm/zhtw-v1")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(args.model).convert_tokens_to_ids("<|endoftext|>")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_parts, val_parts, stats = [], [], {}
    with mp.Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for source in args.sources:
            val, train = [], []
            n_val = n_train = docs = 0
            with open(Path(args.train_dir) / f"{source}.jsonl", encoding="utf-8") as f:
                for ids in pool.imap(_encode, f, chunksize=64):
                    ids = [eos] + ids
                    docs += 1
                    if n_val < args.val_tokens:
                        val.append(np.asarray(ids, dtype=np.uint32))
                        n_val += len(ids)
                    else:
                        train.append(np.asarray(ids, dtype=np.uint32))
                        n_train += len(ids)
                    if n_train >= args.tokens_per_source:
                        break
            train_parts.append(np.concatenate(train))
            val_parts.append(np.concatenate(val))
            stats[source] = {"docs": docs, "train_tokens": n_train, "val_tokens": n_val}
            print(f"{source}: {stats[source]}", flush=True)
    train_arr = np.concatenate(train_parts)
    val_arr = np.concatenate(val_parts)
    train_arr.tofile(out_dir / "train.bin")
    val_arr.tofile(out_dir / "val.bin")
    meta = {"model": args.model, "eos": eos, "sources": stats}
    meta |= {"train_tokens": int(len(train_arr)), "val_tokens": int(len(val_arr))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
