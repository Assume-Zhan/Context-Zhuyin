"""Train the character n-gram used by the first pass beam decoder.

Reads the eval-filtered training documents (data/corpus/train/<source>.jsonl)
and counts n-grams over pure Han runs, so punctuation, digits and Latin text
act as sentence boundaries. Lower order models are the first k count tables
of this model (CharNgram.load(path, order=k)), so one run serves the bigram,
trigram and 4-gram comparisons.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from zhuyin_rescore.ngram import CharNgram
from zhuyin_rescore.textproc import HAN

HAN_RUN = re.compile(f"[{HAN}]+")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="data/corpus/train")
    ap.add_argument("--sources", nargs="+", default=["news", "wiki", "ptt", "web"])
    ap.add_argument("--max-chars-per-source", type=float, default=1.5e8)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2, help="prune 3-grams and up below this count")
    ap.add_argument("--out", default="outputs/ngram/zhtw-o4")
    args = ap.parse_args()

    t0 = time.time()
    runs: list[str] = []
    stats = {}
    for source in args.sources:
        chars = 0
        with open(Path(args.train_dir) / f"{source}.jsonl", encoding="utf-8") as f:
            for line in f:
                for run in HAN_RUN.findall(json.loads(line)["text"]):
                    runs.append(run)
                    chars += len(run)
                if chars >= args.max_chars_per_source:
                    break
        stats[source] = chars
        print(f"{source}: {chars / 1e6:.0f}M Han chars", flush=True)
    min_count = {k: args.min_count for k in range(3, args.order + 1)}
    model = CharNgram.train(runs, order=args.order, min_count=min_count)
    del runs
    model.save(args.out)
    sizes = {f"order{k + 1}": int(len(model.keys[k])) for k in range(model.order)}
    meta = {"sources": stats, "vocab": len(model.chars), "ngrams": sizes, "min_count": args.min_count}
    (Path(args.out) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
