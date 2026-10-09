"""Convert a trained char n-gram (counts) to the compact form the IME loads.

Precomputes the stupid backoff score of every n-gram, quantizes it through a
codebook per order (--bits 16: two bytes per entry, decoding unchanged;
--bits 8: one byte), optionally drops n-grams seen fewer than --min-count
times, and stores keys of order <= 2 in uint32.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from zhuyin_rescore.ngram import CharNgram


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="outputs/ngram/zhtw-o4")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-count",
        nargs="*",
        default=[],
        help="order=count pairs, e.g. 3=3 4=3 drops 3- and 4-grams seen fewer than 3 times",
    )
    ap.add_argument(
        "--bits", type=int, default=16, help="16: decodes as the counts model; 8: smaller, slightly worse"
    )
    args = ap.parse_args()
    min_count = {int(k): int(v) for k, v in (x.split("=") for x in args.min_count)}

    t0 = time.time()
    model = CharNgram.load(args.src)
    small = model.compact(min_count=min_count, bits=args.bits)
    out = Path(args.out)
    small.save(out)
    meta = json.loads((Path(args.src) / "meta.json").read_text())
    meta.update(
        {
            "compact_from": args.src,
            "compact_min_count": min_count,
            "compact_bits": args.bits,
            "ngrams": {f"order{k + 1}": len(small.keys[k]) for k in range(small.order)},
        }
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(f.stat().st_size for f in out.glob("*.npy"))
    print(f"{out}: {meta['ngrams']}, {size / 2**20:.0f} MiB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
