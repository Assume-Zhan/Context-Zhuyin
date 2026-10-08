"""Baseline CER and Oracle@k for each candidate source and input condition.

Reads outputs/candidates/<split>.<condition>.jsonl and prints a markdown
table. Run this before any LM work: if Oracle@k is close to the 1-best,
rescoring cannot help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.metrics import Accumulator, edit_distance, oracle_errors
from zhuyin_rescore.zhuyin import CONDITIONS


def source_lists(row: dict) -> dict[str, list[str]]:
    return {
        "pool": [c["text"] for c in row["candidates"]],
        "tab": row["tab"] or [row["onebest"]],
        "sub": row["sub"] or [row["onebest"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand-dir", default="outputs/candidates")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 30])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    report = {}
    print(
        "| condition | source | n | valid input | 1-best CER | 1-best SentAcc | "
        + " | ".join(f"Oracle@{k} CER" for k in args.ks)
        + " | gen p50 ms | gen p95 ms |"
    )
    print("|" + " --- |" * (7 + len(args.ks)))
    for cond in args.conditions:
        path = Path(args.cand_dir) / f"{args.split}.{cond}.jsonl"
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        base = Accumulator()
        for row in rows:
            base.add(row["text"], row["onebest"])
        gen_ms = np.array([row["gen_ms"] for row in rows])
        valid = float(np.mean([row["valid_input"] for row in rows]))
        for source in ("pool", "tab", "sub"):
            oracles = {k: Accumulator() for k in args.ks}
            for row in rows:
                hyps = source_lists(row)[source]
                errs = [edit_distance(row["text"], h) for h in hyps]
                for k in args.ks:
                    oracles[k].add_errors(row["text"], min(errs[:k]))
            report[f"{cond}/{source}"] = {
                "n": len(rows),
                "valid_input": valid,
                "base": base.summary(),
                "oracle": {k: acc.summary() for k, acc in oracles.items()},
                "gen_ms_p50": float(np.percentile(gen_ms, 50)),
                "gen_ms_p95": float(np.percentile(gen_ms, 95)),
            }
            cells = " | ".join(f"{oracles[k].cer:.4f}" for k in args.ks)
            print(
                f"| {cond} | {source} | {len(rows)} | {valid:.3f} | {base.cer:.4f} | {base.sent_acc:.3f} | "
                f"{cells} | {np.percentile(gen_ms, 50):.1f} | {np.percentile(gen_ms, 95):.1f} |"
            )
        # Sanity: the pool always contains the 1-best.
        assert all(
            oracle_errors(r["text"], [c["text"] for c in r["candidates"]])
            <= edit_distance(r["text"], r["onebest"])
            for r in rows
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
