"""Filter training documents against every evaluation clause.

A training document is dropped if any of its clauses equals a clause from
the evaluation sets (data/eval/*, data/news) or any Common Voice sentence.
News syndication and text reuse across sites would otherwise leak test
sentences into the n-gram and the fine-tuned LM.

Reads data/corpus/raw/<source>.train.jsonl, writes
data/corpus/train/<source>.jsonl and a stats file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
from pathlib import Path

from huggingface_hub import hf_hub_download

from fetch_corpora import SOURCES
from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.textproc import PURE_HAN, iter_clauses

_EVAL: set[str] = set()


def eval_clauses(min_len: int = 5) -> set[str]:
    """Every clause an evaluation set can contain (all sets use 5 to 20 chars)."""
    out = set()
    for path in glob.glob("data/eval/*/*.jsonl") + glob.glob("data/news/*.jsonl"):
        out.update(ex["text"] for ex in read_jsonl(path))
    repo, files = SOURCES["cv"]
    with open(hf_hub_download(repo, files[0], repo_type="dataset"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            text = row["sentence"].strip().rstrip("。！？!?.，,、")
            if PURE_HAN.match(text or "-"):
                out.add(text)
    return {c for c in out if len(c) >= min_len}


def _init(clauses: set[str]) -> None:
    global _EVAL
    _EVAL = clauses


def _clean(line: str) -> str | None:
    doc = json.loads(line)
    for _, clause in iter_clauses(doc["text"]):
        if clause in _EVAL:
            return None
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/corpus/raw")
    ap.add_argument("--out-dir", default="data/corpus/train")
    ap.add_argument("--sources", nargs="+", default=["news", "wiki", "ptt", "web"])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    clauses = eval_clauses()
    print(f"{len(clauses)} evaluation clauses", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"eval_clauses": len(clauses), "sources": {}}
    with mp.Pool(args.workers, initializer=_init, initargs=(clauses,)) as pool:
        for source in args.sources:
            kept = dropped = 0
            with (
                open(Path(args.raw_dir) / f"{source}.train.jsonl", encoding="utf-8") as fin,
                open(out_dir / f"{source}.jsonl", "w", encoding="utf-8") as fout,
            ):
                for line in pool.imap(_clean, fin, chunksize=512):
                    if line is None:
                        dropped += 1
                        continue
                    fout.write(line)
                    kept += 1
            stats["sources"][source] = {"kept_docs": kept, "dropped_docs": dropped}
            print(f"{source}: kept {kept}, dropped {dropped} docs containing an eval clause", flush=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
