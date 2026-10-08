"""Download the raw zh-TW text sources used for training and evaluation.

Only selected files are fetched (the full repos are far larger than needed).
Files land in the HF cache; later steps read them through hf_hub_download,
which returns the cached path. News shard 2 is excluded on purpose: the
phase 0 dev and test sets were built from it.
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download

SOURCES = {
    # Taiwan news, about 2024. Shard 2 holds the phase 0 dev and test sets.
    "news": (
        "liswei/news-collection-zhtw",
        [f"data/train-0000{i}-of-00005.parquet" for i in (0, 1, 3, 4)],
    ),
    # zh-TW Wikipedia, August 2026 dump.
    "wiki": (
        "yuhuanstudio/wikipedia-zh-tw",
        [f"2608/train-0000{i}-of-00007.parquet" for i in range(7)],
    ),
    # PTT posts, colloquial Taiwan forum text.
    "ptt": ("yuhuanstudio/PTT-pretrain-zhtw", ["ppt_pretrain.json"]),
    # Traditional Chinese Common Crawl, 2025-33 crawl (August 2025).
    "web": (
        "jed351/Traditional-Chinese-Common-Crawl-Filtered",
        [f"2025_33/2025_33_C4_Traditional_Chinese-0000{i}-of-00008.parquet" for i in (1, 2)],
    ),
    # Common Voice 25 zh-TW sentence list (text only, CC0).
    "cv": ("OpenFormosa/common_voice_25_zh-TW", ["source/validated_sentences.tsv"]),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args()
    for name in args.sources:
        repo, files = SOURCES[name]
        for filename in files:
            path = hf_hub_download(repo, filename, repo_type="dataset")
            print(f"{name}: {repo}/{filename} -> {path}", flush=True)


if __name__ == "__main__":
    main()
