"""Clean the raw zh-TW sources into documents with a train / heldout split.

Writes data/corpus/raw/<source>.<split>.jsonl with {"id", "source", "year",
"text"}; text is cleaned paragraphs joined by newlines. Heldout documents
are only used to build evaluation sets; train documents are filtered again
against every evaluation clause before any model sees them
(scripts/build_train_text.py).

Heldout policy per source:
- wiki: the newest articles (highest page ids), i.e. text written recently
- ptt: a hash sample of posts from 2022 or later
- web: a hash sample of pages (Taiwan domains from the 2025-33 crawl)
- news: none, the phase 0 news sets come from a shard that is not used here
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import re
import zlib
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from huggingface_hub import hf_hub_download

from fetch_corpora import SOURCES
from zhuyin_rescore.textproc import clean_paragraphs

TW_HOSTS = {
    "ettoday.net",
    "ptt.cc",
    "pixnet.net",
    "mobile01.com",
    "setn.com",
    "chinatimes.com",
    "ctwant.com",
    "udn.com",
    "storm.mg",
    "thenewslens.com",
    "vocus.cc",
    "nownews.com",
    "mirrormedia.mg",
    "upmedia.mg",
    "techbang.com",
    "gamer.com.tw",
    "newtalk.tw",
    "businesstoday.com.tw",
    "cw.com.tw",
    "ltn.com.tw",
    "cna.com.tw",
    "tvbs.com.tw",
    "pts.org.tw",
    "rti.org.tw",
    "ebc.net.tw",
    "bnext.com.tw",
    "inside.com.tw",
    "dcard.tw",
    "104.com.tw",
    "591.com.tw",
    "books.com.tw",
    "pchome.com.tw",
    "managertoday.com.tw",
    "commonhealth.com.tw",
    "womany.net",
    "kknews.cc",
}
SPAM_WORDS = ["娛樂城", "博彩", "百家樂", "色情", "成人影片", "av女優", "二胎", "借款"]
SPAM_WORDS += ["代辦貸款", "威而鋼", "私密處", "外約", "約砲", "免費看"]
SPAM_RE = re.compile("|".join(SPAM_WORDS))
PTT_HEADER_RE = re.compile(r"時間: \w{3} \w{3} +\d+ [\d:]+ (\d{4})")


def is_taiwan_host(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host.endswith(".tw"):
        return True
    return any(host == h or host.endswith("." + h) for h in TW_HOSTS)


def bucket(key: str, mod: int) -> int:
    return zlib.crc32(key.encode("utf-8")) % mod


def _clean_doc(item: tuple[str, str, int | None, str]) -> dict | None:
    doc_id, source, year, text = item
    # PTT lines are short chat style fragments; keep them.
    paras = clean_paragraphs(text, min_chars=4 if source == "ptt" else 10)
    if source == "web":
        paras = [p for p in paras if not SPAM_RE.search(p)]
    body = "\n".join(paras)
    if len(body) < 100:
        return None
    return {"id": doc_id, "source": source, "year": year, "text": body}


def iter_news() -> Iterator[tuple[str, str, int | None, str]]:
    repo, files = SOURCES["news"]
    for filename in files:
        df = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"), columns=["text"])
        shard = filename.split("-")[1]
        for i, text in enumerate(df["text"]):
            yield f"news-{shard}-{i:06d}", "news", None, text or ""


def iter_wiki() -> Iterator[tuple[str, str, int | None, str]]:
    repo, files = SOURCES["wiki"]
    for filename in files:
        df = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"), columns=["id", "text"])
        for pid, text in zip(df["id"], df["text"], strict=True):
            # Drop markdown headings; the first line repeats the title.
            lines = [ln for ln in (text or "").splitlines()[1:] if not ln.lstrip().startswith("#")]
            yield f"wiki-{int(pid)}", "wiki", None, "\n".join(lines)


def iter_ptt() -> Iterator[tuple[str, str, int | None, str]]:
    repo, files = SOURCES["ptt"]
    with open(hf_hub_download(repo, files[0], repo_type="dataset"), encoding="utf-8") as f:
        posts = json.load(f)
    for i, post in enumerate(posts):
        text = post.get("text") or ""
        m = PTT_HEADER_RE.search(text)
        year = int(m.group(1)) if m else None
        title = re.search(r"標題: (.*)", text)
        body = text.split("內文:", 1)[1] if "內文:" in text else ""
        # Posts were flattened; spaces mark the original line breaks.
        body = re.sub(r" {1,}", "\n", body)
        head = re.sub(r"^\s*(Re:|Fw:)?\s*\[[^\]]*\]\s*", "", title.group(1)) if title else ""
        yield f"ptt-{i:07d}", "ptt", year, head + "\n" + body


def iter_web() -> Iterator[tuple[str, str, int | None, str]]:
    repo, files = SOURCES["web"]
    seen_urls: set[str] = set()
    for filename in files:
        df = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"), columns=["url", "text"])
        for url, text in zip(df["url"], df["text"], strict=True):
            if not url or url in seen_urls or not is_taiwan_host(url):
                continue
            seen_urls.add(url)
            yield f"web-{zlib.crc32(url.encode()):08x}-{len(seen_urls)}", "web", 2025, text or ""


ITERS = {"news": iter_news, "wiki": iter_wiki, "ptt": iter_ptt, "web": iter_web}


def is_heldout(doc: dict, wiki_min_id: int) -> bool:
    src = doc["source"]
    if src == "wiki":
        return int(doc["id"].split("-")[1]) >= wiki_min_id and len(doc["text"]) >= 300
    if src == "ptt":
        return (doc["year"] or 0) >= 2022 and bucket(doc["id"], 8) == 0
    if src == "web":
        return bucket(doc["id"], 50) == 0
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=list(ITERS), choices=list(ITERS))
    ap.add_argument("--out-dir", default="data/corpus/raw")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-train-chars", type=float, default=4e8, help="per source cap on train text")
    ap.add_argument("--wiki-heldout-quantile", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        with mp.Pool(args.workers) as pool:
            docs = [d for d in pool.imap(_clean_doc, ITERS[source](), chunksize=256) if d]
        if source == "web":
            # Boilerplate lines repeat across pages of a site; keep the first copy.
            seen: set[int] = set()
            for d in docs:
                kept = []
                for line in d["text"].splitlines():
                    h = hash(line)
                    if h not in seen:
                        seen.add(h)
                        kept.append(line)
                d["text"] = "\n".join(kept)
            docs = [d for d in docs if len(d["text"]) >= 100]
        wiki_min_id = 0
        if source == "wiki":
            ids = sorted(int(d["id"].split("-")[1]) for d in docs)
            wiki_min_id = ids[int(len(ids) * args.wiki_heldout_quantile)]
        heldout = [d for d in docs if is_heldout(d, wiki_min_id)] if source != "news" else []
        held_ids = {d["id"] for d in heldout}
        train = [d for d in docs if d["id"] not in held_ids]
        random.Random(args.seed).shuffle(train)
        capped, total = [], 0
        for d in train:
            if total >= args.max_train_chars:
                break
            capped.append(d)
            total += len(d["text"])
        for split, rows in (("train", capped), ("heldout", heldout)):
            with open(out_dir / f"{source}.{split}.jsonl", "w", encoding="utf-8") as f:
                for d in rows:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(
            f"{source}: {len(docs)} docs, train {len(capped)} docs {total / 1e6:.0f}M chars "
            f"(of {len(train)}), heldout {len(heldout)} docs"
            + (f", wiki heldout min id {wiki_min_id}" if source == "wiki" else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
