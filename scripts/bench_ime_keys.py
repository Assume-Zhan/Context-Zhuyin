"""Per keystroke latency of the IME conversion server (decoder only or with reranker).

Starts the server as a subprocess, types dev sentences key by key through
the client, and reports p50 / p95 / max round trip time per key, which is
what a front end waits for before it can draw, plus the server's memory
after typing: resident (peak, anonymous, file backed) and GPU memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from zhuyin_ime.client import ServerClient
from zhuyin_rescore.zhuyin import syllable_keys


def gpu_process_mb(pid: int) -> float | None:
    """GPU memory held by one process (driver's view, so it includes the CUDA
    context), 0 if it holds none, None without nvidia-smi. Per process
    rather than a before / after difference: the GPU may be shared."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    used = 0.0
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            used += float(parts[1])
    return used


def process_memory_mb(pid: int) -> dict:
    """Resident memory of a process: total, peak, anonymous (heap, private)
    and file backed (mmap'd models, shared libraries; reclaimable)."""
    kb: dict[str, int] = {}
    for name in ("status", "smaps_rollup"):
        with open(f"/proc/{pid}/{name}") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if len(parts) == 2 and parts[1] == "kB":
                    kb[key] = int(parts[0])
    return {
        "rss_mb": round(kb["VmRSS"] / 1024, 1),
        "peak_rss_mb": round(kb["VmHWM"] / 1024, 1),
        "anon_mb": round(kb["Anonymous"] / 1024, 1),
        "file_mb": round((kb["Rss"] - kb["Anonymous"]) / 1024, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/eval/cv/dev.jsonl")
    ap.add_argument("--sentences", type=int, default=200)
    ap.add_argument("--condition", default="full", choices=["full", "notone"])
    ap.add_argument("--server-args", default="", help="extra args, e.g. '--reranker outputs/lm/...'")
    ap.add_argument("--settle-ms", type=float, default=0.0, help="wait for reranker pushes before Enter")
    args = ap.parse_args()

    sock = os.path.join(tempfile.mkdtemp(), "ime.sock")
    cmd = [sys.executable, "-m", "zhuyin_ime.server", "--socket", sock, "--nice", "0"]
    cmd += args.server_args.split()
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(1200):
            if os.path.exists(sock):
                break
            time.sleep(0.1)
        client = ServerClient(sock, timeout=10)
        rows = [json.loads(line) for line in open(args.data)][: args.sentences]
        lat, ok = [], 0
        for r in rows:
            for syl in r[f"zhuyin_{args.condition}"]:
                keys, needs_space = syllable_keys(syl)
                for ch in keys + (" " if needs_space else ""):
                    t0 = time.perf_counter()
                    client.key(char=ch)
                    lat.append((time.perf_counter() - t0) * 1000)
            if args.settle_ms:
                time.sleep(args.settle_ms / 1000)
                client.read_available()
                client.take_pushes()
            st = client.key(name="Return")
            ok += st["commit"] == r["text"]
        stats = client.request("stats").get("reranker")
        memory = process_memory_mb(server.pid)
        memory["gpu_mb"] = gpu_process_mb(server.pid)
        lat = np.array(lat)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "keys": len(lat),
                    "p50_ms": round(float(np.percentile(lat, 50)), 2),
                    "p95_ms": round(float(np.percentile(lat, 95)), 2),
                    "max_ms": round(float(lat.max()), 2),
                    "sentence_acc": round(ok / len(rows), 3),
                    "reranker": stats,
                    "memory": memory,
                }
            )
        )
    finally:
        server.terminate()
        # Wait so a following run starts on an idle machine.
        server.wait(timeout=60)


if __name__ == "__main__":
    main()
