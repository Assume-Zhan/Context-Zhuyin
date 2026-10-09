"""Conversion server: owns the engine and serves IME front ends over a Unix socket.

Protocol: JSON lines. A front end sends requests with an integer "id"
  {"op": "key", "id": 1, "char": "q", "name": "", "shift": false, "ctrl": false, "alt": false}
  {"op": "reset", "id": 2}  {"op": "focus_out", "id": 3}  {"op": "ping", "id": 4}
and gets one reply with the same id: {"op": "state", "id": 1, ...State fields}.
When the LM reranker finds a better sentence for the current input, the
server pushes {"op": "state", "id": null, "push": true, ...} on its own.

Each connection is one input context with its own Session. The server
blocks in recv when idle and the reranker worker blocks on a condition
variable, so nothing spins (see the resource-benchmark skill).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path

from zhuyin_ime.session import Engine, Key, Session, State


class Reranker:
    """Debounced LM reranking shared by all sessions."""

    def __init__(self, scorer, debounce_ms: float = 100.0, homophone: bool = False):
        """homophone: call scorer.score_homophone with the typed syllables
        (character LM with homophone normalization) instead of score_cached."""
        self.scorer = scorer
        self.homophone = homophone
        self.debounce = debounce_ms / 1000.0
        self.cond = threading.Condition()
        self.pending: dict[int, tuple[float, Connection]] = {}
        self.last_ms = 0.0
        threading.Thread(target=self._run, daemon=True, name="reranker").start()

    def request(self, conn: Connection) -> None:
        with self.cond:
            self.pending[id(conn)] = (time.monotonic() + self.debounce, conn)
            self.cond.notify()

    def _run(self) -> None:
        while True:
            with self.cond:
                while True:
                    if self.pending:
                        key, (due, conn) = min(self.pending.items(), key=lambda kv: kv[1][0])
                        wait = due - time.monotonic()
                        if wait <= 0:
                            del self.pending[key]
                            break
                        self.cond.wait(wait)
                    else:
                        self.cond.wait()
            with conn.lock:
                req = conn.session.rerank_request()
            if req is None:
                continue
            version, history, kbest, syllables = req
            t0 = time.perf_counter()
            texts = [d.text for d in kbest]
            try:
                if self.homophone:
                    scores = self.scorer.score_homophone(history[-64:], texts, syllables)
                else:
                    scores = self.scorer.score_cached(history[-64:], texts)
            except Exception as exc:  # never take the IME down because of the reranker
                print(f"reranker error: {exc}", flush=True)
                continue
            self.last_ms = (time.perf_counter() - t0) * 1000
            with conn.lock:
                st = conn.session.apply_rerank(version, scores)
            if st is not None:
                conn.send({"op": "state", "id": None, "push": True, **st.to_json()})


class Connection:
    def __init__(self, sock: socket.socket, engine: Engine, reranker: Reranker | None):
        self.sock = sock
        self.session: Session = engine.new_session()
        self.reranker = reranker
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()

    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self.send_lock:
            try:
                self.sock.sendall(data)
            except OSError:
                pass

    def handle(self, msg: dict) -> State:
        op = msg.get("op")
        with self.lock:
            if op == "key":
                key = Key(
                    char=msg.get("char", ""),
                    name=msg.get("name", ""),
                    shift=bool(msg.get("shift")),
                    ctrl=bool(msg.get("ctrl")),
                    alt=bool(msg.get("alt")),
                )
                before = self.session.version
                st = self.session.process_key(key)
                changed = self.session.version != before
            elif op == "reset":
                st, changed = self.session.reset(), False
            elif op == "focus_out":
                st, changed = self.session.focus_out(), False
            else:
                st, changed = State(handled=False, version=self.session.version), False
        if changed and self.reranker is not None and self.session.syllables:
            self.reranker.request(self)
        return st

    def serve(self) -> None:
        buf = b""
        with self.sock:
            while True:
                try:
                    data = self.sock.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    st = self.handle(msg)
                    self.send({"op": "state", "id": msg.get("id"), **st.to_json()})


def serve(socket_path: str, engine: Engine, reranker: Reranker | None = None, ready=None) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    os.chmod(path, 0o600)
    srv.listen(8)
    if ready is not None:
        ready.set()
    while True:
        conn, _ = srv.accept()
        c = Connection(conn, engine, reranker)
        threading.Thread(target=c.serve, daemon=True).start()


def build_engine(ngram: str, dict_dir: str, beam: int, fusion: tuple[float, float] = (0.05, 2.0)) -> Engine:
    from zhuyin_rescore.lexicon import Lexicon, load_entries
    from zhuyin_rescore.ngram import CharNgram

    return Engine(Lexicon(load_entries(dict_dir), "mixed"), CharNgram.load(ngram), beam=beam, fusion=fusion)


def build_scorer(model: str, device: str, dtype: str, threads: int, graph: bool = False):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import torch

    from zhuyin_rescore.scorer import LMScorer

    scorer = LMScorer(model, device=device, dtype=dtype)
    torch.set_num_threads(threads)
    if graph:
        from zhuyin_rescore.graph_scorer import GraphScorer

        return GraphScorer(scorer)
    return scorer


def build_charlm_scorer(path: str, engine: Engine, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import torch

    from zhuyin_rescore.charlm import CharLMScorer

    torch.set_num_threads(threads)
    lexicon = engine.lexicon

    def homophones(syl: str) -> list[str]:
        return [p for p, _ in lexicon.lookup_span((syl,))]

    return CharLMScorer(path, device="cpu", homophones=homophones)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zhuyin IME conversion server")
    ap.add_argument("--socket", default="outputs/run/zhuyin-ime.sock")
    ap.add_argument("--ngram", default="outputs/ngram/zhtw-o4")
    ap.add_argument("--dict-dir", default="outputs/dict", help="libchewing dictionary dump (CSV) cache dir")
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--reranker", default=None, help="causal LM path or HF id; omit for decoder only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for the reranker")
    ap.add_argument("--graph", action="store_true", help="replay the scoring forward as a CUDA graph")
    ap.add_argument("--reranker-type", default="qwen", choices=["qwen", "charlm"], help="charlm: CPU")
    ap.add_argument("--normalizer", default="homophone", choices=["homophone", "full"], help="charlm only")
    ap.add_argument("--debounce-ms", type=float, default=100.0)
    # Fusion score = lm + a * ngram - beta * [not the decoder 1-best]; tuned on
    # dev for the zh-TW model (use about a=0.75 with the base Qwen model).
    ap.add_argument("--fusion-a", type=float, default=0.05)
    ap.add_argument("--fusion-beta", type=float, default=2.0)
    ap.add_argument("--nice", type=int, default=5, help="lower the process priority")
    args = ap.parse_args()

    os.nice(args.nice)
    engine = build_engine(args.ngram, args.dict_dir, args.beam, (args.fusion_a, args.fusion_beta))
    reranker = None
    if args.reranker and args.reranker_type == "charlm":
        scorer = build_charlm_scorer(args.reranker, engine, args.threads)
        reranker = Reranker(scorer, args.debounce_ms, homophone=args.normalizer == "homophone")
    elif args.reranker:
        scorer = build_scorer(args.reranker, args.device, args.dtype, args.threads, args.graph)
        reranker = Reranker(scorer, args.debounce_ms)
    print(f"zhuyin-ime server on {args.socket} (reranker: {args.reranker or 'off'})", flush=True)
    serve(args.socket, engine, reranker)


if __name__ == "__main__":
    main()
