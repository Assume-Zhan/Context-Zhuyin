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
from collections import Counter, deque
from pathlib import Path

from zhuyin_ime.pool import FUSION_PRESETS, Fusion, PoolBuilder, fuse
from zhuyin_ime.session import Engine, Key, Session, State


class Reranker:
    """Debounced LM reranking shared by all sessions.

    For each request it builds the candidate pool (zhuyin_ime.pool), scores
    every candidate with the n-gram and the LM, fuses them with the dev tuned
    weights for that pool type, and pushes the winner if the input is still
    the same.
    """

    def __init__(
        self,
        scorer,
        engine: Engine,
        pool: PoolBuilder | None = None,
        fusion: dict[str, Fusion] | None = None,
        debounce_ms: float = 100.0,
        homophone: bool = False,
    ):
        """homophone: call scorer.score_homophone with the typed syllables
        (character LM with homophone normalization) instead of score_cached."""
        self.scorer = scorer
        self.engine = engine
        self.pool = pool or PoolBuilder()
        self.fusion = fusion or FUSION_PRESETS["qwen"]
        self.homophone = homophone
        self.debounce = debounce_ms / 1000.0
        self.cond = threading.Condition()
        self.pending: dict[int, tuple[float, Connection]] = {}
        self.warm_context: str | None = None
        self.recent_ms: deque[float] = deque(maxlen=1000)
        self.pool_types: Counter[str] = Counter()
        threading.Thread(target=self._run, daemon=True, name="reranker").start()

    def request(self, conn: Connection) -> None:
        with self.cond:
            self.pending[id(conn)] = (time.monotonic() + self.debounce, conn)
            self.cond.notify()

    def warm(self, context: str) -> None:
        """Build the context cache for freshly committed text in the background,
        so the next clause's rerank does not pay for it."""
        with self.cond:
            self.warm_context = context
            self.cond.notify()

    def _warm_now(self, context: str) -> None:
        build = getattr(self.scorer, "set_context", None) or self.scorer.context_cache
        try:
            build(context)
        except Exception as exc:
            print(f"reranker warm up error: {exc}", flush=True)

    def stats(self) -> dict:
        ms = sorted(self.recent_ms)
        pick = (lambda q: round(ms[min(len(ms) - 1, int(q * len(ms)))], 2)) if ms else (lambda q: None)
        out = {"calls": len(ms), "p50_ms": pick(0.5), "p95_ms": pick(0.95)}
        return out | {"pool_types": dict(self.pool_types)}

    def choose(self, history: str, decoder_texts: list[str], syllables: list[str]) -> tuple[str, str]:
        """Pool type and the fused best candidate."""
        context = history[-64:]
        kind, texts = self.pool.build(decoder_texts, syllables)
        if len(texts) < 2:
            return kind, texts[0] if texts else ""
        ngram = self.engine.ngram.score_batch(texts, history=context).tolist()
        if self.homophone:
            lm = self.scorer.score_homophone(context, texts, syllables)
        else:
            lm = self.scorer.score_cached(context, texts)
        return kind, texts[fuse(lm, ngram, self.fusion[kind])]

    def _run(self) -> None:
        # One throw away call so the first real rerank does not pay for lazy
        # initialization (CUDA graph capture, int8 kernels, allocator warm up).
        try:
            self.scorer.score_cached("", ["我們", "我門"])
        except Exception as exc:
            print(f"reranker warm up error: {exc}", flush=True)
        while True:
            with self.cond:
                warm, self.warm_context = self.warm_context, None
            if warm is not None:
                self._warm_now(warm)
                continue
            with self.cond:
                while True:
                    if self.warm_context is not None:
                        conn = None
                        break
                    if self.pending:
                        key, (due, conn) = min(self.pending.items(), key=lambda kv: kv[1][0])
                        wait = due - time.monotonic()
                        if wait <= 0:
                            del self.pending[key]
                            break
                        self.cond.wait(wait)
                    else:
                        self.cond.wait()
            if conn is None:
                continue  # a warm up request arrived; handle it first
            with conn.lock:
                req = conn.session.rerank_request()
            if req is None:
                continue
            version, history, kbest, syllables = req
            t0 = time.perf_counter()
            try:
                kind, best = self.choose(history, [d.text for d in kbest], syllables)
            except Exception as exc:  # never take the IME down because of the reranker
                print(f"reranker error: {exc}", flush=True)
                continue
            self.recent_ms.append((time.perf_counter() - t0) * 1000)
            self.pool_types[kind] += 1
            with conn.lock:
                st = conn.session.apply_choice(version, best)
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

    def handle(self, msg: dict) -> State | dict:
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
            elif op == "stats":
                return {"reranker": self.reranker.stats() if self.reranker else None}
            else:
                st, changed = State(handled=False, version=self.session.version), False
        if self.reranker is not None:
            if st.commit:
                self.reranker.warm(self.session.history[-64:])
            if changed and self.session.syllables:
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
                    if isinstance(st, dict):
                        self.send({"op": "stats", "id": msg.get("id"), **st})
                    else:
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


def build_engine(ngram: str, dict_dir: str, beam: int) -> Engine:
    from zhuyin_rescore.lexicon import Lexicon, load_entries
    from zhuyin_rescore.ngram import CharNgram

    return Engine(Lexicon(load_entries(dict_dir), "mixed"), CharNgram.load(ngram), beam=beam)


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


def build_charlm_scorer(path: str, engine: Engine, threads: int, int8: bool = False):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import torch

    from zhuyin_rescore.charlm import CharLMScorer

    torch.set_num_threads(threads)
    lexicon = engine.lexicon

    def homophones(syl: str) -> list[str]:
        return [p for p, _ in lexicon.lookup_span((syl,))]

    scorer = CharLMScorer(path, device="cpu", homophones=homophones)
    if int8:
        scorer.quantize_dynamic_int8()
    return scorer


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
    ap.add_argument("--normalizer", default="full", choices=["homophone", "full"], help="charlm only")
    ap.add_argument("--int8", action="store_true", help="charlm only: int8 dynamic quantization")
    ap.add_argument("--debounce-ms", type=float, default=100.0)
    ap.add_argument(
        "--fusion-preset",
        default="auto",
        choices=["auto", "qwen", "charlm"],
        help="dev tuned fusion weights per pool type; auto follows --reranker-type",
    )
    ap.add_argument(
        "--chewing-lib",
        default=os.environ.get("CHEWING14_LIB"),
        help="libchewing 0.14 shared library; adds its n-best to the pool for toned input",
    )
    ap.add_argument("--chewing-syspath", default=os.environ.get("CHEWING14_PATH"), help="its dictionary dir")
    ap.add_argument("--pool-k", type=int, default=30, help="decoder candidates reranked when not toned")
    ap.add_argument("--nice", type=int, default=5, help="lower the process priority")
    args = ap.parse_args()

    os.nice(args.nice)
    engine = build_engine(args.ngram, args.dict_dir, args.beam)
    reranker = None
    if args.reranker:
        preset = args.reranker_type if args.fusion_preset == "auto" else args.fusion_preset
        if args.chewing_lib and not os.path.exists(args.chewing_lib):
            print(f"libchewing 0.14 not found at {args.chewing_lib}; decoder pool only", flush=True)
            args.chewing_lib = None
        pool = PoolBuilder(args.chewing_lib, args.chewing_syspath, k=args.pool_k)
        if args.reranker_type == "charlm":
            scorer = build_charlm_scorer(args.reranker, engine, args.threads, args.int8)
        else:
            scorer = build_scorer(args.reranker, args.device, args.dtype, args.threads, args.graph)
        homophone = args.reranker_type == "charlm" and args.normalizer == "homophone"
        reranker = Reranker(scorer, engine, pool, FUSION_PRESETS[preset], args.debounce_ms, homophone)
    chewing = "with libchewing n-best" if reranker and reranker.pool.chewing else "decoder pool only"
    print(f"zhuyin-ime server on {args.socket} (reranker: {args.reranker or 'off'}, {chewing})", flush=True)
    serve(args.socket, engine, reranker)


if __name__ == "__main__":
    main()
