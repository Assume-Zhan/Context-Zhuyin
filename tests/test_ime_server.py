import threading
import time
from pathlib import Path

import pytest

NGRAM = Path(__file__).parents[1] / "outputs" / "ngram" / "zhtw-o4"


class PreferSecond:
    """Stub LM: always prefers the second candidate by a wide margin."""

    def score_cached(self, context, texts):
        return [1000.0 if i == 1 else 0.0 for i in range(len(texts))]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not NGRAM.exists():
        pytest.skip("n-gram model not built (scripts/train_ngram.py)")
    try:
        from zhuyin_rescore.lexicon import load_entries

        load_entries()
    except Exception as exc:
        pytest.skip(f"lexicon unavailable: {exc}")
    from zhuyin_ime.server import Reranker, build_engine, serve

    path = str(tmp_path_factory.mktemp("ime") / "ime.sock")
    engine = build_engine(str(NGRAM), "outputs/dict", beam=32)
    ready = threading.Event()
    reranker = Reranker(PreferSecond(), engine, debounce_ms=20)
    threading.Thread(target=serve, args=(path, engine, reranker, ready), daemon=True).start()
    assert ready.wait(30)
    return path


def test_typing_over_socket_and_push(server):
    from zhuyin_ime.client import ServerClient

    c = ServerClient(server, timeout=5)
    st = None
    for ch in "ji3au/6wu0 ":
        st = c.key(char=ch)
    assert st["handled"] and st["preedit"] == "我明天"
    # The stub reranker prefers candidate 2 and pushes a new preedit.
    deadline = time.time() + 5
    pushes = []
    while not pushes and time.time() < deadline:
        time.sleep(0.05)
        c.read_available()
        pushes = c.take_pushes()
    assert pushes and pushes[-1]["preedit"] != "我明天"
    assert pushes[-1]["version"] == st["version"]
    st = c.key(name="Return")
    assert st["commit"] == pushes[-1]["preedit"]


def test_unhandled_keys_pass_through(server):
    from zhuyin_ime.client import ServerClient

    c = ServerClient(server, timeout=5)
    assert not c.key(name="Return")["handled"]
    assert not c.key(char=" ")["handled"]
    assert not c.key(char="c", ctrl=True)["handled"]
