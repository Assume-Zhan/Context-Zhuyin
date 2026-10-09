"""Headless end-to-end check of the IBus front end.

Run under a private session bus (tests/test_ibus_e2e.py does this):
    dbus-run-session -- python3 tests/e2e_ibus.py

Starts ibus-daemon without panel or config, the conversion server, and the
engine (which registers itself on the bus), then drives an IBus input
context with real key events and prints one JSON object with the commits and
the last preedit it saw.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, IBus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pump(seconds: float) -> None:
    ctx = GLib.MainContext.default()
    end = time.time() + seconds
    while time.time() < end:
        while ctx.pending():
            ctx.iteration(False)
        time.sleep(0.01)


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="ibus-e2e-")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HOME=tmp, XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp)
    procs = []
    try:
        daemon = subprocess.Popen(
            ["ibus-daemon", "--panel=disable", "--config=disable", "--emoji-extension=disable", "--replace"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(daemon)
        address = ""
        for _ in range(100):
            out = subprocess.run(["ibus", "address"], env=env, capture_output=True, text=True).stdout.strip()
            if out and out != "(null)":
                address = out
                break
            time.sleep(0.1)
        if not address:
            raise RuntimeError("ibus-daemon did not start")
        env["IBUS_ADDRESS"] = address
        os.environ["IBUS_ADDRESS"] = address

        sock = os.path.join(tmp, "ime.sock")
        # The engine needs PyGObject (system Python); the server needs numpy,
        # which may live in another interpreter (conda in the CUDA 12.4 image).
        server_python = os.environ.get("ZHUYIN_SERVER_PYTHON", sys.executable)
        server = subprocess.Popen(
            [server_python, "-m", "zhuyin_ime.server", "--socket", sock, "--nice", "0"],
            env=env,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(server)
        for _ in range(300):
            if os.path.exists(sock):
                break
            time.sleep(0.1)
        engine = subprocess.Popen(
            [sys.executable, "-m", "zhuyin_ime.ibus_engine", "--socket", sock],
            env=env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(engine)

        IBus.init()
        bus = IBus.Bus()
        for _ in range(100):
            names = [e.get_name() for e in bus.list_engines()] if bus.is_connected() else []
            if "zhuyin-lm" in names:
                break
            pump(0.1)
        ic = bus.create_input_context("e2e")
        commits: list[str] = []
        preedits: list[str] = []
        ic.connect("commit-text", lambda _ic, text: commits.append(text.get_text()))
        ic.connect("update-preedit-text", lambda _ic, text, cursor, visible: preedits.append(text.get_text()))
        tables: list[dict] = []

        def on_table(_ic, table, visible) -> None:
            vertical = table.get_orientation() == IBus.Orientation.VERTICAL
            tables.append({"cursor": table.get_cursor_pos(), "vertical": vertical})

        ic.connect("update-lookup-table", on_table)
        caps = IBus.Capabilite.PREEDIT_TEXT | IBus.Capabilite.FOCUS | IBus.Capabilite.LOOKUP_TABLE
        ic.set_capabilities(caps)
        ic.focus_in()
        ic.set_engine("zhuyin-lm")
        pump(0.5)

        def press(keyval: int, state: int = 0) -> bool:
            handled = ic.process_key_event(keyval, 0, state)
            ic.process_key_event(keyval, 0, state | IBus.ModifierType.RELEASE_MASK)
            pump(0.05)
            return handled

        # wo3 ming2 tian1 zai4 qu4 xue2 xiao4 on the Dai Chien layout, then Enter.
        for ch in "ji3au/6wu0 y94fm4vm,6vul4":
            press(IBus.unicode_to_keyval(ch))
        last_preedit = preedits[-1] if preedits else ""
        press(IBus.KEY_Return)
        pump(0.3)
        # Ctrl+, keeps a comma in the preedit; Down opens the candidate window
        # and Down again moves the highlight; Enter picks it, Enter commits.
        for ch in "ji3":
            press(IBus.unicode_to_keyval(ch))
        press(IBus.KEY_comma, IBus.ModifierType.CONTROL_MASK)
        for ch in "au/6wu0 ":
            press(IBus.unicode_to_keyval(ch))
        comma_preedit = preedits[-1] if preedits else ""
        press(IBus.KEY_Down)
        press(IBus.KEY_Down)
        table = tables[-1] if tables else {}
        press(IBus.KEY_Return)
        press(IBus.KEY_Return)
        pump(0.3)
        result = {"commits": commits, "last_preedit": last_preedit, "comma_preedit": comma_preedit}
        print(json.dumps(result | {"table": table}, ensure_ascii=False))
    finally:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
