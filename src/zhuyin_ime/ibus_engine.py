"""IBus front end: a thin engine that forwards keys to the conversion server.

Runs in the user's desktop session (where ibus-daemon runs) and needs only
the standard library and PyGObject with the IBus typelib. All conversion
happens in the server (python -m zhuyin_ime.server), reachable through a
Unix socket; if the server is down, keys pass through untouched.

Run directly to register the engine on the running IBus bus (no root
needed), then switch to it with `ibus engine zhuyin-lm`:

    python3 -m zhuyin_ime.ibus_engine --socket /path/to/zhuyin-ime.sock

ibus-daemon starts it with --ibus when the component XML is installed
(scripts/ibus/install.sh --system).
"""

from __future__ import annotations

import argparse
import os
import sys

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, GObject, IBus  # noqa: E402

from zhuyin_ime.client import ServerClient  # noqa: E402

ENGINE_NAME = "zhuyin-lm"
BUS_NAME = "org.freedesktop.IBus.ZhuyinLM"
DEFAULT_SOCKET = os.environ.get(
    "ZHUYIN_IME_SOCKET", os.path.join(os.path.dirname(__file__), "../../outputs/run/zhuyin-ime.sock")
)

SPECIAL_KEYS = {
    IBus.KEY_Return: "Return",
    IBus.KEY_KP_Enter: "Return",
    IBus.KEY_BackSpace: "BackSpace",
    IBus.KEY_Delete: "Delete",
    IBus.KEY_Escape: "Escape",
    IBus.KEY_Left: "Left",
    IBus.KEY_Right: "Right",
    IBus.KEY_Up: "Up",
    IBus.KEY_Down: "Down",
    IBus.KEY_Page_Up: "Page_Up",
    IBus.KEY_Page_Down: "Page_Down",
    IBus.KEY_Home: "Home",
    IBus.KEY_End: "End",
    IBus.KEY_Tab: "Tab",
}
MODIFIER_KEYS = {
    IBus.KEY_Shift_L, IBus.KEY_Shift_R, IBus.KEY_Control_L, IBus.KEY_Control_R,
    IBus.KEY_Alt_L, IBus.KEY_Alt_R, IBus.KEY_Super_L, IBus.KEY_Super_R,
    IBus.KEY_Meta_L, IBus.KEY_Meta_R, IBus.KEY_Caps_Lock,
}  # fmt: skip

SOCKET_PATH = DEFAULT_SOCKET


class ZhuyinLmEngine(IBus.Engine):
    __gtype_name__ = "ZhuyinLmEngine"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = ServerClient(SOCKET_PATH, timeout=0.5)
        self.table = IBus.LookupTable.new(9, 0, True, True)
        # Up and Down move the highlight, so list the candidates vertically.
        self.table.set_orientation(IBus.Orientation.VERTICAL)
        self.watch_id = 0
        self.version = 0

    # ------------------------------------------------------------ connection
    def _ensure_connected(self) -> bool:
        if self.client.sock is not None:
            return True
        try:
            self.client.connect()
        except OSError:
            return False
        events = GLib.IO_IN | GLib.IO_HUP | GLib.IO_ERR
        fd = self.client.fileno()
        self.watch_id = GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, events, self._on_readable)
        return True

    def _disconnect(self) -> None:
        if self.watch_id:
            GLib.source_remove(self.watch_id)
            self.watch_id = 0
        self.client.close()

    def _on_readable(self, fd, condition) -> bool:
        if condition & (GLib.IO_HUP | GLib.IO_ERR):
            self.watch_id = 0
            self.client.close()
            return False
        try:
            self.client.read_available()
        except (OSError, ConnectionError, ValueError):
            self.watch_id = 0
            self.client.close()
            return False
        self._apply_pushes()
        return True

    def _apply_pushes(self) -> None:
        for push in self.client.take_pushes():
            if push.get("version", 0) >= self.version:
                self._apply(push)

    # ---------------------------------------------------------------- drawing
    def _apply(self, st: dict) -> None:
        self.version = max(self.version, st.get("version", 0))
        if st.get("commit"):
            self.commit_text(IBus.Text.new_from_string(st["commit"]))
        preedit = st.get("preedit", "")
        if preedit:
            text = IBus.Text.new_from_string(preedit)
            text.append_attribute(IBus.AttrType.UNDERLINE, IBus.AttrUnderline.SINGLE, 0, len(preedit))
            self.update_preedit_text(text, min(st.get("cursor", len(preedit)), len(preedit)), True)
        else:
            self.hide_preedit_text()
        cands = st.get("candidates") or []
        if cands:
            self.table.clear()
            for i, c in enumerate(cands):
                self.table.append_candidate(IBus.Text.new_from_string(c))
                self.table.set_label(i, IBus.Text.new_from_string(f"{i + 1}."))
            self.table.set_cursor_pos(min(st.get("highlight", 0), len(cands) - 1))
            self.update_lookup_table(self.table, True)
            page = f"{st.get('page', 0) + 1}/{max(st.get('pages', 1), 1)}"
            self.update_auxiliary_text(IBus.Text.new_from_string(page), True)
        else:
            self.hide_lookup_table()
            self.hide_auxiliary_text()

    def _request(self, op: str, **fields) -> dict | None:
        if not self._ensure_connected():
            return None
        try:
            st = self.client.request(op, **fields)
        except (OSError, ConnectionError, ValueError):
            self._disconnect()
            return None
        self._apply(st)
        self._apply_pushes()
        return st

    # ----------------------------------------------------------------- events
    def do_process_key_event(self, keyval, keycode, state):
        if state & IBus.ModifierType.RELEASE_MASK or keyval in MODIFIER_KEYS:
            return False
        name = SPECIAL_KEYS.get(keyval, "")
        char = ""
        if not name:
            # PyGObject maps gunichar to a one character str ("\x00" if none).
            uni = IBus.keyval_to_unicode(keyval)
            char = uni if uni and uni != "\x00" else ""
        st = self._request(
            "key",
            char=char,
            name=name,
            shift=bool(state & IBus.ModifierType.SHIFT_MASK),
            ctrl=bool(state & IBus.ModifierType.CONTROL_MASK),
            alt=bool(state & IBus.ModifierType.MOD1_MASK),
        )
        return bool(st and st.get("handled"))

    def do_focus_out(self):
        self._request("focus_out")

    def do_reset(self):
        self._request("reset")

    def do_disable(self):
        self._request("reset")
        self._disconnect()


def make_component(exec_path: str) -> IBus.Component:
    component = IBus.Component(
        name=BUS_NAME,
        description="Zhuyin input with n-gram decoding and LM reranking",
        version="0.1",
        license="Apache-2.0",
        author="Phonetic-Symbols-Candidate",
        homepage="",
        command_line=exec_path,
        textdomain="",
    )
    engine = IBus.EngineDesc(
        name=ENGINE_NAME,
        longname="Zhuyin LM",
        description="Zhuyin with n-gram decoding and LM reranking",
        language="zh_TW",
        license="Apache-2.0",
        author="Phonetic-Symbols-Candidate",
        icon="",
        layout="us",
        symbol="ZY",
        rank=0,
    )
    component.add_engine(engine)
    return component


def main() -> None:
    global SOCKET_PATH
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ibus", action="store_true", help="started by ibus-daemon from the component XML")
    ap.add_argument("--socket", default=DEFAULT_SOCKET, help="conversion server socket")
    args = ap.parse_args()
    SOCKET_PATH = os.path.abspath(args.socket)

    IBus.init()
    bus = IBus.Bus()
    if not bus.is_connected():
        sys.exit("cannot connect to ibus-daemon")
    bus.connect("disconnected", lambda *_: GLib.MainLoop().quit() or sys.exit(0))
    factory = IBus.Factory.new(bus.get_connection())
    factory.add_engine(ENGINE_NAME, GObject.type_from_name("ZhuyinLmEngine"))
    if args.ibus:
        bus.request_name(BUS_NAME, 0)
    else:
        exec_path = " ".join([sys.executable, "-m", "zhuyin_ime.ibus_engine", "--ibus"])
        bus.register_component(make_component(exec_path))
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
