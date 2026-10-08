"""Client side of the conversion server protocol (standard library only).

Front ends (the IBus engine) use this to send key events and receive the
state to draw. Replies are matched by id; unsolicited pushes from the
reranker are queued and handed out by take_pushes().
"""

from __future__ import annotations

import json
import socket


class ServerClient:
    def __init__(self, path: str, timeout: float = 0.5):
        self.path = path
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.buf = b""
        self.next_id = 0
        self.pushes: list[dict] = []

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.path)
        self.sock = sock
        self.buf = b""

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def fileno(self) -> int:
        return self.sock.fileno() if self.sock else -1

    def _read_line(self) -> dict | None:
        while b"\n" not in self.buf:
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("server closed the connection")
            self.buf += data
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line) if line.strip() else None

    def request(self, op: str, **fields) -> dict:
        """Send a request and wait for its reply (pushes received meanwhile are queued)."""
        if self.sock is None:
            self.connect()
        self.next_id += 1
        rid = self.next_id
        msg = {"op": op, "id": rid, **fields}
        self.sock.sendall((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        while True:
            reply = self._read_line()
            if reply is None:
                continue
            if reply.get("id") == rid:
                return reply
            if reply.get("push"):
                self.pushes.append(reply)

    def key(self, char: str = "", name: str = "", shift=False, ctrl=False, alt=False) -> dict:
        return self.request("key", char=char, name=name, shift=shift, ctrl=ctrl, alt=alt)

    def read_available(self) -> None:
        """Drain whatever the server already sent (call when the socket is readable)."""
        if self.sock is None:
            return
        self.sock.setblocking(False)
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    raise ConnectionError("server closed the connection")
                self.buf += data
        except BlockingIOError:
            pass
        finally:
            self.sock.settimeout(self.timeout)
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.strip():
                msg = json.loads(line)
                if msg.get("push"):
                    self.pushes.append(msg)

    def take_pushes(self) -> list[dict]:
        out, self.pushes = self.pushes, []
        return out
