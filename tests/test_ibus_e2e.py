import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_ibus_engine_end_to_end():
    if not (shutil.which("dbus-run-session") and shutil.which("ibus-daemon")):
        pytest.skip("dbus-run-session or ibus-daemon not installed")
    if subprocess.run(["/usr/bin/python3", "-c", "import gi"], capture_output=True).returncode:
        pytest.skip("PyGObject not installed for the system python")
    if not (ROOT / "outputs" / "ngram" / "zhtw-o4").exists():
        pytest.skip("n-gram model not built")
    out = subprocess.run(
        ["dbus-run-session", "--", "/usr/bin/python3", str(ROOT / "tests" / "e2e_ibus.py")],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, ZHUYIN_SERVER_PYTHON=sys.executable),  # the server runs with numpy
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert lines, out.stdout + out.stderr
    result = json.loads(lines[-1])
    assert result["last_preedit"] == "我明天再去學校"
    assert result["comma_preedit"] == "我，明天"
    assert result["table"] == {"cursor": 1, "vertical": True}
    first, second, third = result["commits"]
    assert first == "我明天再去學校"
    assert second.startswith("我，") and len(second) == 4
    # Shift taps: English, Chinese, English, Chinese; Shift+< in between is not a tap.
    assert result["english_passthrough"]
    assert result["mixed_preedit"] == third == "我，ok"
    assert [m for m in result["modes"] if m in ("中", "英")][-4:] == ["英", "中", "英", "中"]
