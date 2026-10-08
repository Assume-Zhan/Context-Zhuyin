import json
import shutil
import subprocess
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
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert lines, out.stdout + out.stderr
    result = json.loads(lines[-1])
    assert result["last_preedit"] == "我明天再去學校"
    assert result["commits"] == ["我明天再去學校"]
