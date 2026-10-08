#!/usr/bin/env bash
# Install the Zhuyin LM IBus front end for the current host user.
#
# The front end is a thin IBus engine (needs only python3, PyGObject and the
# IBus typelib on the host). Conversion runs in the conversion server, usually
# inside the dev container; both meet at outputs/run/zhuyin-ime.sock in the repo.
#
# Usage: scripts/ibus/install.sh            # launcher only, register at runtime
#        scripts/ibus/install.sh --system   # also install the component XML (sudo)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="${BIN_DIR}/zhuyin-ime-ibus"
SOCKET="${REPO}/outputs/run/zhuyin-ime.sock"

if ! /usr/bin/python3 -c 'import gi; gi.require_version("IBus", "1.0")' 2>/dev/null; then
    echo "error: install python3-gi and gir1.2-ibus-1.0 (Debian/Ubuntu) first" >&2
    exit 1
fi

mkdir -p "${BIN_DIR}"
cat > "${LAUNCHER}" <<LAUNCH
#!/usr/bin/env bash
export PYTHONPATH="${REPO}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
exec /usr/bin/python3 -m zhuyin_ime.ibus_engine --socket "\${ZHUYIN_IME_SOCKET:-${SOCKET}}" "\$@"
LAUNCH
chmod +x "${LAUNCHER}"
echo "wrote ${LAUNCHER}"

if [ "${1:-}" = "--system" ]; then
    XML=/usr/share/ibus/component/zhuyin-lm.xml
    sed -e "s#@EXEC@#${LAUNCHER} --ibus#" "${REPO}/scripts/ibus/zhuyin-lm.xml.in" | sudo tee "${XML}" > /dev/null
    echo "wrote ${XML}; restarting ibus"
    ibus restart || true
    echo "Add 'Zhuyin LM' under Settings > Keyboard > Input Sources (Chinese (Taiwan))."
else
    cat <<MSG

Runtime registration (no root):
  ${LAUNCHER} &
  ibus engine zhuyin-lm
Switch back with: ibus engine xkb:us::eng
MSG
fi

cat <<MSG

Start the conversion server in the dev container (from the repo root on the host):
  docker compose -f docker/docker-compose.yml exec phonetic-candidate-dev \\
      python -m zhuyin_ime.server --reranker outputs/lm/qwen2.5-0.5b-zhtw
MSG
