"""Export a character LM to ONNX Runtime graphs (fp32 and int8) for the IME.

Writes context.onnx, candidates.onnx and their *.int8.onnx versions next to
model.pt (src/zhuyin_rescore/charlm_ort.py). Needs torch and onnx; the
server then needs only onnxruntime. The server also exports on first use
when the files are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zhuyin_rescore.charlm_ort import export_onnx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="char LM directories, e.g. outputs/charlm/small")
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 quantized graphs")
    args = ap.parse_args()
    for model in args.models:
        export_onnx(model, int8=not args.no_int8)
        print(", ".join(sorted(p.name for p in Path(model).glob("*.onnx"))), flush=True)


if __name__ == "__main__":
    main()
