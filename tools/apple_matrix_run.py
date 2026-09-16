#!/usr/bin/env python3
"""Run the Apple matrix: each model x three modes, classify each cell.

Executes `neurobrix run` per (model, mode) with OWNED caches (the swap/replay
guards apply), reads each output, and classifies triton / triton-sequential
against the compiled reference via `apple_matrix.classify_*`. Emits a JSON
row per model. GPU-bound and memory-hungry -- run it in a memory window (the
VM idle), one model at a time, and it names an OOM or an over-budget artefact
as a NOT_MEASURED cell with its number rather than crashing the sweep.

    tools/apple_matrix_run.py --models swin2SR-classical-sr-x2-64,whisper-... \
        --out validation_outputs/apple_matrix_<date>/rows.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apple_matrix import (MODES, REFERENCE_MODE, Verdict,          # noqa: E402
                          classify_image, classify_text, not_measured)
from check_measurement_environment import owned_cache_env          # noqa: E402

_FLAG = {"compiled": [], "triton": ["--triton"],
         "triton-sequential": ["--triton-sequential"]}


def _run(model: str, mode: str, in_arg: list[str], out: Path, timeout: int) -> int:
    cache = Path(tempfile.mkdtemp(prefix=f"mx_{mode}_"))
    env = owned_cache_env(cache)
    import os
    e = {**os.environ, **env, "PYTHONPATH": "src"}
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model,
           *in_arg, *_FLAG[mode], "--output", str(out)]
    try:
        r = subprocess.run(cmd, env=e, timeout=timeout,
                           capture_output=True, text=True)
        return r.returncode
    except subprocess.TimeoutExpired:
        return 124
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)


def _read(path: Path, kind: str):
    if not path.exists():
        return None
    if kind == "image":
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(path).convert("RGB"))
    return path.read_text(errors="replace")


def run_model(model: str, in_arg: list[str], kind: str, out_dir: Path,
              timeout: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rc, out = {}, {}
    for mode in MODES:
        ext = "png" if kind == "image" else "txt"
        p = out_dir / f"{model}_{mode}.{ext}"
        rc[mode] = _run(model, mode, in_arg, p, timeout)
        out[mode] = _read(p, kind)

    ref = out[REFERENCE_MODE]
    row = {"model": model, "modes": {}}
    for mode in MODES:
        if out[mode] is None:
            row["modes"][mode] = {"verdict": Verdict.NOT_MEASURED.value,
                                  "detail": f"no output (rc={rc[mode]}; "
                                            f"137=OOM, 124=timeout)"}
            continue
        if mode == REFERENCE_MODE:
            row["modes"][mode] = {"verdict": "reference", "detail": "the arm"}
            continue
        if ref is None:
            row["modes"][mode] = {"verdict": Verdict.NOT_MEASURED.value,
                                  "detail": "reference arm produced no output"}
            continue
        cell = (classify_image(ref, out[mode]) if kind == "image"
                else classify_text(ref, out[mode]))
        row["modes"][mode] = {"verdict": cell.verdict.value,
                              "detail": cell.detail, "number": cell.number}
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--image", default="benchmarks/assets/apple_448.png")
    ap.add_argument("--audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    in_arg = (["--audio", a.audio] if a.audio else ["--input-image", a.image])
    kind = "text" if a.audio else "image"
    out_dir = Path(a.out).parent / "outputs"
    rows = [run_model(m.strip(), in_arg, kind, out_dir, a.timeout)
            for m in a.models.split(",") if m.strip()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} model(s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
