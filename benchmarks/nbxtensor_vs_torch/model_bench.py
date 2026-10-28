"""Model level of the nbxtensor benchmark: the gate cells through the engine's entry point, per device.

Each model runs its gate request (`tools/trace_request.gate_request`) through `python -m neurobrix run` with the
machine's own profile, alone, once per engine:
- `torch_eager`: the engine's ATen branch (the default mode: torch's eager kernels, one call per op);
- `nbxtensor`: the Triton branch (`--triton`), the engine that becomes nbxtensor.
`torch.compile` has no engine mode at this level: the row says so rather than inventing one.

Each row: the run's wall time, the engine's own `[Timings]` (load, time to first output, per-token time when it
decodes) and `[Peak] device_mb`, the output's sha256, and whether the two engines' outputs agree (byte-identical, or
the image/video PSNR when they differ). A run that fails is a row with its return code. Rows are appended as each
cell ends; a rerun skips the cells already written.

    python benchmarks/nbxtensor_vs_torch/model_bench.py --models A,B --out <results/<device>/model.tsv> \\
        --outputs <a directory for the outputs>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

FIELDS = ["model", "contender", "rc", "wall_s", "load_s", "ttft_s", "tpot_ms", "peak_mb", "sha256",
          "agrees_with_torch_eager", "command", "note"]
ENGINES = {"torch_eager": [], "nbxtensor": ["--triton"]}
_EXT = {"image": "png", "video": "mp4", "audio": "wav", "tts": "wav", "upscaler": "png"}


def _family(model: str) -> str:
    import trace_request as TR
    return TR.Z.family_of(model)


def _timings(log: str) -> dict:
    out = {}
    m = re.findall(r"\[Timings\] (.*)", log)
    if m:
        for kv in m[-1].split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k] = v
    p = re.findall(r"\[Peak\] device_mb=(\S+)", log)
    out["peak_mb"] = p[-1] if p else ""
    return out


def _sha(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agreement(a: Path, b: Path) -> str:
    """'identical', the PSNR in dB of two images of one size, or 'differs'."""
    if not (a.exists() and b.exists()):
        return ""
    if _sha(a) == _sha(b):
        return "identical"
    if a.suffix == ".png":
        try:
            import numpy as np
            from PIL import Image
            x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
            y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
            if x.shape == y.shape:
                mse = float(((x - y) ** 2).mean())
                return f"PSNR {10 * np.log10(255.0 ** 2 / mse):.1f} dB" if mse else "identical pixels"
        except Exception as e:                       # said in the row
            return f"differs ({type(e).__name__})"
    return "differs"


def run_cell(model: str, contender: str, outdir: Path, python: str) -> dict:
    import trace_request as TR
    request = TR.gate_request(model)
    fam = _family(model)
    ext = _EXT.get(fam, "txt")
    out = outdir / f"{model}.{contender}.{ext}"
    cmd = [python, "-m", "neurobrix", "run", "--model", model, *request, *ENGINES[contender], "--output", str(out)]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    wall = time.time() - t0
    log = p.stdout + p.stderr
    (outdir / f"{model}.{contender}.log").write_text(log)
    t = _timings(log)
    return {"model": model, "contender": contender, "rc": p.returncode, "wall_s": f"{wall:.1f}",
            "load_s": t.get("load_s", ""), "ttft_s": t.get("ttft_s", ""), "tpot_ms": t.get("tpot_ms", ""),
            "peak_mb": t.get("peak_mb", ""), "sha256": _sha(out)[:16], "command": " ".join(cmd),
            "note": "" if p.returncode == 0 else log.strip().splitlines()[-1][:300] if log.strip() else ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True, help="comma-separated, or @file with one model per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    models = (Path(a.models[1:]).read_text().split() if a.models.startswith("@") else a.models.split(","))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    outdir = Path(a.outputs)
    outdir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["model"], r["contender"]) for r in csv.DictReader(f, delimiter="\t")}
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new:
            w.writeheader()
            f.write("\t".join(["*", "torch_compile", "", "", "", "", "", "", "", "", "",
                               "no engine mode runs torch.compile: not measured at the model level"]) + "\n")
        for model in models:
            rows = {}
            for contender in ENGINES:
                if (model, contender) in done:
                    continue
                rows[contender] = run_cell(model, contender, outdir, a.python)
            if "nbxtensor" in rows:
                ext = _EXT.get(_family(model), "txt")
                rows["nbxtensor"]["agrees_with_torch_eager"] = _agreement(
                    outdir / f"{model}.torch_eager.{ext}", outdir / f"{model}.nbxtensor.{ext}")
            for r in rows.values():
                w.writerow(r)
                f.flush()
                print(f"[model] {r['model']} {r['contender']} rc={r['rc']} wall={r['wall_s']}s "
                      f"peak={r['peak_mb']} {r.get('agrees_with_torch_eager', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
