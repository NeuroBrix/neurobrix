"""Per-cell gated matrix: the gate is per CELL, like Prism per model.

Before each model's cell, read available_mb and the cell's NEED from Prism's own
plan (`--explain-plan` prints "planned memory N MB"); run the 3 modes if it fits
(recording the memory it ran in), else emit a row "not measured, need N, avail M".
bench_would_swap remains the runtime guard. The estimate is Prism's, calibrated
on the fused graph — not a constant.
"""
import subprocess, sys, re, json, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apple_matrix import MODES, REFERENCE_MODE, Verdict, classify_image, classify_text
from apple_matrix_run import run_model, _read  # reuse the 3-mode runner + classifier

def available_mb():
    from neurobrix.core.host_memory import memory_state
    return memory_state().available_mb

def prism_need_mb(model, in_flag, in_val):
    """Prism's planned memory for this model, MB — the cell's estimated need."""
    env = dict(os.environ); env["PYTHONPATH"] = "src"
    try:
        r = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--model", model,
                            in_flag, in_val, "--explain-plan"],
                           capture_output=True, text=True, timeout=120, env=env)
        m = re.search(r"planned memory\s+([0-9]+)\s*MB", r.stdout)
        if m: return int(m.group(1))
        m = re.search(r"([0-9]+)\s*MB planned", r.stdout)
        if m: return int(m.group(1))
    except Exception as e:
        return None
    return None

MODELS = [
    ("swin2SR-classical-sr-x2-64", "--input-image", "benchmarks/assets/apple_448.png", "image"),
    ("real-esrgan-x2",             "--input-image", "benchmarks/assets/apple_448.png", "image"),
    ("hat-s-x4",                   "--input-image", "benchmarks/assets/apple_448.png", "image"),
    ("swinir-classical-x2",        "--input-image", "benchmarks/assets/apple_448.png", "image"),
    ("whisper-large-v3-turbo",     "--audio",       "benchmarks/assets/jfk_11s.wav",      "text"),
]
MARGIN = 1.25  # need headroom; bench_would_swap is the hard guard

def main():
    out_dir = Path("validation_outputs/apple_matrix_percell_2026_09_16")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, flag, val, kind in MODELS:
        avail = available_mb()
        need = prism_need_mb(model, flag, val)
        gate = None
        if need is not None and need * MARGIN > avail:
            row = {"model": model, "gated_out": True, "need_mb": need, "available_mb": avail,
                   "detail": f"not measured, need {need} MB * {MARGIN} > available {avail} MB"}
            print(f"[GATED] {model}: need {need}MB avail {avail}MB -> not measured", flush=True)
            rows.append(row); Path(out_dir/"rows.json").write_text(json.dumps(rows, indent=1)); continue
        print(f"[RUN] {model}: need {need}MB avail {avail}MB -> fits, running 3 modes", flush=True)
        in_arg = ([flag, val])
        r = run_model(model, in_arg, kind, out_dir/"outputs", timeout=1200)
        r["available_mb_at_run"] = avail
        r["need_mb"] = need
        rows.append(r); Path(out_dir/"rows.json").write_text(json.dumps(rows, indent=1))
    print(f"DONE {len(rows)} cells -> {out_dir}/rows.json", flush=True)

if __name__ == "__main__":
    sys.exit(main())
