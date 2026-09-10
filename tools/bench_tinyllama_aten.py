#!/usr/bin/env python3
"""TinyLlama on Apple Silicon, ATen branch, under the Dell's locked protocol.

The first complete model measured on this machine. It runs the **ATen /
compiled** branch, where torch is the engine rather than a forbidden import —
R33 governs the Triton branch, and this is the other one. Nothing here is
claimed publicly.

## The protocol, and where it must differ from the Dell

Copied from `benchmarks/harness/bench_row.py`: five reps, **arms interleaved
within each rep** so drift hits every arm equally, one fresh process per rep
so every rep pays a cold load, temperature 0, the same prompt and the same
token budget for every arm, and each engine measured with **its own native
timer**.

Two things the Dell does that this machine cannot, named rather than quietly
dropped:

* **No clock lock.** `nvidia-smi -lgc` has no Apple counterpart: macOS exposes
  no GPU clock pinning at all. The Dell refuses a campaign whose clock drifted
  mid-rep; here that guard does not exist, so thermal drift is a real source
  of spread and the spread is reported instead of hidden.
* **No exclusivity check.** There is no `--query-compute-apps` to prove
  nothing else holds the GPU. The machine was otherwise idle; that is an
  assertion, not a measurement.

## The two numbers, and what each means

* **cold** — wall-clock seconds for the whole process: load, prefill, decode.
  What a user waits for on a first request.
* **warm** — decode rate in tokens/s, from the engine's own per-token
  timestamps, after discarding the first `--warm` tokens. The Dell's
  definition exactly: `(len(ts) - 1) / (ts[-1] - ts[0])`.

For ollama the rate is `eval_count / eval_duration` from **its** timers, over
all generated tokens, with `keep_alive=0` so each rep loads cold like ours.
Each engine gets its best native measurement; the definitions differ and that
is stated rather than averaged away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The Dell's BASE_ENV, minus the triton replay flags that only apply to that
# branch. The seed stays: same seed, same prompt, same budget, every arm.
BASE_ENV = {"NBX_FORCE_RAND_SEED": "1234"}


def rate_from_progress(path: str, warm: int):
    """Decode tokens/s after the first `warm` tokens. The Dell's definition."""
    stamps = []
    try:
        for line in open(path):
            found = re.search(r"t=([0-9.]+)", line)
            if found:
                stamps.append(float(found.group(1)))
    except OSError:
        return None
    if len(stamps) < warm + 5:
        return None
    stamps = stamps[warm:]
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else None


def run_once_nbx(args, arm: str, tag: str, outdir: Path) -> dict:
    progress = outdir / f"prog_{tag}.txt"
    progress.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["NBX_DECODE_PROGRESS"] = str(progress)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    out_path = outdir / f"out_{tag}.txt"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", args.model, "--prompt", args.prompt,
           "--max-tokens", str(args.max_tokens),
           "--temperature", args.temperature,
           "--output", str(out_path), f"--{arm}"]

    started = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=2400, cwd=REPO_ROOT)
    wall = time.time() - started

    text = out_path.read_text() if out_path.exists() else ""
    engine = ""
    found = re.search(r"Engine:\s*(\w+)", proc.stdout)
    if found:
        engine = found.group(1)
    return {
        "tag": tag, "arm": arm, "rc": proc.returncode,
        "cold_wall_s": round(wall, 3),
        "warm_decode_tok_s": rate_from_progress(str(progress), args.warm),
        "engine": engine,
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "chars": len(text),
        "stderr_tail": proc.stderr.strip().splitlines()[-1][:160]
                       if proc.returncode else "",
    }


def run_once_ollama(args, tag: str, outdir: Path) -> dict:
    import urllib.request

    body = json.dumps({
        "model": args.ollama_model, "prompt": args.prompt, "stream": False,
        "keep_alive": 0,
        "options": {"temperature": float(args.temperature),
                    "num_predict": args.max_tokens, "seed": 1234},
    }).encode()
    request = urllib.request.Request(
        args.ollama_url + "/api/generate", data=body,
        headers={"Content-Type": "application/json"})

    started = time.time()
    with urllib.request.urlopen(request, timeout=2400) as response:
        payload = json.loads(response.read())
    wall = time.time() - started

    text = payload.get("response", "")
    (outdir / f"out_{tag}.txt").write_text(text)
    evaluated = payload.get("eval_count", 0)
    duration = payload.get("eval_duration", 0)
    return {
        "tag": tag, "arm": "ollama", "rc": 0,
        "cold_wall_s": round(wall, 3),
        "warm_decode_tok_s": evaluated / (duration / 1e9) if duration else None,
        "engine": "OLLAMA",
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "chars": len(text),
        "ollama": {"eval_count": evaluated, "eval_duration_ns": duration,
                   "prompt_eval_count": payload.get("prompt_eval_count"),
                   "prompt_eval_duration_ns": payload.get("prompt_eval_duration"),
                   "load_duration_ns": payload.get("load_duration"),
                   "total_duration_ns": payload.get("total_duration")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--arm", action="append", default=None,
                    help="compiled | sequential | ollama (repeatable)")
    ap.add_argument("--ollama-model", default="tinyllama")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = args.arm or ["compiled", "sequential"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    # Interleaved: every arm runs once per rep before any arm runs twice.
    for rep in range(1, args.reps + 1):
        for arm in arms:
            tag = f"{arm}_r{rep}"
            print(f"  {tag} ...", end="", flush=True)
            if arm == "ollama":
                record = run_once_ollama(args, tag, outdir)
            else:
                record = run_once_nbx(args, arm, tag, outdir)
            records.append(record)
            rate = record["warm_decode_tok_s"]
            print(f" rc={record['rc']} cold={record['cold_wall_s']}s "
                  f"warm={rate if rate is None else round(rate, 2)} tok/s "
                  f"sha={record['sha256'][:8]}", flush=True)

    (outdir / "records.json").write_text(json.dumps(
        {"model": args.model, "prompt": args.prompt,
         "max_tokens": args.max_tokens, "temperature": args.temperature,
         "reps": args.reps, "warm": args.warm, "arms": arms,
         "records": records}, indent=1))
    print(f"\nwritten: {outdir / 'records.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
