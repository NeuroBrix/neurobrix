#!/usr/bin/env python3
"""S5 VLM/multimodal row harness — capacity + latency, fidelity-gated.

Modes (per row, declared in the S5 prereg):
  vqa   : --image + fixed question -> answer must phrase-contain ALL
          --must words (normalized), n=5.
  t2i   : fixed prompt -> PNG; gate = decodable + size + mean in the
          pre-declared band [80,220] + non-uniform (std>5), n per prereg.
  audio : --audio + prompt -> transcript phrase-containment (jfk file).

Arms: nbx triton / nbx compiled (vendor arms are per-model venv
scripts when the vendored stack serves inference — wired like
bench_stt's venv runners when a row declares one).
"""
from __future__ import annotations

import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_stt import (_locked, _norm_words, _run_capped, check_exclusive,
                       gpu_state, lock_clocks, unlock_clocks)

REPO = Path(__file__).resolve().parents[2]


def run_once_nbx(args, arm_env, tag, outdir):
    out = outdir / (f"out_{tag}.png" if args.mode == "t2i"
                    else f"out_{tag}.wav" if args.mode == "audio"
                    else f"out_{tag}.txt")
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", args.model, "--hardware", args.hardware,
           "--output", str(out),
           "--temperature", "0", "--top-k", "0", "--top-p", "1"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.image:
        cmd += ["--input-image", args.image]
    if args.audio:
        cmd += ["--audio", args.audio]
    if args.mode == "t2i":
        cmd += ["--mode", "image"]
    if args.mode == "audio":
        cmd += ["--mode", "audio"]
    if args.steps:
        cmd += ["--steps", str(args.steps)]
    if args.max_tokens:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if arm_env.get("ARM_ENGINE") == "":
        cmd.append("--triton")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = str(REPO / "src")
    t0 = time.time()
    r = _locked(args, lambda: _run_capped(lambda: subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=2400)))
    wall = time.time() - t0
    if r.returncode != 0:
        (outdir / f"err_{tag}.log").write_text(
            (r.stdout or "")[-3000:] + "\n===\n" + (r.stderr or "")[-6000:])
    # speech-to-speech rows gate on the pipeline's own transcription line
    (outdir / f"log_{tag}.txt").write_text((r.stdout or "")[-8000:])
    return {"tag": tag, "rc": r.returncode, "wall_s": round(wall, 2),
            "log": str(outdir / f"log_{tag}.txt"),
            "out": str(out) if out.exists() else None,
            "gpu_after": gpu_state(args.gpu)}


def gate(args, res) -> tuple:
    """(ok, why). Family fidelity gates, mutation-provable."""
    if res["rc"] != 0 or not res.get("out"):
        return False, "run failed / no output"
    p = Path(res["out"])
    if args.mode == "audio" and p.suffix == ".wav":
        # speech-to-speech: the WAV is the R29 artifact; the fidelity
        # gate reads the pipeline's [Output] Transcription: line.
        log = Path(res["log"]).read_text() if res.get("log") else ""
        lines = [l.split("Transcription:", 1)[1] for l in log.splitlines()
                 if "Transcription:" in l]
        if not lines:
            return False, "no transcription line in run log"
        text = lines[-1]
        got = _norm_words(text)
        for m in args.must:
            w = _norm_words(m)
            if not any(got[i:i+len(w)] == w
                       for i in range(len(got) - len(w) + 1)):
                return False, f"speech lacks '{m}': {text.strip()[:90]}"
        return True, None
    if args.mode in ("vqa", "audio"):
        text = p.read_text()
        got = _norm_words(text)
        for m in args.must:
            w = _norm_words(m)
            if not any(got[i:i+len(w)] == w
                       for i in range(len(got) - len(w) + 1)):
                return False, f"answer lacks '{m}': {text.strip()[:90]}"
        return True, None
    # t2i
    import numpy as np
    from PIL import Image
    try:
        a = np.asarray(Image.open(p).convert("RGB")).astype(float)
    except Exception as e:
        return False, f"png undecodable: {e}"
    if args.expect_size and (a.shape[1], a.shape[0]) != tuple(args.expect_size):
        return False, f"size {a.shape[1]}x{a.shape[0]} != expected"
    if not (80.0 <= a.mean() <= 220.0):
        return False, f"mean {a.mean():.1f} outside the declared [80,220]"
    if a.std() <= 5.0:
        return False, f"uniform image (std {a.std():.2f})"
    return True, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["vqa", "t2i", "audio"])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--must", action="append", default=[],
                    help="normalized phrase the answer must contain "
                         "(vqa/audio); repeatable")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--expect-size", type=int, nargs=2, default=None)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arm", nargs=2, action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock-clock", type=int, default=1290)
    args = ap.parse_args()

    check_exclusive()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    start = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    print(f"  campaign start {start} GPU {args.gpu} mode={args.mode} "
          f"T={gpu_state(args.gpu).get('temp_c','?')}C", flush=True)
    arms = []
    for name, envspec in args.arm:
        env = {}
        if envspec != "-":
            for kv in envspec.split(","):
                k, _, v = kv.partition("=")
                env[k] = v
        arms.append((name, env))

    if args.lock_clock:
        lock_clocks(args.gpu, args.lock_clock)
    results = {n: [] for n, _ in arms}
    gate_fail = False
    try:
        dead_arms = set()
        for rep in range(0, args.reps + 1):
            for name, env in arms:
                if name in dead_arms:
                    continue
                res = run_once_nbx(args, env, f"{name}_{rep}", outdir)
                if res.get("rc") == 124:
                    dead_arms.add(name)
                    print(f"  ARM {name} TIMED OUT — recorded, skipped "
                          f"for the rest of the campaign", flush=True)
                if rep == 0:
                    continue
                ok, why = gate(args, res)
                res["gate_ok"] = ok
                if not ok:
                    gate_fail = True
                    print(f"  GATE FAIL {name} rep{rep}: {why}", flush=True)
                results[name].append(res)
                print(f"  {name} rep{rep} rc={res['rc']} "
                      f"wall={res['wall_s']}s gate={ok}", flush=True)
    finally:
        if args.lock_clock:
            unlock_clocks(args.gpu)

    report = {"model": args.model, "mode": args.mode,
              "prompt": args.prompt, "must": args.must,
              "reps": args.reps, "lock_clock_mhz": args.lock_clock,
              "campaign_start_utc": start,
              "campaign_end_utc": time.strftime("%Y-%m-%d %H:%M UTC",
                                                time.gmtime()),
              "arms": {}}
    for name, _ in arms:
        ok = [r for r in results[name] if r["rc"] == 0]
        walls = [r["wall_s"] for r in ok]
        if walls:
            report["arms"][name] = {
                "n": len(walls),
                "median_wall_s": round(statistics.median(walls), 2),
                "min": min(walls), "max": max(walls),
                "gate_ok": all(r.get("gate_ok") for r in ok),
                "runs": results[name]}
            print(f"{name:8s} n={len(walls)} median "
                  f"{statistics.median(walls):8.2f}s gate="
                  f"{report['arms'][name]['gate_ok']}")
        else:
            report["arms"][name] = {"n": 0, "runs": results[name]}
            print(f"{name:8s} NO SUCCESSFUL REPS")
    if gate_fail:
        report["gate_verdict"] = "FAIL"
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report -> {outdir}/report.json")
    return 1 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
