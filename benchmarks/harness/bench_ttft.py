"""PINNED serve-mode TTFT protocol — the only way a headline TTFT is
produced (hub benchmark S1, 2026-08-25).

Measures time-to-first-token in SERVE mode on both sides, weights
loaded (the real-usage, fair comparison): the NeuroBrix serving daemon
(engine per arm) vs an ollama server. Shares the bench_row discipline:
SM clocks locked + watchdog, machine exclusivity, campaign time slot +
start temperature logged, n + median + min-max per cell.

Protocol specifics of TTFT (pre-registered in the S1 prereg):

- COLD-PROMPT RULE: every rep uses a FRESH same-length prompt variant
  (deterministic per rep index) so no server-side prompt cache can
  serve a rep. A cache hit is DETECTED, not assumed absent: the
  ollama arm records prompt_eval_count per rep and a rep whose count
  is ~0 is REFUSED (the campaign stops — the variant generator is
  broken). Prior art: reference_rows VERDICT.md's 246,420 tok/s
  nonsense prefill on a repeated prompt.
- REQUEST-SPLIT: request-1 (first request after daemon ready) and
  requests-2+ are reported as SEPARATE cells on both sides. Known
  prior art on the nbx side: the warm re-freeze class (per-request
  plan re-freeze); the split IS the serve-audit measurement.
- SEQUENTIAL ARMS, same campaign: two warm daemons do not fit one
  32 GB card, so arms run as back-to-back blocks on the SAME GPU
  under the SAME clock lock, block start times logged. TTFT cells are
  per-arm absolutes (not interleaved ratio judgments); the slot label
  carries the residual inter-block drift risk.

Usage:
  python3 benchmarks/harness/bench_ttft.py \
      --model <nbx-model> --gpu 2 --reps 5 --lock-clock 1290 \
      --prompt-file benchmarks/harness/prompts/long_ctx.txt \
      --max-tokens 8 \
      --arm nbx ARM_ENGINE=nbx-serve,NBX_SERVE_FLAG=--triton \
      --arm ollama ARM_ENGINE=ollama-serve,OLLAMA_MODEL=qwen3-coder:30b \
      --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS_DIR, "..", ".."))
from benchmarks.harness.bench_row import (  # noqa: E402
    ClockWatch, check_exclusive, gpu_state, lock_clocks, unlock_clocks)

REPO_SRC = os.path.abspath(os.path.join(HARNESS_DIR, "..", "..", "src"))


def prompt_variant(base: str, rep: int) -> str:
    """Fresh same-length prompt per rep, so no server-side prompt
    cache can serve a rep (the change is at the FRONT — prefix caches
    miss from byte 0). Long base: a deterministic header replaces an
    equal-length prefix, keeping length ~constant. Short base: the
    header is prepended (every rep gets the same header length, so
    reps stay same-length with each other; the small delta vs the
    bare decode-row prompt is recorded in the report's prompt_sha
    being the BASE's)."""
    header = f"Session {rep:04d} review. "
    if len(base) < 4 * len(header):
        return header + base
    return header + base[len(header):]


def ttft_stream_nbx(prompt: str, max_tokens: int) -> dict:
    """One request against the running NeuroBrix daemon; returns
    request wall, TTFT, tokens."""
    sys.path.insert(0, REPO_SRC)
    from neurobrix.serving.client import DaemonClient
    c = DaemonClient()
    c.connect()
    t0 = time.time()
    first = None
    n = 0
    for kind, _ev in c.generate_stream(prompt, max_tokens=max_tokens,
                                       temperature=0):
        if kind == "token":
            n += 1
            if first is None:
                first = time.time() - t0
    total = time.time() - t0
    c.close()
    return {"ttft_s": first, "total_s": total, "tokens": n}


def ttft_stream_ollama(url: str, model: str, prompt: str,
                       max_tokens: int) -> dict:
    import urllib.request
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": max_tokens,
                    "seed": 1234},
    }).encode()
    t0 = time.time()
    first = None
    n = 0
    final = {}
    with urllib.request.urlopen(
            urllib.request.Request(url + "/api/generate", data=body,
                                   headers={"Content-Type":
                                            "application/json"}),
            timeout=2400) as resp:
        for line in resp:
            msg = json.loads(line)
            if msg.get("response"):
                n += 1
                if first is None:
                    first = time.time() - t0
            if msg.get("done"):
                final = msg
    return {"ttft_s": first, "total_s": time.time() - t0, "tokens": n,
            "prompt_eval_count": final.get("prompt_eval_count"),
            "prompt_eval_duration_ns": final.get("prompt_eval_duration"),
            "load_duration_ns": final.get("load_duration")}


def run_nbx_arm(args, env_spec: dict, outdir: Path) -> list:
    """Start the daemon warm, run reps with fresh variants, shut down."""
    flag = env_spec.get("NBX_SERVE_FLAG", "--triton")
    env = dict(os.environ)
    # Per-arm placement / build overrides (2026-09-02): the two engines of one
    # campaign may need DIFFERENT builds and placements — the int4 build is
    # triton-only (the compiled engine refuses the weight encoding) and the
    # fp16 30B build block-scatters over the whole node. NBX_SERVE_GPUS =
    # CUDA_VISIBLE_DEVICES for this arm (default args.gpu); NBX_SERVE_MODEL =
    # the build (default args.model); NBX_SERVE_HARDWARE = the profile
    # ("auto" omits --hardware and lets the detected profile place it).
    env["CUDA_VISIBLE_DEVICES"] = env_spec.get("NBX_SERVE_GPUS", args.gpu).replace(":", ",")  # "0:1:2:3" (the arm spec is comma-separated)
    model = env_spec.get("NBX_SERVE_MODEL", args.model)
    hardware = env_spec.get("NBX_SERVE_HARDWARE", args.hardware)
    cmd = ["python3", "-u", "-m", "neurobrix", "serve", "--model", model]
    if hardware and hardware != "auto":
        cmd += ["--hardware", hardware]
    cmd += [flag, "--foreground"]
    # NBX_SERVE_CMD_PREFIX (2026-09-02, serve TTFT reconciliation): a
    # shell-split prefix in front of the daemon command — the profiler
    # wraps the SAME daemon the locked protocol measures ("nsys profile
    # -t cuda -o …"); the daemon log keeps the NBX_PHASE_TRACE stamps.
    prefix = env_spec.get("NBX_SERVE_CMD_PREFIX", "")
    if prefix:
        import shlex
        cmd = shlex.split(prefix) + cmd
    proc = subprocess.Popen(
        cmd, env=env, stdout=open(outdir / f"daemon_{flag.strip('-')}.log", "w"),
        stderr=subprocess.STDOUT, text=True)
    sys.path.insert(0, REPO_SRC)
    from neurobrix.serving.client import DaemonClient
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"REFUSED: nbx daemon died during load "
                             f"(see daemon log in {outdir})")
        try:
            c = DaemonClient()
            c.connect()
            if c.send("status").get("loaded"):
                ready = True
                c.close()
                break
        except Exception:
            time.sleep(3)
    if not ready:
        proc.kill()
        raise SystemExit("REFUSED: nbx daemon never became ready")
    print(f"  [nbx {flag}] daemon warm "
          f"(T={gpu_state(args.gpu).get('temp_c')}C)", flush=True)
    recs = []
    try:
        for rep in range(1, args.reps + 1):
            p = prompt_variant(args.prompt, rep)
            r = ttft_stream_nbx(p, args.max_tokens)
            r["rep"] = rep
            r["request_index"] = rep  # request-1 vs 2+ split
            recs.append(r)
            print(f"  nbx rep{rep} "
                  f"ttft={r['ttft_s'] and round(r['ttft_s'], 3)}s "
                  f"tokens={r['tokens']}", flush=True)
    finally:
        # The daemon must NEVER outlive its arm block — a leaked warm
        # daemon breaks the next campaign's exclusivity check (leaked
        # once on the 2026-08-25 long-OOM rep).
        try:
            c = DaemonClient()
            c.connect()
            c.shutdown()
        except Exception:
            proc.kill()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    return recs


def run_ollama_arm(args, env_spec: dict, outdir: Path) -> list:
    model = env_spec["OLLAMA_MODEL"]
    url = env_spec.get("OLLAMA_URL", "http://127.0.0.1:11434")
    # Warm the weights once (excluded from cells; request-1 on ollama
    # is defined AFTER weights are resident, symmetric to the nbx
    # daemon whose load is also excluded).
    warm = ttft_stream_ollama(url, model, "warmup.", 1)
    print(f"  [ollama {model}] warm load={warm['load_duration_ns'] and warm['load_duration_ns']/1e9:.1f}s",
          flush=True)
    recs = []
    for rep in range(1, args.reps + 1):
        p = prompt_variant(args.prompt, 1000 + rep)
        r = ttft_stream_ollama(url, model, p, args.max_tokens)
        r["rep"] = rep
        r["request_index"] = rep
        pec = r.get("prompt_eval_count") or 0
        if pec < max(8, 0.5 * (args.expect_prompt_tokens or 0)):
            raise SystemExit(
                f"REFUSED: ollama rep{rep} prompt_eval_count={pec} — "
                f"prompt-cache hit (cold-prompt rule violated); the "
                f"variant generator failed, the campaign stops")
        ld = (r.get("load_duration_ns") or 0) / 1e9
        if ld > 2.0:
            raise SystemExit(
                f"REFUSED: ollama rep{rep} load_duration={ld:.1f}s — "
                f"weights were NOT warm; keep_alive failed")
        recs.append(r)
        print(f"  ollama rep{rep} ttft={r['ttft_s'] and round(r['ttft_s'], 3)}s "
              f"prompt_eval={pec}", flush=True)
    # release the weights so the next arm/campaign starts clean
    import urllib.request
    urllib.request.urlopen(urllib.request.Request(
        url + "/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"}), timeout=120).read()
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--expect-prompt-tokens", type=int, default=0,
                    help="approx token length of the prompt; used by the "
                         "ollama cache-hit refusal threshold")
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("NAME", "ENV"))
    ap.add_argument("--lock-clock", type=int, default=1290)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()
    if not args.prompt:
        ap.error("--prompt or --prompt-file required")

    check_exclusive()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    campaign_start_utc = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    start_state = gpu_state(args.gpu)
    print(f"  campaign start {campaign_start_utc}  GPU {args.gpu} "
          f"T={start_state.get('temp_c', '?')}C", flush=True)

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
        print(f"  clocks LOCKED at {args.lock_clock} MHz", flush=True)
    results = {}
    try:
        # SEQUENTIAL blocks (two warm daemons do not fit one card),
        # same lock, same campaign; block start logged per arm.
        for name, env in arms:
            print(f"  == arm {name} block start "
                  f"{time.strftime('%H:%M:%S UTC', time.gmtime())}",
                  flush=True)
            if args.lock_clock:
                with ClockWatch(args.gpu, args.lock_clock) as watch:
                    if env.get("ARM_ENGINE") == "ollama-serve":
                        recs = run_ollama_arm(args, env, outdir)
                    else:
                        recs = run_nbx_arm(args, env, outdir)
                if watch.violations:
                    raise SystemExit(
                        f"REFUSED: clock lock broke during arm {name} — "
                        f"sampled {watch.violations[:5]}")
            else:
                if env.get("ARM_ENGINE") == "ollama-serve":
                    recs = run_ollama_arm(args, env, outdir)
                else:
                    recs = run_nbx_arm(args, env, outdir)
            results[name] = recs
    finally:
        if args.lock_clock:
            unlock_clocks(args.gpu)

    report = {"model": args.model, "max_tokens": args.max_tokens,
              "prompt_sha": __import__("hashlib").sha256(
                  args.prompt.encode()).hexdigest()[:12],
              "reps": args.reps, "lock_clock_mhz": args.lock_clock,
              "campaign": {"start_utc": campaign_start_utc,
                           "end_utc": time.strftime(
                               "%Y-%m-%d %H:%M UTC", time.gmtime()),
                           "gpu_start": start_state,
                           "gpu_end": gpu_state(args.gpu)},
              "arms": {}}
    print()
    for name, recs in results.items():
        t1 = [r["ttft_s"] for r in recs if r["request_index"] == 1
              and r["ttft_s"]]
        t2 = [r["ttft_s"] for r in recs if r["request_index"] > 1
              and r["ttft_s"]]
        cell = {"runs": recs}
        if t1:
            cell["request1_ttft_s"] = round(t1[0], 3)
        if t2:
            cell["request2plus"] = {
                "n": len(t2), "median": round(statistics.median(t2), 3),
                "min": round(min(t2), 3), "max": round(max(t2), 3)}
            print(f"{name:10s} request-1 {t1 and round(t1[0], 3)}s | "
                  f"request-2+ n={len(t2)} median "
                  f"{statistics.median(t2):.3f}s "
                  f"[{min(t2):.3f}–{max(t2):.3f}]")
        report["arms"][name] = cell
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\ncampaign {campaign_start_utc} -> "
          f"{report['campaign']['end_utc']}  (headline TTFT cells carry "
          f"this slot)\nreport -> {outdir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
