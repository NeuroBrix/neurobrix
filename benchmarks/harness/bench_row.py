"""PINNED measurement protocol for row decode rates — the only way a
headline number gets produced.

Born from a measured incident (2026-08-23): the same artifact, the same
day, both campaigns labelled "paired, machine exclusive", produced
9.074 and 8.448 tok/s as baselines — a 7% gap larger than half the
deltas we judge. Two causes found:

  1. `NBX_DECODE_PROGRESS` wrote timestamps at 0.1 s resolution, so a
     60-token rate was QUANTISED to ~2% steps (49/5.4 = 9.074,
     49/5.8 = 8.448 — every published value maps to an integer number
     of deciseconds) and single-arm repeatability looked perfect while
     the floor of a single rep was ±2%. Fixed: %.3f.
  2. Residual inter-campaign drift (clocks, machine state) that no run
     logged, so nothing could adjudicate it after the fact.

The protocol this script pins:

  - N repetitions per arm (default 5), ARMS INTERLEAVED (a1 b1 a2 b2 …)
    so drift lands on both arms;
  - SM clocks LOCKED for the whole campaign (default 1380 MHz via
    passwordless-sudo `nvidia-smi -lgc`, verified held; a sampler
    polls during every rep and any excursion REFUSES the campaign —
    drift killed at the source, not documented). `--lock-clock 0`
    opts out. NOTE: locked numbers are ~10% below boost-clock numbers;
    compare like with like (the report records `lock_clock_mhz`), and
    re-anchor baselines once per lock regime;
  - SM clock + temperature + persistence logged via nvidia-smi before
    and after every single run;
  - rate computed from millisecond timestamps over the post-warm steps;
  - the report prints, per arm: n, MEDIAN, MIN–MAX — and every headline
    number quoted from it must carry n and the dispersion;
  - machine exclusivity checked before starting (refuses like the gate
    runner does);
  - AMBIENT-THERMAL CLAUSE (2026-08-25): the server cabinet is cooled
    by ambient air — warm by day, cool at night. The campaign start
    time (UTC) and the GPU's starting temperature are logged in the
    report (`campaign` block) and printed with the summary. Every
    HEADLINE ABSOLUTE number quoted from a report must carry its time
    slot alongside n and dispersion; absolute numbers from different
    slots are compared across campaigns only with the slot visible.
    RATIOS are already protected: the same-campaign interleaved-arms
    rule means thermal drift lands on both arms of a judgment.

Usage:
  python3 benchmarks/harness/bench_row.py \
      --model <name> --prompt-file <f>|--prompt <s> --max-tokens 60 \
      --gpu 2 --reps 5 --warm 10 \
      --arm name1 ENV1=V1,ENV2=V2 --arm name2 ENV1=V0 \
      --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE_ENV = {
    "NBX_TRITON_REPLAY": "1",
    "NBX_REPLAY_KV_DECODE": "1",
    "NBX_REPLAY_GRAPH": "1",
    "NBX_FORCE_RAND_SEED": "1234",
}


def lock_clocks(gpu: str, mhz: int) -> None:
    """Pin SM clocks to a single frequency for the whole campaign.

    Kills clock drift at the source instead of documenting it (the
    2026-08-23 campaigns logged dips to 1485/1425/1290 MHz mid-rep).
    Requires passwordless sudo for nvidia-smi (verified on driver
    535.309.01: `-lgc M,M` holds min=max even at idle). A lock REQUEST
    that cannot be satisfied is a REFUSAL, not a silent unlocked run.
    `gpu` may be a comma list — every listed GPU is locked (multi-GPU
    cells for over-card models, prereg amendment 2026-08-26)."""
    for g in [x.strip() for x in str(gpu).split(",")]:
        r = subprocess.run(["sudo", "-n", "nvidia-smi", "-i", g,
                            "-lgc", f"{mhz},{mhz}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"REFUSED: cannot lock GPU {g} clocks to {mhz} MHz "
                f"({(r.stderr or r.stdout).strip()}) — pass "
                f"--lock-clock 0 to measure unlocked (drift "
                f"documented, not killed)")
        got = gpu_state(g).get("sm_clock", "")
        if not got.startswith(str(mhz)):
            unlock_clocks(gpu)
            raise SystemExit(
                f"REFUSED: lock did not take on GPU {g} "
                f"(clocks.sm={got!r}, wanted {mhz} MHz)")


def unlock_clocks(gpu: str) -> None:
    for g in [x.strip() for x in str(gpu).split(",")]:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-i", g, "-rgc"],
                       capture_output=True, text=True)


class ClockWatch:
    """Samples clocks.sm during a run; any sample off the locked value
    marks the rep contaminated (a before/after pair would miss a
    transient mid-rep dip — power events can override even min=max).
    `gpu` may be a comma list ("0,1,2,3" — multi-GPU cells for
    over-card models): every listed GPU is watched."""

    def __init__(self, gpu: str, mhz: int, period_s: float = 2.0):
        import threading
        self.gpus = [g.strip() for g in str(gpu).split(",")]
        self.mhz, self.period = mhz, period_s
        self.violations: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.period):
            for g in self.gpus:
                s = gpu_state(g).get("sm_clock", "")
                if not s.startswith(str(self.mhz)):
                    self.violations.append(f"gpu{g}:{s}")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=5)


def gpu_state(gpu: str) -> dict:
    # A comma list (multi-GPU cell) records the FIRST listed GPU here;
    # ClockWatch and lock_clocks cover every listed GPU individually.
    gpu = str(gpu).split(",")[0].strip()
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=persistence_mode,temperature.gpu,clocks.sm,power.draw",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    parts = [p.strip() for p in out.split(",")]
    return {"persistence": parts[0], "temp_c": parts[1],
            "sm_clock": parts[2], "power": parts[3]} if len(parts) == 4 else {"raw": out}


def check_exclusive() -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"REFUSED: compute apps running ({out}) — the "
                         f"protocol requires machine exclusivity")


def rate_from_progress(path: str, warm: int):
    ts = []
    for line in open(path):
        m = re.search(r"t=([0-9.]+)", line)
        if m:
            ts.append(float(m.group(1)))
    if len(ts) < warm + 5:
        return None
    ts = ts[warm:]
    span = ts[-1] - ts[0]
    return (len(ts) - 1) / span if span > 0 else None


def run_once(args, arm_env: dict, tag: str, outdir: Path) -> dict:
    prog = outdir / f"prog_{tag}.txt"
    prog.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["NBX_DECODE_PROGRESS"] = str(prog)
    # Engine selector (hub-benchmark S1): ARM_ENGINE=compiled runs the
    # PyTorch/compiled branch (no --triton flag); default stays the
    # triton branch. Mirrors the existing ARM_ENGINE=ollama pattern.
    cmd = ["python3", "-u", "-m", "neurobrix", "run",
           "--hardware", args.hardware, "--model", args.model,
           "--prompt", args.prompt, "--max-tokens", str(args.max_tokens),
           "--temperature", args.temperature,
           "--output", str(outdir / f"out_{tag}.txt")]
    if arm_env.get("ARM_ENGINE") != "compiled":
        cmd.insert(-2, "--triton")
    before = gpu_state(args.gpu)
    t0 = time.time()
    if args.lock_clock:
        with ClockWatch(args.gpu, args.lock_clock) as watch:
            r = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=2400)
        if watch.violations:
            raise SystemExit(
                f"REFUSED: clock lock broke during rep {tag} — sampled "
                f"{watch.violations[:5]} against the {args.lock_clock} MHz "
                f"lock; the rep is contaminated and the campaign stops "
                f"(re-run, or lower --lock-clock to a frequency this "
                f"thermal envelope can hold)")
    else:
        r = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=2400)
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    rate = rate_from_progress(str(prog), args.warm) if r.returncode == 0 else None
    sha = ""
    op = outdir / f"out_{tag}.txt"
    if op.exists():
        import hashlib
        sha = hashlib.sha256(op.read_bytes()).hexdigest()[:12]
    return {"tag": tag, "rc": r.returncode, "rate": rate, "wall_s": round(wall, 1),
            "sha": sha, "gpu_before": before, "gpu_after": after}


def _wait_gpu_free(gpu: str, timeout_s: int = 180) -> None:
    """Poll until no compute app holds the GPU (an ollama arm with
    keep_alive=0 unloads asynchronously after the response; the next
    nbx rep must not contend)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        if not out:
            return
        time.sleep(2)
    raise SystemExit(
        f"REFUSED: GPU {gpu} still held by a compute app {timeout_s}s "
        f"after an ollama rep — unload did not complete; the campaign "
        f"cannot keep its exclusivity contract")


def run_once_ollama(args, arm_env: dict, tag: str, outdir: Path) -> dict:
    """One ollama rep through its native API — the cross-engine arm of
    the locked protocol (2026-08-24 re-anchor: no published gap number
    may rest on a ratio transfer; both sides measured on this machine
    under the same clock lock).

    Fairness notes, recorded not hidden:
      - keep_alive=0 -> cold load every rep, like the nbx arm.
      - decode rate = eval_count / eval_duration from ollama's OWN
        timers (its native definition, over all generated tokens); the
        nbx arm's rate is post-warm from ms timestamps. Each engine
        gets its best native measurement.
      - temperature 0, same versioned prompt, same token budget.
    TTFT fields (load_duration, prompt_eval_*) land in the run record.
    """
    import json as _json
    import urllib.request
    model = arm_env.get("OLLAMA_MODEL")
    if not model:
        raise SystemExit("ollama arm needs OLLAMA_MODEL=<tag> in its env")
    url = arm_env.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    body = _json.dumps({
        "model": model, "prompt": args.prompt, "stream": False,
        "keep_alive": 0,
        "options": {"temperature": float(args.temperature),
                    "num_predict": args.max_tokens, "seed": 1234},
    }).encode()
    before = gpu_state(args.gpu)
    t0 = time.time()
    if args.lock_clock:
        with ClockWatch(args.gpu, args.lock_clock) as watch:
            with urllib.request.urlopen(
                    urllib.request.Request(
                        url, data=body,
                        headers={"Content-Type": "application/json"}),
                    timeout=2400) as resp:
                r = _json.loads(resp.read())
        if watch.violations:
            raise SystemExit(
                f"REFUSED: clock lock broke during ollama rep {tag} — "
                f"sampled {watch.violations[:5]} against "
                f"{args.lock_clock} MHz")
    else:
        with urllib.request.urlopen(
                urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"}),
                timeout=2400) as resp:
            r = _json.loads(resp.read())
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    text = r.get("response", "")
    (outdir / f"out_{tag}.txt").write_text(text)
    import hashlib
    sha = hashlib.sha256(text.encode()).hexdigest()[:12]
    ev_n = r.get("eval_count", 0)
    ev_d = r.get("eval_duration", 0)
    rate = ev_n / (ev_d / 1e9) if ev_d else None
    rec = {"tag": tag, "rc": 0, "rate": rate, "wall_s": round(wall, 1),
           "sha": sha, "gpu_before": before, "gpu_after": after,
           "ollama": {
               "eval_count": ev_n, "eval_duration_ns": ev_d,
               "prompt_eval_count": r.get("prompt_eval_count"),
               "prompt_eval_duration_ns": r.get("prompt_eval_duration"),
               "load_duration_ns": r.get("load_duration"),
               "total_duration_ns": r.get("total_duration")}}
    _wait_gpu_free(args.gpu)
    return rec


def run_once_vllm(args, arm_env: dict, tag: str, outdir: Path) -> dict:
    """One vLLM rep — cold load per rep like every other arm, decode
    rate and TTFT from vLLM's OWN RequestMetrics (each engine gets its
    best native measurement). Arm env: VLLM_MODEL (HF id or local
    path), VLLM_PYTHON (venv interpreter, default
    ~/venvs/vllm073/bin/python), optional VLLM_DTYPE (default
    float16 — Volta has no bf16)."""
    py = arm_env.get("VLLM_PYTHON",
                     os.path.expanduser("~/venvs/vllm073/bin/python"))
    model = arm_env.get("VLLM_MODEL")
    if not model:
        raise SystemExit("vllm arm needs VLLM_MODEL=<hf-id-or-path>")
    dtype = arm_env.get("VLLM_DTYPE", "float16")
    script = r'''
import json, sys, hashlib
from vllm import LLM, SamplingParams
model, dtype, prompt_path, max_tokens = sys.argv[1:5]
prompt = open(prompt_path).read()
llm = LLM(model=model, dtype=dtype, enforce_eager=False,
          disable_log_stats=True)
sp = SamplingParams(temperature=0, max_tokens=int(max_tokens), seed=1234)
out = llm.generate([prompt], sp)[0]
m = out.metrics
n = len(out.outputs[0].token_ids)
text = out.outputs[0].text
rate = ((n - 1) / (m.finished_time - m.first_token_time)
        if (m and m.first_token_time and m.finished_time
            and m.finished_time > m.first_token_time and n > 1) else None)
print("VLLM_RESULT " + json.dumps({
    "rate": rate, "tokens": n,
    "ttft_s": (m.first_token_time - m.arrival_time) if m else None,
    "sha": hashlib.sha256(text.encode()).hexdigest()[:12]}))
'''
    prompt_file = outdir / f"vllm_prompt_{tag}.txt"
    prompt_file.write_text(args.prompt)
    env = dict(os.environ)
    env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    before = gpu_state(args.gpu)
    t0 = time.time()

    def _run():
        return subprocess.run(
            [py, "-c", script, model, dtype, str(prompt_file),
             str(args.max_tokens)],
            env=env, capture_output=True, text=True, timeout=2400)

    if args.lock_clock:
        with ClockWatch(args.gpu, args.lock_clock) as watch:
            r = _run()
        if watch.violations:
            raise SystemExit(
                f"REFUSED: clock lock broke during vllm rep {tag} — "
                f"sampled {watch.violations[:5]}")
    else:
        r = _run()
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    rate, sha, vres = None, "", {}
    for line in (r.stdout or "").splitlines():
        if line.startswith("VLLM_RESULT "):
            vres = json.loads(line[len("VLLM_RESULT "):])
            rate, sha = vres.get("rate"), vres.get("sha", "")
    if r.returncode != 0 and not vres:
        (outdir / f"vllm_err_{tag}.log").write_text(
            (r.stdout or "")[-4000:] + "\n===\n" + (r.stderr or "")[-8000:])
    return {"tag": tag, "rc": r.returncode, "rate": rate,
            "wall_s": round(wall, 1), "sha": sha, "vllm": vres,
            "gpu_before": before, "gpu_after": after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("NAME", "ENV"),
                    help="arm name + comma-separated ENV=VAL list ('-' for none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock-clock", type=int, default=1380,
                    help="pin SM clocks to this MHz for the whole campaign "
                         "(default 1380 — held with margin at decode-class "
                         "loads; 8k+ prefills draw harder and hold 1290, "
                         "not 1380 — the watchdog refused a 1380 campaign "
                         "on a 1335 MHz sample during an 8k prefill. The "
                         "max boost 1530 is never holdable. 0 = measure "
                         "unlocked (drift documented per rep, not killed).")
    args = ap.parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()
    if not args.prompt:
        ap.error("--prompt or --prompt-file required")

    check_exclusive()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ambient-thermal clause: log when the campaign ran and how warm
    # the GPU started — the cabinet is ambient-air cooled, so absolute
    # numbers carry their time slot (ratios are covered by
    # interleaving).
    campaign_start_utc = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    start_state = gpu_state(args.gpu)
    print(f"  campaign start {campaign_start_utc}  GPU {args.gpu} "
          f"T={start_state.get('temp_c', '?')}C "
          f"clk={start_state.get('sm_clock', '?')} "
          f"persistence={start_state.get('persistence', '?')}", flush=True)

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
        print(f"  clocks LOCKED at {args.lock_clock} MHz (GPU {args.gpu})",
              flush=True)
    results = {name: [] for name, _ in arms}
    try:
        # INTERLEAVED: rep 1 of every arm, then rep 2 of every arm, ...
        for rep in range(1, args.reps + 1):
            for name, env in arms:
                runner = {"ollama": run_once_ollama,
                          "vllm": run_once_vllm}.get(
                    env.get("ARM_ENGINE", ""), run_once)
                res = runner(args, env, f"{name}_{rep}", outdir)
                results[name].append(res)
                print(f"  {name} rep{rep} rc={res['rc']} "
                      f"rate={res['rate'] and round(res['rate'], 3)} "
                      f"sha={res['sha']} "
                      f"clk={res['gpu_after'].get('sm_clock', '?')} "
                      f"T={res['gpu_after'].get('temp_c', '?')}C", flush=True)
    finally:
        if args.lock_clock:
            unlock_clocks(args.gpu)
            print(f"  clocks unlocked (GPU {args.gpu})", flush=True)

    report = {"model": args.model, "max_tokens": args.max_tokens,
              "prompt_sha": __import__("hashlib").sha256(
                  args.prompt.encode()).hexdigest()[:12],
              "reps": args.reps, "warm": args.warm,
              "lock_clock_mhz": args.lock_clock,
              "campaign": {
                  "start_utc": campaign_start_utc,
                  "end_utc": time.strftime("%Y-%m-%d %H:%M UTC",
                                           time.gmtime()),
                  "gpu_start": start_state,
                  "gpu_end": gpu_state(args.gpu)},
              "arms": {}}
    print(f"\ncampaign {campaign_start_utc} -> "
          f"{report['campaign']['end_utc']}  "
          f"start T={start_state.get('temp_c', '?')}C  "
          f"(headline absolutes must quote this time slot)")
    for name, _ in arms:
        rates = [r["rate"] for r in results[name] if r["rate"]]
        shas = {r["sha"] for r in results[name] if r["sha"]}
        if rates:
            med = statistics.median(rates)
            report["arms"][name] = {
                "n": len(rates), "median": round(med, 3),
                "min": round(min(rates), 3), "max": round(max(rates), 3),
                "shas": sorted(shas), "runs": results[name]}
            print(f"{name:14s} n={len(rates)} median {med:7.3f} tok/s "
                  f"[{min(rates):.3f} – {max(rates):.3f}]  "
                  f"outputs: {len(shas)} distinct")
    if len(arms) == 2:
        a, b = (report["arms"].get(arms[0][0]), report["arms"].get(arms[1][0]))
        if a and b:
            overlap = not (b["min"] > a["max"] or a["min"] > b["max"])
            print(f"\n{arms[1][0]} / {arms[0][0]} = "
                  f"x{b['median']/a['median']:.3f} "
                  f"({100*(b['median']/a['median']-1):+.1f}%)  "
                  f"{'ARMS OVERLAP' if overlap else 'no overlap'}")
            report["ratio"] = round(b["median"] / a["median"], 4)
            report["arms_overlap"] = overlap
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nreport -> {outdir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
