"""PINNED measurement protocol for STT rows — the S2 sibling of
bench_row.py (decode) and bench_ttft.py (serve TTFT).

Metric: RTFx = clip_audio_seconds / transcribe_seconds, HIGHER is
better (the Open ASR Leaderboard convention; readers coming from
whisper.cpp-style RTF numbers apply 1/x once). The clip duration is
read from the wav header, never hardcoded.

`transcribe_seconds` is each arm's own transcription-only time — the
same discipline as the decode rows, where each engine gets its best
native measurement and the definition is RECORDED per arm:

  - nbx (triton / compiled): the CLI's own `[Timing] Total execution:
    Xs` line (execute phase: input prep + encoder + decoder + output;
    excludes process startup and weight load). The per-component
    `Done in Nms` / `Generated N tokens in Nms` lines land in the run
    record as secondary evidence when present.
  - faster-whisper: wall around the exhausted `transcribe()` iterator
    (model load excluded), measured inside the venv subprocess.
  - whisper.cpp: `whisper_print_timings` total minus load, parsed
    from its own report.
  - NeMo (parakeet): wall around `transcribe()` after model load,
    inside the venv subprocess.
  - openai-whisper: wall around `transcribe()` after `load_model`,
    inside the venv subprocess.

Everything else is the bench_row locked protocol, reused not
reinvented: machine exclusivity, SM clock lock + in-rep watchdog
(any excursion REFUSES the campaign), interleaved arms, n/median/
min-max + transcript sha per cell, ambient-thermal clause (campaign
time slot + starting temperature in the report), report.json as the
only source annex cells may quote (verificateur lesson 2026-08-26).

Determinism cells: the transcript sha is byte-level PER ARM.
Cross-ENGINE (nbx triton vs nbx compiled) byte-identity is the
drift-gate cell; cross-TOOL shas differ trivially (whitespace,
casing) and are compared as labeled text, never as a gate.

Usage:
  python3 benchmarks/harness/bench_stt.py \
      --model whisper-large-v3-turbo --audio benchmarks/assets/jfk_11s.wav \
      --gpu 2 --reps 5 \
      --arm nbx - --arm nbxc ARM_ENGINE=compiled \
      --arm fw ARM_ENGINE=fwhisper,FW_MODEL=large-v2 \
      --out validation_outputs/hub_benchmark_2026_08/s2_...
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
import wave
from pathlib import Path

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS_DIR, "..", ".."))
from benchmarks.harness.bench_row import (  # noqa: E402
    ClockWatch, check_exclusive, gpu_state, lock_clocks, unlock_clocks)


def clip_seconds(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def _sha12(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _locked(args, fn):
    """Run fn() under the in-rep clock watchdog (bench_row semantics:
    an excursion is a campaign REFUSAL, not a footnote)."""
    if not args.lock_clock:
        return fn()
    with ClockWatch(args.gpu, args.lock_clock) as watch:
        r = fn()
    if watch.violations:
        raise SystemExit(
            f"REFUSED: clock lock broke during a rep — sampled "
            f"{watch.violations[:5]} against the {args.lock_clock} MHz lock")
    return r


def run_once_nbx(args, arm_env: dict, tag: str, outdir: Path) -> dict:
    env = dict(os.environ)
    env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    # top-k 0 / top-p 1 neutralize the vendor sampling defaults
    # explicitly: the sampling capability gate refuses defaults-carried
    # top_k/top_p on paths that don't implement them, even under
    # greedy, and greedy argmax is invariant to both anyway.
    cmd = ["python3", "-u", "-m", "neurobrix", "run",
           "--hardware", args.hardware, "--model", args.model,
           "--audio", args.audio, "--temperature", args.temperature,
           "--top-k", "0", "--top-p", "1",
           "--output", str(outdir / f"out_{tag}.txt")]
    if arm_env.get("NBX_PROMPT"):          # audio_llm arms need a prompt
        cmd += ["--prompt", arm_env["NBX_PROMPT"]]
    if arm_env.get("ARM_ENGINE") != "compiled":
        cmd.insert(-2, "--triton")
    before = gpu_state(args.gpu)
    t0 = time.time()
    r = _locked(args, lambda: subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=2400))
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    log = (r.stdout or "") + (r.stderr or "")
    (outdir / f"log_{tag}.txt").write_text(log[-20000:])
    m = re.search(r"\[Timing\] Total execution:\s*([0-9.]+)\s*s", log)
    transcribe_s = float(m.group(1)) if m else None
    stages = dict(re.findall(
        r"\[([\w.]+)\] (?:Done|Generated \d+ tokens) in (\d+)ms", log))
    text = ""
    op = outdir / f"out_{tag}.txt"
    if op.exists():
        text = op.read_text()
    return {"tag": tag, "rc": r.returncode, "transcribe_s": transcribe_s,
            "wall_s": round(wall, 1), "sha": _sha12(text) if text else "",
            "stages_ms": stages or None,
            "gpu_before": before, "gpu_after": after}


_FW_SCRIPT = r'''
import json, sys, time
from faster_whisper import WhisperModel
model_id, compute, lang, audio = sys.argv[1:5]
m = WhisperModel(model_id, device="cuda", compute_type=compute)
t0 = time.time()
segs, info = m.transcribe(audio, beam_size=1, temperature=0,
                          language=(lang or None))
text = "".join(s.text for s in segs)          # exhausting = transcribing
dt = time.time() - t0
print("STT_RESULT " + json.dumps({"transcribe_s": dt, "text": text}))
'''

_OAIW_SCRIPT = r'''
import json, sys, time
import whisper
model_id, lang, audio = sys.argv[1:4]
m = whisper.load_model(model_id)
t0 = time.time()
r = m.transcribe(audio, temperature=0, language=(lang or None))
dt = time.time() - t0
print("STT_RESULT " + json.dumps({"transcribe_s": dt, "text": r["text"]}))
'''

_NEMO_SCRIPT = r'''
import json, sys, time
import nemo.collections.asr as nemo_asr
model_id, audio = sys.argv[1:3]
if model_id.endswith(".nemo"):
    m = nemo_asr.models.ASRModel.restore_from(model_id)
else:
    m = nemo_asr.models.ASRModel.from_pretrained(model_id)
t0 = time.time()
out = m.transcribe([audio])
dt = time.time() - t0
text = out[0].text if hasattr(out[0], "text") else str(out[0])
print("STT_RESULT " + json.dumps({"transcribe_s": dt, "text": text}))
'''


def _run_venv_script(args, arm_env, tag, outdir, py, script, argv) -> dict:
    env = dict(os.environ)
    env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    before = gpu_state(args.gpu)
    t0 = time.time()
    r = _locked(args, lambda: subprocess.run(
        [py, "-c", script, *argv],
        env=env, capture_output=True, text=True, timeout=2400))
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    res = {}
    for line in (r.stdout or "").splitlines():
        if line.startswith("STT_RESULT "):
            res = json.loads(line[len("STT_RESULT "):])
    if r.returncode != 0 and not res:
        (outdir / f"err_{tag}.log").write_text(
            (r.stdout or "")[-4000:] + "\n===\n" + (r.stderr or "")[-8000:])
    text = res.get("text", "")
    if text:
        (outdir / f"out_{tag}.txt").write_text(text)
    return {"tag": tag, "rc": r.returncode,
            "transcribe_s": res.get("transcribe_s"),
            "wall_s": round(wall, 1), "sha": _sha12(text) if text else "",
            "gpu_before": before, "gpu_after": after}


def run_once_fwhisper(args, arm_env, tag, outdir) -> dict:
    py = arm_env.get("FW_PYTHON",
                     os.path.expanduser("~/venvs/fwhisper/bin/python"))
    model = arm_env.get("FW_MODEL") or sys.exit("fwhisper arm needs FW_MODEL")
    return _run_venv_script(
        args, arm_env, tag, outdir, py, _FW_SCRIPT,
        [model, arm_env.get("FW_COMPUTE", "float16"),
         arm_env.get("FW_LANG", "en"), args.audio])


def run_once_oaiwhisper(args, arm_env, tag, outdir) -> dict:
    py = arm_env.get("OAIW_PYTHON",
                     os.path.expanduser("~/venvs/oaiwhisper/bin/python"))
    model = arm_env.get("OAIW_MODEL") or sys.exit(
        "oaiwhisper arm needs OAIW_MODEL")
    return _run_venv_script(
        args, arm_env, tag, outdir, py, _OAIW_SCRIPT,
        [model, arm_env.get("OAIW_LANG", "en"), args.audio])


def run_once_nemo(args, arm_env, tag, outdir) -> dict:
    py = arm_env.get("NEMO_PYTHON",
                     os.path.expanduser("~/venvs/nemo/bin/python"))
    model = arm_env.get("NEMO_MODEL", "nvidia/parakeet-tdt-1.1b")
    return _run_venv_script(
        args, arm_env, tag, outdir, py, _NEMO_SCRIPT, [model, args.audio])


def run_once_whispercpp(args, arm_env, tag, outdir) -> dict:
    binp = arm_env.get("WCPP_BIN") or sys.exit(
        "whispercpp arm needs WCPP_BIN=<path-to-whisper-cli>")
    model = arm_env.get("WCPP_MODEL") or sys.exit(
        "whispercpp arm needs WCPP_MODEL=<ggml path>")
    cmd = [binp, "-m", model, "-f", args.audio, "-nt", "-bs", "1",
           "-l", arm_env.get("WCPP_LANG", "en")]
    if arm_env.get("WCPP_FLASH") == "1":
        cmd.append("-fa")
    env = dict(os.environ)
    env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    before = gpu_state(args.gpu)
    t0 = time.time()
    r = _locked(args, lambda: subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=2400))
    wall = time.time() - t0
    after = gpu_state(args.gpu)
    timings = {k: float(v) for k, v in re.findall(
        r"whisper_print_timings:\s+(\w[\w ]*?) time\s*=\s*([0-9.]+)\s*ms",
        r.stderr or "")}
    total, load = timings.get("total"), timings.get("load", 0.0)
    text = (r.stdout or "").strip()
    if text:
        (outdir / f"out_{tag}.txt").write_text(text)
    if r.returncode != 0 and not text:
        (outdir / f"err_{tag}.log").write_text((r.stderr or "")[-8000:])
    return {"tag": tag, "rc": r.returncode,
            "transcribe_s": (total - load) / 1000 if total else None,
            "wall_s": round(wall, 1), "sha": _sha12(text) if text else "",
            "wcpp_timings_ms": timings or None,
            "gpu_before": before, "gpu_after": after}


RUNNERS = {"": run_once_nbx, "compiled": run_once_nbx,
           "fwhisper": run_once_fwhisper, "whispercpp": run_once_whispercpp,
           "nemo": run_once_nemo, "oaiwhisper": run_once_oaiwhisper}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--temperature", default="0",
                    help="nbx arms only; 0 = greedy (the S2 determinism "
                         "contract — audio_llm paths REFUSE the vendor "
                         "top_k/top_p defaults by capability gate)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("NAME", "ENV"),
                    help="arm name + comma-separated ENV=VAL list ('-' for none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock-clock", type=int, default=1380)
    args = ap.parse_args()

    check_exclusive()
    clip_s = clip_seconds(args.audio)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    campaign_start_utc = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    start_state = gpu_state(args.gpu)
    print(f"  campaign start {campaign_start_utc}  GPU {args.gpu} "
          f"clip={clip_s:.2f}s "
          f"T={start_state.get('temp_c', '?')}C "
          f"clk={start_state.get('sm_clock', '?')}", flush=True)

    arms = []
    for name, envspec in args.arm:
        env = {}
        if envspec != "-":
            for kv in envspec.split(","):
                k, _, v = kv.partition("=")
                env[k] = v
        engine = env.get("ARM_ENGINE", "")
        if engine not in RUNNERS:
            raise SystemExit(f"unknown ARM_ENGINE {engine!r} "
                             f"(known: {sorted(RUNNERS)})")
        arms.append((name, env))

    if args.lock_clock:
        lock_clocks(args.gpu, args.lock_clock)
        print(f"  clocks LOCKED at {args.lock_clock} MHz (GPU {args.gpu})",
              flush=True)
    results = {name: [] for name, _ in arms}
    try:
        for rep in range(1, args.reps + 1):     # interleaved, like bench_row
            for name, env in arms:
                res = RUNNERS[env.get("ARM_ENGINE", "")](
                    args, env, f"{name}_{rep}", outdir)
                res["rtfx"] = (round(clip_s / res["transcribe_s"], 3)
                               if res.get("transcribe_s") else None)
                results[name].append(res)
                print(f"  {name} rep{rep} rc={res['rc']} "
                      f"rtfx={res['rtfx']} "
                      f"transcribe={res['transcribe_s']}s sha={res['sha']} "
                      f"clk={res['gpu_after'].get('sm_clock', '?')} "
                      f"T={res['gpu_after'].get('temp_c', '?')}C", flush=True)
    finally:
        if args.lock_clock:
            unlock_clocks(args.gpu)
            print(f"  clocks unlocked (GPU {args.gpu})", flush=True)

    import hashlib
    report = {"model": args.model, "audio": args.audio,
              "audio_sha": hashlib.sha256(
                  Path(args.audio).read_bytes()).hexdigest()[:12],
              "clip_s": round(clip_s, 2),
              "metric": "RTFx = clip_s / transcribe_s (higher better); "
                        "per-arm transcribe_s definitions in the harness "
                        "header",
              "reps": args.reps, "lock_clock_mhz": args.lock_clock,
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
        vals = [r["rtfx"] for r in results[name] if r.get("rtfx")]
        shas = {r["sha"] for r in results[name] if r["sha"]}
        if vals:
            med = statistics.median(vals)
            report["arms"][name] = {
                "n": len(vals), "median_rtfx": round(med, 3),
                "min": round(min(vals), 3), "max": round(max(vals), 3),
                "shas": sorted(shas), "runs": results[name]}
            print(f"{name:14s} n={len(vals)} median RTFx {med:8.3f} "
                  f"[{min(vals):.3f} – {max(vals):.3f}]  "
                  f"outputs: {len(shas)} distinct")
        else:
            report["arms"][name] = {"n": 0, "runs": results[name]}
            print(f"{name:14s} NO SUCCESSFUL REPS (runs recorded)")
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nreport -> {outdir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
