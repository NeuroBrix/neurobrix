#!/usr/bin/env python3
"""S2-TTS row harness — latency + RTFx + R29 WAV + independent STT probe.

Twin of bench_stt.py (imports its lock/exclusive/normalization
helpers). Metric: wall_s per generation of the VERSIONED fixed
sentence; RTFx = produced_audio_s / wall_s (higher better). The
family fidelity gate is the CONTENT PROBE: an independent STT engine
(whisper-large-v3-turbo, nbx compiled) transcribes each kept WAV and
the transcript must phrase-CONTAIN the versioned probe phrase — a WAV
of the right length says nothing about WHAT was said. Mutation-provable
like every gate in the family.

Usage:
  python3 benchmarks/harness/bench_tts.py --model Kokoro-82M \
    --sentence benchmarks/assets/tts_sentence.txt \
    --probe-phrase benchmarks/assets/tts_probe_phrase.txt \
    --reps 5 --gpu 2 --out <dir> \
    --arm nbxt ARM_ENGINE= --arm nbxc ARM_ENGINE=compiled \
    [--arm kokoro ARM_ENGINE=kokoro_pip,KOKORO_PYTHON=...]
"""
from __future__ import annotations

import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_stt import (_locked, _norm_words, check_exclusive, clip_seconds,
                       gpu_state, lock_clocks, unlock_clocks)

REPO = Path(__file__).resolve().parents[2]


def run_once_nbx(args, arm_env, tag, outdir):
    wav = outdir / f"tts_{tag}.wav"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", args.model, "--prompt", args.sentence_text,
           "--output", str(wav), "--hardware", args.hardware,
           "--temperature", "0", "--top-k", "0", "--top-p", "1"]
    if arm_env.get("ARM_ENGINE") == "":
        cmd.append("--triton")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = str(REPO / "src")
    t0 = time.time()
    r = _locked(args, lambda: subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=1800))
    wall = time.time() - t0
    audio_s = clip_seconds(str(wav)) if wav.exists() else None
    if r.returncode != 0:
        (outdir / f"err_{tag}.log").write_text(
            (r.stdout or "")[-3000:] + "\n===\n" + (r.stderr or "")[-6000:])
    return {"tag": tag, "rc": r.returncode, "wall_s": round(wall, 2),
            "audio_s": round(audio_s, 2) if audio_s else None,
            "rtfx": round(audio_s / wall, 3) if audio_s and wall else None,
            "wav": str(wav) if wav.exists() else None,
            "gpu_after": gpu_state(args.gpu)}


_KOKORO_SCRIPT = r'''
import json, sys, time, soundfile as sf
from kokoro import KPipeline
text, out = sys.argv[1:3]
p = KPipeline(lang_code="a")
t0 = time.time()
chunks = [a for (_, _, a) in p(text, voice="af_heart")]
dt = time.time() - t0
import numpy as np
audio = np.concatenate(chunks)
sf.write(out, audio, 24000)
print("TTS_RESULT " + json.dumps({"gen_s": dt}))
'''

_CHATTERBOX_SCRIPT = r'''
import json, sys, time
import torchaudio
from chatterbox.tts import ChatterboxTTS
text, out = sys.argv[1:3]
m = ChatterboxTTS.from_pretrained(device="cuda")
t0 = time.time()
wav = m.generate(text)
dt = time.time() - t0
torchaudio.save(out, wav, m.sr)
print("TTS_RESULT " + json.dumps({"gen_s": dt}))
'''


def _run_pip_arm(args, arm_env, tag, outdir, py, script):
    wav = outdir / f"tts_{tag}.wav"
    env = dict(os.environ); env.update(arm_env)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    t0 = time.time()
    r = _locked(args, lambda: subprocess.run(
        [py, "-c", script, args.sentence_text, str(wav)],
        env=env, capture_output=True, text=True, timeout=1800))
    wall = time.time() - t0
    res = {}
    for line in (r.stdout or "").splitlines():
        if line.startswith("TTS_RESULT "):
            res = json.loads(line[len("TTS_RESULT "):])
    audio_s = clip_seconds(str(wav)) if wav.exists() else None
    gen = res.get("gen_s") or wall
    if r.returncode != 0:
        (outdir / f"err_{tag}.log").write_text(
            (r.stdout or "")[-3000:] + "\n===\n" + (r.stderr or "")[-6000:])
    return {"tag": tag, "rc": r.returncode, "wall_s": round(gen, 2),
            "audio_s": round(audio_s, 2) if audio_s else None,
            "rtfx": round(audio_s / gen, 3) if audio_s and gen else None,
            "wav": str(wav) if wav.exists() else None,
            "gpu_after": gpu_state(args.gpu)}


def run_once_kokoro(args, arm_env, tag, outdir):
    py = arm_env.get("KOKORO_PYTHON",
                     os.path.expanduser("~/venvs/kokoro/bin/python"))
    return _run_pip_arm(args, arm_env, tag, outdir, py, _KOKORO_SCRIPT)


def run_once_chatterbox(args, arm_env, tag, outdir):
    py = arm_env.get("CB_PYTHON",
                     os.path.expanduser("~/venvs/chatterbox/bin/python"))
    return _run_pip_arm(args, arm_env, tag, outdir, py, _CHATTERBOX_SCRIPT)


RUNNERS = {"": run_once_nbx, "compiled": run_once_nbx,
           "kokoro_pip": run_once_kokoro, "chatterbox_pip": run_once_chatterbox}


def stt_probe(args, wav_path: str, outdir, tag) -> dict:
    """Independent STT content probe: nbx whisper-turbo compiled."""
    txt = outdir / f"probe_{tag}.txt"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", "whisper-large-v3-turbo", "--audio", wav_path,
           "--output", str(txt), "--hardware", args.hardware]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = str(REPO / "src")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=900)
    text = txt.read_text() if txt.exists() else ""
    phrase = _norm_words(Path(args.probe_phrase).read_text())
    got = _norm_words(text)
    ok = any(got[i:i+len(phrase)] == phrase
             for i in range(len(got) - len(phrase) + 1))
    return {"probe_rc": r.returncode, "probe_ok": ok,
            "transcript": text.strip()[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sentence", required=True)
    ap.add_argument("--probe-phrase", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arm", nargs=2, action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock-clock", type=int, default=1290)
    args = ap.parse_args()
    args.sentence_text = Path(args.sentence).read_text().strip()

    check_exclusive()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    start = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    print(f"  campaign start {start} GPU {args.gpu} "
          f"T={gpu_state(args.gpu).get('temp_c','?')}C", flush=True)

    arms = []
    for name, envspec in args.arm:
        env = {}
        if envspec != "-":
            for kv in envspec.split(","):
                k, _, v = kv.partition("=")
                env[k] = v
        if env.get("ARM_ENGINE", "") not in RUNNERS:
            raise SystemExit(f"unknown ARM_ENGINE {env.get('ARM_ENGINE')!r}")
        arms.append((name, env))

    if args.lock_clock:
        lock_clocks(args.gpu, args.lock_clock)
        print(f"  clocks LOCKED {args.lock_clock}", flush=True)
    results = {n: [] for n, _ in arms}
    gate_fail = False
    try:
        for rep in range(0, args.reps + 1):        # rep 0 = warmup
            for name, env in arms:
                res = RUNNERS[env.get("ARM_ENGINE", "")](
                    args, env, f"{name}_{rep}", outdir)
                if rep == 0:
                    continue
                # Content probe on the FIRST measured rep's wav (R29
                # artifact); wall cells stay pure generation.
                if rep == 1 and res.get("wav"):
                    res.update(stt_probe(args, res["wav"], outdir,
                                         f"{name}_{rep}"))
                    if res.get("probe_ok") is False:
                        gate_fail = True
                        print(f"  CONTENT PROBE FAIL {name}: "
                              f"'{res.get('transcript','')[:80]}'",
                              flush=True)
                results[name].append(res)
                print(f"  {name} rep{rep} rc={res['rc']} wall={res['wall_s']}s "
                      f"audio={res['audio_s']}s rtfx={res['rtfx']} "
                      f"probe={res.get('probe_ok','-')}", flush=True)
    finally:
        if args.lock_clock:
            unlock_clocks(args.gpu)

    report = {"model": args.model, "sentence_file": args.sentence,
              "probe_phrase_file": args.probe_phrase,
              "metric": "wall_s per generation; RTFx = audio_s/wall_s "
                        "(higher better); content probe = independent "
                        "whisper-turbo transcript phrase-containment",
              "reps": args.reps, "lock_clock_mhz": args.lock_clock,
              "campaign_start_utc": start,
              "campaign_end_utc": time.strftime("%Y-%m-%d %H:%M UTC",
                                                time.gmtime()),
              "arms": {}}
    for name, _ in arms:
        ok = [r for r in results[name] if r.get("rc") == 0 and r.get("wall_s")]
        walls = [r["wall_s"] for r in ok]
        if walls:
            report["arms"][name] = {
                "n": len(walls), "median_wall_s": round(statistics.median(walls), 2),
                "min": min(walls), "max": max(walls),
                "median_rtfx": round(statistics.median(
                    [r["rtfx"] for r in ok if r.get("rtfx")] or [0]), 3),
                "probe_ok": next((r.get("probe_ok") for r in ok
                                  if "probe_ok" in r), None),
                "runs": results[name]}
            print(f"{name:10s} n={len(walls)} median wall "
                  f"{statistics.median(walls):7.2f}s  probe="
                  f"{report['arms'][name]['probe_ok']}")
        else:
            report["arms"][name] = {"n": 0, "runs": results[name]}
            print(f"{name:10s} NO SUCCESSFUL REPS")
    if gate_fail:
        report["content_probe_verdict"] = "FAIL"
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report -> {outdir}/report.json")
    return 1 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
