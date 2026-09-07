#!/usr/bin/env python3
"""TinyLlama on Apple Silicon, any arm, under the Dell's locked protocol.

Runs whichever arms it is given — `--compiled` / `--sequential` (the ATen
branch, where torch is the engine rather than a forbidden import), `--triton`
/ `--triton-sequential` (the R33 branch), `ollama` — under one protocol, and
WRITES WHAT IT MEASURED: every arm's text in its own file, a `records.json`,
and a dated `RESULTS.md`. Nothing here is claimed publicly.

It refuses to report a conclusion it has not written to disk. A number quoted
from a run whose output file does not exist is not a measurement, and this
tool exits non-zero rather than print one — the check is at the end of
`main`, and it names the files it expected.

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
import datetime
import hashlib
import platform
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

# The three caches that decide what a run measures: the Metal shader cache,
# Triton's own, and the engine's persisted autotune sweep. A campaign that
# leaves them warm measures its own history — identical code gave different
# results across runs on 2026-09-06 — so the tool clears them itself before
# every process rather than trusting the shell that launched it, and records
# in RESULTS.md that it did.
_CACHE_DIRS = (Path.home() / ".cache" / "triton_msl",
               Path.home() / ".triton" / "cache")
_CACHE_GLOB = (Path.home() / ".neurobrix" / "replay_cache", "autotune_configs_*.json")


def clear_caches() -> list:
    import shutil
    cleared = []
    for path in _CACHE_DIRS:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        cleared.append(str(path))
    root, pattern = _CACHE_GLOB
    if root.exists():
        for f in root.glob(pattern):
            f.unlink()
            cleared.append(str(f))
    return cleared


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
    ap.add_argument("--reference", action="append", default=None,
                    help="LABEL=SHA of an earlier campaign to compare against, "
                         "e.g. 'ATen 2026-09-05=9f12fd3966c1bce2' (repeatable)")
    ap.add_argument("--title", default=None,
                    help="RESULTS.md title; defaults to the model and the arms")
    args = ap.parse_args()
    arms = args.arm or ["compiled", "sequential"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    campaign = outdir / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)

    cleared = clear_caches()
    records = []
    # Interleaved: every arm runs once per rep before any arm runs twice.
    for rep in range(1, args.reps + 1):
        for arm in arms:
            tag = f"{arm}_r{rep}"
            print(f"  {tag} ...", end="", flush=True)
            # Cold on purpose, per PROCESS: the caches go before each run, not
            # once before the campaign, so rep 5 measures what rep 1 did.
            clear_caches()
            if arm == "ollama":
                record = run_once_ollama(args, tag, campaign)
            else:
                record = run_once_nbx(args, arm, tag, campaign)
            record["out_file"] = f"campaign/out_{tag}.txt"
            records.append(record)
            rate = record["warm_decode_tok_s"]
            print(f" rc={record['rc']} cold={record['cold_wall_s']}s "
                  f"warm={rate if rate is None else round(rate, 2)} tok/s "
                  f"sha={record['sha256'][:8]}", flush=True)

    references = {}
    for spec in (args.reference or []):
        label, _, sha = spec.partition("=")
        if sha:
            references[label.strip()] = sha.strip()

    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "tool": "tools/bench_tinyllama_arms.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "model": args.model, "prompt": args.prompt,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "reps": args.reps, "warm": args.warm, "arms": arms,
        "caches_cleared_before_every_run": cleared,
        "references": references,
        "records": records,
    }
    (outdir / "records.json").write_text(json.dumps(document, indent=1))
    (outdir / "RESULTS.md").write_text(_results_markdown(document, args))

    # REFUSE TO CONCLUDE WITHOUT THE FILES.
    #
    # Every number this tool prints has to exist on disk, in the file the
    # table names, or the run is not evidence. Twice a figure was reported
    # from a run whose output was never written down; this is the check that
    # makes that impossible rather than a habit.
    missing = [r["out_file"] for r in records
               if not (outdir / r["out_file"]).exists()]
    for required in ("records.json", "RESULTS.md"):
        if not (outdir / required).exists():
            missing.append(required)
    if missing:
        print(f"\nREFUSING to conclude: {len(missing)} file(s) the results "
              f"name were not written: {missing}", flush=True)
        return 2
    failed = [r["tag"] for r in records if r["rc"] != 0]
    print(f"\nwritten: {outdir / 'RESULTS.md'} "
          f"({len(records)} runs, {len(failed)} failed)")
    return 1 if failed else 0


def _results_markdown(document, args) -> str:
    """The dated table, in the same shape as the attention-regimes file."""
    records = document["records"]
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    shas = sorted({r["sha256"] for r in records})

    lines = []
    title = args.title or f"{document['model']} — {', '.join(document['arms'])}"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated **{document['generated']}** by "
                 f"`{document['tool']}` on {document['machine']}.")
    lines.append("**No public claim is made from any number here.**")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"* prompt (verbatim): `{document['prompt']}`")
    lines.append(f"* max-tokens **{document['max_tokens']}**, temperature "
                 f"**{document['temperature']}**, seed 1234")
    lines.append(f"* **{document['reps']} reps, arms interleaved** — every arm "
                 f"runs once per rep before any arm runs twice")
    lines.append("* one fresh process per run, so every run pays a cold load")
    lines.append("* the three caches are cleared by this tool before **every** "
                 "run, not once per campaign:")
    for c in document["caches_cleared_before_every_run"]:
        lines.append(f"  * `{c}`")
    lines.append(f"* warm rate discards the first **{document['warm']}** tokens, "
                 f"then `(len(ts)-1)/(ts[-1]-ts[0])` from the engine's own "
                 f"per-token timestamps")
    lines.append("")
    lines.append("## Output identity")
    lines.append("")
    lines.append("| arm | reps | distinct sha256 | sha256 |")
    lines.append("|---|---:|---:|---|")
    for arm in document["arms"]:
        rows = by_arm.get(arm, [])
        arm_shas = sorted({r["sha256"] for r in rows})
        lines.append(f"| `--{arm}` | {len(rows)} | {len(arm_shas)} | "
                     f"{', '.join('`%s`' % s for s in arm_shas)} |")
    for label, sha in document.get("references", {}).items():
        lines.append(f"| {label} (earlier campaign) | — | 1 | `{sha}` |")
    lines.append("")
    everything = set(shas) | set(document.get("references", {}).values())
    if len(everything) == 1:
        lines.append(f"**All {len(records)} runs and every reference carry the "
                     f"same sha256, `{shas[0]}`** — the arms agree with each "
                     f"other and with the earlier campaign, byte for byte.")
    else:
        lines.append(f"**The outputs are NOT all identical**: "
                     f"{len(everything)} distinct sha256 across the runs and "
                     f"references — {sorted(everything)}.")
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    lines.append("| arm | engine | cold wall (s) | warm decode (tok/s) |")
    lines.append("|---|---|---|---|")
    for arm in document["arms"]:
        rows = [r for r in by_arm.get(arm, []) if r["rc"] == 0]
        if not rows:
            lines.append(f"| `--{arm}` | — | (no successful run) | — |")
            continue
        cold = sorted(r["cold_wall_s"] for r in rows)
        warm = sorted(r["warm_decode_tok_s"] for r in rows
                      if r["warm_decode_tok_s"] is not None)
        med = lambda v: v[len(v) // 2]
        warm_txt = (f"median **{med(warm):.2f}** (min {warm[0]:.2f}, "
                    f"max {warm[-1]:.2f})") if warm else "—"
        lines.append(f"| `--{arm}` | {rows[0]['engine'] or '—'} | "
                     f"median **{med(cold):.2f}** (min {cold[0]:.2f}, "
                     f"max {cold[-1]:.2f}) | {warm_txt} |")
    lines.append("")
    lines.append("## Every run, and the file it wrote")
    lines.append("")
    lines.append("| tag | rc | sha256 | chars | cold (s) | warm (tok/s) | file |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for r in records:
        rate = r["warm_decode_tok_s"]
        lines.append(f"| `{r['tag']}` | {r['rc']} | `{r['sha256']}` | "
                     f"{r['chars']} | {r['cold_wall_s']} | "
                     f"{'—' if rate is None else round(rate, 2)} | "
                     f"`{r['out_file']}` |")
    lines.append("")
    first = records[0]
    text_path = Path(args.out) / first["out_file"]
    if text_path.exists():
        lines.append("## The generated text")
        lines.append("")
        lines.append("```")
        lines.append(text_path.read_text().strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
