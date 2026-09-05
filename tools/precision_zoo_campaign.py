#!/usr/bin/env python3
"""Whole-zoo measurement of one engine lever: the DtypeEngine calibration.

For every cached model of a family (or a list), on one pinned GPU:
  1. `neurobrix calibrate` — the record (one request on the conservative path);
  2. arm A — the same request on the conservative path (`NBX_ACTIVATIONS_FP16_SAFE=0`);
  3. arm B — the same request on the calibration record (islands dumped);
  4. the quality gate by output kind — text: byte identity (first differing
     character otherwise); image: PSNR / SSIM / moved-pixel fraction against A
     (tools/image_fidelity.py); audio: SNR of B against A; video: sha only;
  5. the cold execute time of both arms (`[Timing] Total execution`), each arm
     run twice (A B A B) and the min kept — the first execute of a cold
     process reads the weights from disk; outputs checked run-to-run identical.
  The census and the arms are cold CLI processes: no serving daemon may run
  meanwhile (the engine runs one task at a time) — the locked warm rows of the
  reference models are a separate, serialized stage (benchmarks/harness/run_bench.py).
Artefacts per model under <out>/<model>/ (R29); `table` renders the per-lever
table: who won, by how much, who did not move, who regressed.

    python tools/precision_zoo_campaign.py run --family stt --gpu 1
    python tools/precision_zoo_campaign.py run --models TinyLlama-1.1B-Chat --gpu 1
    python tools/precision_zoo_campaign.py table

The R33 lever (`--probe`): one complete `--triton` request per model under
tools/r33_sys_modules_probe.py — torch in sys.modules at exit, the first
import path when it is there, the run's exit code and the output's sha.
`--src <dir>` puts a frozen worktree's src on the probe's PYTHONPATH.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NBX = "/home/mlops/ml/venv/bin/neurobrix"
PY = "/home/mlops/ml/venv/bin/python"
CACHE = Path(os.path.expanduser("~")) / ".neurobrix" / "cache"
ASSETS = REPO / "benchmarks" / "assets"
OUT_DEFAULT = REPO / "validation_outputs" / "precision_zoo_2026_09_05"

# Family-level request inputs the calibration section cannot carry (media).
_MEDIA = {
    "stt": ["--audio", str(ASSETS / "jfk_11s.wav")],
    "audio_llm": ["--audio", str(ASSETS / "jfk_11s.wav"), "--prompt", "Transcribe this audio."],
    "vlm": ["--input-image", str(ASSETS / "apple_448.png"), "--prompt", "Describe this image in one sentence."],
    "upscaler": ["--input-image", str(ASSETS / "apple_448.png")],
}
# audio_llm: the model's own sampling contract (an explicit --temperature turns
# the vendor's top_k into an "explicit" parameter the path refuses).
_TEXT_BOUND = {"llm": ["--max-tokens", "64", "--temperature", "0"],
               "vlm": ["--max-tokens", "64", "--temperature", "0"],
               "audio_llm": ["--max-tokens", "64"],
               "multimodal": ["--max-tokens", "64", "--temperature", "0"]}


def manifest(model: str) -> dict:
    return json.loads((CACHE / model / "manifest.json").read_text())


def weight_gb(model: str) -> float:
    """Bytes of every safetensors shard under the model's cache directory
    (the profiles of transformers-format artifacts carry no weight size)."""
    total = 0
    for root, _dirs, files in os.walk(CACHE / model):
        for f in files:
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def family_of(model: str) -> str:
    return manifest(model)["family"]


def family_stimulus(family: str) -> list:
    """The family's `calibration:` section as explicit request flags — the
    same request for `calibrate` and both arms (the calibrate command fills
    them itself; `run` does not)."""
    import yaml
    cfg = yaml.safe_load((REPO / "src" / "neurobrix" / "config" / "families" / f"{family}.yml").read_text()) or {}
    out = []
    for k, v in (cfg.get("calibration") or {}).items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def request_args(model: str, family: str, extra: list) -> list:
    args = family_stimulus(family) + list(_MEDIA.get(family, []))
    bound = list(_TEXT_BOUND.get(family, []))
    for i in range(0, len(bound), 2):           # a family stimulus value wins over the campaign bound
        if bound[i] not in args:
            args += bound[i:i + 2]
    if family == "multimodal":
        topo = json.loads((CACHE / model / "topology.json").read_text())
        gen = ((topo.get("flow") or {}).get("generation") or {}).get("type", "")
        mode = "image" if gen == "autoregressive_image" else "text"
        args += ["--mode", mode]
        if mode == "text":
            args += ["--input-image", str(ASSETS / "apple_448.png"), "--prompt", "Describe this image in one sentence."]
    return args + list(extra)


def run(cmd, env, log: Path, timeout: int) -> tuple:
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write(f"\nTIMEOUT after {timeout}s\n")
    return rc, time.time() - t0


def exec_time(log: Path):
    m = re.findall(r"\[Timing\] Total execution: ([0-9.]+)s", log.read_text(errors="replace"))
    return float(m[-1]) if m else None


def islands_from_log(log: Path) -> dict:
    out = {}
    for m in re.finditer(r"\[DtypeEngine\] (\S+): precision contract from calibration \S+ \((\d+) pass.*?: (\d+) op\(s\) islanded", log.read_text(errors="replace")):
        out[m.group(1)] = {"passes": int(m.group(2)), "islands": int(m.group(3))}
    for m in re.finditer(r"\[DtypeEngine\] (\S+): no calibration record", log.read_text(errors="replace")):
        out.setdefault(m.group(1), {"islands": None})
    return out


def gate(a: Path, b: Path) -> dict:
    if not a.exists() or not b.exists():
        return {"kind": "missing", "pass": False}
    ext = a.suffix.lower()
    if ext == ".txt":
        ta, tb = a.read_bytes(), b.read_bytes()
        if ta == tb:
            return {"kind": "text", "identical": True, "pass": True, "chars": len(ta)}
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        return {"kind": "text", "identical": False, "pass": False, "first_diff_at": k, "chars": len(ta)}
    if ext == ".png":
        r = subprocess.run([PY, str(REPO / "tools" / "image_fidelity.py"), str(a), str(b), "--json"],
                           capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
        except Exception:
            return {"kind": "image", "pass": False, "error": r.stdout[-300:] + r.stderr[-300:]}
        d["kind"] = "image"
        d["pass"] = bool(d.get("identical")) or float(d.get("psnr_db", 0)) >= 30.0
        return d
    if ext == ".wav":
        import numpy as np
        import soundfile as sf
        xa, sra = sf.read(str(a)); xb, srb = sf.read(str(b))
        n = min(len(xa), len(xb))
        xa, xb = np.asarray(xa[:n], dtype=np.float64), np.asarray(xb[:n], dtype=np.float64)
        noise = float(np.sum((xa - xb) ** 2)); sig = float(np.sum(xa ** 2))
        snr = float("inf") if noise == 0 else 10 * np.log10(sig / noise) if sig > 0 else float("nan")
        return {"kind": "audio", "pass": snr >= 30.0 or noise == 0, "snr_db": snr, "identical": noise == 0,
                "len_a": len(xa), "len_b": len(xb), "sr": sra}
    import hashlib
    ha, hb = hashlib.sha256(a.read_bytes()).hexdigest()[:12], hashlib.sha256(b.read_bytes()).hexdigest()[:12]
    return {"kind": ext.lstrip("."), "identical": ha == hb, "pass": ha == hb, "sha_a": ha, "sha_b": hb}


def one_model(model: str, gpu, out: Path, extra: list, timeout: int) -> dict:
    """gpu=None → the whole rig visible (Prism free): the stage for models
    whose weights do not fit one card; an int → one pinned card."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    ext = {"llm": ".txt", "stt": ".txt", "vlm": ".txt", "audio_llm": ".txt", "tts": ".wav",
           "video": ".mp4", "image": ".png", "upscaler": ".png"}.get(fam, ".txt")
    req = request_args(model, fam, extra)
    if fam == "multimodal" and "--mode" in req and req[req.index("--mode") + 1] == "image":
        ext = ".png"
    base_env = {**os.environ}
    if gpu is None:
        base_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = {"model": model, "family": fam, "model_name": manifest(model).get("model_name"), "request": req,
           "weight_gb": round(weight_gb(model), 2), "config": "machine" if gpu is None else f"pinned:{gpu}"}
    rc, wall = run([NBX, "calibrate", "--model", model] + req, base_env, d / "calibrate.log", timeout)
    res["calibrate"] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / "calibrate.log")}
    if rc:
        # No record → the arms would both run the conservative path and the
        # table would read a speed-up that does not exist. Fail the model.
        res["A"] = {"rc": rc, "exec_s": None}; res["B"] = {"rc": rc, "exec_s": None}
        res["islands"] = {}; res["gate"] = {"kind": "missing", "pass": False}; res["speedup"] = None
        log = (d / "calibrate.log").read_text(errors="replace")
        res["error"] = ("triton-only build" if "UNSUPPORTED PATH" in log and "encoding" in log
                        else "calibrate failed")
        (d / "result.json").write_text(json.dumps(res, indent=1))
        return res
    # Each arm twice, A B A B: a cold process reads the weights from disk on
    # its first execute (NFS-cold vs page-cache), so the kept execute time
    # is the MIN of the two runs — the cold-execute protocol with the disk
    # out of the picture. Outputs are from the second run; the first run's
    # output must be byte-identical to it (determinism check).
    arms = (("A", {**base_env, "NBX_ACTIVATIONS_FP16_SAFE": "0"}),
            ("B", {**base_env, "NBX_PRECISION_ISLANDS": str(d / "islands.tsv")}))
    for rep in (1, 2):
        for arm, env in arms:
            if arm == "B":
                (d / "islands.tsv").unlink(missing_ok=True)
            outp = d / f"{arm}{ext}"
            if rep == 1:
                outp = d / f"{arm}.run1{ext}"
            rc, wall = run([NBX, "run", "--model", model] + req + ["--output", str(outp)], env,
                           d / f"{arm}.run{rep}.log", timeout)
            e = exec_time(d / f"{arm}.run{rep}.log")
            prev = res.get(arm)
            res[arm] = {"rc": rc if prev is None else (prev["rc"] or rc),
                        "exec_runs": (prev["exec_runs"] if prev else []) + [e],
                        "output": str(d / f"{arm}{ext}")}
    for arm, _ in arms:
        runs = [x for x in res[arm]["exec_runs"] if x is not None]
        res[arm]["exec_s"] = min(runs) if runs else None
        r1, r2 = d / f"{arm}.run1{ext}", d / f"{arm}{ext}"
        res[arm]["run_to_run_identical"] = (r1.exists() and r2.exists() and r1.read_bytes() == r2.read_bytes())
    res["stochastic_reference"] = not res["A"]["run_to_run_identical"]
    (d / "B.log").write_text((d / "B.run2.log").read_text(errors="replace")) if (d / "B.run2.log").exists() else None
    res["islands"] = islands_from_log(d / "B.log")
    res["gate"] = gate(d / f"A{ext}", d / f"B{ext}")
    a, b = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (a / b) if a and b else None
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def launcher_ab(model: str, gpu, out: Path, extra: list, timeout: int) -> dict:
    """The launcher gate on one model: `--triton` with upstream's launcher
    (NBX_LAUNCHER=triton) vs the NeuroBrix launcher, outputs byte-compared."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    ext = {"llm": ".txt", "stt": ".txt", "vlm": ".txt", "audio_llm": ".txt", "tts": ".wav",
           "video": ".mp4", "image": ".png", "upscaler": ".png"}.get(fam, ".txt")
    req = request_args(model, fam, extra) + ["--triton"]
    base_env = {**os.environ}
    if gpu is None:
        base_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "launcher"}
    for arm, env in (("A", {**base_env, "NBX_LAUNCHER": "triton"}), ("B", {**base_env, "NBX_LAUNCHER": "nbx"})):
        outp = d / f"{arm}{ext}"
        rc, wall = run([NBX, "run", "--model", model] + req + ["--output", str(outp)], env, d / f"{arm}.log", timeout)
        res[arm] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / f"{arm}.log"), "output": str(outp)}
    a, b = d / f"A{ext}", d / f"B{ext}"
    same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
    res["gate"] = {"kind": "bytes", "identical": same, "pass": same}
    res["islands"] = {}
    x, y = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (x / y) if x and y else None
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def r33_probe(model: str, gpu, out: Path, extra: list, timeout: int, src: Path = None) -> dict:
    """The R33 proof on one model: a complete `--triton` request in-process
    under the sys.modules probe; the verdict is whether torch is in
    sys.modules at exit, with the first import path when it is."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    ext = {"llm": ".txt", "stt": ".txt", "vlm": ".txt", "audio_llm": ".txt", "tts": ".wav",
           "video": ".mp4", "image": ".png", "upscaler": ".png"}.get(fam, ".txt")
    req = request_args(model, fam, extra)
    env = {**os.environ, "PYTHONPATH": str((src or (REPO / "src")).resolve())}
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    outp = d / f"probe{ext}"
    log = d / "probe.log"
    probe = str((src.parent if src else REPO) / "tools" / "r33_sys_modules_probe.py")
    if not Path(probe).exists():
        probe = str(REPO / "tools" / "r33_sys_modules_probe.py")
    rc, wall = run([PY, probe, "--triton", "--model", model] + req + ["--output", str(outp)], env, log, timeout)
    text = log.read_text(errors="replace")
    m = re.search(r"torch in sys\.modules at exit: (True|False)", text)
    torch_at_exit = None if m is None else (m.group(1) == "True")
    m = re.search(r"the run exited (-?\d+)", text)
    run_rc = int(m.group(1)) if m else (0 if torch_at_exit is not None else rc)
    site = None
    blk = re.search(r"the stack that requested it:(.*?)\[R33 probe\]", text, re.S)
    if blk:
        frames = re.findall(r'File "[^"]*?/src/neurobrix/([^"]+)", line (\d+)', blk.group(1))
        if frames:
            site = f"{frames[-1][0]}:{frames[-1][1]}"
    sha = None
    if outp.exists():
        import hashlib
        sha = hashlib.sha256(outp.read_bytes()).hexdigest()[:12]
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "r33",
           "src": str(src) if src else str(REPO / "src"),
           "torch_at_exit": torch_at_exit, "run_rc": run_rc, "probe_rc": rc, "wall_s": wall,
           "first_import_site": site, "output_sha": sha, "exec_s": exec_time(log),
           "A": {"rc": run_rc, "exec_s": exec_time(log)}, "B": {"rc": run_rc, "exec_s": None},
           "gate": {"kind": "r33", "pass": (torch_at_exit is False and run_rc == 0)}}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def verdict(r: dict) -> str:
    if r.get("lever") == "r33":
        if r.get("run_rc"):
            return f"FAILED (the run exited {r['run_rc']})" + (f"; torch via {r['first_import_site']}" if r.get("torch_at_exit") else "")
        if r.get("torch_at_exit") is False:
            return "NO TORCH"
        if r.get("torch_at_exit") is True:
            return f"TORCH ({r.get('first_import_site') or '?'})"
        return "FAILED (no verdict in the log)"
    if r.get("lever") == "launcher":
        if r["A"]["rc"] or r["B"]["rc"]:
            return "FAILED (a --triton arm did not run)"
        return "IDENTICAL" if r["gate"].get("identical") else "DIFFERENT"
    if r.get("error") == "calibrate failed":
        try:
            log = (OUT_DEFAULT / r["model"] / "calibrate.log").read_text(errors="replace")
            if "UNSUPPORTED PATH" in log and "encoding" in log:
                r["error"] = "triton-only build"
        except OSError:
            pass
    if r.get("error") == "triton-only build":
        return "N/A (triton-only build: the census needs the compiled reference)"
    if r.get("error") == "calibrate failed":
        return "FAILED (calibrate)"
    if r["A"]["rc"] or r["B"]["rc"] or not r.get("speedup"):
        return "FAILED"
    if r.get("stochastic_reference"):
        return "UNGATED (A differs run to run)"   # the request draws its own randomness: no byte gate
    if not r["gate"].get("pass"):
        return "REGRESSED (gate)"
    s = r["speedup"]
    if s >= 1.05:
        return "won"
    if s <= 0.95:
        return "REGRESSED (slower)"
    return "no move"


def table(out: Path) -> str:
    rows = []
    results = [json.loads(rj.read_text()) for rj in sorted(out.glob("*/result.json"))]
    if results and all(r.get("lever") == "r33" for r in results):
        head = ("| model | family | weights, config | run | torch in sys.modules at exit | first import path | exec (s) | output sha | verdict |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            e = r.get("exec_s")
            rows.append(f"| {r['model']} | {r['family']} | {r.get('weight_gb', '?')} GB, {r.get('config', '?')} | "
                        f"{'ok' if not r.get('run_rc') else 'exit ' + str(r.get('run_rc'))} | "
                        f"{'NO' if r.get('torch_at_exit') is False else ('YES' if r.get('torch_at_exit') else '?')} | "
                        f"{r.get('first_import_site') or '—'} | {e if e is None else f'{e:.2f}'} | {r.get('output_sha') or '—'} | {verdict(r)} |")
        return head + "\n".join(rows) + "\n"
    for r in results:
        isl = ", ".join(f"{c}:{v.get('islands')}" for c, v in (r.get("islands") or {}).items()) or "—"
        g = r.get("gate") or {}
        if g.get("kind") == "image":
            gs = f"{g.get('psnr_db', 0):.1f} dB / {g.get('ssim', 0):.3f}"
        elif g.get("kind") == "text":
            gs = "identical" if g.get("identical") else f"diff @{g.get('first_diff_at')}"
        elif g.get("kind") == "audio":
            gs = "identical" if g.get("identical") else f"SNR {g.get('snr_db', 0):.1f} dB"
        else:
            gs = "identical" if g.get("identical") else g.get("kind", "?")
        a, b = r["A"].get("exec_s"), r["B"].get("exec_s")
        rows.append(f"| {r['model']} | {r['family']} | {r.get('weight_gb', '?')} GB, {r.get('config', '?')} | {isl} | {a if a is None else f'{a:.2f}'} | "
                    f"{b if b is None else f'{b:.2f}'} | {('×%.2f' % r['speedup']) if r.get('speedup') else '—'} | {gs} | {verdict(r)} |")
    head = ("| model | family | weights, config | islands per component | A conservative (cold s) | B calibrated (cold s) | A/B | gate vs A | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family")
    r.add_argument("--models")
    r.add_argument("--gpu", type=int, default=None, help="one pinned card; omit for the whole rig (--machine)")
    r.add_argument("--machine", action="store_true", help="whole rig visible, Prism free (models that do not fit one card)")
    r.add_argument("--max-weight-gb", type=float, default=None,
                   help="pinned stage: skip (and list) models whose weights exceed this — they belong to the machine stage")
    r.add_argument("--min-weight-gb", type=float, default=None, help="machine stage: only models above this")
    r.add_argument("--out", default=str(OUT_DEFAULT))
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--extra", default="", help="extra request args, space separated (e.g. '--num-frames 9')")
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--probe", action="store_true",
                   help="the R33 lever: one complete --triton request per model under the sys.modules probe")
    r.add_argument("--src", default=None, help="a frozen worktree's src for the probe's PYTHONPATH (default: this repo)")
    r.add_argument("--launcher-ab", action="store_true",
                   help="the launcher gate instead of the precision lever: --triton with upstream's launcher vs NeuroBrix's, bytes compared")
    t = sub.add_parser("table")
    t.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    if args.cmd == "table":
        print(table(Path(args.out)))
        return 0
    out = Path(args.out)
    if args.models:
        models = [m for m in args.models.split(",") if m]
    else:
        models = sorted(m.name for m in CACHE.iterdir() if (m / "manifest.json").exists()
                        and family_of(m.name) == args.family)
    extra = args.extra.split() if args.extra else []
    if not args.machine and args.gpu is None:
        ap.error("--gpu <n> for the pinned stage, or --machine for the whole rig")
    gpu = None if args.machine else args.gpu
    for m in models:
        if args.skip_done and (out / m / "result.json").exists():
            print(f"[zoo] {m}: done, skipped"); continue
        lock = out / m / ".running"
        if lock.exists():
            print(f"[zoo] {m}: running elsewhere ({lock.read_text().strip()}), skipped"); continue
        wgb = weight_gb(m)
        if args.max_weight_gb is not None and wgb > args.max_weight_gb:
            print(f"[zoo] {m}: {wgb:.1f} GB of weights > {args.max_weight_gb} GB — machine stage", flush=True); continue
        if args.min_weight_gb is not None and wgb < args.min_weight_gb:
            print(f"[zoo] {m}: {wgb:.1f} GB of weights < {args.min_weight_gb} GB — pinned stage", flush=True); continue
        print(f"[zoo] {m} ({family_of(m)}, {wgb:.1f} GB) on {'the whole rig' if gpu is None else f'GPU {gpu}'} …", flush=True)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"gpu={gpu} pid={os.getpid()} {time.strftime('%H:%M:%S')}\n")
        try:
            if args.probe:
                res = r33_probe(m, gpu, out, extra, args.timeout, Path(args.src) if args.src else None)
                print(f"[zoo] {m}: {verdict(res)} exec={res.get('exec_s')} sha={res.get('output_sha')}", flush=True)
            else:
                res = (launcher_ab if args.launcher_ab else one_model)(m, gpu, out, extra, args.timeout)
                print(f"[zoo] {m}: {verdict(res)} A={res['A']['exec_s']} B={res['B']['exec_s']} gate={res['gate']}", flush=True)
        except Exception as e:  # one model's failure never stops the campaign
            print(f"[zoo] {m}: ERROR {type(e).__name__}: {e}", flush=True)
            (out / m).mkdir(parents=True, exist_ok=True)
            (out / m / "result.json").write_text(json.dumps({"model": m, "family": family_of(m), "error": str(e),
                                                              "A": {"rc": 1}, "B": {"rc": 1}, "gate": {}}, indent=1))
        finally:
            lock.unlink(missing_ok=True)
    print(table(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
