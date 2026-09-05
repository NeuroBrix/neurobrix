#!/usr/bin/env python3
"""Whole-zoo measurement of one engine lever: the DtypeEngine calibration.

For every cached model of a family (or a list), on one pinned GPU:
  1. `neurobrix calibrate` — the record (one request on the conservative path);
  2. arm A — the same request on the conservative path (`NBX_ACTIVATIONS_FP16_SAFE=0`);
  3. arm B — the same request on the calibration record (islands dumped);
  4. the quality gate by output kind — text: byte identity (first differing
     character otherwise); image: PSNR / SSIM / moved-pixel fraction against A
     (tools/image_fidelity.py); audio: SNR of B against A; video: sha only;
  5. the cold execute time of both arms (`[Timing] Total execution`).
Artefacts per model under <out>/<model>/ (R29); `table` renders the per-lever
table: who won, by how much, who did not move, who regressed.

    python tools/precision_zoo_campaign.py run --family stt --gpu 1
    python tools/precision_zoo_campaign.py run --models TinyLlama-1.1B-Chat --gpu 1
    python tools/precision_zoo_campaign.py table
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
_TEXT_BOUND = {"llm": ["--max-tokens", "64", "--temperature", "0"],
               "vlm": ["--max-tokens", "64", "--temperature", "0"],
               "audio_llm": ["--max-tokens", "64", "--temperature", "0"],
               "multimodal": ["--max-tokens", "64", "--temperature", "0"]}


def manifest(model: str) -> dict:
    return json.loads((CACHE / model / "manifest.json").read_text())


def family_of(model: str) -> str:
    return manifest(model)["family"]


def request_args(model: str, family: str, extra: list) -> list:
    args = list(_MEDIA.get(family, [])) + list(_TEXT_BOUND.get(family, []))
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


def one_model(model: str, gpu: int, out: Path, extra: list, timeout: int) -> dict:
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    ext = {"llm": ".txt", "stt": ".txt", "vlm": ".txt", "audio_llm": ".txt", "tts": ".wav",
           "video": ".mp4", "image": ".png", "upscaler": ".png"}.get(fam, ".txt")
    req = request_args(model, fam, extra)
    if fam == "multimodal" and "--mode" in req and req[req.index("--mode") + 1] == "image":
        ext = ".png"
    base_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    res = {"model": model, "family": fam, "model_name": manifest(model).get("model_name"), "request": req}
    rc, wall = run([NBX, "calibrate", "--model", model] + req, base_env, d / "calibrate.log", timeout)
    res["calibrate"] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / "calibrate.log")}
    for arm, env in (("A", {**base_env, "NBX_ACTIVATIONS_FP16_SAFE": "0"}),
                     ("B", {**base_env, "NBX_PRECISION_ISLANDS": str(d / "islands.tsv")})):
        for k in ("NBX_ACTIVATIONS_FP16_SAFE", "NBX_PRECISION_ISLANDS"):
            env.pop(k, None) if k not in env or env.get(k) is None else None
        if arm == "B":
            (d / "islands.tsv").unlink(missing_ok=True)
        outp = d / f"{arm}{ext}"
        rc, wall = run([NBX, "run", "--model", model] + req + ["--output", str(outp)], env, d / f"{arm}.log", timeout)
        res[arm] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / f"{arm}.log"), "output": str(outp)}
    res["islands"] = islands_from_log(d / "B.log")
    res["gate"] = gate(d / f"A{ext}", d / f"B{ext}")
    a, b = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (a / b) if a and b else None
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def verdict(r: dict) -> str:
    if r["A"]["rc"] or r["B"]["rc"] or not r.get("speedup"):
        return "FAILED"
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
    for rj in sorted(out.glob("*/result.json")):
        r = json.loads(rj.read_text())
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
        rows.append(f"| {r['model']} | {r['family']} | {isl} | {a if a is None else f'{a:.2f}'} | "
                    f"{b if b is None else f'{b:.2f}'} | {('×%.2f' % r['speedup']) if r.get('speedup') else '—'} | {gs} | {verdict(r)} |")
    head = ("| model | family | islands per component | A conservative (cold s) | B calibrated (cold s) | B/A | gate vs A | verdict |\n"
            "|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family")
    r.add_argument("--models")
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--out", default=str(OUT_DEFAULT))
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--extra", default="", help="extra request args, space separated (e.g. '--num-frames 9')")
    r.add_argument("--skip-done", action="store_true")
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
    for m in models:
        if args.skip_done and (out / m / "result.json").exists():
            print(f"[zoo] {m}: done, skipped"); continue
        print(f"[zoo] {m} ({family_of(m)}) on GPU {args.gpu} …", flush=True)
        try:
            res = one_model(m, args.gpu, out, extra, args.timeout)
            print(f"[zoo] {m}: {verdict(res)} A={res['A']['exec_s']} B={res['B']['exec_s']} gate={res['gate']}", flush=True)
        except Exception as e:  # one model's failure never stops the campaign
            print(f"[zoo] {m}: ERROR {type(e).__name__}: {e}", flush=True)
            (out / m).mkdir(parents=True, exist_ok=True)
            (out / m / "result.json").write_text(json.dumps({"model": m, "family": family_of(m), "error": str(e),
                                                              "A": {"rc": 1}, "B": {"rc": 1}, "gate": {}}, indent=1))
    print(table(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
