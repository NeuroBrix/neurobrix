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
import hashlib
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


def _declares_image_input(model: str) -> bool:
    """A container whose topology names `global.image` takes an image (TI2V,
    I2V) — the campaign feeds the asset image, data-driven, never by family."""
    try:
        return '"global.image"' in (CACHE / model / "topology.json").read_text()
    except OSError:
        return False


def request_args(model: str, family: str, extra: list) -> list:
    args = family_stimulus(family) + list(_MEDIA.get(family, []))
    if "--input-image" not in args and family in ("video", "image") and _declares_image_input(model):
        args += ["--input-image", str(ASSETS / "apple_448.png")]
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


def _log_mel(x, sr, n_fft=1024, hop=256, n_mels=80):
    """Log-mel spectrogram (numpy): the perceptual frame of the audio gate."""
    import numpy as np
    win = np.hanning(n_fft)
    frames = [np.abs(np.fft.rfft(x[i:i + n_fft] * win)) ** 2 for i in range(0, max(len(x) - n_fft, 1), hop)]
    power = np.asarray(frames, dtype=np.float64)          # [T, F]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    edges = np.linspace(mel(0.0), mel(sr / 2.0), n_mels + 2)
    hz = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9); dn = (hi - freqs) / max(hi - ce, 1e-9)
        fb[m] = np.clip(np.minimum(up, dn), 0.0, None)
    return np.log10(power @ fb.T + 1e-10)


def audio_gate(a: Path, b: Path) -> dict:
    """The audio gate, in two frames a sample SNR cannot see (D-ZOO-AUDIO-GATE,
    2026-09-05): (1) a LOG-MEL distance — a phase drift of the same speech
    (Kokoro: SNR 2.4 dB, mel distance 0.017) stays close, a broken render does
    not; (2) for an autoregressive synthesis whose sampled token path moves, a
    TRANSCRIPT comparison through the engine's own speech recognizer
    (whisper-large-v3-turbo, compiled) when the mel frame disagrees: the same
    words = the same speech. The bar: mel distance <= 0.05 (a one-frame shift
    of the same signal scores 0.057), else transcript WER <= 0.10."""
    import numpy as np
    import soundfile as sf
    xa, sra = sf.read(str(a)); xb, srb = sf.read(str(b))
    xa = np.asarray(xa, dtype=np.float64); xb = np.asarray(xb, dtype=np.float64)
    if xa.ndim > 1: xa = xa.mean(axis=1)
    if xb.ndim > 1: xb = xb.mean(axis=1)
    n = min(len(xa), len(xb))
    noise = float(np.sum((xa[:n] - xb[:n]) ** 2)); sig = float(np.sum(xa[:n] ** 2))
    snr = float("inf") if noise == 0 else (10 * np.log10(sig / noise) if sig > 0 else float("nan"))
    out = {"kind": "audio", "identical": noise == 0, "snr_db": snr, "len_a": len(xa), "len_b": len(xb), "sr": sra}
    if noise == 0:
        out["pass"] = True
        return out
    ma, mb = _log_mel(xa, sra), _log_mel(xb, srb)
    m = min(len(ma), len(mb))
    mel_dist = float(np.abs(ma[:m] - mb[:m]).mean()) if m else float("inf")
    length_ratio = min(len(xa), len(xb)) / max(len(xa), len(xb), 1)
    out.update({"mel_distance": mel_dist, "length_ratio": length_ratio})
    if mel_dist <= 0.05 and length_ratio >= 0.9:
        out["pass"] = True
        return out
    # The token path moved: the same words are the gate.
    try:
        ta, tb = _transcribe(a), _transcribe(b)
        wer = _wer(ta, tb)
        out.update({"transcript_a": ta, "transcript_b": tb, "wer": wer, "pass": wer <= 0.10})
    except Exception as e:  # noqa: BLE001 — the gate says why it could not judge
        out.update({"pass": False, "error": f"transcribe: {e}"[:300]})
    return out


def _transcribe(path: Path) -> str:
    """The engine's own recognizer, compiled path, on a pinned card."""
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="nbx_gate_")) / "t.txt"
    env = {**os.environ}
    env.pop("NBX_ACTIVATIONS_FP16_SAFE", None)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("NBX_GATE_GPU", "1")   # never the condemned card
    r = subprocess.run([NBX, "run", "--model", "whisper-large-v3-turbo", "--audio", str(path), "--output", str(out)],
                       env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"whisper exit {r.returncode}: {r.stdout[-200:]} {r.stderr[-200:]}")
    return out.read_text().strip().lower()


def _wer(ref: str, hyp: str) -> float:
    import re as _re
    r = _re.findall(r"[a-z0-9']+", ref.lower()); h = _re.findall(r"[a-z0-9']+", hyp.lower())
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def gate(a: Path, b: Path) -> dict:
    if not a.exists() or not b.exists():
        return {"kind": "missing", "pass": False}
    ext = a.suffix.lower()
    if ext == ".mp4":
        # Frame fidelity: both clips decoded (imageio-ffmpeg), the mean and
        # the worst per-frame PSNR of B against A over the common length; the
        # same 30 dB bar as the image gate; sha kept for the record.
        import hashlib
        import numpy as np
        try:
            import imageio.v3 as iio
            fa = np.asarray(list(iio.imiter(str(a))))
            fb = np.asarray(list(iio.imiter(str(b))))
        except Exception as e:  # noqa: BLE001 — the gate says why it could not read
            return {"kind": "video", "pass": False, "error": f"decode: {e}"[:300],
                    "sha_a": hashlib.sha256(a.read_bytes()).hexdigest()[:12],
                    "sha_b": hashlib.sha256(b.read_bytes()).hexdigest()[:12]}
        n = min(len(fa), len(fb))
        if n == 0 or fa.shape[1:] != fb.shape[1:]:
            return {"kind": "video", "pass": False, "error": f"frames {fa.shape} vs {fb.shape}"}
        psnr = []
        for i in range(n):
            x, y = fa[i].astype(np.float64), fb[i].astype(np.float64)
            mse = float(np.mean((x - y) ** 2))
            psnr.append(float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        finite = [v for v in psnr if v != float("inf")]
        mean_psnr = float(np.mean(psnr)) if finite else float("inf")
        min_psnr = float(min(psnr))
        identical = a.read_bytes() == b.read_bytes()
        return {"kind": "video", "pass": identical or min_psnr >= 30.0, "identical": identical,
                "frames_a": int(len(fa)), "frames_b": int(len(fb)), "psnr_mean_db": mean_psnr,
                "psnr_min_db": min_psnr, "sha_a": hashlib.sha256(a.read_bytes()).hexdigest()[:12],
                "sha_b": hashlib.sha256(b.read_bytes()).hexdigest()[:12]}
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
        return audio_gate(a, b)
    import hashlib
    ha, hb = hashlib.sha256(a.read_bytes()).hexdigest()[:12], hashlib.sha256(b.read_bytes()).hexdigest()[:12]
    return {"kind": ext.lstrip("."), "identical": ha == hb, "pass": ha == hb, "sha_a": ha, "sha_b": hb}


def one_model(model: str, gpu, out: Path, extra: list, timeout: int) -> dict:
    """gpu=None → the whole rig visible (Prism free): the stage for models
    whose weights do not fit one card; an int → one pinned card."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra)
    ext = output_ext(fam, req)
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


def output_ext(fam: str, req: list) -> str:
    """The output extension the engine's output dispatch expects for this
    request: by family, and by the request's `--mode` for a multimodal model
    (an autoregressive-image request writes a PNG, a text request a TXT)."""
    if fam == "multimodal" and "--mode" in req:
        return ".png" if req[req.index("--mode") + 1] == "image" else ".txt"
    return {"llm": ".txt", "stt": ".txt", "vlm": ".txt", "audio_llm": ".txt", "tts": ".wav",
            "video": ".mp4", "image": ".png", "upscaler": ".png"}.get(fam, ".txt")


def launcher_ab(model: str, gpu, out: Path, extra: list, timeout: int) -> dict:
    """The launcher gate on one model: `--triton` with upstream's launcher
    (NBX_LAUNCHER=triton) vs the NeuroBrix launcher, outputs byte-compared."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra) + ["--triton"]
    ext = output_ext(fam, req)
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



_EXCLUSION_LINE = re.compile(r"\[AUTOTUNE_SCREEN\].*\bexcluded\b", re.I)          # a config the screen refused
_UNSCREENED_LINE = re.compile(r"\[AUTOTUNE_SCREEN\].*\bnot screened\b|\bgo to the timer unchecked\b", re.I)
_SCREEN_SUMMARY = re.compile(r"correctness screen (on|off): checked (\d+) key\(s\), excluded (\d+) config")


def _sweep_store_entries(store: Path, model: str) -> dict:
    """The model's sweep artifact in one arm's store: {key: config}, or {}."""
    entries = {}
    for art in sorted((store / model).glob("*.json")) if (store / model).exists() else []:
        try:
            doc = json.loads(art.read_text())
        except Exception:  # noqa: BLE001
            continue
        if str(doc.get("format", "")).startswith("nbx-autotune-sweep/"):
            entries.update(doc.get("entries") or {})
    return entries


def _compare_sweep_stores(d: Path, model: str, trees: list) -> dict:
    """Per arm: the number of keys it measured, the configs the screen
    excluded (its log), and — against the first arm — whether every key
    chose the same config, with the first key that did not."""
    first = trees[0][0]
    base = _sweep_store_entries(d / f"{first}_autotune", model)
    out = {}
    for label, _ in trees:
        ent = _sweep_store_entries(d / f"{label}_autotune", model)
        log = (d / f"{label}.log").read_text(errors="replace") if (d / f"{label}.log").exists() else ""
        excluded = [ln.strip() for ln in log.splitlines() if _EXCLUSION_LINE.search(ln)]
        unscreened = [ln.strip() for ln in log.splitlines() if _UNSCREENED_LINE.search(ln)]
        rec = {"keys": len(ent), "exclusions": len(excluded), "excluded_lines": excluded[:20],
               "unscreened": len(unscreened), "unscreened_lines": unscreened[:10]}
        summ = _SCREEN_SUMMARY.findall(log)              # the activation proof: the tree has the screen and it ran
        if summ:
            rec["screen"] = summ[-1][0]
            rec["screened_keys"] = sum(int(m[1]) for m in summ)
            rec["exclusions"] = max(rec["exclusions"], sum(int(m[2]) for m in summ))
        else:
            rec["screen"] = "absent"
        if label != first:
            missing = sorted(set(base) - set(ent))
            extra = sorted(set(ent) - set(base))
            differing = sorted(k for k in set(base) & set(ent) if base[k] != ent[k])
            rec.update({"identical": not missing and not extra and not differing and bool(base),
                        "missing_keys": missing[:10], "extra_keys": extra[:10],
                        "differing_keys": differing[:10],
                        "first_diff": (differing or missing or extra or [None])[0],
                        "first_diff_configs": ({"first": base.get(differing[0]), label: ent.get(differing[0])}
                                               if differing else None)})
        out[label] = rec
    return out

def tree_ab(model: str, gpu, out: Path, extra: list, timeout: int, trees: list, sweep_arms: bool = False) -> dict:
    """The tree gate on one model: the same `--triton` request run from two
    or more frozen source trees (label=path/to/src), outputs byte-compared
    against the first tree. This is how a kernel or launcher change made for
    another backend is proven numerically inert on CUDA: byte identity
    against main on the whole zoo, or a named difference.

    Each arm runs the CLI through the tree's own `neurobrix` package
    (PYTHONPATH=<tree>/src) and records the package path the process saw,
    so a result can never be attributed to a tree that was not the one
    executed.

    `sweep_arms`: every arm sweeps its kernels cold into its OWN sweep store
    (NBX_AUTOTUNE=sweep, NEUROBRIX_AUTOTUNE_STORE=<row>/<label>_autotune), and
    the stores are compared key by key after the bytes — the proof that a
    change to the autotuner (a correctness screen before the stopwatch) leaves
    the CHOICE of every kernel config identical on CUDA, with the sweep's
    overhead measured (exec seconds per arm, both cold) and every config the
    screen excluded counted from the arm's log."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra) + ["--triton"]
    ext = output_ext(fam, req)
    base_env = {**os.environ}
    if gpu is None:
        base_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "tree",
           "trees": {label: str(src) for label, src in trees}, "arms": {}}
    entry = "import sys; from neurobrix.cli import main; sys.exit(main())"
    for label, src in trees:
        env = {**base_env, "PYTHONPATH": str(Path(src).resolve())}
        if sweep_arms:
            env["NBX_AUTOTUNE"] = "sweep"
            env["NEUROBRIX_AUTOTUNE_STORE"] = str(d / f"{label}_autotune")
        seen = subprocess.run([PY, "-c", "import neurobrix, sys; sys.stdout.write(neurobrix.__file__)"],
                              env=env, capture_output=True, text=True).stdout.strip()
        outp = d / f"{label}{ext}"
        rc, wall = run([PY, "-c", entry, "run", "--model", model] + req + ["--output", str(outp)],
                       env, d / f"{label}.log", timeout)
        res["arms"][label] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / f"{label}.log"),
                              "output": str(outp), "package_seen": seen,
                              "sha": hashlib.sha256(outp.read_bytes()).hexdigest()[:12] if outp.exists() else None}
    first = trees[0][0]
    a = d / f"{first}{ext}"
    comp = {}
    for label, _ in trees[1:]:
        b = d / f"{label}{ext}"
        same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        comp[label] = {"identical": same}
        if a.exists() and b.exists() and not same:
            try:
                comp[label]["diff"] = gate(a, b)       # how far, in the family's own measure
            except Exception as e:  # noqa: BLE001
                comp[label]["diff"] = {"error": str(e)}
    if sweep_arms:
        res["autotune"] = _compare_sweep_stores(d, model, trees)
    ran = all(v["rc"] == 0 and v["sha"] for v in res["arms"].values())
    res["gate"] = {"kind": "bytes", "against": first, "arms": comp,
                   "identical": ran and all(v["identical"] for v in comp.values()), "ran": ran}
    res["A"] = res["arms"][first]
    res["B"] = res["arms"][trees[1][0]] if len(trees) > 1 else res["arms"][first]
    x, y = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (x / y) if x and y else None
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def sweep_one(model: str, gpu, out: Path, extra: list, timeout: int) -> dict:
    """The sweep producer on one model: a `--triton --sweep` request from
    this tree; the verdict is the model's sweep artifact for this hardware
    profile (path and measured-shape count), the engine's own output line."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra) + ["--triton", "--sweep"]
    ext = output_ext(fam, req)
    env = {**os.environ}
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    outp = d / f"sweep{ext}"
    rc, wall = run([NBX, "run", "--model", model] + req + ["--output", str(outp)], env, d / "sweep.log", timeout)
    text = (d / "sweep.log").read_text(errors="replace")
    m = re.search(r"\[autotune\] \S+: sweep artifact written (\S+) \((\d+) measured shape", text)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "sweep",
           "artifact": m.group(1) if m else None, "shapes": int(m.group(2)) if m else None,
           "A": {"rc": rc, "exec_s": exec_time(d / "sweep.log"), "wall_s": wall}, "B": {"rc": rc, "exec_s": None},
           "gate": {"kind": "sweep", "pass": bool(m) and rc == 0}}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def drift_one(model: str, gpu, out: Path, extra: list, timeout: int, bound: float = 0.02) -> dict:
    """The drift-site lever on one model: `neurobrix drift` (the ATen oracle
    against the Triton engine, per op) on the family stimulus; the verdict is
    the first drifting op, or none."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra)
    env = {**os.environ}
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rc, wall = run([NBX, "drift", "--model", model, "--out", str(d), "--bound", str(bound)] + req, env, d / "drift.log", timeout)
    oracle_log = (d / "oracle.log").read_text(errors="replace") if (d / "oracle.log").exists() else ""
    triton_only = "UNSUPPORTED PATH" in oracle_log and "encoding" in oracle_log
    rep = {}
    if (d / "drift.json").exists():
        rep = json.loads((d / "drift.json").read_text())
    first = rep.get("first") or {}
    kernel = rep.get("first_same_dtype") or {}
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "drift",
           "rc": rc, "wall_s": wall, "ops": rep.get("ops_a"), "matched": rep.get("matched"),
           "missing": rep.get("missing_in_b"), "over_bound": rep.get("over_bound"), "bound": bound,
           "policy_sites": rep.get("policy_sites"),
           "site": (f"{first.get('component')}/{first.get('op_uid')}" if first else None),
           "site_type": first.get("op_type"), "site_dev": first.get("rel_dev"), "site_index": first.get("index"),
           "kernel_site": (f"{kernel.get('component')}/{kernel.get('op_uid')}" if kernel else None),
           "kernel_site_type": kernel.get("op_type"), "kernel_site_dev": kernel.get("rel_dev"),
           "kernel_site_index": kernel.get("index"),
           "A": {"rc": rc, "exec_s": None}, "B": {"rc": rc, "exec_s": None},
           "gate": {"kind": "drift", "pass": rc == 0 and not first}}
    if triton_only:
        res["error"] = "triton-only build"
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def env_ab(model: str, gpu, out: Path, extra: list, timeout: int, env_b: dict, lever: str) -> dict:
    """An engine lever behind an environment switch, measured on one model:
    arm A = the request as it is, arm B = the same request with `env_b`
    set (e.g. NBX_OPTIM_ALGEBRAIC=1), outputs byte-compared, execution
    times recorded. The engine is chosen by the request args (`--extra
    --triton` for the Triton engine)."""
    fam = family_of(model)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    req = request_args(model, fam, extra)
    ext = output_ext(fam, req)
    base_env = {**os.environ}
    for k in env_b:
        base_env.pop(k, None)
    if gpu is None:
        base_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": lever, "env_b": env_b}
    for arm, env in (("A", base_env), ("B", {**base_env, **env_b})):
        outp = d / f"{arm}{ext}"
        rc, wall = run([NBX, "run", "--model", model] + req + ["--output", str(outp)], env, d / f"{arm}.log", timeout)
        log = (d / f"{arm}.log").read_text(errors="replace")
        m = re.search(r"\[Optim\] algebraic: (\d+) identity ops aliased away", log)
        res[arm] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / f"{arm}.log"), "output": str(outp),
                    "sha": hashlib.sha256(outp.read_bytes()).hexdigest()[:12] if outp.exists() else None,
                    "ops_removed": sum(int(x) for x in re.findall(r"\[Optim\] algebraic: (\d+) identity ops", log)) or None}
    a, b = d / f"A{ext}", d / f"B{ext}"
    same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
    res["gate"] = {"kind": "bytes", "identical": same, "pass": same,
                   "ran": res["A"]["rc"] == 0 and res["B"]["rc"] == 0}
    if a.exists() and b.exists() and not same:
        try:
            res["gate"]["diff"] = gate(a, b)
        except Exception as e:  # noqa: BLE001
            res["gate"]["diff"] = {"error": str(e)}
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
    req = request_args(model, fam, extra)
    ext = output_ext(fam, req)
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
    if r.get("lever") == "drift":
        if r.get("error") == "triton-only build":
            return "N/A (triton-only build: no ATen oracle for this container)"
        if r.get("rc"):
            return f"FAILED (drift exited {r['rc']})"
        if r.get("site"):
            where = (f"kernel site {r['kernel_site']} ({r.get('kernel_site_type')}, {r.get('kernel_site_dev', 0):.3f}, op #{r.get('kernel_site_index')})"
                     if r.get("kernel_site") else f"no kernel site: policy only ({r.get('policy_sites')} dtype-policy sites)")
            return f"DRIFT first at {r['site']} ({r.get('site_type')}, {r.get('site_dev', 0):.3f}, op #{r.get('site_index')}; {r.get('over_bound')} over) — {where}"
        return f"NO DRIFT ({r.get('matched')} ops within {r.get('bound')})"
    if r.get("lever") == "sweep":
        if r.get("A", {}).get("rc"):
            return f"FAILED (the run exited {r['A']['rc']})"
        return f"SWEPT ({r.get('shapes')} shapes)" if r.get("artifact") else "FAILED (no artifact written)"
    if r.get("lever") == "tree":
        g = r.get("gate") or {}
        if not g.get("ran"):
            failed = [k for k, v in (r.get("arms") or {}).items() if v.get("rc") != 0 or not v.get("sha")]
            return f"FAILED (arm {', '.join(failed) or '?'} did not run)"
        at = r.get("autotune") or {}
        tail = ""
        if at:
            first = g.get("against")
            others = {k: v for k, v in at.items() if k != first}
            if all(v.get("identical") for v in others.values()) and others:
                tail = f"; choices identical ({at.get(first, {}).get('keys', 0)} keys)"
            else:
                bad = [f"{k} first differs at {v.get('first_diff')}" for k, v in others.items() if not v.get("identical")]
                tail = "; CHOICES DIFFER (" + "; ".join(bad) + ")"
            excl = {k: v.get("exclusions", 0) for k, v in at.items() if v.get("exclusions")}
            if excl:
                tail += "; SCREEN EXCLUDED " + ", ".join(f"{k}: {n} config(s)" for k, n in excl.items()) + " — a finding to close"
        if g.get("identical"):
            return "IDENTICAL" + tail
        diff = [k for k, v in g.get("arms", {}).items() if not v.get("identical")]
        return "DIFFERENT (" + ", ".join(diff) + ")" + tail
    if r.get("lever", "").startswith("env:"):
        g = r.get("gate") or {}
        if not g.get("ran"):
            return "FAILED (an arm did not run)"
        n = (r.get("B") or {}).get("ops_removed")
        tag = f", {n} ops removed" if n else ""
        return ("IDENTICAL" if g.get("identical") else "DIFFERENT") + tag
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
    if results and all(r.get("lever") == "sweep" for r in results):
        head = ("| model | family | weights, config | exec (s) | measured shapes | artifact | verdict |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in results:
            e = (r.get("A") or {}).get("exec_s")
            rows.append(f"| {r['model']} | {r['family']} | {r.get('weight_gb', '?')} GB, {r.get('config', '?')} | "
                        f"{e if e is None else f'{e:.2f}'} | {r.get('shapes') if r.get('shapes') is not None else '—'} | "
                        f"{r.get('artifact') or '—'} | {verdict(r)} |")
        return head + "\n".join(rows) + "\n"
    if results and all(r.get("lever") == "drift" for r in results):
        head = ("| model | family | weights, config | oracle ops | matched | missing on Triton | over bound (policy) | first site | first KERNEL site | verdict |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            ks = r.get("kernel_site")
            kcell = f"{ks} ({r.get('kernel_site_type')}, {r.get('kernel_site_dev', 0):.3f}, op #{r.get('kernel_site_index')})" if ks else ("none" if r.get("site") else "—")
            rows.append(f"| {r['model']} | {r['family']} | {r.get('weight_gb', '?')} GB, {r.get('config', '?')} | "
                        f"{r.get('ops') if r.get('ops') is not None else '—'} | {r.get('matched') if r.get('matched') is not None else '—'} | "
                        f"{r.get('missing') if r.get('missing') is not None else '—'} | "
                        f"{r.get('over_bound') if r.get('over_bound') is not None else '—'} ({r.get('policy_sites') if r.get('policy_sites') is not None else '?'}) | "
                        f"{r.get('site') or '—'} | {kcell} | {verdict(r)} |")
        return head + "\n".join(rows) + "\n"
    if results and all(r.get("lever") == "tree" for r in results):
        labels = []
        for r in results:
            for k in (r.get("arms") or {}):
                if k not in labels:
                    labels.append(k)
        with_at = any(r.get("autotune") for r in results)
        at_head = " | autotune keys / choices vs first / screen exclusions (per arm) | sweep overhead (s, arm − first)" if with_at else ""
        head = ("| model | family | weights, config | " + " | ".join(f"{l} exec (s) / sha" for l in labels) +
                " | gate vs " + (labels[0] if labels else "?") + at_head + " | verdict |\n|---|---|---|" + "---|" * len(labels)
                + "---|" + ("---|---|" if with_at else "") + "---|\n")
        for r in results:
            cells = []
            for l in labels:
                v = (r.get("arms") or {}).get(l) or {}
                e = v.get("exec_s")
                cells.append(f"{e if e is None else f'{e:.2f}'} / {v.get('sha') or '—'}")
            if with_at:
                at = r.get("autotune") or {}
                parts = []
                for l in labels:
                    a = at.get(l)
                    if not a:
                        parts.append(f"{l}: —"); continue
                    ch = "" if "identical" not in a else (" / identical" if a["identical"] else f" / DIFFER at {a.get('first_diff')}")
                    sc = a.get("screen", "absent")
                    sc = f"screen {sc}" + (f" ({a['screened_keys']} checked)" if a.get("screened_keys") is not None else "")
                    parts.append(f"{l}: {a.get('keys', 0)} keys{ch} / {sc} / {a.get('exclusions', 0)} excluded"
                                 + (f" / {a['unscreened']} not screened" if a.get("unscreened") else ""))
                cells.append("; ".join(parts))
                e0 = ((r.get("arms") or {}).get(labels[0]) or {}).get("exec_s")
                ov = []
                for l in labels[1:]:
                    e1 = ((r.get("arms") or {}).get(l) or {}).get("exec_s")
                    ov.append(f"{l}: {e1 - e0:+.2f}" if e0 is not None and e1 is not None else f"{l}: —")
                cells.append("; ".join(ov) or "—")
            g = (r.get("gate") or {}).get("arms") or {}
            gs = []
            for l, c in g.items():
                if c.get("identical"):
                    gs.append(f"{l}: identical")
                else:
                    dd = c.get("diff") or {}
                    if dd.get("kind") == "image":
                        gs.append(f"{l}: {dd.get('psnr_db', 0):.1f} dB")
                    elif dd.get("kind") == "text":
                        gs.append(f"{l}: diff @{dd.get('first_diff_at')}")
                    elif dd.get("kind") == "audio":
                        gs.append(f"{l}: mel {dd.get('mel_distance', 0):.3f}" + (f", WER {dd['wer']:.2f}" if "wer" in dd else ""))
                    elif dd.get("kind") == "video":
                        gs.append(f"{l}: {dd.get('psnr_min_db', 0):.1f} dB min" if "psnr_min_db" in dd else f"{l}: {dd.get('error', '?')}")
                    else:
                        gs.append(f"{l}: different")
            rows.append(f"| {r['model']} | {r['family']} | {r.get('weight_gb', '?')} GB, {r.get('config', '?')} | " +
                        " | ".join(cells) + f" | {'; '.join(gs) or '—'} | {verdict(r)} |")
        return head + "\n".join(rows) + "\n"
    for r in results:
        isl = ", ".join(f"{c}:{v.get('islands')}" for c, v in (r.get("islands") or {}).items()) or "—"
        g = r.get("gate") or {}
        if g.get("kind") == "image":
            gs = f"{g.get('psnr_db', 0):.1f} dB / {g.get('ssim', 0):.3f}"
        elif g.get("kind") == "text":
            gs = "identical" if g.get("identical") else f"diff @{g.get('first_diff_at')}"
        elif g.get("kind") == "audio":
            if g.get("identical"):
                gs = "identical"
            elif "wer" in g:
                gs = f"mel {g.get('mel_distance', 0):.3f}, WER {g['wer']:.2f}"
            elif "mel_distance" in g:
                gs = f"mel {g['mel_distance']:.3f}"
            else:
                gs = f"SNR {g.get('snr_db', 0):.1f} dB"
        elif g.get("kind") == "video":
            gs = ("identical" if g.get("identical") else
                  (f"{g.get('psnr_min_db', 0):.1f} dB min / {g.get('psnr_mean_db', 0):.1f} dB mean, {g.get('frames_b')} frames"
                   if "psnr_min_db" in g else g.get("error", "?")))
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
    r.add_argument("--gpu", default=None, help="one pinned card, or a comma list of cards (e.g. 2,3) Prism may spread over; "
                                                "omit for the whole rig (--machine)")
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
    r.add_argument("--trees", default=None,
                   help="tree gate: label=path/to/src,label=path/to/src[,...] — the same --triton request from each "
                        "frozen tree, bytes compared against the first (a port's kernel change must be inert on CUDA)")
    r.add_argument("--sweep-arms", action="store_true",
                   help="with --trees: every arm sweeps cold into its own store (NBX_AUTOTUNE=sweep) and the chosen "
                        "config per kernel key is compared across arms, the sweep's overhead measured per arm, and the "
                        "configs a correctness screen excluded counted from each arm's log")
    r.add_argument("--env-ab", default=None, metavar="KEY=VALUE[,KEY=VALUE]",
                   help="an engine lever behind an environment switch: arm A without, arm B with it, bytes compared")
    r.add_argument("--drift", action="store_true",
                   help="drift-site lever: `neurobrix drift` per model (ATen oracle vs Triton engine, per op); "
                        "the verdict is the first drifting op")
    r.add_argument("--drift-bound", type=float, default=0.02)
    r.add_argument("--sweep", action="store_true",
                   help="sweep producer: a --triton --sweep request per model from this tree; the verdict is the "
                        "model's sweep artifact for this hardware profile")
    r.add_argument("--launcher-ab", action="store_true",
                   help="the launcher gate instead of the precision lever: --triton with upstream's launcher vs NeuroBrix's, bytes compared")
    t = sub.add_parser("table")
    t.add_argument("--out", default=str(OUT_DEFAULT))
    g = sub.add_parser("regate", help="re-run the quality gate on the existing A/B outputs of a model (no GPU)")
    g.add_argument("--models", required=True)
    g.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    if args.cmd == "table":
        print(table(Path(args.out)))
        return 0
    if args.cmd == "regate":
        out = Path(args.out)
        for m in args.models.split(","):
            rj = out / m / "result.json"
            r = json.loads(rj.read_text())
            a = Path(r["A"]["output"]); b = Path(r["B"]["output"])
            r["gate"] = gate(a, b)
            rj.write_text(json.dumps(r, indent=1))
            print(f"[zoo] {m}: {verdict(r)} gate={r['gate']}")
        return 0
    out = Path(args.out)
    if args.models:
        models = [m for m in args.models.split(",") if m]
    else:
        models = sorted(m.name for m in CACHE.iterdir() if (m / "manifest.json").exists()
                        and family_of(m.name) == args.family)
    import shlex
    extra = shlex.split(args.extra) if args.extra else []      # quotes honoured: --extra '--prompt "a red fox"'
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
            elif args.env_ab:
                env_b = dict(kv.split("=", 1) for kv in args.env_ab.split(",") if "=" in kv)
                res = env_ab(m, gpu, out, extra, args.timeout, env_b, "env:" + ",".join(env_b))
                print(f"[zoo] {m}: {verdict(res)} A={res['A']['exec_s']} B={res['B']['exec_s']} "
                      f"{('×%.2f' % res['speedup']) if res.get('speedup') else ''}", flush=True)
                continue
            elif args.drift:
                res = drift_one(m, gpu, out, extra, args.timeout, args.drift_bound)
                print(f"[zoo] {m}: {verdict(res)}", flush=True)
                continue
            elif args.sweep:
                res = sweep_one(m, gpu, out, extra, args.timeout)
                print(f"[zoo] {m}: {verdict(res)} exec={res['A'].get('exec_s')} artifact={res.get('artifact')}", flush=True)
                continue
            elif args.trees:
                trees = [(t.split("=", 1)[0], Path(t.split("=", 1)[1])) for t in args.trees.split(",") if t]
                res = tree_ab(m, gpu, out, extra, args.timeout, trees, sweep_arms=args.sweep_arms)
                print(f"[zoo] {m}: {verdict(res)} " + " ".join(f"{k}={v.get('exec_s')}/{v.get('sha')}" for k, v in res["arms"].items()), flush=True)
                continue
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
