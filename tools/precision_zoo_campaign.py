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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NBX = "/home/mlops/ml/venv/bin/neurobrix"


from rig_devices import visible_card, gate_card   # the one brick, not a third copy


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
# The campaign's bound on a request whose length the vendor's default leaves open: a decode
# loop's token budget, a denoiser's step count. The comparison is between two arms on the SAME
# request — the vendor's aesthetic default buys nothing and can make a row ungateable: Allegro's
# 100 steps at 3.5 min each are 5.8 h for ONE arm, 23 h for the four a retrace gate needs, and
# the row timed out at 7200 s twice (2026-09-07/08). Four steps exercise the same graph, every op
# of the loop, in a gateable hour. A family stimulus value wins over the bound (loop below), so a
# family whose YAML names the flag keeps its own.
# audio_llm: the model's own sampling contract (an explicit --temperature turns
# the vendor's top_k into an "explicit" parameter the path refuses).
_REQUEST_BOUND = {"llm": ["--max-tokens", "64", "--temperature", "0"],
                  "vlm": ["--max-tokens", "64", "--temperature", "0"],
                  "audio_llm": ["--max-tokens", "64"],
                  "multimodal": ["--max-tokens", "64", "--temperature", "0"],
                  "video": ["--steps", "4"],
                  # image: the two arms of a retrace gate must render the SAME number of steps.
                  # A container declares its own in `flow.generation`, and a retrace that ADDS
                  # that declaration changes the length under the comparison: Flex.1-alpha's old
                  # container rendered 20 (the runtime's default, nothing declared) against the
                  # new container's 25, and the gate read 17.2 dB — two different renders, not a
                  # graph difference (2026-09-08). Pinned here for both arms and for the vendor
                  # render; 20 is what the image rows already ran.
                  "image": ["--steps", "20"]}


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
    bound = list(_REQUEST_BOUND.get(family, []))
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


def run_group(cmd, env, fh, timeout: int, cwd=None) -> int:
    """The command in its own process group; a timeout kills the WHOLE group — the
    command's children too. A `drift` that timed out at 7200 s on 2026-09-07 left its
    child `run` alive on GPU3 beside the zoo's next model for ten minutes: only the
    direct child was killed. -9 names a timeout."""
    import os
    import signal
    p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
                         start_new_session=True)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        return -9


def run(cmd, env, log: Path, timeout: int) -> tuple:
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = run_group(cmd, env, fh, timeout)
        if rc == -9:
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
    env["CUDA_VISIBLE_DEVICES"] = gate_card()   # never the condemned card, and never outside a pin
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
        base_env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
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
        base_env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
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

    # Last word on the ratio: a lever arm that exercised nothing has not
    # measured, so the cell carries NO ratio rather than a ratio of one
    # (vacuous_lever_reason). Inert where the lever has no such telemetry.
    _vacuous = vacuous_lever_reason(res)
    if _vacuous:
        res["lever_vacuous"] = _vacuous
        res["speedup"] = None
        print(f"[zoo] {model}: LEVER NOT MEASURED — {_vacuous}", flush=True)

    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res



_EXCLUSION_LINE = re.compile(r"\[AUTOTUNE_SCREEN\].*\bexcluded\b", re.I)          # a config the screen refused
_UNSCREENED_LINE = re.compile(r"\[AUTOTUNE_SCREEN\].*\bnot screened\b|\bgo to the timer unchecked\b", re.I)
_SCREEN_SUMMARY = re.compile(r"correctness screen (on|off): checked (\d+) key\(s\), excluded (\d+) config")


def _cfg(entry):
    """The config of a sweep entry without its bench timing (two arms choose the same config with different times)."""
    return None if entry is None else {k: v for k, v in entry.items() if k != "timing"}


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
    stores = {label: _sweep_store_entries(d / f"{label}_autotune", model) for label, _ in trees}
    # A key whose bench margin (second-best over best) is under 10 % in ANY arm is a near-tie the
    # timer can flip on the next run, whatever the arms chose there.
    near_tie = sorted({k for ent in stores.values() for k, e in ent.items()
                       if isinstance(e.get("timing"), dict) and e["timing"].get("margin") is not None
                       and e["timing"]["margin"] < 0.10})
    # The control: a second arm on the FIRST tree — what timing noise alone does to a choice.
    control = next((l for l, src in trees[1:] if str(src) == str(trees[0][1])), None)
    ctrl = stores.get(control, {}) if control else {}
    noise_keys = sorted(k for k in set(base) & set(ctrl) if _cfg(base[k]) != _cfg(ctrl[k])) if control else []
    out = {}
    for label, _ in trees:
        ent = stores[label]
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
            differing = sorted(k for k in set(base) & set(ent) if _cfg(base[k]) != _cfg(ent[k]))
            # A key where the arm agrees with EITHER the first arm or the control is within the
            # run-to-run noise of the timer; only a key that differs from both is the arm's own.
            # ... and a key the control itself disagrees on with the first arm is a demonstrated
            # near-tie of the timer: a third config there is still noise, not the arm's doing.
            beyond = sorted(k for k in differing
                            if k not in near_tie
                            and not (control and label != control and (_cfg(ctrl.get(k)) == _cfg(ent.get(k)) or k in noise_keys)))
            rec.update({"identical": not missing and not extra and not differing and bool(base),
                        "within_noise": (not missing and not extra and bool(base) and (label == control or not beyond)),
                        "missing_keys": missing[:10], "extra_keys": extra[:10],
                        "differing_keys": differing[:10], "beyond_noise_keys": beyond[:10],
                        "noise_keys": noise_keys[:10], "noise_key_count": len(noise_keys),
                        "near_tie_count": len(near_tie), "margins_recorded": sum(1 for e in ent.values() if isinstance(e.get("timing"), dict)),
                        "first_diff": (differing or missing or extra or [None])[0],
                        "first_diff_configs": ({"first": base.get(differing[0]), label: ent.get(differing[0]),
                                                **({control: ctrl.get(differing[0])} if control else {})}
                                               if differing else None)})
        out[label] = rec
    return out

def tree_ab(model: str, gpu, out: Path, extra: list, timeout: int, trees: list, sweep_arms: bool = False,
            oracle_on_diff: bool = False) -> dict:
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
        base_env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "tree",
           "trees": {label: str(src) for label, src in trees}, "arms": {}}
    entry = "import sys; from neurobrix.cli import main; sys.exit(main())"
    for label, src in trees:
        env = {**base_env, "PYTHONPATH": str(Path(src).resolve())}
        if sweep_arms:
            env["NBX_AUTOTUNE"] = "sweep"
            env["NEUROBRIX_AUTOTUNE_STORE"] = str(d / f"{label}_autotune")
            env["NEUROBRIX_REPLAY_CACHE"] = str(d / f"{label}_replay")     # no seeding from the machine cache: a cold sweep per arm
        seen = subprocess.run([PY, "-c", "import neurobrix, sys; sys.stdout.write(neurobrix.__file__)"],
                              env=env, capture_output=True, text=True).stdout.strip()
        outp = d / f"{label}{ext}"
        rc, wall = run([PY, "-c", entry, "run", "--model", model] + req + ["--output", str(outp)],
                       env, d / f"{label}.log", timeout)
        res["arms"][label] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(d / f"{label}.log"),
                              "output": str(outp), "package_seen": seen,
                              "sha": hashlib.sha256(outp.read_bytes()).hexdigest()[:12] if outp.exists() else None}
        if rc != 0 or not outp.exists():
            # The byte gate below needs EVERY arm (`ran` is an `all(...)`), so a
            # verdict is already impossible and every remaining arm buys no
            # information. Allegro cost 16 h of a quiet rig to learn this twice:
            # its A arm timed out at step 1 of 4 (7.8 h/step, ~31 h projected),
            # and the identical B arm was allowed to spend another 8 h reaching
            # the same wall. Say why, and give the rig back.
            skipped = [lab for lab, _ in trees[trees.index((label, src)) + 1:]]
            if skipped:
                res["arms_skipped"] = skipped
                res["skip_reason"] = (
                    f"arm {label} produced no output (rc={rc}"
                    + (", timed out" if rc == -9 else "")
                    + f"); the byte gate needs every arm, so {', '.join(skipped)} "
                      f"could not change the verdict")
                print(f"[zoo] {model}: arm {label} produced no output — skipping "
                      f"{', '.join(skipped)}, the verdict cannot change", flush=True)
            break
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
    if oracle_on_diff and ran and comp and not all(v["identical"] for v in comp.values()):
        # The trees' outputs differ: a fix changed this model's output, so the LAST tree's output
        # is proven against the sequential oracle run from that same tree (the ATen engine, op by
        # op) — the owner's word of 2026-09-07: a model that changes was wrong, and its corrected
        # output must be the oracle's; a model that does not change proves the fix inert.
        last_label, last_src = trees[-1]
        env = {**base_env, "PYTHONPATH": str(Path(last_src).resolve())}
        res["oracle"] = {**_oracle_run(d, model, fam, extra, ext, env, timeout), "tree": last_label}
        outp = Path(res["oracle"]["output"])
        if res["oracle"]["rc"] == 0:
            after = _vs_oracle(outp, d / f"{last_label}{ext}")
            res["oracle"]["corrected_identical"] = after["identical"] is True
            if after["identical"] is False:
                res["oracle"]["diff"] = {k: v for k, v in after.items() if k != "identical"}   # how far the corrected output is from the oracle
            # and how far the output BEFORE the fix was: a fix that moves an output within the
            # family's measure is a config or tiling difference, a fix that brings a wrong output
            # onto the oracle is the defect closed — the row must say which (hat-l-x4, 2026-09-07)
            res["oracle"]["before_diff"] = _vs_oracle(outp, d / f"{trees[0][0]}{ext}")
        else:
            res["oracle"]["corrected_identical"] = False
    res["gate"] = {"kind": "bytes", "against": first, "arms": comp,
                   "identical": ran and all(v["identical"] for v in comp.values()), "ran": ran}
    res["A"] = res["arms"][first]
    res["B"] = res["arms"][trees[1][0]] if len(trees) > 1 else res["arms"][first]
    x, y = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (x / y) if x and y else None

    # Last word on the ratio: a lever arm that exercised nothing has not
    # measured, so the cell carries NO ratio rather than a ratio of one
    # (vacuous_lever_reason). Inert where the lever has no such telemetry.
    _vacuous = vacuous_lever_reason(res)
    if _vacuous:
        res["lever_vacuous"] = _vacuous
        res["speedup"] = None
        print(f"[zoo] {model}: LEVER NOT MEASURED — {_vacuous}", flush=True)

    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def _oracle_run(d: Path, model: str, fam: str, extra: list, ext: str, env: dict, timeout: int) -> dict:
    """The sequential oracle (the ATen engine, op by op) on this model's request, from the tree
    `env` names (PYTHONPATH) — the reference any changed output is measured against."""
    oreq = request_args(model, fam, extra)
    oreq = [a for a in oreq if a != "--triton"] + ["--sequential"]
    outp = d / f"oracle{ext}"
    rc, wall = run([PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "run", "--model", model]
                   + oreq + ["--output", str(outp)], env, d / "oracle.log", timeout)
    return {"rc": rc, "wall_s": wall, "output": str(outp),
            "sha": hashlib.sha256(outp.read_bytes()).hexdigest()[:12] if outp.exists() else None}


def _vs_oracle(oracle: Path, arm: Path) -> dict:
    """One arm against the oracle's output: identical, else the family's own measure."""
    if not (oracle.exists() and arm.exists()):
        return {"identical": None, "error": "an output is missing"}
    if oracle.read_bytes() == arm.read_bytes():
        return {"identical": True}
    try:
        return {"identical": False, **gate(oracle, arm)}
    except Exception as e:  # noqa: BLE001
        return {"identical": False, "error": str(e)}


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
        env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
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


def drift_one(model: str, gpu, out: Path, extra: list, timeout: int, bound: float = 0.02, src: Path = None) -> dict:
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
        env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
    if src is not None:                                  # the walk runs the given tree's package
        env = {**env, "PYTHONPATH": str(Path(src).resolve())}
        cmd = [PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "drift", "--model", model]
    else:
        cmd = [NBX, "drift", "--model", model]
    rc, wall = run(cmd + ["--out", str(d), "--bound", str(bound)] + req, env, d / "drift.log", timeout)
    oracle_log = (d / "oracle.log").read_text(errors="replace") if (d / "oracle.log").exists() else ""
    triton_only = "UNSUPPORTED PATH" in oracle_log and "encoding" in oracle_log
    rep = {}
    if (d / "drift.json").exists():
        rep = json.loads((d / "drift.json").read_text())
    first = rep.get("first") or {}
    kernel = rep.get("first_same_dtype") or {}
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": "drift",
           "src": str(src) if src else None,
           "rc": rc, "wall_s": wall, "ops": rep.get("ops_a"), "matched": rep.get("matched"),
           "missing": rep.get("missing_in_b"), "over_bound": rep.get("over_bound"), "bound": bound,
           "policy_sites": rep.get("policy_sites"),
           "site": (f"{first.get('component')}/{first.get('op_uid')}" if first else None),
           "site_type": first.get("op_type"), "site_dev": first.get("rel_dev"), "site_index": first.get("index"),
           "kernel_site": (f"{kernel.get('component')}/{kernel.get('op_uid')}" if kernel else None),
           "kernel_site_type": kernel.get("op_type"), "kernel_site_dev": kernel.get("rel_dev"),
           "kernel_site_index": kernel.get("index"),
           "origin_class": rep.get("origin_class"),
           "float_before": (f"{(rep.get('float_before') or {}).get('component')}/{(rep.get('float_before') or {}).get('op_uid')}"
                            if rep.get("float_before") else None),
           "float_before_dev": (rep.get("float_before") or {}).get("rel_dev"),
           "producer_missing": rep.get("producer_missing"), "site_abs": first.get("abs_dev"), "abs_before": rep.get("abs_before"),
           "A": {"rc": rc, "exec_s": None}, "B": {"rc": rc, "exec_s": None},
           "gate": {"kind": "drift", "pass": rc == 0 and not first}}
    if triton_only:
        res["error"] = "triton-only build"
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def _certified_entries(src: Path) -> dict:
    """Every entry of the tree's certified directory keyed as (kernel name, shape key) — the
    replay cache names a key `<kernel module path>::<shape key>`; the entry carries the
    setting, its proof (best_ms, second_ms, deviation, tolerance) and the excluded settings."""
    root = Path(os.environ.get("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR") or (Path(src) / "neurobrix" / "config" / "autotune"))
    entries = {}
    for f in root.glob("*/*/*.json"):
        try:
            j = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        kernel = f.name.split(".")[0]
        ents = j.get("entries") or j.get("settings") or j
        if isinstance(ents, dict):
            for k, v in ents.items():
                if isinstance(k, str) and k.startswith("(") and isinstance(v, dict):
                    entries[(kernel, k)] = v
    return entries


def _cfg_of(e: dict) -> tuple:
    return (json.dumps((e or {}).get("kwargs"), sort_keys=True), (e or {}).get("num_warps"), (e or {}).get("num_stages"))


NEAR_TIE_MARGIN = 0.10      # the campaign's convention (the sweep-store comparison): second-best within 10 % of best


def _choices_ab(d: Path, src: Path) -> dict:
    """The two cold arms' replay caches compared key by key, and every differing key the
    certified directory holds classified by the entry's own proof: NEAR-TIE when the certifier's
    second-best was within 10 % of its best (the runtime sweep's pick is the timer's noise among
    settings all within tolerance — CogVideoX-2b, 2026-09-07: six matmul/addmm keys, margins
    0.001–0.9 %); EXCLUDED-PICKED when the runtime picked a setting the certifier had excluded
    for its deviation (the unscreened runtime sweep let an out-of-tolerance setting through — a
    finding); CONTRADICTED when the runtime picked another setting on a key with a clear margin
    (a finding: the certification or the runtime's timing is wrong on this machine)."""
    stores = {}
    for arm in ("A", "B"):
        ent = {}
        # The arm's replay directories: `<arm>_replay` at --paired 1, and
        # `<arm>_replay_r<rep>` once every repetition gets its own (which it
        # must, else repetition 2 replays what repetition 1 swept). Every
        # repetition of a cold arm sweeps the same keys, so the union is the
        # arm's choices; a key present in several repetitions keeps the last
        # read, which is the same setting.
        for dirname in sorted(d.glob(f"{arm}_replay*")):
            for f in dirname.glob("*.json"):
                try:
                    j = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(j, dict):
                    ent.update(j)
        stores[arm] = ent
    if not stores["B"] and not stores["A"]:
        return {}
    certified = _certified_entries(src) if src is not None else {}
    def entry(k):
        mod, _, shape = k.partition("::")
        return certified.get((mod.rsplit(".", 1)[-1], shape))
    keys = sorted(set(stores["A"]) | set(stores["B"]))
    def pick(arm, k):
        """The arm's setting for the key: its runtime sweep's when it swept, else the
        directory's entry — an arm that served the key certified left no replay record
        (the machine band's arm A served every key: its replay is empty by design)."""
        chosen = stores[arm].get(k)
        if chosen is None and (e := entry(k)):
            return _cfg_of(e.get("config"))
        return _cfg_of(chosen)
    differ = [k for k in keys if pick("A", k) != pick("B", k)]
    near_tie, contradicted, excluded_picked, uncertified = [], [], [], []
    for k in differ:
        e = entry(k)
        if not e:
            uncertified.append(k); continue
        pr = e.get("proof") or {}
        picks = {pick("A", k), pick("B", k)} - {_cfg_of(e.get("config"))}
        if any(_cfg_of(x.get("config")) in picks for x in (e.get("excluded") or [])):
            excluded_picked.append(k); continue
        best, second = pr.get("best_ms"), pr.get("second_ms")
        margin = (second / best - 1.0) if best and second else None
        (near_tie if margin is not None and margin < NEAR_TIE_MARGIN else contradicted).append(
            {"key": k, "margin": margin, "best_ms": best, "delta_ms": (second - best) if best and second else None}
            if margin is not None else k)
    return {"keys": len(keys), "certified": sum(1 for k in keys if entry(k)), "differ": len(differ),
            "near_tie": near_tie[:20], "near_tie_count": len(near_tie),
            "contradicted": contradicted[:20], "contradicted_count": len(contradicted),
            "excluded_picked": excluded_picked[:20], "excluded_picked_count": len(excluded_picked),
            "differ_uncertified": uncertified[:20], "differ_uncertified_count": len(uncertified)}


def env_ab(model: str, gpu, out: Path, extra: list, timeout: int, env_b: dict, lever: str, cold: bool = False, src: Path = None,
           oracle_on_diff: bool = False, paired: int = 1) -> dict:
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
        base_env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
    res = {"model": model, "family": fam, "weight_gb": round(weight_gb(model), 2),
           "config": "machine" if gpu is None else f"pinned:{gpu}", "request": req, "lever": lever, "env_b": env_b,
           "src": str(src) if src else None, "cold": cold, "paired": paired}
    # Paired: the arms interleaved A B A B … `paired` times in the same minutes, the row's time per
    # arm the MEDIAN of its repeats — the order-tax control of 2026-09-07 showed a lone A-then-B
    # pair reads a cold first arm or the host's drift as a lever gain (canary ×1.84 → none).
    # The bytes of every repeat of an arm must agree: a repeat that differs is the model's own
    # nondeterminism, said in the row, never a lever verdict.
    reps = {"A": [], "B": []}
    for rep in range(max(1, paired)):
        for arm, env in (("A", base_env), ("B", {**base_env, **env_b})):
            outp = d / f"{arm}{ext}"
            if cold:
                # A cold start per ARM **and per REPETITION**. The path used to
                # carry only {arm}: repetition 1 swept and wrote here, and every
                # later repetition of the same arm read its own leftovers back
                # instead of sweeping. On 2026-09-10 that turned
                # deepseek-moe-16b-chat into `speedup 1.0288` — its control arm
                # took all eight keys `from the local replay cache` it had
                # written itself one repetition earlier, while repetition 1, the
                # only one that measured, read 67,01 s against 95,94 s.
                #
                # This never touches the machine's own cache
                # (~/.neurobrix/replay_cache): the arm is given a directory of
                # its own, so nothing is set aside and nothing is destroyed.
                env = {**env, "NEUROBRIX_REPLAY_CACHE": str(d / f"{arm}_replay_r{rep}")}
            if src is not None:                          # the request runs the given tree's package
                env = {**env, "PYTHONPATH": str(Path(src).resolve())}
                cmd = [PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "run", "--model", model]
            else:
                cmd = [NBX, "run", "--model", model]
            # One log per REPETITION. It used to be one per arm, opened "w"
            # each time, so only the LAST repetition survived on disk — and the
            # repetition that actually measured left no trace. That is what made
            # the 2026-09-10 contamination read as a cache problem: B.log said
            # "8 from the local replay cache" because repetition 3 was speaking,
            # while repetition 1 had swept its eight and was already overwritten.
            logp = d / (f"{arm}.log" if paired <= 1 else f"{arm}.r{rep}.log")
            rc, wall = run(cmd + req + ["--output", str(outp)], env, logp, timeout)
            log = logp.read_text(errors="replace")
            cert = re.search(r"certified directory: (\d+) key\(s\) served without a sweep, "
                             r"(\d+) swept at runtime \(kept locally\), "
                             r"(\d+) from the local replay cache", log)
            # An encoded build (int4) is refused by the compiled engine at its capability gate:
            # not an arm that failed, a row the lever does not apply to on that engine.
            unsupported = re.search(r"UNSUPPORTED PATH: (.*?encoding '[^']*'[^.]*)", log)
            res[arm] = {"rc": rc, "wall_s": wall, "exec_s": exec_time(logp), "output": str(outp),
                        "log": logp.name,
                        "n_a": unsupported.group(1).strip()[:160] if unsupported else None,
                        "sha": hashlib.sha256(outp.read_bytes()).hexdigest()[:12] if outp.exists() else None,
                        "ops_removed": sum(int(x) for x in re.findall(r"\[Optim\] algebraic: (\d+) identity ops", log)) or None,
                        "certified_served": int(cert.group(1)) if cert else None,
                        "swept": int(cert.group(2)) if cert else None,
                        # The number that named the 2026-09-10 contamination: an
                        # arm that "swept 0" may simply have read back what an
                        # earlier repetition of ITSELF wrote.
                        "from_replay": int(cert.group(3)) if cert else None,
                        "announced_missing": len(re.findall(r"\[autotune\] no certified setting for", log)),
                        "contradictions": len(re.findall(r"\[AUTOTUNE_SCREEN\] CONTRADICTION", log)),
                        "screen_excluded": len(re.findall(r"\[AUTOTUNE_SCREEN\] .*config excluded", log))}
            if paired > 1:
                reps[arm].append({"exec_s": res[arm]["exec_s"], "sha": res[arm]["sha"], "rc": rc})
                if rep < paired - 1 and outp.exists():
                    (d / f"{arm}.rep{rep + 1}{ext}").write_bytes(outp.read_bytes())
    if paired > 1:
        import statistics
        for arm in ("A", "B"):
            xs = [r["exec_s"] for r in reps[arm] if r["exec_s"] is not None]
            res[arm]["reps"] = reps[arm]
            res[arm]["exec_s"] = statistics.median(xs) if xs else None
            shas = {r["sha"] for r in reps[arm] if r["sha"]}
            res[arm]["repeat_identical"] = len(shas) <= 1

    a, b = d / f"A{ext}", d / f"B{ext}"
    same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
    res["gate"] = {"kind": "bytes", "identical": same, "pass": same,
                   "ran": res["A"]["rc"] == 0 and res["B"]["rc"] == 0}
    if paired > 1 and not (res["A"].get("repeat_identical", True) and res["B"].get("repeat_identical", True)):
        res["gate"]["nondeterministic"] = [arm for arm in ("A", "B") if not res[arm].get("repeat_identical", True)]
    na = res["A"].get("n_a") or res["B"].get("n_a")
    if na:
        res["gate"] = {"kind": "n/a", "reason": na, "pass": None, "ran": False}
    if a.exists() and b.exists() and not same:
        try:
            res["gate"]["diff"] = gate(a, b)
        except Exception as e:  # noqa: BLE001
            res["gate"]["diff"] = {"error": str(e)}
    if oracle_on_diff and res["gate"]["ran"] and not same:
        # The lever changed this model's bytes: both arms against the sequential oracle from the
        # same tree, without the lever — which arm the oracle stands on, and how far the other is,
        # in the family's own measure (VibeVoice under the fusion lever, 2026-09-07: 39 dB apart).
        env = {**base_env, "PYTHONPATH": str(Path(src).resolve())} if src is not None else base_env
        res["oracle"] = _oracle_run(d, model, fam, extra, ext, env, timeout)
        if res["oracle"]["rc"] == 0:
            outp = Path(res["oracle"]["output"])
            res["oracle"]["A"] = _vs_oracle(outp, a)
            res["oracle"]["B"] = _vs_oracle(outp, b)
    if cold:
        # both arms cold with their own replay caches: their kernel choices side by side
        res["choices"] = _choices_ab(d, src)
    res["islands"] = {}
    x, y = res["A"]["exec_s"], res["B"]["exec_s"]
    res["speedup"] = (x / y) if x and y else None

    # Last word on the ratio: a lever arm that exercised nothing has not
    # measured, so the cell carries NO ratio rather than a ratio of one
    # (vacuous_lever_reason). Inert where the lever has no such telemetry.
    _vacuous = vacuous_lever_reason(res)
    if _vacuous:
        res["lever_vacuous"] = _vacuous
        res["speedup"] = None
        print(f"[zoo] {model}: LEVER NOT MEASURED — {_vacuous}", flush=True)

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
        env["CUDA_VISIBLE_DEVICES"] = visible_card(gpu)
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
            oc = r.get("origin_class")
            if oc == "discrete":
                where = (f"origin DISCRETE (an integer tensor flipped; largest float deviation before it "
                         f"{r.get('float_before')} {r.get('float_before_dev') or 0:.3f})")
            elif oc == "kernel":
                where = (f"origin KERNEL site {r['site']} ({r.get('site_type')}, {r.get('site_dev', 0):.3f})"
                         + (f"; largest float deviation before it {r.get('float_before')} {r.get('float_before_dev') or 0:.3f}" if r.get("float_before") else ""))
            elif oc == "scale":
                where = (f"origin a SCALE crossing at {r['site']} ({r.get('site_type')}): abs deviation {r.get('site_abs') or 0:.3g} vs "
                         f"{r.get('abs_before') or 0:.3g} before it — no new error here, inherited; largest float deviation before it "
                         f"{r.get('float_before')} {r.get('float_before_dev') or 0:.3f}")
            elif oc == "policy":
                where = (f"origin a POLICY site (dtypes differ); first same-dtype site after it "
                         f"{r.get('kernel_site')} ({r.get('kernel_site_type')}, {r.get('kernel_site_dev') or 0:.3f})"
                         if r.get("kernel_site") else f"origin a POLICY site (dtypes differ); no same-dtype site ({r.get('policy_sites')} policy sites)")
            elif oc == "carrier":
                where = (f"origin a CARRIER (its input's deviation; largest float before it {r.get('float_before')} "
                         f"{r.get('float_before_dev') or 0:.3f})")
            else:
                where = (f"kernel site {r['kernel_site']} ({r.get('kernel_site_type')}, {r.get('kernel_site_dev', 0):.3f}, op #{r.get('kernel_site_index')})"
                         if r.get("kernel_site") else f"no kernel site: policy only ({r.get('policy_sites')} dtype-policy sites)")
            if r.get("producer_missing"):
                where += "; its PRODUCER has no record on the engine side (fused or skipped there — read the fusion)"
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
            elif all(v.get("within_noise") for v in others.values()) and others:
                nk = max((v.get("noise_key_count") or 0) for v in others.values())
                nt = max((v.get("near_tie_count") or 0) for v in others.values())
                tail = (f"; choices within the timer's noise ({at.get(first, {}).get('keys', 0)} keys; {nk} moved on the control run"
                        + (f", {nt} near-ties by bench margin" if nt else "") + ")")
            else:
                bad = [f"{k} differs beyond the noise at {(v.get('beyond_noise_keys') or [v.get('first_diff')])[0]}"
                       for k, v in others.items() if not v.get("identical") and not v.get("within_noise")]
                tail = "; CHOICES DIFFER (" + "; ".join(bad) + ")"
            idle = [k for k, v in others.items() if v.get("screen") == "on" and not v.get("screened_keys")]
            if idle:
                tail += "; NOT PROVEN (screen on but checked 0 keys in " + ", ".join(idle) + ")"
            excl = {k: v.get("exclusions", 0) for k, v in at.items() if v.get("exclusions")}
            if excl:
                tail += "; SCREEN EXCLUDED " + ", ".join(f"{k}: {n} config(s)" for k, n in excl.items()) + " — a finding to close"
        if g.get("identical"):
            return "IDENTICAL" + tail
        diff = [k for k, v in g.get("arms", {}).items() if not v.get("identical")]
        o = r.get("oracle")
        if o:
            if o.get("rc") != 0:
                tail += "; the oracle did not run"
            elif o.get("corrected_identical"):
                tail += f"; corrected output IDENTICAL to the sequential oracle"
            else:
                dd = o.get("diff") or {}
                m = dd.get("psnr_db", dd.get("snr_db", dd.get("psnr_mean_db")))
                tail += f"; corrected output vs the sequential oracle: {dd.get('kind', '?')} " + \
                        ("PASS" if dd.get("pass") else "DIFFERENT") + (f" ({m})" if m is not None else "")
            bd = o.get("before_diff") or {}
            if bd:
                mb = bd.get("psnr_db", bd.get("snr_db", bd.get("psnr_mean_db")))
                tail += "; before the fix vs the oracle: " + ("IDENTICAL" if bd.get("identical") else (("PASS" if bd.get("pass") else "DIFFERENT") + (f" ({mb})" if mb is not None else "")))
        return "DIFFERENT (" + ", ".join(diff) + ")" + tail
    if r.get("lever", "").startswith("env:"):
        g = r.get("gate") or {}
        if g.get("kind") == "n/a":
            return f"N/A ({g.get('reason')})"
        if not g.get("ran"):
            return "FAILED (an arm did not run)"
        n = (r.get("B") or {}).get("ops_removed")
        tag = f", {n} ops removed" if n else ""
        A, B = r.get("A") or {}, r.get("B") or {}
        if (r.get("paired") or 1) > 1:
            tag += f"; paired ×{r['paired']}, medians"
            nd = g.get("nondeterministic")
            if nd:
                tag += f"; NONDETERMINISTIC repeats on arm(s) {', '.join(nd)} — the model's own, not the lever's"
        if A.get("certified_served") is not None or B.get("certified_served") is not None:
            tag += (f"; A certified {A.get('certified_served', 0)} / swept {A.get('swept', 0)}"
                    f", B certified {B.get('certified_served', 0)} / swept {B.get('swept', 0)}")
            if A.get("exec_s") and B.get("exec_s"):
                tag += f"; cold start A {A['exec_s']:.1f} s vs B {B['exec_s']:.1f} s"
            c = (A.get("contradictions") or 0) + (B.get("contradictions") or 0)
            if c:
                tag += f"; {c} CONTRADICTION(S) — a finding"
            x = (A.get("screen_excluded") or 0) + (B.get("screen_excluded") or 0)
            if x:
                tag += f"; the screen excluded {x} config(s) — a finding to close"
        ch = r.get("choices") or {}
        if ch:
            if not ch.get("differ"):
                tag += f"; every kernel choice alike on {ch['keys']} keys ({ch['certified']} certified)"
            else:
                parts = [f"choices differ on {ch['differ']} of {ch['keys']} keys ({ch['certified']} certified)"]
                if ch.get("excluded_picked_count"):
                    parts.append(f"EXCLUDED SETTING PICKED AT RUNTIME on {ch['excluded_picked_count']} key(s) — a finding: {ch['excluded_picked'][0]}")
                if ch.get("contradicted_count"):
                    c0 = ch["contradicted"][0]
                    scale = ""
                    if isinstance(c0, dict) and c0.get("best_ms") is not None:
                        # the scale of the certifier's margin: a 15 % margin on a 21 µs kernel is 3 µs
                        scale = f" (the certifier's margin {c0['margin'] * 100:.0f} % = {c0['delta_ms'] * 1000:.1f} µs on a {c0['best_ms'] * 1000:.1f} µs kernel)"
                    parts.append(f"CERTIFIED CHOICE CONTRADICTED on {ch['contradicted_count']} key(s) — a finding: "
                                 f"{c0['key'] if isinstance(c0, dict) else c0}{scale}")
                if ch.get("near_tie_count"):
                    ms = [x["margin"] for x in ch["near_tie"] if isinstance(x, dict) and x.get("margin") is not None]
                    parts.append(f"{ch['near_tie_count']} certified near-tie(s) (the certifier's second-best within "
                                 + (f"{max(ms) * 100:.1f} %" if ms else "10 %") + " of its best; the runtime's pick is the timer's noise)")
                if ch.get("differ_uncertified_count"):
                    parts.append(f"{ch['differ_uncertified_count']} uncertified (the runtime sweep's own variance)")
                tag += "; " + "; ".join(parts)
        if not g.get("identical"):
            dd = g.get("diff") or {}
            m = dd.get("psnr_db", dd.get("snr_db", dd.get("psnr_mean_db")))
            if m is not None:
                tag += f"; arms {m:.1f} dB apart"
            o = r.get("oracle")
            if o:
                if o.get("rc") != 0:
                    tag += "; the oracle did not run"
                else:
                    def _arm(v):
                        if v.get("identical"):
                            return "IDENTICAL"
                        mm = v.get("psnr_db", v.get("snr_db", v.get("psnr_mean_db")))
                        return ("PASS" if v.get("pass") else "DIFFERENT") + (f" {mm:.1f} dB" if mm is not None else "")
                    tag += f"; vs the sequential oracle: A {_arm(o.get('A') or {})}, B {_arm(o.get('B') or {})}"
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
        at_head = " | autotune keys / choices vs first / screen / exclusions (per arm) | sweep overhead (s, arm − control; the first arm pays the kernel compile)" if with_at else ""
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
                    ch = "" if "identical" not in a else (" / identical" if a["identical"] else (" / within noise" if a.get("within_noise") else f" / DIFFER at {(a.get('beyond_noise_keys') or [a.get('first_diff')])[0]}"))
                    sc = a.get("screen", "absent")
                    sc = f"screen {sc}" + (f" ({a['screened_keys']} checked)" if a.get("screened_keys") is not None else "")
                    parts.append(f"{l}: {a.get('keys', 0)} keys{ch} / {sc} / {a.get('exclusions', 0)} excluded"
                                 + (f" / {a['unscreened']} not screened" if a.get("unscreened") else ""))
                cells.append("; ".join(parts))
                trees = r.get("trees") or {}
                ctrl = next((l for l in labels[1:] if trees.get(l) == trees.get(labels[0])), labels[0])
                e0 = ((r.get("arms") or {}).get(ctrl) or {}).get("exec_s")
                ov = []
                for l in labels[1:]:
                    if l == ctrl:
                        continue
                    e1 = ((r.get("arms") or {}).get(l) or {}).get("exec_s")
                    ov.append(f"{l} − {ctrl}: {e1 - e0:+.2f}" if e0 is not None and e1 is not None else f"{l}: —")
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



def lock_holder_alive(text: str) -> bool:
    """A `.running` lock names its holder (`gpu=N pid=P HH:MM:SS`); the holder is alive when
    that pid still exists. A lock without a pid is trusted (an older writer)."""
    import re
    m = re.search(r"\bpid=(\d+)", text)
    if not m:
        return True
    return Path(f"/proc/{m.group(1)}").exists()


ARM_LABELS = ("A", "B")


def vacuous_lever_reason(record: dict):
    """Why this cell measured nothing, or None if it measured.

    A lever cell exists to make one arm pay a cost the other does not. When the
    paying arm reports it served nothing from the certified directory AND swept
    nothing at runtime, it paid nothing: the row's ratio is one because both
    arms did the same work, not because the lever is worth one.

    `deepseek-moe-16b-chat` on 2026-09-10 is the live shape — `speedup 1.0288`,
    A 34,34 s against B 33,38 s, and a control arm that took all eight of its
    keys `from the local replay cache` its OWN first repetition had written.
    Its first repetition, the only one that measured, read 67,01 against 95,94.

    Same class as a gate that never passed and a runner that counted segments
    instead of cells: a measurement must prove it took place. A ratio of one
    from an arm that did no work is worse than a missing row, because a missing
    row is visibly missing.

    Judged only on the arms' own telemetry. A cell that declares no lever is
    none of this guard's business, and an arm that never reached the autotune
    recap (a crash, an unsupported path) is a failure the byte gate already
    reports, not a vacuous measurement.
    """
    if not record.get("lever"):
        return None
    arms = {a: (record.get(a) or {}) for a in ARM_LABELS}
    told = [a for a, v in arms.items()
            if v.get("certified_served") is not None or v.get("swept") is not None]
    if not told:
        return None                      # no telemetry at all: not this guard's call
    # The lever arm is the one that carries env_b — B by construction of the
    # arm loop. A is the reference and is SUPPOSED to be served without
    # sweeping; only B's silence is a defect.
    lever_arm = ARM_LABELS[1]
    if lever_arm not in told:
        return None
    v = arms[lever_arm]
    if (v.get("certified_served") or 0) or (v.get("swept") or 0):
        return None
    detail = ", ".join(
        f"{a}: served {arms[a].get('certified_served') or 0}, "
        f"swept {arms[a].get('swept') or 0}, "
        f"{arms[a].get('from_replay') or 0} from the replay cache"
        for a in told)
    return (f"arm {lever_arm} exercised nothing of the lever {record['lever']} — {detail}. "
            f"The ratio this cell would report is one because both arms did "
            f"the same work, not because the lever is worth one"
            + (f" (paired={record['paired']}: a repetition after the first "
               f"replays what the first swept unless every repetition gets its "
               f"own cache)" if (record.get("paired") or 1) > 1 else ""))


def cell_cost_estimate(model_out: Path, timeout: int):
    """(seconds, basis) this cell is expected to cost, or None if unmeasured.

    Read from the cell's own previous record — the only honest source. A model
    nobody has run has no estimate and must NOT be refused: the guard's job is
    to stop a KNOWN cost, never to guess an unknown one.

    An arm with `rc < 0` was killed (SIGKILL at the timeout, or SIGTERM), so its
    wall is a LOWER bound, not a measurement: the cell costs at least the
    timeout for every arm it still has to run. `Allegro` on 2026-09-10 is the
    live shape — A killed at 28 800 s, B killed at 24 958 s, ~31 h per arm
    projected from its own log against an 8 h timeout.
    """
    p = Path(model_out) / "result.json"
    if not p.exists():
        return None
    try:
        record = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    walls, killed = [], []
    for label in ARM_LABELS:
        arm = record.get(label)
        if not isinstance(arm, dict):
            continue
        w = arm.get("wall_s")
        if w is None:
            continue
        walls.append(float(w))
        try:
            if int(arm.get("rc", 0)) < 0:
                killed.append(label)
        except (TypeError, ValueError):
            pass
    if not walls:
        return None
    if killed:
        est = float(timeout) * max(len(walls), len(ARM_LABELS))
        return est, (f"arm(s) {', '.join(killed)} were killed at the wall "
                     f"({max(walls):.0f} s, timeout {timeout} s) — the cost is a "
                     f"lower bound, at least {est:.0f} s for the arms")
    est = sum(walls)
    return est, (f"{len(walls)} measured arm(s), "
                 f"{', '.join(f'{w:.0f} s' for w in walls)} — {est:.0f} s")


def budget_refusal(model_out: Path, budget_s, timeout: int):
    """The reason to refuse this cell at the door, or None to let it in.

    Opt-in: a campaign that declares no budget keeps the previous behaviour
    exactly. A cell whose KNOWN cost exceeds the whole campaign's budget can
    never fit inside it, whatever the order of the models, so it is refused
    before its first arm starts — loudly, with its number.
    """
    if budget_s is None:
        return None
    got = cell_cost_estimate(model_out, timeout)
    if got is None:
        return None
    est, basis = got
    if est <= float(budget_s):
        return None
    return (f"estimated {est:.0f} s against a campaign budget of "
            f"{float(budget_s):.0f} s — {basis}")


def has_verdict(model_out: Path) -> bool:
    """True when `<model_out>/result.json` records a gate that actually RAN.

    `--skip-done` asks "is this model MEASURED?", and only a gate that ran
    answers it. A crashed arm pair still writes result.json — `gate.ran` False,
    both arms `rc=1` — and reading the file's mere existence as done makes a
    fixable failure permanent: every later container prints "done, skipped" and
    the model is never measured again. Three models of the certified-directory
    proof sat that way on 2026-09-10 (a negative-size malloc, an OOM, and a
    tiling contract that failed in six seconds), each hiding a distinct defect.

    A DIFFERENT verdict IS a measurement and does count — the question is
    whether the arms were compared, not whether they agreed. A model that is
    genuinely impossible belongs to the queue's `impossible_when` predicate,
    with its reason written, not to a crash promoted to a result.

    Present-but-broken returns False rather than raising: here the fallback is
    to MEASURE AGAIN, which is self-correcting and can never silently skip —
    unlike a manifest read, where a silent default would decide flags.
    """
    p = Path(model_out) / "result.json"
    if not p.exists():
        return False
    try:
        record = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return bool((record.get("gate") or {}).get("ran"))


def held_by_retrace(retrace_out: Path) -> set:
    """Every model of a retrace campaign whose recorded gate is not a PASS: listed in its
    models file (phase lists) or holding a state.json without a PASS gate — its cache slot may
    change under a measurement (a restore of the previous object, an install of the new
    build), so a lever or a proof must not measure it until it is gated."""
    held = set()
    for lst in retrace_out.glob("*_models.txt"):
        held |= {m.strip() for m in lst.read_text().replace("\n", ",").split(",") if m.strip()}
    for st in retrace_out.glob("*/state.json"):
        try:
            steps = json.loads(st.read_text()).get("steps") or {}
        except (json.JSONDecodeError, OSError):
            continue
        gate = steps.get("gate") or {}
        if str(gate.get("verdict", "")).startswith("PASS") and gate.get("ok") is True:
            held.discard(st.parent.name)
        else:
            held.add(st.parent.name)
    return held


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
    r.add_argument("--budget", type=float, default=None, metavar="SECONDS",
                   help="the campaign's card budget. A cell whose KNOWN cost "
                        "(from its own previous record) exceeds it is refused "
                        "at the door, before its first arm starts. Opt-in: "
                        "without it nothing is refused.")
    r.add_argument("--hold-from", default=None, metavar="RETRACE_OUT",
                   help="skip every model a retrace campaign at this path still holds (listed in its phase files or "
                        "carrying a state without a PASS gate): its cache slot may change under the measurement")
    r.add_argument("--probe", action="store_true",
                   help="the R33 lever: one complete --triton request per model under the sys.modules probe")
    r.add_argument("--src", default=None, help="a frozen worktree's src for the probe's PYTHONPATH (default: this repo)")
    r.add_argument("--trees", default=None,
                   help="tree gate: label=path/to/src,label=path/to/src[,...] — the same --triton request from each "
                        "frozen tree, bytes compared against the first (a port's kernel change must be inert on CUDA)")
    r.add_argument("--oracle-on-diff", action="store_true",
                   help="when the arms' outputs differ, run the sequential oracle (no lever) from the tree and measure the "
                        "arms against it — with --trees the last tree's output must land on the oracle (a fix that changes "
                        "an output); with --env-ab both arms are measured (which one the oracle stands on, how far the other)")
    r.add_argument("--sweep-arms", action="store_true",
                   help="with --trees: every arm sweeps cold into its own store (NBX_AUTOTUNE=sweep) and the chosen "
                        "config per kernel key is compared across arms, the sweep's overhead measured per arm, and the "
                        "configs a correctness screen excluded counted from each arm's log")
    r.add_argument("--env-ab", default=None, metavar="KEY=VALUE[,KEY=VALUE]",
                   help="an engine lever behind an environment switch: arm A without, arm B with it, bytes compared")
    r.add_argument("--paired", type=int, default=1, metavar="N",
                   help="with --env-ab: interleave the arms A B A B … N times and time each arm by the median of its "
                        "repeats (a lone pair reads a cold first arm as a gain); the repeats' bytes must agree")
    r.add_argument("--cold-arms", action="store_true",
                   help="every arm of every repetition starts cold on THREE things, "
                        "each named: (1) the model's weights and the host page "
                        "cache; (2) the engine's replay cache — the arm runs "
                        "with NEUROBRIX_REPLAY_CACHE pointed at a directory of "
                        "its own, so it sweeps instead of reading back, and the "
                        "machine's own cache at ~/.neurobrix/replay_cache is "
                        "never moved, seeded or destroyed; (3) the same holds "
                        "PER REPETITION under --paired, so repetition 2 does not "
                        "replay what repetition 1 swept. Without it, a control "
                        "arm can report a gain of one having done no work.")
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
    held = held_by_retrace(Path(args.hold_from)) if args.hold_from else set()
    for m in models:
        if args.skip_done:
            if has_verdict(out / m):
                print(f"[zoo] {m}: done, skipped"); continue
            if (out / m / "result.json").exists():
                # A record without a gate that ran: the arms crashed. Say so and
                # measure again — see has_verdict.
                print(f"[zoo] {m}: a record with no verdict (the arms did not run) — measured again",
                      flush=True)
        # The guard at the door. There is already one INSIDE the cell (an arm
        # that produces no output ends the container); this is the one that
        # never lets the first arm start. Allegro cost ~15 h of a quiet rig on
        # 2026-09-10 for a verdict already known impossible and already
        # written — the next flight excluded it by hand, so the lesson was
        # learnt after the spend rather than before it.
        _refusal = budget_refusal(out / m, args.budget, args.timeout)
        if _refusal:
            print(f"[zoo] {m}: REFUSED at the door — {_refusal}", flush=True)
            (out / m).mkdir(parents=True, exist_ok=True)
            (out / m / "refused_budget.txt").write_text(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} refused before any arm ran\n{_refusal}\n")
            continue
        if m in held:
            # A container still short of its retrace gate: its cache slot moves (a restore, an
            # install) — CogVideoX-2b's proof row straddled an install at 14:34 on 2026-09-07.
            print(f"[zoo] {m}: held by the retrace campaign (no PASS gate yet in {args.hold_from}), skipped", flush=True); continue
        lock = out / m / ".running"
        if lock.exists():
            held = lock.read_text().strip()
            if lock_holder_alive(held):
                print(f"[zoo] {m}: running elsewhere ({held}), skipped"); continue
            # the holder is gone (a power loss, a kill): the lock is stale, the model runs again
            print(f"[zoo] {m}: stale lock of a dead run ({held}), removed", flush=True); lock.unlink()
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
                res = env_ab(m, gpu, out, extra, args.timeout, env_b, "env:" + ",".join(env_b), cold=args.cold_arms,
                             src=Path(args.src) if args.src else None, oracle_on_diff=args.oracle_on_diff,
                             paired=args.paired)
                print(f"[zoo] {m}: {verdict(res)} A={res['A']['exec_s']} B={res['B']['exec_s']} "
                      f"{('×%.2f' % res['speedup']) if res.get('speedup') else ''}", flush=True)
                continue
            elif args.drift:
                res = drift_one(m, gpu, out, extra, args.timeout, args.drift_bound, src=Path(args.src) if args.src else None)
                print(f"[zoo] {m}: {verdict(res)}", flush=True)
                continue
            elif args.sweep:
                res = sweep_one(m, gpu, out, extra, args.timeout)
                print(f"[zoo] {m}: {verdict(res)} exec={res['A'].get('exec_s')} artifact={res.get('artifact')}", flush=True)
                continue
            elif args.trees:
                trees = [(t.split("=", 1)[0], Path(t.split("=", 1)[1])) for t in args.trees.split(",") if t]
                res = tree_ab(m, gpu, out, extra, args.timeout, trees, sweep_arms=args.sweep_arms,
                              oracle_on_diff=args.oracle_on_diff)
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
