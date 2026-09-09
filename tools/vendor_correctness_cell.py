#!/usr/bin/env python3
"""Vendor-correctness cell — run the permanent justness column of the protocol.

Reads `benchmarks/harness/vendor_cells.yml` and, for each cell, runs OUR engine
and the ORIGINAL model on its own vendor stack, then compares the two outputs by
the cell's declared metric against the cell's declared bound.

The point of this cell is what it can see that nothing else we own can: our
sequential oracle replays the same graph the runtime executes, so a defect OF
THE GRAPH is identical on both arms and every byte gate reports IDENTICAL. Only
a comparison against the vendor's own stack can catch it.

A cell whose vendor stack is absent reports NOT-RUN with the reason. NOT-RUN is
a finding, not a pass — the summary counts it separately and never folds it into
the green count.

Usage:
    python3 tools/vendor_correctness_cell.py --cells benchmarks/harness/vendor_cells.yml \
        --out validation_outputs/vendor_cells_<date> [--family llm] [--id llm-dense-tinyllama]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Tuple

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── comparison metrics ──────────────────────────────────────────────────────
# Each returns (value, passed). The bound comes from the cell, never from here.

def _words(t: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (t or "").lower())


def m_text_common_prefix_words(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    a, b = _words(ours), _words(theirs)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n, n >= float(bound)


def m_psnr_db(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    """Delegates to tools/image_fidelity.py — the metric brick, not a copy."""
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "image_fidelity.py"),
                        theirs, ours, "--json"], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr[-400:]}, False
    d = json.loads(p.stdout)
    # The key is `psnr_db` — `psnr` was never emitted, so this read fell back to
    # 0.0 on every render and the cell could not report AGREES whatever the two
    # images were. A metric that cannot pass is not a gate, it is an alarm.
    if "psnr_db" not in d:
        return {"error": f"image_fidelity emitted no psnr_db: keys={sorted(d)}"}, False
    return d, float(d["psnr_db"]) >= float(bound)


def _wer_words(t: str) -> List[str]:
    """Words for WER: case and punctuation carry no transcription error."""
    return [w for w in re.sub(r"[^\w\s']", " ", str(t).lower()).split() if w]


def m_wer(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    """Word error rate of our transcript against the vendor's, Levenshtein on words."""
    r, h = _wer_words(theirs), _wer_words(ours)
    if not r:
        return {"error": "vendor transcript empty"}, False
    # Classic DP; the reference is the vendor's transcript, as the cell's whole
    # point is that the vendor is the truth and we are the thing measured.
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    wer = prev[len(h)] / len(r)
    return ({"wer": round(wer, 5), "ref_words": len(r), "hyp_words": len(h),
             "edits": prev[len(h)], "ours": ours[:200], "theirs": theirs[:200]},
            wer <= float(bound))


def m_psnr_db_first_frame(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    """First frame of each render, compared with the same brick as a still.

    The first frame is where a temporal defect shows up cleanly: a frozen frame
    count or a dropped frame changes what frame 0 IS, while the still-image
    metric stays interpretable. Both arms hand over a PNG of frame 0, so this
    delegates to the same image_fidelity brick rather than owning a second copy
    of PSNR.
    """
    return m_psnr_db(ours, theirs, bound)


METRICS = {
    "text_common_prefix_words": m_text_common_prefix_words,
    "psnr_db": m_psnr_db,
    "psnr_db_first_frame": m_psnr_db_first_frame,
    "wer": m_wer,
}


# ── vendor runners ──────────────────────────────────────────────────────────

def vendor_ollama(cell: Dict[str, Any], defaults: Dict[str, Any],
                  host: str) -> Dict[str, Any]:
    req = cell.get("request", {})
    body = json.dumps({
        "model": cell["vendor"]["ref"],
        "messages": [{"role": "user", "content": req["prompt"]}],
        "stream": False,
        "options": {"temperature": req.get("temperature", 0),
                    "seed": defaults.get("seed", 42),
                    "num_predict": req.get("max_tokens", defaults.get("max_tokens", 48)),
                    # CPU: the cell must never contend with a timed campaign.
                    "num_gpu": 0},
    }).encode()
    r = urllib.request.Request(f"{host}/api/chat", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            d = json.load(resp)
        return {"output": d.get("message", {}).get("content", ""), "error": None}
    except Exception as e:
        return {"output": None, "error": f"ollama unreachable or model absent: {e!r}"}


def vendor_available(cell: Dict[str, Any]) -> str:
    """Why this cell's vendor stack cannot run, or "" if it can.

    Checked BEFORE our arm, because a `latent_pinned` cell must run ours first
    (it produces the latent the vendor starts from) and loading a model for a
    vendor that was never installed would be waste, not evidence.
    """
    kind = cell["vendor"]["kind"]
    if kind == "ollama":
        return ""
    if kind == "faster_whisper":
        venv = cell["vendor"].get("venv", "")
        if not os.path.isfile(os.path.join(venv, "bin", "python")):
            return f"vendor venv absent: {venv}"
        if not os.path.isdir(cell["vendor"]["ref"]):
            return f"vendor snapshot absent: {cell['vendor']['ref']}"
        return ""
    if kind in ("diffusers", "diffusers_video"):
        venv = cell["vendor"].get("venv", "")
        if not os.path.isfile(os.path.join(venv, "bin", "python")):
            return f"vendor venv absent: {venv}"
        if not os.path.isdir(cell["vendor"]["ref"]):
            return f"vendor snapshot absent: {cell['vendor']['ref']}"
        return ""
    return (f"no runner wired for vendor.kind={kind!r} "
            f"({cell['vendor'].get('ref')}) — declared, not yet runnable")


def vendor_diffusers(cell: Dict[str, Any], defaults: Dict[str, Any],
                     out_dir: str, latents: str = "") -> Dict[str, Any]:
    v, req = cell["vendor"], cell.get("request", {})
    venv = v.get("venv", "")
    py = os.path.join(venv, "bin", "python")
    if not os.path.isfile(py):
        return {"output": None, "error": f"vendor venv absent: {venv}"}
    if not os.path.isdir(v["ref"]):
        return {"output": None, "error": f"vendor snapshot absent: {v['ref']}"}
    png = os.path.join(out_dir, f"{cell['id']}_vendor.png")
    cmd = [py, os.path.join(REPO, "tools", "vendor_image_repro.py"),
           "--snapshot", v["ref"], "--prompt", req["prompt"],
           "--seed", str(defaults.get("seed", 42)),
           "--steps", str(req.get("steps", 20)),
           "--guidance", str(req.get("guidance", 4.5)),
           "--height", str(req.get("height", 1024)),
           "--width", str(req.get("width", 1024)),
           "--out", png]
    if latents:
        # The PRIMARY gate: the vendor denoises OUR initial noise, so PSNR
        # compares two computations instead of two valid samples.
        cmd += ["--latents", latents]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not os.path.isfile(png):
        return {"output": None, "error": f"diffusers run failed: {p.stderr[-500:]}"}
    return {"output": png, "error": None}


FWHISPER_DRIVER = r"""
import glob, json, os, sys
from faster_whisper import WhisperModel
root, audio, beam = sys.argv[1], sys.argv[2], int(sys.argv[3])
snaps = glob.glob(os.path.join(root, "snapshots", "*"))
model = WhisperModel(snaps[0] if snaps else root, device="cuda", compute_type="float16")
segs, info = model.transcribe(audio, beam_size=beam, temperature=0)
print(json.dumps({"text": "".join(s.text for s in segs).strip(),
                  "language": info.language}))
"""


def vendor_faster_whisper(cell: Dict[str, Any], defaults: Dict[str, Any],
                          *_a, **_k) -> Dict[str, Any]:
    """The vendor transcript, from the model's own CTranslate2 runtime.

    Greedy (beam 1, temperature 0) on both sides so the comparison is of the
    computation, not of a decoder's search.
    """
    v, req = cell["vendor"], cell.get("request", {})
    py = os.path.join(v.get("venv", ""), "bin", "python")
    if not os.path.isfile(py):
        return {"output": None, "error": f"vendor venv absent: {v.get('venv')}"}
    if not os.path.isdir(v["ref"]):
        return {"output": None, "error": f"vendor snapshot absent: {v['ref']}"}
    audio = req.get("audio", "")
    if not os.path.isfile(audio):
        return {"output": None, "error": f"cell audio absent: {audio}"}
    p = subprocess.run([py, "-c", FWHISPER_DRIVER, v["ref"], audio, "1"],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"output": None, "error": f"faster-whisper failed: {p.stderr[-400:]}"}
    try:
        return {"output": json.loads(p.stdout.strip().splitlines()[-1])["text"], "error": None}
    except Exception as exc:
        return {"output": None, "error": f"unreadable vendor output ({exc}): {p.stdout[-200:]}"}


def vendor_diffusers_video(cell: Dict[str, Any], defaults: Dict[str, Any],
                           out_dir: str, latents: str = "") -> Dict[str, Any]:
    """The vendor's first frame, from its own diffusers video pipeline."""
    v, req = cell["vendor"], cell.get("request", {})
    py = os.path.join(v.get("venv", ""), "bin", "python")
    if not os.path.isfile(py):
        return {"output": None, "error": f"vendor venv absent: {v.get('venv')}"}
    if not os.path.isdir(v["ref"]):
        return {"output": None, "error": f"vendor snapshot absent: {v['ref']}"}
    png = os.path.join(out_dir, f"{cell['id']}_vendor.png")
    clip = os.path.join(out_dir, f"{cell['id']}_vendor.mp4")
    cmd = [py, os.path.join(REPO, "tools", "vendor_video_repro.py"),
           "--snapshot", v["ref"], "--prompt", req["prompt"],
           "--seed", str(defaults.get("seed", 42)),
           "--steps", str(req.get("steps", 4)),
           "--frames", str(req.get("frames", 17)),
           "--height", str(req.get("height", 480)),
           "--width", str(req.get("width", 832)),
           "--out", png, "--clip-out", clip]
    if latents:
        cmd += ["--latents", latents]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not os.path.isfile(png):
        return {"output": None, "error": f"diffusers_video run failed: {p.stderr[-500:]}"}
    # Frame 0 out of the vendor's CLIP, by the same path ours takes. Comparing a
    # frame that crossed an mp4 encode against one taken straight from the array
    # charges the engine for the codec: measured 36.6 dB on the vendor frame
    # round-tripped alone, against a 30 dB bound.
    if os.path.isfile(clip):
        err = _first_frame(clip, png)
        if err:
            return {"output": None, "error": f"vendor clip unreadable: {err}"}
    return {"output": png, "error": None}


def vendor_unavailable(cell: Dict[str, Any], *_args, **_kwargs) -> Dict[str, Any]:
    return {"output": None,
            "error": f"no runner wired for vendor.kind={cell['vendor']['kind']!r} "
                     f"({cell['vendor'].get('ref')}) — declared, not yet runnable"}


def _first_frame(clip: str, png: str) -> str:
    """Frame 0 of a rendered clip, written as a PNG. Returns "" or the reason.

    Media file-I/O at the harness boundary, which is I/O and not compute (R34).
    """
    try:
        import imageio.v3 as iio
        frame = iio.imread(clip, index=0, plugin="pyav")
    except Exception:
        try:
            import imageio.v2 as iio2
            with iio2.get_reader(clip) as rd:
                frame = rd.get_data(0)
        except Exception as exc:
            return f"could not read frame 0 of {os.path.basename(clip)}: {exc}"
    try:
        from PIL import Image
        Image.fromarray(frame).save(png)
    except Exception as exc:
        return f"could not write frame 0: {exc}"
    return ""


# ── our side ────────────────────────────────────────────────────────────────

def run_ours(cell: Dict[str, Any], defaults: Dict[str, Any], mode: str,
             src: str, out_dir: str, timeout: int,
             latent_dir: str = "") -> Dict[str, Any]:
    req = cell.get("request", {})
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    if latent_dir:
        # Our arm runs first and writes the raw N(0,1) draw the vendor will be
        # handed. The seam lives at the one place both modes synthesize a
        # `randn` variable, so this works in compiled and in triton (R30).
        env["NBX_DUMP_INIT_LATENT"] = latent_dir
    cmd = [sys.executable, "-c",
           "import sys; from neurobrix.cli import main; sys.exit(main())",
           "run", "--model", cell["model"], "--seed", str(defaults.get("seed", 42))]
    if "prompt" in req:
        cmd += ["--prompt", req["prompt"]]
    if req.get("audio"):
        cmd += ["--audio", req["audio"]]
    if req.get("temperature") is not None:
        cmd += ["--temperature", str(req["temperature"])]
    if not (cell["compare"]["metric"].startswith("psnr")) and (
            req.get("max_tokens") or defaults.get("max_tokens")):
        # A render has no token budget; passing the text default to a diffusion
        # request puts a flag in the command that means nothing there.
        cmd += ["--max-tokens", str(req.get("max_tokens", defaults["max_tokens"]))]
    is_image = cell["compare"]["metric"].startswith("psnr")
    is_video = cell["compare"]["metric"] == "psnr_db_first_frame"
    png = os.path.join(out_dir, f"{cell['id']}_ours.png")
    # A video family writes a clip, not a still: handing it a .png output made
    # our arm fail rc=1 and the cell called it NOT-RUN for a reason that was the
    # harness's. It renders the clip, and frame 0 is extracted for the metric.
    clip = os.path.join(out_dir, f"{cell['id']}_ours.mp4")
    if is_video:
        cmd += ["--output", clip]
    elif is_image:
        cmd += ["--output", png]
    if is_image or is_video:
        # The request's render config belongs to BOTH arms, whichever kind of
        # render it is. Ours used to take the model's defaults while the vendor
        # rendered the declared step count — the same defect the retrace gate
        # closed for its image row. Guarding this on `is_image` alone put the
        # video cell straight back into it: Wan rendered its default 81 frames
        # against a 17-frame vendor arm, which is not slow, it is a different
        # request.
        if req.get("steps") is not None:
            cmd += ["--steps", str(req["steps"])]
        if req.get("guidance") is not None:
            cmd += ["--cfg", str(req["guidance"])]
        if req.get("height") is not None:
            cmd += ["--height", str(req["height"])]
        if req.get("width") is not None:
            cmd += ["--width", str(req["width"])]
        if is_video and req.get("frames") is not None:
            cmd += ["--num-frames", str(req["frames"])]
    if mode == "triton":
        cmd.append("--triton")
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"output": None, "error": f"our engine timed out after {timeout}s"}
    if p.returncode != 0:
        return {"output": None, "error": f"our engine failed rc={p.returncode}: "
                                         f"{p.stderr[-500:]}"}
    if is_video:
        if not os.path.isfile(clip):
            return {"output": None, "error": "our engine wrote no clip"}
        err = _first_frame(clip, png)
        return ({"output": png, "error": None} if not err
                else {"output": None, "error": err})
    if is_image:
        return ({"output": png, "error": None} if os.path.isfile(png)
                else {"output": None, "error": "our engine wrote no image"})
    # One extractor for the CLI's framing, shared with the verdict table. The
    # copy that lived here knew "Generated text:" but not the form the CLI
    # actually prints ("Generated <n> tokens"), so a correct TinyLlama
    # generation came back as the whole banner and the cell reported DIVERGES
    # on a harness bug. A cell that cries wolf is worth less than no cell.
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from moe_verdict_table import engine_text
    return {"output": engine_text(p.stdout), "error": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(REPO, "benchmarks", "harness",
                                                    "vendor_cells.yml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="/home/mlops/nbx_converge_mac/src")
    ap.add_argument("--mode", default="triton", choices=["triton", "compiled"])
    ap.add_argument("--family", default=None)
    ap.add_argument("--id", dest="cell_id", default=None)
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.cells))
    defaults = spec.get("defaults", {})
    os.makedirs(args.out, exist_ok=True)

    results = []
    for cell in spec["cells"]:
        if args.family and cell["family"] != args.family:
            continue
        if args.cell_id and cell["id"] != args.cell_id:
            continue
        # A cell whose comparison the harness cannot compute is NOT-RUN, never a
        # pass — the ladder cell below is driven by moe_real_path_check.py.
        metric = cell["compare"]["metric"]
        row: Dict[str, Any] = {
            "id": cell["id"], "family": cell["family"], "model": cell["model"],
            "vendor": cell["vendor"], "metric": metric,
            "bound": cell["compare"]["bound"], "blind_to": cell.get("blind_to", "").strip(),
        }
        t0 = time.time()
        if metric not in METRICS:
            # A cell can declare its own driver when the generic one-shot path
            # cannot express it (the MoE ladder needs eight loads at eight
            # lengths). Naming the command is the honest report; silently
            # counting it green would be the dishonest one.
            driver = cell.get("driven_by")
            row.update(verdict="DELEGATED" if driver else "NOT-RUN",
                       reason=(f"driven by its own tool: {driver}" if driver else
                               f"metric {metric!r} has no implementation in this harness"))
            results.append(row)
            print(f"[cell] {cell['id']:32s} {row['verdict']:9s} ({row['reason']})",
                  flush=True)
            continue

        # The vendor stack is checked BEFORE either arm runs: a `latent_pinned`
        # cell must run OURS first (it produces the latent the vendor starts
        # from), and loading a model for an absent vendor is waste, not evidence.
        unavailable = vendor_available(cell)
        if unavailable:
            row.update(verdict="NOT-RUN", reason=unavailable,
                       seconds=round(time.time() - t0, 1))
            results.append(row)
            print(f"[cell] {cell['id']:32s} NOT-RUN  ({unavailable[:90]})", flush=True)
            continue

        gate = cell.get("gate", {})
        gate_kind = gate.get("kind", "deterministic_output")
        row["gate"] = gate_kind
        pinned = gate_kind == "latent_pinned"
        latent_dir = os.path.join(args.out, f"{cell['id']}_latent") if pinned else ""

        ours = run_ours(cell, defaults, args.mode, args.src, args.out, args.timeout,
                        latent_dir=latent_dir)
        if ours["error"]:
            row.update(verdict="NOT-RUN", reason=ours["error"],
                       seconds=round(time.time() - t0, 1))
            results.append(row)
            print(f"[cell] {cell['id']:32s} NOT-RUN  ({ours['error'][:90]})", flush=True)
            continue

        latents = ""
        if pinned:
            dumped = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
            if not dumped:
                # The primary gate cannot be run without the latent, and the
                # weaker one is not its equivalent: say so rather than silently
                # comparing two independent samples and calling it a divergence.
                row.update(verdict="NOT-RUN", seconds=round(time.time() - t0, 1),
                           reason="the primary gate is latent_pinned and our arm "
                                  "dumped no latent (NBX_DUMP_INIT_LATENT wrote "
                                  f"nothing to {latent_dir}); comparing independent "
                                  "samples would not be this gate")
                results.append(row)
                print(f"[cell] {cell['id']:32s} NOT-RUN  (no latent dumped — primary "
                      f"gate not runnable)", flush=True)
                continue
            latents = dumped[0]
            row["latent"] = latents

        kind = cell["vendor"]["kind"]
        if kind == "ollama":
            ven = vendor_ollama(cell, defaults, args.ollama_host)
        elif kind == "faster_whisper":
            ven = vendor_faster_whisper(cell, defaults)
        elif kind == "diffusers":
            ven = vendor_diffusers(cell, defaults, args.out, latents=latents)
        elif kind == "diffusers_video":
            ven = vendor_diffusers_video(cell, defaults, args.out, latents=latents)
        else:
            ven = vendor_unavailable(cell)
        if ven["error"]:
            row.update(verdict="NOT-RUN", reason=ven["error"],
                       seconds=round(time.time() - t0, 1))
            results.append(row)
            print(f"[cell] {cell['id']:32s} NOT-RUN  ({ven['error'][:90]})", flush=True)
            continue

        value, passed = METRICS[metric](ours["output"], ven["output"],
                                        cell["compare"]["bound"])
        row.update(verdict="AGREES" if passed else "DIVERGES", value=value,
                   ours=ours["output"] if not metric.startswith("psnr") else ours["output"],
                   theirs=ven["output"], seconds=round(time.time() - t0, 1))
        results.append(row)
        print(f"[cell] {cell['id']:32s} {row['verdict']:8s} {metric}={value} "
              f"bound={cell['compare']['bound']}", flush=True)

    path = os.path.join(args.out, "vendor_cells.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    agree = sum(1 for r in results if r["verdict"] == "AGREES")
    diverge = sum(1 for r in results if r["verdict"] == "DIVERGES")
    notrun = sum(1 for r in results if r["verdict"] == "NOT-RUN")
    deleg = sum(1 for r in results if r["verdict"] == "DELEGATED")
    print(f"\n{agree} agree · {diverge} diverge · {notrun} not-run · "
          f"{deleg} delegated (not-run is a finding, never a pass)")
    print(f"written: {path}")
    return 1 if diverge else 0


if __name__ == "__main__":
    raise SystemExit(main())
