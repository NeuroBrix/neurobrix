#!/usr/bin/env python
"""The instruments that judge an artefact from OUTSIDE the engine (R29, hardened
by the owner 2026-09-16): bytes identical between two arms of this engine prove
the two arms agree, not that either is right. What is outside the engine:

* `stt`     — faster-whisper (a third-party ASR, /home/mlops/venvs/fwhisper)
              transcribes a WAV. Used two ways: as the control for our own
              transcription, and as the reader of what our TTS synthesised.
* `wer`     — word error rate between a text and a text known in advance.
* `image`   — the degeneracy facts a flat or white frame cannot survive
              (standard deviation, distinct colours, uniform rows/columns) and,
              when a reference is given, the correlation with its bicubic
              upscale. The CONTENT is judged by looking at the file; these
              numbers accompany that look, they do not replace it.
* `video`   — the same per frame, plus the change between consecutive frames
              (a frozen sequence has none), and a contact sheet to look at.
* `audio`   — duration, RMS, peak, longest silence (a mute file has no level).

Every subcommand prints a JSON record and exits 1 when its own mechanical test
fails, so a chain stops on a degenerate artefact instead of recording it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _norm(t: str) -> list:
    keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in t)
    return keep.split()


def wer(expected: str, got: str) -> dict:
    a, b = _norm(expected), _norm(got)
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return {"words_expected": len(a), "edits": d[len(a)][len(b)],
            "wer": d[len(a)][len(b)] / max(1, len(a)), "expected": " ".join(a), "got": " ".join(b)}


def stt(path: str, language=None, model="base") -> dict:
    """faster-whisper, in its own interpreter — nothing of this engine is loaded."""
    code = (
        "import json,sys\n"
        "from faster_whisper import WhisperModel\n"
        f"m=WhisperModel({model!r}, device='cpu', compute_type='int8')\n"
        f"segs,info=m.transcribe({path!r}, language={language!r}, beam_size=5)\n"
        "t=' '.join(s.text.strip() for s in segs)\n"
        "print(json.dumps({'text':t,'language':info.language,'duration':info.duration}))\n"
    )
    r = subprocess.run(["/home/mlops/venvs/fwhisper/bin/python", "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr.strip()[-400:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def image_facts(path: str, reference=None) -> dict:
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(path).convert("RGB")).astype(np.float64)
    flat = a.reshape(-1, 3)
    rows_uniform = int(sum(1 for r in range(a.shape[0]) if a[r].std() < 0.5))
    cols_uniform = int(sum(1 for c in range(a.shape[1]) if a[:, c].std() < 0.5))
    out = {"path": path, "size": [a.shape[1], a.shape[0]], "std": float(a.std()),
           "distinct_colours": int(len(np.unique(flat.astype(np.uint8), axis=0))),
           "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
           "uniform_rows": rows_uniform, "uniform_cols": cols_uniform,
           "uniform_row_fraction": rows_uniform / a.shape[0]}
    if reference:
        ref = Image.open(reference).convert("RGB").resize((a.shape[1], a.shape[0]), Image.BICUBIC)
        b = np.array(ref).astype(np.float64)
        x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
        out["reference"] = reference
        out["corr_with_bicubic_reference"] = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))
    out["degenerate"] = bool(out["std"] < 1.0 or out["distinct_colours"] < 16
                             or out["uniform_row_fraction"] > 0.98)
    return out


def video_facts(path: str, sheet=None) -> dict:
    import numpy as np
    import imageio.v3 as iio
    frames = np.stack(list(iio.imread(path, plugin="pyav"))).astype(np.float64)
    per = [{"frame": i, "std": float(frames[i].std()),
            "distinct": int(len(np.unique(frames[i].reshape(-1, 3).astype(np.uint8), axis=0)))}
           for i in range(frames.shape[0])]
    deltas = [float(np.abs(frames[i + 1] - frames[i]).mean()) for i in range(frames.shape[0] - 1)]
    if sheet:
        from PIL import Image
        n = frames.shape[0]
        cols = min(4, n); rows = (n + cols - 1) // cols
        h, w = frames.shape[1], frames.shape[2]
        s = Image.new("RGB", (cols * w, rows * h))
        for i in range(n):
            s.paste(Image.fromarray(frames[i].astype(np.uint8)), ((i % cols) * w, (i // cols) * h))
        s.save(sheet)
    out = {"path": path, "frames": int(frames.shape[0]), "size": [frames.shape[2], frames.shape[1]],
           "per_frame": per, "mean_abs_change_between_frames": deltas,
           "contact_sheet": sheet}
    out["degenerate"] = bool(min(p["std"] for p in per) < 1.0
                             or max(p["distinct"] for p in per) < 16
                             or (deltas and max(deltas) < 0.5))
    return out


def audio_facts(path: str) -> dict:
    import wave
    import numpy as np
    w = wave.open(path)
    n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    win = max(1, sr // 50)
    env = np.abs(a[: len(a) // win * win].reshape(-1, win)).max(1)
    silent = env < 1e-3
    longest = 0; run = 0
    for s_ in silent:
        run = run + 1 if s_ else 0
        longest = max(longest, run)
    out = {"path": path, "seconds": n / sr, "sample_rate": sr, "channels": ch,
           "rms": float(np.sqrt((a * a).mean())), "peak": float(np.abs(a).max()),
           "longest_silence_s": longest * win / sr}
    out["degenerate"] = bool(out["rms"] < 1e-4 or out["peak"] < 1e-3
                             or out["longest_silence_s"] > 0.9 * out["seconds"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stt"); p.add_argument("wav"); p.add_argument("--language"); p.add_argument("--model", default="base")
    p = sub.add_parser("wer"); p.add_argument("expected_file"); p.add_argument("got_file")
    p = sub.add_parser("image"); p.add_argument("png"); p.add_argument("--reference")
    p = sub.add_parser("video"); p.add_argument("mp4"); p.add_argument("--sheet")
    p = sub.add_parser("audio"); p.add_argument("wav")
    a = ap.parse_args()
    if a.cmd == "stt":
        rec = stt(a.wav, a.language, a.model)
    elif a.cmd == "wer":
        rec = wer(open(a.expected_file).read(), open(a.got_file).read())
    elif a.cmd == "image":
        rec = image_facts(a.png, a.reference)
    elif a.cmd == "video":
        rec = video_facts(a.mp4, a.sheet)
    else:
        rec = audio_facts(a.wav)
    print(json.dumps(rec, indent=1))
    return 1 if (rec.get("degenerate") or rec.get("error")) else 0


if __name__ == "__main__":
    sys.exit(main())
