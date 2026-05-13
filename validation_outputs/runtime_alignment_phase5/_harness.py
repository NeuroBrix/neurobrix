"""Phase 5 harness — run a single model smoke and write R29 artefacts.

Usage: python _harness.py <slug>
Reads the model spec from MODELS dict, runs neurobrix CLI, computes stats,
writes prompt.txt / stats.json / output.<ext> / verdict.md.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5")
NBX_BIN = "/home/mlops/ml/venv/bin/neurobrix"
TEST_WAV = "/home/mlops/NeuroBrix_System/test_speech_ref.wav"

# Per-model spec: model name, family, mode, prompt/audio, extra flags, expected verdict
MODELS = {
    "TinyLlama": {
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "family": "llm",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--prompt", "What is 2+2?", "--max-tokens", "30"],
        "expected": "PASS",
    },
    "DeepSeek-MoE": {
        "model": "deepseek-moe-16b-chat",
        "family": "llm",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--prompt", "What is 2+2?", "--max-tokens", "30"],
        "expected": "PASS",
    },
    "Qwen3-30B": {
        "model": "Qwen3-30B-A3B-Thinking-2507",
        "family": "llm",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--prompt", "What is 2+2?", "--max-tokens", "30"],
        "expected": "PASS",
    },
    "PixArt-XL": {
        "model": "PixArt-XL-2-1024-MS",
        "family": "image",
        "mode": "t2i",
        "ext": ".png",
        "flags": ["--prompt", "a red apple on a white plate", "--steps", "12"],
        "expected": "PASS",
    },
    "PixArt-Sigma": {
        "model": "PixArt-Sigma-XL-2-1024-MS",
        "family": "image",
        "mode": "t2i",
        "ext": ".png",
        "flags": ["--prompt", "a red apple on a white plate", "--steps", "12"],
        "expected": "PASS",
    },
    "Sana-1024-MultiLing": {
        "model": "Sana_1600M_1024px_MultiLing",
        "family": "image",
        "mode": "t2i",
        "ext": ".png",
        "flags": ["--prompt", "a red apple on a white plate", "--steps", "12"],
        "expected": "PASS",
    },
    "Janus-image": {
        "model": "Janus-Pro-7B",
        "family": "multimodal",
        "mode": "image",
        "ext": ".png",
        "flags": ["--mode", "image", "--prompt", "a red cat"],
        "expected": "PASS",
    },
    "Janus-text": {
        "model": "Janus-Pro-7B",
        "family": "multimodal",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--mode", "text", "--prompt", "describe a cat"],
        "expected": "SKIP",  # build is image-only; expect clear error
    },
    "Whisper-V3-Turbo": {
        "model": "whisper-large-v3-turbo",
        "family": "stt",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--audio", TEST_WAV],
        "expected": "PASS",
    },
    "Voxtral": {
        "model": "Voxtral-Mini-3B-2507",
        "family": "audio_llm",
        "mode": "text",
        "ext": ".txt",
        "flags": ["--audio", TEST_WAV, "--prompt", "what is being said?", "--max-tokens", "40"],
        "expected": "PASS",
    },
    "Chatterbox": {
        "model": "chatterbox",
        "family": "tts",
        "mode": "audio",
        "ext": ".wav",
        "flags": ["--prompt", "Hello world"],
        "expected": "PASS",
    },
    "Orpheus": {
        "model": "orpheus-3b-0.1-ft",
        "family": "tts",
        "mode": "audio",
        "ext": ".wav",
        "flags": ["--prompt", "Hello world"],
        "expected": "PASS",  # may FAIL on independent GQA bug
    },
    "Sana-4Kpx": {
        "model": "Sana_1600M_4Kpx_BF16",
        "family": "image",
        "mode": "t2i",
        "ext": ".png",
        "flags": ["--prompt", "a red apple on a white plate", "--steps", "12"],
        "expected": "FAIL",  # known OOM, out of scope
    },
}


def compute_stats_image(path: Path):
    from PIL import Image
    import numpy as np
    img = np.array(Image.open(path).convert("RGB"))
    return {
        "shape": list(img.shape),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": int(img.min()),
        "max": int(img.max()),
        "size_bytes": path.stat().st_size,
    }


def compute_stats_audio(path: Path, slug_dir: Path):
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        n_frames = w.getnframes()
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n_frames)
    duration = n_frames / rate
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        arr = arr.reshape(-1, ch)[:, 0]
    arr_norm = arr / (2 ** (8 * sw - 1))
    rms = float(np.sqrt(np.mean(arr_norm ** 2)))
    var_window = 0.0
    win = max(1, len(arr_norm) // 20)
    if len(arr_norm) >= win * 2:
        chunks = [arr_norm[i:i + win] for i in range(0, len(arr_norm) - win, win)]
        chunk_means = [float(c.mean()) for c in chunks]
        var_window = float(np.var(chunk_means))
    stats = {
        "duration_s": round(duration, 3),
        "sample_rate": rate,
        "channels": ch,
        "rms": round(rms, 6),
        "variance_temporal_windows": round(var_window, 8),
        "size_bytes": path.stat().st_size,
    }
    # Slice to 10s if longer (R29)
    if duration > 10.0:
        try:
            sliced = slug_dir / "output_10s.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(path), "-t", "10", "-c", "copy", str(sliced)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeort=30)
            if r.returncode == 0:
                stats["sliced_to_10s"] = sliced.name
            else:
                stats["slice_warning"] = "ffmpeg copy failed; full wav kept"
        except Exception as e:
            stats["slice_warning"] = f"slice unavailable ({type(e).__name__}); full wav kept"
    return stats


def compute_stats_text(path: Path):
    text = path.read_text(errors="replace")
    if len(text) > 2000:
        text = text[:2000]
        path.write_text(text)
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "preview": text[:200],
        "size_bytes": path.stat().st_size,
    }


def run_model(slug: str, spec: dict):
    slug_dir = ROOT / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    output_path = slug_dir / f"output{spec['ext']}"
    # Clean previous output
    if output_path.exists():
        output_path.unlink()
    # Write prompt.txt
    prompt = " ".join(spec["flags"])
    (slug_dir / "prompt.txt").write_text(prompt + "\n")
    # Build CLI command
    cmd = [NBX_BIN, "run", "--model", spec["model"], "--output", str(output_path)] + spec["flags"]
    # Run
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeort=600)
    duration = time.time() - t0
    stdort_tail = proc.stdort[-3000:] if proc.stdort else ""
    stderr_tail = proc.stderr[-3000:] if proc.stderr else ""
    # Save logs
    (slug_dir / "stdort.log").write_text(stdort_tail)
    (slug_dir / "stderr.log").write_text(stderr_tail)
    # Compute stats
    stats = {
        "duration_s": round(duration, 2),
        "exit_code": proc.returncode,
        "expected": spec["expected"],
        "family": spec["family"],
        "mode": spec["mode"],
    }
    verdict = "FAIL"
    fail_reason = ""
    if proc.returncode != 0:
        if spec["expected"] == "FAIL":
            verdict = "FAIL_EXPECTED"
            fail_reason = stderr_tail.split("\n")[-2] if stderr_tail else "exit != 0"
        elif spec["expected"] == "SKIP":
            # SKIP for Janus text-mode: we WANT a clear error. Check it's clean.
            if "ZERO FALLBACK" in stdort_tail or "ZERO FALLBACK" in stderr_tail or "ERROR:" in stdort_tail:
                verdict = "SKIP_EXPECTED"
            else:
                verdict = "FAIL"
                fail_reason = "Expected clear ZERO FALLBACK error, got different failure"
        else:
            verdict = "FAIL"
            fail_reason = (stderr_tail.split("\n")[-2] if stderr_tail else "") or "exit != 0"
    else:
        if spec["expected"] == "SKIP":
            verdict = "FAIL"
            fail_reason = "Expected SKIP/error, got success"
        elif output_path.exists():
            try:
                if spec["ext"] == ".png":
                    stats.update({"output": compute_stats_image(output_path)})
                    img = stats["output"]
                    # Generors mean band (white-plate compositions skew bright);
                    # std is the real discriminator vs noise/uniform-blob.
                    if 30 <= img["mean"] <= 240 and img["std"] > 30:
                        verdict = "PASS"
                    else:
                        verdict = "FAIL"
                        fail_reason = f"image stats ort of range: mean={img['mean']:.1f}, std={img['std']:.1f}"
                elif spec["ext"] == ".wav":
                    stats.update({"output": compute_stats_audio(output_path, slug_dir)})
                    aud = stats["output"]
                    if aud["duration_s"] > 0.5 and aud["rms"] > 1e-3 and aud["variance_temporal_windows"] > 1e-7:
                        verdict = "PASS"
                    else:
                        verdict = "FAIL"
                        fail_reason = f"audio stats ort of range: dur={aud['duration_s']}, rms={aud['rms']}, var={aud['variance_temporal_windows']}"
                elif spec["ext"] == ".txt":
                    stats.update({"output": compute_stats_text(output_path)})
                    txt = stats["output"]
                    if txt["words"] >= 3:
                        verdict = "PASS"
                    else:
                        verdict = "FAIL"
                        fail_reason = f"text too short: {txt['words']} words"
            except Exception as e:
                verdict = "FAIL"
                fail_reason = f"stats compute error: {type(e).__name__}: {e}"
        else:
            verdict = "FAIL"
            fail_reason = "no output file produced"
    stats["verdict"] = verdict
    if fail_reason:
        stats["fail_reason"] = fail_reason
    (slug_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    # verdict.md
    md = [
        f"# {slug} — verdict",
        "",
        f"**Verdict agent**: {verdict}",
        f"**Family**: {spec['family']}  •  **Mode**: {spec['mode']}",
        f"**Duration**: {duration:.1f}s  •  **Exit**: {proc.returncode}",
    ]
    if fail_reason:
        md.append(f"**Reason**: {fail_reason}")
    md.extend([
        "",
        "**Relaunch**:",
        "```",
        " ".join(cmd),
        "```",
        "",
        "Hocine validation: TODO",
    ])
    (slug_dir / "verdict.md").write_text("\n".join(md) + "\n")
    return stats, verdict, duration


def main():
    if len(sys.argv) < 2:
        print("Usage: _harness.py <slug>")
        sys.exit(1)
    slug = sys.argv[1]
    if slug not in MODELS:
        print(f"Unknown slug: {slug}")
        sys.exit(1)
    spec = MODELS[slug]
    print(f"\n=== {slug} ({spec['family']}/{spec['mode']}) ===")
    stats, verdict, dur = run_model(slug, spec)
    print(f"  → {verdict} in {dur:.1f}s")
    if "fail_reason" in stats:
        print(f"  reason: {stats['fail_reason']}")


if __name__ == "__main__":
    main()
