#!/usr/bin/env python3
"""Per-cell gated matrix for the PROMPT families (tts, llm) — the rows the
image/audio runner does not cover. Same gate (Prism's planned memory vs
available_mb), same three modes, artefacts written to disk for R29 judgment:
a TTS wav is judged by passing it back through an STT; an LLM text is read.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

MODES = ("compiled", "triton", "triton-sequential")
_FLAG = {"compiled": [], "triton": ["--triton"], "triton-sequential": ["--triton-sequential"]}

PROMPT_LLM = "In one sentence, what is the capital of France?"
PROMPT_TTS = "The quick brown fox jumps over the lazy dog."

MODELS = [
    ("Kokoro-82M", PROMPT_TTS, "wav"),
    ("TinyLlama-1.1B-Chat", PROMPT_LLM, "txt"),
    ("TinyLlama-1.1B-Chat-v1.0", PROMPT_LLM, "txt"),
]
MARGIN = 1.25


def available_mb():
    from neurobrix.core.host_memory import memory_state
    return memory_state().available_mb


def prism_need_mb(model, prompt):
    env = dict(os.environ); env["PYTHONPATH"] = "src"
    try:
        r = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--model", model,
                            "--prompt", prompt, "--explain-plan"],
                           capture_output=True, text=True, timeout=180, env=env)
        m = re.search(r"planned memory\s+([0-9]+)\s*MB", r.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def run_one(model, prompt, ext, mode, out_dir, timeout=1200):
    out = out_dir / f"{model}_{mode}.{ext}"
    env = dict(os.environ); env["PYTHONPATH"] = "src"
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model,
           "--prompt", prompt, *_FLAG[mode], "--output", str(out)]
    try:
        r = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
        return r.returncode, (str(out) if out.exists() else None)
    except subprocess.TimeoutExpired:
        return 124, (str(out) if out.exists() else None)


def main():
    out_dir = Path("validation_outputs/apple_matrix_percell_2026_09_16/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    res_path = Path("validation_outputs/apple_matrix_percell_2026_09_16/prompt_rows.json")
    for model, prompt, ext in MODELS:
        avail = available_mb(); need = prism_need_mb(model, prompt)
        if need is not None and need * MARGIN > avail:
            print(f"[GATED] {model}: need {need}MB avail {avail}MB", flush=True)
            rows.append({"model": model, "gated_out": True, "need_mb": need, "available_mb": avail})
            res_path.write_text(json.dumps(rows, indent=1)); continue
        print(f"[RUN] {model}: need {need}MB avail {avail}MB -> fits", flush=True)
        row = {"model": model, "prompt": prompt, "need_mb": need,
               "available_mb_at_run": avail, "modes": {}}
        for mode in MODES:
            t0 = time.time()
            rc, art = run_one(model, prompt, ext, mode, out_dir)
            row["modes"][mode] = {"rc": rc, "artefact": art, "seconds": round(time.time() - t0, 1)}
            print(f"    {mode}: rc={rc} artefact={art}", flush=True)
        rows.append(row); res_path.write_text(json.dumps(rows, indent=1))
    print(f"DONE {len(rows)} prompt cells", flush=True)


if __name__ == "__main__":
    sys.exit(main())
