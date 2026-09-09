#!/usr/bin/env python3
"""MoE routing — verdict on the real path, against the vendor.

The alert: a MoE trace burns the trace batch's per-expert token counts into
`aten::slice` bounds. If those bounds are executed, every generation routes with
the trace's expert histogram and the text is silently wrong.

Our sequential oracle replays the SAME graph, so it cannot see a defect of the
graph itself. The reference here is the ORIGINAL model served by ollama, same
prompt, greedy on both sides.

Two independent signals, so the verdict does not rest on the vendor alone:

  1. LENGTH SWEEP — frozen bounds would make coherence depend on the sequence
     length, and the trace length (23) is the only one where the burned bounds
     match the buffer. Prompts render to 22 / 23 / 24 tokens (one word apart)
     plus 10/12/16/43/68: a defect shows as "coherent at 23, broken elsewhere".
  2. RUNTIME ROUTING SHAPES — `NBX_MOE_DIAG=1` prints the shapes the fused op
     actually routes. Recomputed bounds give [seq_len, top_k] at the REAL
     seq_len; trace bounds would stay pinned at the trace's 23.

Usage:
    python3 tools/moe_real_path_check.py --model DeepSeek-Coder-V2-Lite-Instruct \
        --out validation_outputs/moe_routing_2026_09_09 [--mode triton] [--diag]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Rendered length under the container's own chat template (bos + "User: " +
# content + "\n\n" + "Assistant:"), measured with the tokenizer embedded in the
# .nbx — never a guess, and 23 is present ON PURPOSE as the control.
PROMPTS = [
    (10, "Say hi."),
    (12, "Name three primary colours."),
    (16, "Write a Python function that reverses a string."),
    (22, "Explain what a mixture-of-experts layer does inside a transformer model."),
    (23, "Explain briefly what a mixture-of-experts layer does inside a transformer model."),
    (24, "Explain in detail what a mixture-of-experts layer does inside a transformer model."),
    (43, "In the context of large language models, describe how a router assigns each "
         "token to a small number of expert feed-forward networks, and explain why that "
         "reduces the compute per token."),
    (68, "In the context of large language models, describe how a router assigns each "
         "token to a small number of expert feed-forward networks, explain why that "
         "reduces the compute cost per token, and mention one practical difficulty that "
         "arises when the experts are spread across several accelerators during both "
         "training and inference of the network."),
]

DIAG_RE = re.compile(r"\[MOE_DIAG\]\s+(\S+)\s+shape=(\[[^\]]*\])")


def run_engine(model: str, prompt: str, mode: str, max_tokens: int,
               seed: int, diag: bool, src: str, timeout: int) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    if diag:
        env["NBX_MOE_DIAG"] = "1"
    cmd = [sys.executable, "-c",
           "import sys; from neurobrix.cli import main; sys.exit(main())",
           "run", "--model", model, "--prompt", prompt,
           "--temperature", "0", "--seed", str(seed),
           "--max-tokens", str(max_tokens)]
    if mode == "triton":
        cmd.append("--triton")
    elif mode == "sequential":
        cmd.append("--sequential")
    elif mode == "triton_sequential":
        cmd.append("--triton-sequential")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, err, rc = (e.stdout or b"").decode("utf8", "replace"), \
                       (e.stderr or b"").decode("utf8", "replace"), -9
    # Routing shapes the fused op actually used, first occurrence per label.
    shapes: Dict[str, str] = {}
    for label, shape in DIAG_RE.findall(err):
        shapes.setdefault(label, shape)
    return {"returncode": rc, "seconds": round(time.time() - t0, 1),
            "stdout": out, "stderr_tail": err[-4000:], "routing_shapes": shapes}


def run_ollama(tag: str, prompt: str, max_tokens: int, seed: int,
               host: str, timeout: int) -> Dict[str, Any]:
    """The ORIGINAL model. Greedy, same seed, ollama applies the vendor template."""
    body = json.dumps({
        "model": tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return {"text": d.get("message", {}).get("content", ""),
                "seconds": round(time.time() - t0, 1), "error": None}
    except Exception as e:
        return {"text": "", "seconds": round(time.time() - t0, 1), "error": repr(e)}


def extract_generation(stdout: str) -> str:
    """The CLI frames the generated text; keep everything after the last marker."""
    for marker in ("Generated text:", "Output:", "=== OUTPUT ==="):
        if marker in stdout:
            return stdout.split(marker)[-1].strip()
    return stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DeepSeek-Coder-V2-Lite-Instruct")
    ap.add_argument("--ollama-tag", default="deepseek-coder-v2:16b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--src", default="/home/mlops/nbx_converge_mac/src")
    ap.add_argument("--mode", default="triton",
                    choices=["triton", "compiled", "sequential", "triton_sequential"])
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--diag", action="store_true",
                    help="NBX_MOE_DIAG=1 — routing shapes on the real path")
    ap.add_argument("--only-lengths", default=None,
                    help="comma-separated rendered lengths to run")
    ap.add_argument("--skip-vendor", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keep = ({int(x) for x in args.only_lengths.split(",")}
            if args.only_lengths else None)
    rows = []
    for length, prompt in PROMPTS:
        if keep is not None and length not in keep:
            continue
        print(f"[moe] len={length} mode={args.mode} diag={args.diag} …", flush=True)
        eng = run_engine(args.model, prompt, args.mode, args.max_tokens,
                         args.seed, args.diag, args.src, args.timeout)
        row = {"rendered_len": length, "prompt": prompt,
               "engine_mode": args.mode,
               "engine_rc": eng["returncode"], "engine_seconds": eng["seconds"],
               "engine_text": extract_generation(eng["stdout"]),
               "routing_shapes": eng["routing_shapes"],
               "engine_stderr_tail": eng["stderr_tail"]}
        if not args.skip_vendor:
            ven = run_ollama(args.ollama_tag, prompt, args.max_tokens,
                             args.seed, args.ollama_host, 600)
            row["vendor_text"] = ven["text"]
            row["vendor_error"] = ven["error"]
            row["vendor_seconds"] = ven["seconds"]
        rows.append(row)
        tag = f"len{length}_{args.mode}"
        with open(os.path.join(args.out, f"row_{tag}.json"), "w") as f:
            json.dump(row, f, indent=2)
        print(f"[moe] len={length} rc={eng['returncode']} "
              f"shapes={eng['routing_shapes']} "
              f"text={row['engine_text'][:120]!r}", flush=True)

    path = os.path.join(args.out, f"sweep_{args.mode}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
