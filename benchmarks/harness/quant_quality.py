"""Quantized-tier quality harness — row-verdict clause 2 instruments.

Two measurements, quantized variant vs its full-precision sibling:

1. PERPLEXITY on the pinned corpus (benchmarks/assets/
   wikitext2_raw_test.parquet, sha recorded in the report): the causal
   LM is teacher-forced by PREFILL — one forward over each W-token
   window yields logits at every position; CE(next token) accumulates
   over all windows. Both artifacts run the SAME windows on the SAME
   engine (triton).
2. GREEDY ANSWER SHAS on the reference prompts: `neurobrix run
   --triton` per prompt per artifact; the answer text's sha256 and the
   text itself land in the report (the eye reads the answers, the sha
   pins them).

Usage:
  python3 benchmarks/harness/quant_quality.py \
      --model Qwen3-Coder-30B-A3B-Instruct-int4g128 \
      --reference Qwen3-Coder-30B-A3B-Instruct \
      --gpu 2 --hardware v100-32g --windows 8 --window-len 512 \
      --out validation_outputs/<dir>/quality.json

The fp16 reference of a >32G model cannot ride one card — pass
--skip-reference-ppl and the report carries the documented reason;
greedy shas for the reference then come from its own multi-GPU
config (--reference-gpu '').
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PROMPTS = [
    "Write a Python function that reverses a string.",
    "Explain the difference between a list and a tuple in Python.",
    "What is the capital of France?",
    "Write a SQL query that selects the top 5 rows of a table.",
]


def corpus_windows(n_windows: int, window_len: int, tokenizer):
    import pyarrow.parquet as pq
    path = REPO / "benchmarks" / "assets" / "wikitext2_raw_test.parquet"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    text = "\n".join(t for t in pq.read_table(path).column("text").to_pylist()
                     if t.strip())
    ids = tokenizer.encode(text)
    windows = []
    for i in range(n_windows):
        start = i * window_len
        chunk = ids[start: start + window_len + 1]
        if len(chunk) < window_len + 1:
            break
        windows.append(chunk)
    return windows, sha


def measure_ppl(model_name: str, gpu: str, hardware: str,
                n_windows: int, window_len: int) -> dict:
    """Teacher-forced perplexity via prefill logits.

    Requires the engine's teacher-forced scoring entry (`score
    window -> all-position logits` on the autoregressive flow) —
    the NAMED increment before the row VERDICT. Refuses loudly until
    it lands (never a fake number)."""
    raise NotImplementedError(
        "ppl instrument pending the engine scoring entry (all-position "
        "lm_head over a prefill window on the triton autoregressive "
        "flow) — named increment 'quant-ppl-scoring'; the row verdict "
        "waits for it. Use --shas-only meanwhile.")


def greedy_shas(model_name: str, gpu: str, hardware: str,
                out_dir: Path, tag: str) -> list:
    import os
    rows = []
    for i, prompt in enumerate(PROMPTS):
        out = out_dir / f"greedy_{tag}_{i}.txt"
        env = dict(os.environ)
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [sys.executable, "-m", "neurobrix", "run",
               "--model", model_name, "--prompt", prompt,
               "--max-tokens", "48", "--triton", "--output", str(out)]
        if hardware:
            cmd[4:4] = ["--hardware", hardware]
        r = subprocess.run(cmd, env=env, cwd=str(REPO),
                           capture_output=True, text=True, timeout=1800)
        text = out.read_text() if out.exists() else f"<rc={r.returncode}>"
        rows.append({"prompt": prompt,
                     "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                     "answer": text})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--reference-gpu", default=None,
                    help="GPU set for the reference (default: same; '' = all)")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--reference-hardware", default=None)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--skip-reference-ppl", action="store_true")
    ap.add_argument("--ppl-only", action="store_true")
    ap.add_argument("--shas-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "reference": args.reference}

    if not args.shas_only:
        print(f"== ppl {args.model}")
        report["ppl_quant"] = measure_ppl(
            args.model, args.gpu, args.hardware,
            args.windows, args.window_len)
        if args.skip_reference_ppl:
            report["ppl_reference"] = {
                "skipped": "full-precision build exceeds one-GPU VRAM "
                           "(documented row note)"}
        else:
            print(f"== ppl {args.reference}")
            report["ppl_reference"] = measure_ppl(
                args.reference, args.gpu,
                args.reference_hardware or args.hardware,
                args.windows, args.window_len)

    if not args.ppl_only:
        print(f"== greedy shas {args.model}")
        report["greedy_quant"] = greedy_shas(
            args.model, args.gpu, args.hardware, out_path.parent, "quant")
        print(f"== greedy shas {args.reference}")
        ref_gpu = args.gpu if args.reference_gpu is None else args.reference_gpu
        report["greedy_reference"] = greedy_shas(
            args.reference, ref_gpu,
            args.reference_hardware or "", out_path.parent, "ref")

    out_path.write_text(json.dumps(report, indent=1))
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
