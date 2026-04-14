"""
HuggingFace DeepSeek-MoE-16B Benchmark
Compare with NeuroBrix using same hardware (multi-GPU, Accelerate device_map=auto)

Prerequisites
-------------
  * A HuggingFace access token with permission to pull
    `deepseek-ai/deepseek-moe-16b-chat`.
  * Set `HF_TOKEN` in the shell, OR put it in a `.env` file at the repo
    root (gitignored by default). Example `.env`:

        HF_TOKEN=hf_xxx

  * `pip install transformers accelerate torch` (plus optionally
    `python-dotenv` to get .env auto-loading).

Usage
-----
    python benchmarks/profile_hf_deepseek.py
"""
import os
import time


# Auto-load .env if present (optional, safe no-op when dotenv missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not set.\n"
        "Either export it in the shell (`export HF_TOKEN=hf_xxx`) or put it "
        "in a `.env` file at the repo root (which is gitignored). Installing "
        "python-dotenv enables .env auto-loading."
    )

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_ID = "deepseek-ai/deepseek-moe-16b-chat"


def main():
    print("=" * 60)
    print("HuggingFace DeepSeek-MoE-16B Benchmark")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=HF_TOKEN)

    print("Loading model (fp16, device_map=auto — Accelerate multi-GPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    if hasattr(model, "hf_device_map"):
        n_gpus = len({str(d) for d in model.hf_device_map.values()
                      if str(d).startswith("cuda")})
        print(f"  Device map: {n_gpus} GPU(s) used")

    prompt = "Hello"
    print(f"\nPrompt: {prompt!r}")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Warm up
    print("\nWarm-up run...")
    with torch.inference_mode():
        _ = model.generate(**inputs, generation_config=gen_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed run
    print("Timed run...")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=gen_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    tok_per_s = n_new / dt if dt > 0 else 0
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Generated {n_new} tokens in {dt:.2f}s  ({tok_per_s:.2f} tok/s)")
    print(f"  Output: {text!r}")


if __name__ == "__main__":
    main()
