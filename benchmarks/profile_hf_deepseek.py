"""
HuggingFace DeepSeek-MoE-16B Benchmark
Compare with NeuroBrix using same hardware (multi-GPU, Accelerate device_map=auto)

Run with: /tmp/deepseek_bench_env/bin/python benchmarks/profile_hf_deepseek.py
"""
import os
import torch
import time

# Requires HF_TOKEN in the environment — set it before running, e.g.
#   export HF_TOKEN=hf_xxx
# Deliberately not hardcoded so this file can be committed to a public repo.
if not os.environ.get('HF_TOKEN'):
    raise RuntimeError(
        "HF_TOKEN not set. Export a HuggingFace access token in the shell "
        "before running this benchmark.")

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_ID = "deepseek-ai/deepseek-moe-16b-chat"

def main():
    print("=" * 60)
    print("HuggingFace DeepSeek-MoE-16B Benchmark")
    print("=" * 60)

    # Load
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model (fp16, device_map=auto — Accelerate multi-GPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # V100 doesn't support flash_attn (needs sm80+)
    )
    model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Device map
    if hasattr(model, "hf_device_map"):
        from collections import Counter
        devices = Counter(model.hf_device_map.values())
        print("\nAccelerate device distribution:")
        for dev, count in devices.most_common():
            print(f"  {dev}: {count} layers/params")

    print("\nVRAM usage:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        name = torch.cuda.get_device_properties(i).name
        print(f"  GPU {i} ({name}): {alloc:.1f}/{total:.1f} GB")

    # Warmup
    print("\nWarmup...")
    inputs = tokenizer("Hi", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    warmup_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Warmup OK: {warmup_text[:50]}")

    # Benchmark
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis step by step.",
    ]

    print(f"\nBenchmark (100 tokens, {len(prompts)} runs):")
    print("-" * 60)

    all_tps = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        t = time.time() - t0

        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        tps = new_tokens / t if t > 0 else 0
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:100]
        all_tps.append(tps)
        print(f"  Gen {i+1}: {t:.2f}s | {new_tokens} tok | {tps:.1f} tok/s")
        print(f"    {decoded}...")

    avg_tps = sum(all_tps) / len(all_tps)
    print(f"\n{'=' * 60}")
    print(f"HuggingFace Average: {avg_tps:.1f} tok/s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
