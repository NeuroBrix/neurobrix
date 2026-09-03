#!/usr/bin/env python3
"""Vendor-of-record transcription for the Voxtral class (transformers stack
the artifact is traced from): fp16, greedy, the processor's own
long-form contract (`apply_transcription_request` → fixed 30 s windows
stacked on the batch axis, one BEGIN_AUDIO). Writes the transcript and
the wall time; the engine's long-form gate (D-AUDIOLLM-LONGFORM) compares
word for word after the bench_stt normalization.

  python3 benchmarks/harness/vendor_voxtral_transcribe.py --snapshot <dir> \
      --audio benchmarks/assets/librivox_aow0506_600s.wav --out <txt> [--lang en]
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lang", default="en")
ap.add_argument("--max-new-tokens", type=int, default=4096)
a = ap.parse_args()

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# The 7.5k-token prefill of a 600 s recording: the math SDPA path materialises
# the fp32 scores (6.7 GiB per layer) — force the memory-efficient kernel (V100 fp16).
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

proc = AutoProcessor.from_pretrained(a.snapshot)
model = VoxtralForConditionalGeneration.from_pretrained(
    a.snapshot, torch_dtype=torch.float16, device_map="auto",
    max_memory={0: "10GiB", 1: "26GiB"})   # weights split over two cards: the 7.5k-token attention needs ~7 GiB beside them
inputs = proc.apply_transcription_request(
    language=a.lang, audio=a.audio, model_id=a.snapshot, return_tensors="pt")
inputs = inputs.to(model.device, dtype=torch.float16)
print(f"input_features {tuple(inputs['input_features'].shape)} "
      f"input_ids {tuple(inputs['input_ids'].shape)}", flush=True)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
torch.cuda.synchronize()
dt = time.time() - t0
text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
open(a.out, "w").write(text.strip() + "\n")
print(f"generated {out.shape[1] - inputs['input_ids'].shape[1]} tokens in {dt:.1f}s -> {a.out}")
print(text[:300])
