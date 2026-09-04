#!/usr/bin/env python3
"""Warm-request GPU timeline for a NeuroBrix model — the served number's truth.

Loads the serving engine in-process, runs one warm-up request, then wraps
the SECOND request in cudaProfilerStart/Stop so `nsys profile
--capture-range=cudaProfilerApi` records exactly one served request
(no load, no first-request allocations). The CLI cold path profiles the
wrong thing for a warm daemon (2026-09-04: 10 s of allocs/syncs that the
locked protocol never pays).

Usage (under nsys):
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
       -t cuda,osrt -o OUT python tools/profile_warm_request.py MODEL --prompt ... --steps N --seed S
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--prompt", default="a red apple on a wooden table")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="compiled")
    ap.add_argument("--output", default="/tmp/warm_profiled.png")
    ap.add_argument("--audio", default=None, help="audio_path for STT / audio-LLM rows")
    args = ap.parse_args()

    import torch
    from neurobrix.serving.engine import InferenceEngine

    eng = InferenceEngine(args.model, args.hardware, args.mode)
    t0 = time.perf_counter(); eng.load(); print(f"[warm] load {time.perf_counter()-t0:.2f}s", flush=True)
    kw = dict(steps=args.steps, seed=args.seed)
    if args.audio:
        kw["audio_path"] = args.audio
    t0 = time.perf_counter(); eng.generate(args.prompt, **kw); print(f"[warm] warm-up request {time.perf_counter()-t0:.2f}s", flush=True)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter(); res = eng.generate(args.prompt, **kw); torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[warm] PROFILED request {wall:.3f}s timing={res.get('timing')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
