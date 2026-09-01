> **HISTORICAL STARTING POINT — pre-mandate numbers, kept verbatim as the zero-point of the optimization curve. The current one-pager is `../SUMMARY_2026_08_30.md` (capacity first, trajectory section names this file as its starting column).**

# Benchmark results — 2026-08-05 (row: llm_dense_tinyllama, hardened harness)

Re-measure of the NeuroBrix columns ONLY, after paying the two harness
debts recorded on 2026-08-04: the daemon now wire-streams per-token
events (`generate` + `stream=true` → real TTFT at the client wall) and
the runner reads the daemon's exact generated-token count (it always
existed under `tokens`; the 08-04 runner read the wrong keys). Decode
rate now uses the SAME formula as the vLLM cell:
`(n-1)/(t_last - t0 - ttft)` — decode-only, prefill excluded.

The competitor columns are carried over from
`../2026_08_04/SUMMARY.md` unchanged (their cells were streamed and
exact from day one — nothing in this change touches them).

TinyLlama-1.1B-Chat-v1.0, fp16 everywhere, greedy, 128 new tokens
(all 128 generated — exact counts, cross-checked against streamed
events), warm serving, N=5 (median), GPU 3 (V100-SXM2-32GB), driver
535.309.01. Engine tree = `de31b89` + the harness-hardening change
committed with this artifact (streaming sink proven byte-inert:
`validation_outputs/bench_harness_hardening_2026_08_05/`).

| column | tok/s decode (med) | TTFT (med) | wall/req (med) | peak VRAM | cold start |
|---|---|---|---|---|---|
| vLLM 0.7.3 (XFormers, V0)¹ | 205.4 | 35 ms | 0.65 s | 30,059 MiB | 39.1 s |
| Ollama 0.24.0 (f16 GGUF)¹ | 209.7 | 61 ms | 0.65 s | 2,477 MiB | 16.2 s |
| NeuroBrix compiled | **21.6** | **183 ms** | 6.16 s | 2,859 MiB | 6.0 s |
| NeuroBrix triton | **5.87** | **587 ms** | 22.19 s | 2,495 MiB | 7.0 s |

¹ Carried over from 2026-08-04 (measured with the same client-side
  streamed-wall methodology; vLLM peak = its 0.9 prealloc policy).

## Deltas vs the 08-04 estimated row

- compiled: ~19 tok/s (wall/128 estimate) → 21.6 tok/s measured
  decode-only. The estimate UNDERSTATED by ~14% because it charged
  prefill + RPC framing to every token. TTFT 183 ms is now a fact.
- triton: ~5.7 → 5.87 tok/s decode-only, TTFT 587 ms. Wall unchanged
  within noise (22.19 s vs 22.57 s) — the per-token streaming sink
  costs nothing measurable, as the inertness gate predicted.

## The gap, restated on hard numbers

Decode: compiled 9.5× behind vLLM, triton 35.0×. TTFT: compiled 5.2×
behind, triton 16.8×. These are the zero-point coordinates the
optimization curve moves from; Phase 2 const_fold measures against
THIS file.
