> **HISTORICAL STARTING POINT — pre-mandate numbers, kept verbatim as the zero-point of the optimization curve. The current one-pager is `../SUMMARY_2026_08_30.md` (capacity first, trajectory section names this file as its starting column).**

# Benchmark results — 2026-08-04 (row: llm_dense_tinyllama)

TinyLlama-1.1B-Chat-v1.0, fp16 everywhere, greedy, 128 new tokens
requested, warm serving, N=5 (median reported), GPU 3 (V100-SXM2-32GB),
driver 535.309.01, engine commit `e205699`. Raw per-request timings in
the per-cell JSONs beside this file; environment in
`env_manifest.json`.

| column | tok/s (med) | TTFT (med) | wall/req (med) | peak VRAM | cold start |
|---|---|---|---|---|---|
| vLLM 0.7.3 (XFormers, V0) | 205.4 | 35 ms | 0.65 s | 30,059 MiB¹ | 39.1 s |
| Ollama 0.24.0 (f16 GGUF) | 209.7 | 61 ms | 0.65 s | 2,477 MiB | 16.2 s |
| NeuroBrix compiled | ~19² | n/a³ | 6.75 s | 2,859 MiB | 6.0 s |
| NeuroBrix triton | ~5.7² | n/a³ | 22.57 s | 2,795 MiB | 7.0 s |

¹ vLLM pre-allocates 90% of VRAM for its KV cache by design
  (`gpu_memory_utilization=0.9` default) — its peak measures the
  allocator policy, not model need.
² Estimated from wall / 128 requested tokens: the serving daemon does
  not yet report generated-token counts (recorded harness debt — a
  serving-side counter lands with the daemon streaming work). Ollama
  stopped at EOS (113 tokens, its own eval_count/eval_duration used);
  vLLM forced exactly 128 via ignore_eos.
³ Daemon RPC is not wire-streamed yet — TTFT measurable only on the
  competitor columns this round.

## Reading (the honest one)

The competitor baselines sit at ~205-210 tok/s on this dense 1.1B row;
our compiled engine is ~10x behind and triton ~35x behind. This is the
gap the optimization program exists to close, and the Phase 1 gains
map already names the levers on this exact model: per-op launch tax
(the `replay` pass), 88 vertical-fusion sites, 199 const-foldable ops,
plus the known Volta Triton matmul ceiling. Every optimization pass
now has its baseline; the benchmarks re-run after each landed phase.

## Incidents recorded

- vLLM venv needed an era pin `transformers==4.49.0` (0.7.3 declares
  `>=4.48.2` unbounded; transformers 5.x removed
  `all_special_tokens_extended`) — recorded in
  `benchmarks/config/backends.yml`.
- Pinned NeuroBrix serving requires the matching single-GPU hardware
  profile (`--hardware v100-32g`) — the battery closure-config
  pattern, now enforced by the runner.
