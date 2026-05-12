# P-PRISM-NEVER-REFUSE — S2 (P-NATIVE-SEQUENTIAL-CPU-DEBUG)

Sub-chantier: native `--sequential` mode adapted for runtime seq_len != trace_seq_len, unblocking CPU and validating GPU parity.

| Slug | Family | Mode | Verdict agent | Stats key | Size | Link | Hocine OK |
|---|---|---|---|---|---|---|---|
| tinyllama_cpu_sequential | llm | sequential (cpu-only-x86) | PASS (coherent text, 36.7s, 8 tok) | duration=36.69s | 33B | [output.txt](tinyllama_cpu_sequential/output.txt) | TODO |
| tinyllama_gpu_sequential | llm | sequential (v100-32g) | PASS anti-régression (coherent text, 3.8s, 8 tok) | duration=3.82s | 36B | [output.txt](tinyllama_gpu_sequential/output.txt) | TODO |

## Fix summary

Root cause: trace-time `aten::slice(cos_cached, dim=0, end=trace_seq_len)` is patched by `_patch_seq_len_in_ops` to `end=runtime_seq_len` at run-time. During decode (`runtime_seq_len=1`) the sliced cos/sin table only covers position 0, so `aten::index(sliced, position_ids=[N])` for `N >= 1` raises IndexError.

Fix: detect RoPE cache slices (input weight matches `cos_cached`/`sin_cached`/`cos_cache`/`sin_cache`) and substitute slice `end` with the full table size, mirroring the compiled-mode RoPE fix in `compiled_sequence._promote_seq_len_scalars_to_symbolic`.

Also added `_adapt_seq_dependent_weights` (mirror of `CompiledSequence.update_seq_dependent_constants`) for models that store RoPE under `constant_T_*` naming — currently a no-op for TinyLlama (no constant_T_*), reserved for future families.
