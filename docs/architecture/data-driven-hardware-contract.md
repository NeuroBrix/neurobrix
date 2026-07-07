# NeuroBrix Hardware-Universal Data-Driven Contract

## TL;DR

Every parameter that depends on hardware (SMEM budget, block sizes, dtype
support, tile dimensions, register limits, warp/wavefront size, etc.) has
its source of truth in
`src/neurobrix/config/vendors/<vendor>/<architecture>.yml`. New hardware
support = add a YAML; no Python branches per arch.

The wrapper code consuming these parameters has TWO acceptable
materialization modes:

1. **Runtime YAML lookup** via `get_vendor_config()` (cached @lru_cache) —
   for sites that are NOT in a kernel hot path (e.g. setup-time decisions,
   one-shot configuration, planner code).
2. **Manual alignment with the YAML** — for sites IN a Triton kernel hot
   path, until the empirical Python-frame drift documented below is
   resolved.

## The Python-frame drift on Volta + Triton SDPA hot path

### Empirical observation

Adding ANY function call (helper, cache lookup, list literal allocation,
helper closure) to `scaled_dot_product_attention_wrapper` between argument
parsing and kernel launch causes run-to-run output drift on Volta + Triton,
even when:

- The added code returns identical values across runs (purely deterministic
  from inputs).
- The Triton kernel cache is stable across runs (verified empirically: 0
  new compiled kernels between warm runs).
- The pre-Layer-6 baseline produces 4/4 identical output ("Certainly! Here")
  across cold + 3 warm runs at commit 06d26c2.

Example Layer 6 inputs that drifted vs the bit-perfect pre-Layer-6
baseline, all ruled in via systematic bisect:

- Module-level `_FA_SMEM_CACHE` global dict + `_fa_max_smem` driver query.
- Compiled cascade-if picker function called per SDPA call.
- `_resolve_blocks_and_dtype` cached lookup (even with 100% cache hit rate
  after the first call).
- List literal `candidates = [(16, 64), ...]` allocated per call (Layer 6
  original DRIFT cause, isolated by Step 3e bisect).

The empirical signature is the same in every case: 3 successive runs of
the same `python -m neurobrix run --model TinyLlama --temperature 0
--prompt Hello --max-tokens 5 --triton` produce 3 different 5-token
outputs, while the equivalent test on the pre-Layer-6 baseline (or on a
strictly-minimal inline cascade-if) produces 3 identical "Certainly! Here".

### What we ruled out (cache investigation, T_warmcache test)

- **Triton kernel cache invalidation**: tested via `rm -rf ~/.triton/cache`
  + 1 cold + 3 warm runs. Cache stable across warm runs (0 new compiled
  files). Pre-Layer-6 stable across all 4. Layer 6 drifts across all 4
  (cold output ≠ each warm output). Conclusion: drift is NOT in the
  compilation path.

### Hypotheses not yet eliminated (open follow-up)

A. **CUDA stream / atomics non-determinism** — flash attention's softmax
   accumulation could rely on an order that the CUDA runtime serializes
   slightly differently each launch. Diagnostic: dump the kernel binary
   loaded at each call via `triton.compiler` introspection, compare across
   runs. If identical → runtime non-determinism in the kernel itself.

B. **Driver state perturbation by Python frame allocations** — the
   sub-microsecond Python work between arg parsing and kernel launch may
   alter the CUDA driver's internal state (stream creation, memory pool
   selection) enough to perturb kernel output. Hard to disprove without a
   custom CUPTI trace.

C. **`@triton.heuristics` lambdas re-evaluated per call** — even with
   stable cache, the heuristic lambdas attached to the kernel decorator
   re-execute on every launch. They may resolve to slightly different
   constexpr values depending on the timing of when args are inspected.

D. **Thread-local state in Triton or CUDA** — variable initialization
   ordering could matter.

The mechanism remains unexplained after systematic bisect across 6
investigation rounds. A future Layer X will revisit with the diagnostic
listed in (A) — kernel binary diff between runs.

## Current pattern (Layer 6.bis and beyond, until Layer X)

For SDPA block selection — the only hot path empirically affected — the
wrapper contains a strictly minimal inline cascade-if:

```python
BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
if seqlen_q <= 16:
    BLOCK_M = 16
    BLOCK_N = 64 if BLOCK_HEADDIM < 128 else (64 if BLOCK_HEADDIM < 256 else 32)
elif BLOCK_HEADDIM >= 512:        # Layer 6.bis: PixArt VAE
    BLOCK_M = 16
    BLOCK_N = 16
elif BLOCK_HEADDIM >= 256:
    BLOCK_M = 32
    BLOCK_N = 32
elif BLOCK_HEADDIM >= 128:
    BLOCK_M = 64
    BLOCK_N = 64
else:
    BLOCK_M = 128
    BLOCK_N = 64
```

The vendor YAMLs (`src/neurobrix/config/vendors/nvidia/{volta,ampere,hopper}.yml`,
`src/neurobrix/config/vendors/amd/cdna.yml`) declare the same values and
serve as the documentary source of truth. When adding a new architecture:

1. Create the vendor YAML with the architecture's optimal block sizes,
   SMEM budget, etc.
2. Manually align the wrapper's inline cascade-if with the YAML values.
3. Run V_drift (3 successive `--triton` calls produce identical output)
   and V_regression (LLM harness 14/14 + Sana 1024 zero regression) on the
   new hardware before merging.

This manual-alignment dance is acceptable transitional pattern — each
architecture is added rarely, and the YAML keeps the source-of-truth
property that future-proofs against the wrapper drifting from intended
behavior. Layer X will retire this pattern once the Python-frame drift
root cause is isolated.

## Note on `vendors/` gitignore

At time of writing (Layer 6.bis), `src/neurobrix/config/vendors/` is in
`.gitignore` — the YAMLs are local-only. This is a separate architectural
issue: documentary source-of-truth files should be tracked. Removing the
gitignore is a follow-up and out of scope for the SDPA fix. The values
documented here in the cascade-if were captured from the live local YAMLs
during the Layer 6.bis investigation.

## Anti-patterns (always reject)

- Hardcode tile sizes / SMEM budgets per-arch in Python `if vendor == ...`
  branches.
- `triton.runtime.driver.active.utils.get_device_properties()` or any
  Triton/CUDA driver query in a kernel hot path. Use the YAML.
- List literals (`candidates = [(16, 64), ...]`) allocated on every call
  inside a hot path — this was the Layer 6 DRIFT cause.

## Cross-references

- [Symbolic Shapes Contract](symbolic-shapes-contract.md) — the parallel
  master contract for spatial-adaptive graphs.
