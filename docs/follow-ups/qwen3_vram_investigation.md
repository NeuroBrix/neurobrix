# Qwen3-30B VRAM Investigation — follow-up to Item 3

## Motivation

After Item 3 (in-kernel fp16→fp32 weight promotion, Path A'), Qwen3-30B
`v100-16g --triton` still shows ~10.8 GB peak VRAM. The original
Item 3 brief targeted ≤ 6.0 GB based on an "8.3 GB baseline" that
does not reproduce at HEAD. Item 3 removed the per-call weight widen
as a contributor (measured -64 MB on Qwen3); the remaining ~10.7 GB
is dominated by other sources. This document scopes the investigation
that must precede any fix session aimed at driving the Qwen3 peak
below the current level.

Entry points for the investigation:

- `tests/scratch/matmul_item3/REPORT.md` — Phase 3 measurements that
  established the current baseline and attributed the residual VRAM
  qualitatively.
- `CHANGELOG.md` "Deferred" section — the official scope statement.

## Known contributors (from Phase 3 analysis, Item 3 session)

Per `tests/scratch/matmul_item3/REPORT.md` Phase 3 Scenario 2, the
~10.8 GB peak on Qwen3-30B `v100-16g --triton --prompt "2+2=" --max-tokens 4`
comes from (rough hierarchy, exact magnitudes TBD by instrumentation
in this chantier):

1. **Two block-windows of expert weights resident under zero3 ratchet
   pipelining.** `Zero3Strategy` (see `CHANGELOG.md` Added-section
   entry in this release) maintains a sliding 2-block window across
   the compiled op sequence: block N's weights are bound to the arena
   while block N+1's are prefetched on a dedicated transfer stream.
   For Qwen3-30B-A3B with MoE (128 experts, top_k = 8), per-block
   weight mass varies by op — one block of the transformer holds the
   attention projections + MoE router + per-expert gate/up/down
   weights for all 128 experts. Rough order-of-magnitude: several
   hundred MB per block × 2 resident windows.

2. **Non-block weights on `cuda:0`**: `lm_head`, token embeddings,
   final norm — loaded through the regular GPU path (not CPU-offloaded
   under zero3) because flow handlers access them directly (see
   `triton/weight_loader.py::_load_to_pinned_cpu` vs non-block path).

3. **MoE expert promotion buffers**: `triton/moe.py::execute_moe_fused`
   detects CPU-backed expert weights under zero3 and promotes them to
   the activation device via `NBXTensor.to_cuda(act_dev)` before
   `_build_ptr_tables` bakes raw pointers into the GPU int64 table.
   Explicit `del` + `gc.collect` at function exit releases the
   promoted lists, but the transient peak during the MoE op is still
   substantial (~768 MB of expert weights per release, per the zero3
   ratchet release notes).

4. **Prefill activation working-set** for `"2+2="` prompt — typically
   a couple hundred MB for a small prompt but scales with sequence
   length.

5. **KV cache**, growing with decode step count. At 4 tokens the KV
   contribution is relatively small; at larger token counts this
   would dominate.

All five are independent of the per-call fp32 weight widen that
Item 3 eliminated.

## Investigation tasks (DO NOT PERFORM IN THIS DOC; dedicated session)

1. **VRAM attribution histogram.** Run Qwen3-30B `v100-16g --triton`
   with `NBX_UNLOAD_DIAG=1` and instrumented `DeviceAllocator` hooks
   (allocation site, size, live interval). Produce a per-moment peak
   attribution showing, at the ~10.8 GB peak, which allocations are
   alive and how much each contributes. Expected output: a table
   keyed by `(op_idx, allocation_site)` with live bytes at peak.

2. **Reproduce or reject the 8.3 GB baseline.** Search `git log` and
   any prior session reports (`tests/scratch/**/REPORT*.md`) for the
   configuration that produced 8.3 GB. Candidate hypotheses: shorter
   prompt, different strategy, older zero3 implementation, no MoE
   fusion. If reproducible — document the config. If not — confirm
   the target was miscalibrated and update future targets
   accordingly.

3. **Identify the top-2 contributors** from task 1's histogram.
   Evaluate the following levers quantitatively — each a separate
   mini-chantier with before/after measurements:

   - **Zero3 sliding window size**: can it be reduced from 2 blocks
     to 1? Cost: per-block H2D transfer latency cannot be hidden by
     pipelining anymore; throughput likely regresses. Measure.
   - **MoE expert promotion buffer reuse**: instead of allocating a
     fresh GPU buffer per MoE op call, can `execute_moe_fused`
     recycle a persistent buffer? Requires the buffer to be sized
     for the largest expert, invalidates `_ptr_cache` on reuse.
   - **KV cache compression**: fp8 quantization, attention sinks,
     sliding window. Significant research-engineering scope — not a
     line-of-code fix.
   - **Non-block weight eviction** between forward passes: `lm_head`
     is accessed once per decode step; can it be streamed in the
     same way as block weights? Breaks the current contract that
     flow handlers access non-block weights directly — requires flow
     handler redesign.

4. **Each lever → scope into its own mini-chantier** with a decision
   tree (fix if measured gain > X GB and no correctness regression
   and no throughput regression > Y %).

## Targets (to calibrate AFTER investigation)

Once the top-contributors are quantified, realistic per-lever targets
can be set. The "≤ 6.0 GB" headline number from the Item 3 brief is
a guess that predates this investigation — it may or may not be
achievable on this hardware with any of the levers above. Do not
anchor future work to it until task 2 (reproduce or reject) closes.

## Session rules (apply here and in any follow-up)

- No fix before investigation — tasks 1–3 must complete before any
  code edit.
- No commit before measurement — every lever change lands with a
  before/after VRAM + throughput table.
- Audit R1 on any touched file (no orphan code, no duplicated
  helpers).
- Native + triton parity for any zero3 / MoE change — the two code
  paths must match semantically.
- Zero torch in the triton path (no `torch.Tensor`, no `torch.dtype`,
  no `torch.cuda` in `src/neurobrix/triton/`).

## Not in scope for this follow-up

- Re-opening Item 3 — the in-kernel tile promotion is correct and
  shipped. Do not revert or extend it here.
- Quantization of weights (AWQ / GPTQ / int4) — separate chantier,
  not a zero3 / MoE residency concern.
- Anything that touches hardware other than V100 16 GB — this
  investigation is bounded to that configuration.
