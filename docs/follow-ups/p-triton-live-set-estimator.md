# P-TRITON-LIVE-SET-ESTIMATOR — Prism estimator per-engine fidelity + margin consolidation

**Status:** open — named continuation of P-TRITON-LIVE-SET (DETTE D5.1).
The CAPACITY core of D5.1 landed (F1 data-driven drain policy, F2 no-cache
decode retention, F7 attention-bias-cache eviction). The estimator-fidelity
and margin-consolidation findings are deferred here **with reason**, not
dropped.

**Surfaced:** engine audit #2
(`docs/audits/engine_audit_2_memory_correctness_portability_2026_07_05.md`,
section 1, findings F3/F4/F5/F6/F8).

## Why deferred (the plan is SHARED across engines)

Prism solves **once** per run — `solver.solve_smart(container, hw_profile,
input_config)` at `cli/commands/run.py:283`, `serving/engine.py:108`,
`cli/commands/upscale.py:170` — with **no execution-mode argument**. The
resulting `ExecutionPlan` places BOTH the compiled and the triton engine
(CLAUDE.md: "the same Prism allocation plan"). Therefore any change to the
estimator's `peak_bytes` is applied to the compiled placement too, and can
flip a validated plan (e.g. `single_gpu+tiling → weight_sharding`) across
the whole model zoo. The audit's own F3 note records exactly this failure
class (the CFG batch-binding incident that doubled the Sana 4Kpx VAE
estimate and flipped its plan). A capacity chantier whose gate is "capacity
must only improve or hold" must not ship a plan-flipping change without a
full-zoo plan-diff gate on BOTH engines — which is broader than the
watermark-measurement gate that governs the landed core.

Additionally, **the landed F1 pressure gate directly guards the OOM that
estimator fidelity would prevent**. The estimator under-predicts the triton
peak by up to the deferred-queue cliff (2 GB) — but only when the queue is
full, and F1's pressure branch drains the queue EARLY whenever a driver-free
probe shows the device within `pressure_reserve` (6 GiB) of capacity. So in
exactly the regime where the estimator's under-prediction would cause an OOM
(the device near capacity), the triton engine reclaims the queue before free
memory reaches the danger zone. When there is headroom the 2 GB excess is
harmless (there is room for it), so the estimator's under-prediction is
inconsequential there. The practical urgency of F3/F4 is thus low. F5/F6
must NOT be consolidated in isolation: the audit shows F5's over-reserve and
F3/F4's under-reserve "coincidentally partially cancel", so removing the
over-reserve without first fixing the under-reserve would REMOVE the
compensation and risk OOM.

## Scope (findings, with file:line from the 2026-07-05 tree)

- **F3 (HIGH)** — `core/prism/profiler.py:531-582` simulates free-at-last-use
  (the COMPILED engine's semantics) and `estimate_peak_memory(mode=...)`
  already carries a `mode` param that `PrismSolver._compute_memory`
  (`solver.py:1474/1514/1536`) never passes. A per-ENGINE retention model
  would add the triton deferred-queue budget (== `wrappers.
  deferred_drain_budget_bytes()`, so the two stay self-consistent) to the
  triton peak ONLY. **Precondition:** thread the execution mode from the CLI
  through `solve_smart → solve → _compute_memory → estimate_peak_memory`
  WITHOUT breaking the shared-plan contract — either (a) compute both
  engines' peaks and place the max, or (b) make the runtime re-solve
  per-mode. Both are architectural decisions for the maintainer.

- **F4 (MED-HIGH)** — `core/prism/profiler.py:584-620`: op workspace
  (`estimate_op_workspace_bytes` — cuDNN conv, SDPA lse) is computed only in
  the overflow-op flagging scan, never added to the running `current_bytes`/
  `peak_bytes`. The placement figure excludes memory live at the peak op.
  Same shared-plan plan-flip risk as F3 (it raises every component's peak on
  both engines).

- **F5 (MED)** — stacked hardcoded margins: `solver.py:357-360`
  (`safety_margin` 0.95, from `PRISM_DEFAULTS`), `1568/1821-1822`
  (`overhead_factor`/`overhead_pct` 0.05), `1832` (`_OOM_RESERVE_MB = 3072`),
  `1754` (attention `safety = 1.2`). Consolidate into profile-derived
  reserves (R7/R24) — but only jointly with F3/F4 (see cancellation note).

- **F6 (MED)** — `solver.py:1551/1554` `activation_bytes = weight_bytes*0.5`
  fallback guess when a component has no graph / the profiler raised.

- **F8 (MED-LOW)** — `_base`-chain views can pin whole deferred parent
  blocks past the drain (`triton/sequence.py` liveness). Audit item: verify
  whether any live model retains a parent block via a `_base` view beyond
  its kill point; the PixArt inter-run arena bug
  (`docs/follow-ups/pixart_triton_arena_inter_run_bug.md`) is an adjacent
  `_base`-pinning symptom.

## Gate required before landing any of the above

Full-zoo Prism plan-diff on BOTH engines (compiled + triton): the placement
strategy and per-component device assignment must be unchanged for every
closed model, OR every change must be an intended capacity improvement with
a recorded verdict. This is strictly broader than the watermark-measurement
gate used for the landed F1/F2/F7 core.
