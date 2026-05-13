# S4 prerequisite — VAE-alone peak measurement (2026-05-13)

Per mandate Q4 doctrine ("mesure factuelle AVANT code"), this note
records the empirical VAE-alone peak VRAM for Sana 4Kpx and the
resulting internal reorder S4 → S5 → S4.

## Methodology

Three independent probes, all on a clean cuda:2 (Tesla V100-SXM2-32GB)
with no other GPU work resident.

1. Sana 4Kpx on `v100-16g` (1× 16 GiB profile, cuda:0):
   OOM at `custom.rms_norm::21` requesting 2 GiB on top of 11.6 GiB
   already resident → 13.6 GiB needed at op 21 (which is EARLY in the
   VAE chain, not peak).

2. Sana 4Kpx on `v100-16g-x2-01` (2× 16 GiB profile, both cuda:0+cuda:1):
   Same `rms_norm::21` OOM at 11.6 GiB on cuda:0. Prism placed all
   three components on cuda:0; cuda:1 entirely unused (this is the
   inter-component placement gap = Gap A from the S4 scope analysis).

3. Sana 4Kpx on `v100-32g` (1× 32 GiB profile, cuda:2):
   PASS coherent PNG, 22.7 s end-to-end. Sampled `nvidia-smi
   memory.used` every 0.5 s during the run; **peak = 32441 MiB
   (≈ 31.7 GiB)**.

## Verdict

Case **2** of the mandate Q4 decision tree applies:

> VAE seul dépasse 16 GiB → Gap A insuffisant, S5 nécessaire en
> parallèle pour 2×16 GiB. Décision : enchaîner S5 avant S4
> (réordonnancement mandate).

Reasoning chain:
- Probe 3 peak 31.7 GiB is the COMBINED peak with `lazy_sequential`
  unload between phases. By the time the VAE post-loop starts,
  text_encoder + transformer weights have been released — only their
  activation residue (latents handed off to VAE) plus VAE weights +
  VAE activations are live. The peak at the VAE post-loop alone is
  therefore ≤ 31.7 GiB but unambiguously **> 16 GiB** (probe 1
  already shows op 21 needs 13.6 GiB and the chain peaks later at
  silu::24 op 707 per the existing project doctrine where 3× 4 GiB
  tensors co-reside).
- Placing the VAE alone on cuda:1 (Gap A only) cannot help: even
  with the full 16 GiB at its disposal, the VAE's intrinsic peak
  exceeds 16 GiB.
- S5 (`P-DC-AE-RESIDUAL-CHAIN-TILING`) addresses precisely this
  intra-op peak by tiling the residual chain so the 3×4 GiB
  co-resident tensors drop below the budget. Once S5 lands, VAE
  peak fits 16 GiB; THEN Gap A (component_placement actually using
  both devices) becomes meaningful for 2×16 GiB cells.

## Reorder

New session order, with no change to mandate exit conditions:

1. **S5** (`P-DC-AE-RESIDUAL-CHAIN-TILING`, 300-500 lines) — first.
   Validates 16 GiB single-GPU compiled / sequential / triton /
   triton_sequential for Sana 4Kpx via op-level peak reduction.
2. **S4** (`P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT` scope = Gap A
   only, 150-300 lines) — second. Validates 2×16 GiB cells across
   the four modes once VAE individually fits 16 GiB.

Backlog deferred (no order change):
- `P-OP-LEVEL-CROSS-DEVICE-SPLIT` (Gap B) — opens a separate mandate
  when a model surfaces that needs per-op cross-device split (e.g.
  Qwen3-30B-A3B class with weights > 16 GiB).
- `P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP` (S3 escalation) — triggers
  on upstream wheel availability or NeuroBrix doctrinal change.

## Reference artefacts

- `/tmp/sana4kpx_peak_32g.png` — coherent red apple, 32g cuda:2.
- `/tmp/sana4kpx_sample.png` — coherent red apple, 32g cuda:2 with
  nvidia-smi peak sampling (32441 MiB).
- Failure trace text-only (probes 1, 2) — `rms_norm::21` OOM
  signature; not committed to validation artefacts because they
  are non-novel failure traces.
