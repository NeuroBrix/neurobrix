# Dtype-mirror walk — P-SANA-4KPX-RUNTIME 2026-05-09

R29 inspectable artefact for Hocine's "NBX must mirror PyTorch op-by-op" directive.

## Methodology

`/tmp/dtype_mirror_walk.py` — monkey-patches `NativeATenDispatcher.dispatch`
and `TritonSequentialDispatcher.dispatch` to log per-op dtypes via
frame inspection (op_uid lives in the calling executor's frame). Runs
both modes end-to-end on `vae_isolation_input.pt`, collects logs,
cross-references into TSV.

NO interceptor REPLACEMENT — observation only, runtime path preserved.
NO force-fp32. NO override. We capture what the runtime actually does
under each engine's natural dtype rules.

## Results

| metric | value |
|---|---|
| total VAE ops | 737 |
| sequential captured | 686 (run completed) |
| triton_sequential captured | 670 (OOM at deep upsample chain — late ops) |
| ops with full match (in+out dtypes equal) | 562 |
| ops with divergence (any) | **175** |

## Files

- `dtype_walk.tsv` — full per-op TSV: op_idx, op_uid, op_type, seq_in_dts, seq_out_dt, tri_in_dts, tri_out_dt, match
- `first_divergences.json` — JSON of all 175 divergences sorted by op_idx
- `first5_categorization.md` — categorization (a/b/c) of the first 5 divergences + the cleanest kernel-level divergence (`custom.rms_norm::0`)

## Top-line findings

The 175 divergences cluster around **two structural patterns**:

1. **Pattern (c) inheritance from input::z dtype**: input::z (the saved
   latent) enters the SEQ runtime store as **fp16** but the TRI runtime
   store as **fp32**. This propagates through the metadata chain
   (unsqueeze→expand→clone→view) which is dtype-passthrough in BOTH
   engines. Once a binary op (add, mm, ...) downstream sees the fp32
   from view::0, the difference cascades. ~100 divergences are
   downstream propagations of this single root mismatch.

2. **Pattern (a) NBX defensive fp32-keep at custom.rms_norm**: same
   inputs (fp32, fp16) in both modes, but **seq casts output back to
   fp16, NBX keeps fp32**. NBX `rms_norm` wrapper has an opt-in
   cast-back gated by `activations_fp16_safe` flag (default False);
   seq's `DtypeEngine` casts back unconditionally. Direct kernel-level
   dtype divergence at the wrapper.

## Awaiting Hocine direction

Per directive: "Pas de fix de code dans ce commit, juste le diagnostic."
The two patterns above suggest different fix surfaces:

- For pattern (c): trace `input::z` runtime entry; identify why seq
  casts to fp16 and tri doesn't. Likely a missing cast in
  `RuntimeExecutor` input loading for triton mode.
- For pattern (a): align `rms_norm` cast-back default — either flip
  Sana to `activations_fp16_safe=True` in model_registry, or remove
  the opt-in gate and always cast back to compute_dtype.

Hocine arbitrage requested before any fix.
