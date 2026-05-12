# triton-cpu — known upstream coverage gaps (2026-05-12)

This document records the open upstream issues in `triton-lang/triton-cpu`
that block specific NeuroBrix code paths on the CPU triton backend, the
NeuroBrix-side markers gating them (so we don't paper over them with
silent fallbacks), and the follow-up chantier
`P-TRITON-CPU-FP16-UPSTREAM-FOLLOWUP` whose sole job is to monitor and
re-test when these close.

Per project doctrine (R25 — no internal fork of upstream when blocked,
P-PRISM-NEVER-REFUSE v2 / S3 escalation clause), NeuroBrix does NOT fix
these gaps by maintaining a private triton-cpu fork; we document, gate,
and escalate.

## Active blockers

| Upstream issue | Subject | Affected NeuroBrix path | NeuroBrix marker |
|---|---|---|---|
| [triton-cpu #147](https://github.com/triton-lang/triton-cpu/issues/147) | fp16 `tl.dot` Dot3D accuracy gap (open since 2024-09-18) | mm / bmm / addmm in fp16 paths; any model that uses fp16 attention or fp16 GEMM on CPU | `TRITON_CPU_FP16_UPSTREAM_BLOCKED` in `cpu_backend.py` |
| [triton-cpu #222](https://github.com/triton-lang/triton-cpu/issues/222) | `make_block_ptr` fast-path GEMM does not support masks | conv2d implicit-gemm halo/tail masking; fast-path matmul-with-mask patterns | `TRITON_CPU_MASKED_BLOCKPTR_GEMM_BLOCKED` in `cpu_backend.py` |
| [triton-cpu #229](https://github.com/triton-lang/triton-cpu/issues/229) | AVX512-BF16 matmul performance / tuning | bf16 matmul perf (correctness OK, slow) | accepted — perf-only, no gate |
| [triton-cpu #58](https://github.com/triton-lang/triton-cpu/issues/58) | Round-to-zero fp32 conversion not supported | very narrow surface, no NeuroBrix path hits it today | none |
| [triton-cpu #144](https://github.com/triton-lang/triton-cpu/issues/144) | `LLVM Translation failed for builtin.unrealized_conversion_cast` on mixed precision | regressions on mixed-precision ops adjacent to #147 | covered by `TRITON_CPU_FP16_UPSTREAM_BLOCKED` |
| [triton-cpu #233](https://github.com/triton-lang/triton-cpu/issues/233) | torch 2.6+ build incompatibility on the build path | install gate only (manifest in `docs/triton_cpu_install.md`) | none — install-time |

## Supported NeuroBrix paths on triton-cpu today

**Validated (or expected to validate)**:

- TinyLlama (fp32 registry default) under `--triton` and
  `--triton-sequential` on a CPU-only profile.
- Sana 1024 BF16 (`Sana_1600M_1024px_MultiLing`) under `--triton` and
  `--triton-sequential` on a CPU-only profile. Expect slow matmul
  (issue #229).

**Blocked upstream — escalated per P-PRISM-NEVER-REFUSE v2 mandate
"épuisement technique" clause**:

- Sana 1600M 4Kpx fp16 (`Sana_1600M_4Kpx_BF16` configured at fp16) under
  `--triton` / `--triton-sequential` on a CPU-only profile. Requires
  fp16 Dot3D parity (#147) AND masked `make_block_ptr` GEMM (#222) for
  the implicit-gemm conv2d halo path. NeuroBrix cannot close these
  upstream gaps without a private fork (R25 forbids).

**Alternative for fp16 CPU-pure inference today**: `--compiled` mode,
which uses the mature PyTorch CPU backend and has no equivalent fp16
gaps. The triton-cpu fp16 path will reopen automatically when
`P-TRITON-CPU-FP16-UPSTREAM-FOLLOWUP` flips the marker constants in
`cpu_backend.py`.

## Maintenance procedure

When a new triton-cpu release lands:

1. Run `pip install -U triton-cpu` in a sandbox venv.
2. Re-run the NeuroBrix S3 validation matrix:
   - TinyLlama fp32 `--triton` CPU pure
   - Sana 1024 bf16 `--triton` CPU pure
3. If passing, attempt fp16 Sana 4Kpx CPU pure (the upstream-blocked
   cell). On numerical PASS:
   - Edit `cpu_backend.py` and flip
     `TRITON_CPU_FP16_UPSTREAM_BLOCKED = False` and (if #222 also
     closed) `TRITON_CPU_MASKED_BLOCKPTR_GEMM_BLOCKED = False`.
   - Re-pin the minimum triton-cpu version in `docs/triton_cpu_install.md`.
   - Close `P-TRITON-CPU-FP16-UPSTREAM-FOLLOWUP`.
4. If failing, leave markers in place and update the "Active blockers"
   table above with the new upstream issue ID if the failure mode
   changed.
