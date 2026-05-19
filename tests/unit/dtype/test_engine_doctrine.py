"""Ch7 P-DTYPE-ENGINE-DOCTRINE-RECONCILE — doctrine pins.

Pins the live DtypeEngine doctrine that Ch7 canonized (Resolution R1,
see docs/audits/dtype_engine_doctrine_audit.md). The pins are
intentionally MEMBERSHIP-only — they don't enumerate every op in
every set; they assert the load-bearing entries are present so any
future silent removal trips the test. Behavioral pins assert the
two distinguishing patterns the audit identified:

  (i) the `_to_copy` fp32→fp16 narrowing clamp clips a >65504 source
      to ±65504 (finite max-representable) instead of letting it
      saturate to ±Inf — the documented physics for that site.
 (ii) the env-gated NBX_DTYPE_CLAMP_DIAG fires exactly once per call
      site when an overflow is observed, and stays silent otherwise.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_engine_doctrine.py
"""
from __future__ import annotations

from typing import Iterable

import torch

from neurobrix.core.dtype import engine as engine_module
from neurobrix.core.dtype.engine import (
    AMP_FP32_OPS,
    AMP_FP16_OPS,
    AMP_PROMOTE_OPS,
    DtypeEngine,
    _FP16_NEED_FP32,
)


# ─────────────────────────────────────────────────────────────────────
# Doctrine membership pins (live engine — audit Section 3)
# ─────────────────────────────────────────────────────────────────────

# Load-bearing FP32-precision ops. Removing any of these silently from
# AMP_FP32_OPS would let fp16 numerical instability surface on the
# affected ops — exactly the regression class lesson 004 documented.
_FP32_SENTINELS: Iterable[str] = (
    "pow", "rsqrt",                                      # RMSNorm / RoPE
    "layer_norm", "native_layer_norm",
    "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "cudnn_batch_norm", "instance_norm",
    "softmax", "_softmax", "log_softmax",                # attention scores
    "sum", "prod", "cumsum", "cumprod",                  # accumulation
    "polar", "view_as_complex",                          # complex32 absent on CUDA → fp32
    "upsample_nearest2d", "upsample_bilinear2d",         # diffusion VAE
)

# Load-bearing FP16-class compute ops. These must stay in AMP_FP16_OPS
# so the engine knows to wrap them (either pure input-cast on bf16
# hardware, or fp32 upcast on V100 fp16 for the _FP16_NEED_FP32 subset).
_FP16_SENTINELS: Iterable[str] = (
    "addmm", "mm", "bmm", "baddbmm", "matmul", "linear", "einsum",
    "conv1d", "conv2d", "conv3d", "convolution",
    "div",                                               # epsilon underflow on V100
)

# The V100-specific fp32 upcast subset. Audit Section 3: mm/bmm/addmm
# need fp32 because inner-dim accumulation exceeds fp16 max; div needs
# fp32 because epsilon 1e-15 rounds to 0 in fp16 (min ~6e-8).
# Pinned EXACTLY — adding to this set is a doctrine change that
# warrants its own chantier.
_EXPECTED_FP16_NEED_FP32 = frozenset({"mm", "bmm", "div", "addmm"})

_PROMOTE_SENTINELS: Iterable[str] = (
    "addcdiv", "addcmul", "atan2", "index_put", "scatter_add",
)


def test_amp_fp32_ops_contains_load_bearing_entries():
    missing = [op for op in _FP32_SENTINELS if op not in AMP_FP32_OPS]
    assert not missing, (
        f"AMP_FP32_OPS lost load-bearing entries: {missing}. "
        "Audit doc Section 3 / lesson 004 — removing FP32-precision "
        "protection on these ops re-exposes fp16 numerical "
        "instability (NaN, marble effect)."
    )


def test_amp_fp16_ops_contains_load_bearing_entries():
    missing = [op for op in _FP16_SENTINELS if op not in AMP_FP16_OPS]
    assert not missing, (
        f"AMP_FP16_OPS lost load-bearing entries: {missing}."
    )


def test_fp16_need_fp32_subset_is_exact():
    """The V100-specific upcast subset is doctrine-frozen. Audit Section 3."""
    assert _FP16_NEED_FP32 == _EXPECTED_FP16_NEED_FP32, (
        f"_FP16_NEED_FP32 doctrine drift: got {set(_FP16_NEED_FP32)}, "
        f"expected {set(_EXPECTED_FP16_NEED_FP32)}. Any change here is "
        "a doctrine change — requires its own chantier."
    )


def test_amp_promote_ops_contains_load_bearing_entries():
    missing = [op for op in _PROMOTE_SENTINELS if op not in AMP_PROMOTE_OPS]
    assert not missing, f"AMP_PROMOTE_OPS lost entries: {missing}"


def test_fp32_and_fp16_sets_are_disjoint():
    """A given op cannot be both FP32 and FP16 class — that would be
    a wrapper-routing ambiguity bug (the compile_op cascade would
    take the first match and the runtime behaviour would depend on
    declaration order in the source)."""
    overlap = AMP_FP32_OPS & AMP_FP16_OPS
    assert not overlap, f"FP32/FP16 set overlap: {overlap}"


# ─────────────────────────────────────────────────────────────────────
# `_to_copy` clamp behavioral pin (audit Section 4)
# ─────────────────────────────────────────────────────────────────────

def test_to_copy_clamps_fp32_overflow_to_fp16_range_not_inf():
    """A fp32 value > fp16 max (±65504) narrowed via aten::_to_copy
    must produce a FINITE fp16 ±65504, not ±Inf. This is the audit
    Section 4 distinction from the lesson-004-prohibited matmul-output
    clamp: the value MUST become fp16 here, fp16 cannot represent
    anything beyond ±65504, so the choice is finite-65504 vs
    Inf-then-NaN-cascade."""
    eng = DtypeEngine(compute_dtype=torch.float16, graph_dtype=torch.float16)
    fn = eng.compile_op("aten::_to_copy", None,
                        {"output_dtypes": ["float16"]})
    src = torch.tensor([1.0e6, -1.0e6, 100.0, -100.0, 0.0],
                       dtype=torch.float32)
    out = fn(src)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), (
        f"_to_copy fp32→fp16 narrowing produced non-finite values: {out}. "
        "Audit Section 4: the clamp at the _to_copy site exists "
        "precisely to prevent the Inf-then-NaN cascade lesson 004 "
        "described — under a different physics from the matmul-output "
        "clamp lesson 004 prohibited."
    )
    # Overflow positions clamped to fp16 max; finite positions
    # preserved bit-exactly.
    assert out[0].item() == 65504.0
    assert out[1].item() == -65504.0
    assert out[2].item() == 100.0
    assert out[3].item() == -100.0
    assert out[4].item() == 0.0


def test_to_copy_passthrough_branch_clamps_via_kwargs_dtype():
    """The passthrough branch of _make_to_copy (no output_dtypes in
    graph attrs — runtime kwargs supply dtype) must also clamp. Same
    invariant, different code path: engine.py:443-447."""
    eng = DtypeEngine(compute_dtype=torch.float16, graph_dtype=torch.float16)
    fn = eng.compile_op("aten::_to_copy", None, {"output_dtypes": []})
    src = torch.tensor([2.0e6, -2.0e6], dtype=torch.float32)
    out = fn(src, dtype=torch.float16)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert out[0].item() == 65504.0
    assert out[1].item() == -65504.0


def test_to_copy_no_clamp_when_target_not_fp16():
    """Non-fp16 narrowings must not clamp. bf16 has fp32 exponent
    range — clamping there would corrupt values bf16 can legitimately
    represent above 65504 (up to ~3.4e38). Audit Section 3 hardware
    split."""
    eng = DtypeEngine(compute_dtype=torch.bfloat16,
                      graph_dtype=torch.bfloat16)
    fn = eng.compile_op("aten::_to_copy", None,
                        {"output_dtypes": ["bfloat16"]})
    src = torch.tensor([1.0e6], dtype=torch.float32)
    out = fn(src)
    assert out.dtype == torch.bfloat16
    # bf16 represents 1e6 exactly (or close to it); MUST NOT have been
    # clamped to 65504.
    assert out.item() > 65504.0, (
        f"bf16 narrowing was clamped to fp16 range — bug: {out.item()}"
    )


# ─────────────────────────────────────────────────────────────────────
# Env-gated NBX_DTYPE_CLAMP_DIAG behavior (Ch7 Commit 2)
# ─────────────────────────────────────────────────────────────────────

def test_clamp_diag_default_off_silent(capsys):
    """Default-off: zero runtime cost / zero output even when the
    clamp actually fires."""
    # Module-level flag was set at import time from os.environ; force
    # the default-off path here regardless of how the test runner was
    # launched.
    engine_module._CLAMP_DIAG_ENABLED = False
    engine_module._CLAMP_DIAG_FIRED.clear()

    eng = DtypeEngine(compute_dtype=torch.float16, graph_dtype=torch.float16)
    fn = eng.compile_op("aten::_to_copy", None,
                        {"output_dtypes": ["float16"]})
    fn(torch.tensor([1.0e6], dtype=torch.float32))

    captured = capsys.readouterr()
    assert "NBX_DTYPE_CLAMP_DIAG" not in captured.out


def test_clamp_diag_enabled_fires_one_shot_per_site(capsys):
    engine_module._CLAMP_DIAG_ENABLED = True
    engine_module._CLAMP_DIAG_FIRED.clear()

    eng = DtypeEngine(compute_dtype=torch.float16, graph_dtype=torch.float16)
    fn = eng.compile_op("aten::_to_copy", None,
                        {"output_dtypes": ["float16"]})
    # First call: source overflows → should log once.
    fn(torch.tensor([1.0e6], dtype=torch.float32))
    # Second call: source overflows again → MUST stay silent
    # (one-shot per site).
    fn(torch.tensor([2.0e6], dtype=torch.float32))
    # Third call: source in range → no log either.
    fn(torch.tensor([100.0], dtype=torch.float32))

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines()
             if "NBX_DTYPE_CLAMP_DIAG" in l]
    assert len(lines) == 1, (
        f"Expected exactly one diag line (one-shot per site), got "
        f"{len(lines)}: {lines}"
    )
    assert "site=to_copy_target" in lines[0]
    assert "max_abs=1000000.00" in lines[0]
    assert "shape=(1,)" in lines[0]

    # Reset for any subsequent tests.
    engine_module._CLAMP_DIAG_ENABLED = False
    engine_module._CLAMP_DIAG_FIRED.clear()


def test_clamp_diag_enabled_silent_when_no_overflow(capsys):
    engine_module._CLAMP_DIAG_ENABLED = True
    engine_module._CLAMP_DIAG_FIRED.clear()

    eng = DtypeEngine(compute_dtype=torch.float16, graph_dtype=torch.float16)
    fn = eng.compile_op("aten::_to_copy", None,
                        {"output_dtypes": ["float16"]})
    fn(torch.tensor([100.0, -100.0, 0.0], dtype=torch.float32))

    captured = capsys.readouterr()
    assert "NBX_DTYPE_CLAMP_DIAG" not in captured.out

    engine_module._CLAMP_DIAG_ENABLED = False
    engine_module._CLAMP_DIAG_FIRED.clear()
