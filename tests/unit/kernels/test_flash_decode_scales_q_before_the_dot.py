"""flash_decode scales Q before the dot, and its output matches ATen.

Measured 2026-09-12: whisper reached
`aten._scaled_dot_product_efficient_attention::0` -- far past the addmm that
had blocked it -- and was refused with

    FlashAttention that applies an elementwise scale/bias to a tt.dot RESULT
    is not supported by the generic attention lowering — the fused op on the
    dot result is silently dropped / mis-applied. Refusing. Scale Q BEFORE the
    dot instead.

The refusal names its own remedy, and the main `flash_attention` kernel
already follows it, carrying a comment that cites this exact refusal.
`flash_decode` did not: it computed `s = tl.dot(q, kT)` and then
`s = s.to(tl.float32) * sm_scale`.

`(alpha * q) @ kT` and `alpha * (q @ kT)` are the same product and NOT the
same rounding, so the change is not free and is not asserted to be. It is
measured against ATen, which is an independent implementation with no reason
to agree with a wrong kernel.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_flash_decode_scales_q_before_the_dot.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import owned_cache_env       # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Metal attention path")


@pytest.fixture(autouse=True)
def _owns_its_cache(tmp_path, monkeypatch):
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)


def _kernel_source() -> str:
    import neurobrix.kernels.ops.flash_decode as fd
    return Path(fd.__file__).read_text()


def test_no_scale_is_applied_to_the_dot_result():
    """The structural half, by AST so a comment quoting the old form is safe.

    The pattern refused is a multiplication whose left operand derives from
    the `tl.dot` result. Asserting on the SOURCE is what makes this hold for
    the shapes no test exercises -- the decode path has many, and only one of
    them is measured numerically below.
    """
    tree = ast.parse(_kernel_source())
    dot_results = set()
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if "dot" in ast.unparse(node.value.func):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        dot_results.add(t.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = ast.unparse(node.left)
            if any(r in left for r in dot_results) and "scale" in ast.unparse(node.right):
                offenders.append((node.lineno, ast.unparse(node)[:70]))
    assert not offenders, (
        "these scale a dot RESULT:\n  "
        + "\n  ".join(f"line {n}: {s}" for n, s in offenders)
        + "\n\nScale Q before the dot: the same product, and the form the "
          "backend supports.")


def test_the_detector_sees_the_refused_form():
    """Both directions, or the assertion above is satisfied by a finder that
    finds nothing."""
    src = ("def k():\n"
           "    s = tl.dot(q, kt)\n"
           "    s = s.to(tl.float32) * sm_scale\n")
    tree = ast.parse(src)
    dot_results, offenders = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if "dot" in ast.unparse(node.value.func):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        dot_results.add(t.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if any(r in ast.unparse(node.left) for r in dot_results) and \
                    "scale" in ast.unparse(node.right):
                offenders.append(node.lineno)
    assert offenders == [3], f"the refused form must be seen, got {offenders}"


def test_decode_attention_matches_aten():
    """The numerical half, against an independent implementation.

    torch is used deliberately: R33 forbids it in the Triton branch and allows
    it in oracles and diagnostics, and a test that judges a kernel is an
    oracle.
    """
    torch = pytest.importorskip("torch")
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor

    B, H, T_k, D = 1, 4, 128, 64
    rng = np.random.default_rng(20260912)
    q = (rng.standard_normal((B, H, 1, D)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((B, H, T_k, D)) * 0.2).astype(np.float32)
    v = (rng.standard_normal((B, H, T_k, D)) * 0.2).astype(np.float32)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    want = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q).to(dev), torch.from_numpy(k).to(dev),
        torch.from_numpy(v).to(dev)).to("cpu").numpy().astype(np.float64)

    got = W.scaled_dot_product_attention_wrapper(
        NBXTensor.from_numpy(q), NBXTensor.from_numpy(k),
        NBXTensor.from_numpy(v))
    got = np.asarray(got.to_cpu().numpy(), dtype=np.float64)

    scale = float(np.abs(want).max()) or 1.0
    dev_max = float(np.abs(got - want).max() / scale)
    assert dev_max <= 1e-3, (
        f"decode attention differs from ATen by {dev_max:.3e}. Scaling Q "
        f"before the dot is the same product and not the same rounding; this "
        f"is what says the difference stayed inside the noise.")
