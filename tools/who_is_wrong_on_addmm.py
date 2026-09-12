#!/usr/bin/env python3
"""The oracle, our kernel, and an independent third party on the same inputs.

The screen refuses every `addmm_kernel` candidate at
`M=8 N=768 K=128 … fp32,fp32,bf16,fp32` with "the fp64 oracle contradicts
EVERY candidate (10 of 10)". That verdict survives the removal of the raw
device read and survives a genuinely cold replay cache, so it is not an
artefact of either.

"Every candidate is wrong" is the verdict a broken reference gives as readily
as a correct one, and the register now requires the reference to be measured
before it condemns. Two numpy computations would be circular -- the oracle IS
`a @ b + bias` in float64. So the third party is ATen on MPS: a separate
implementation, on the same operands, which has no reason to agree with a
wrong kernel or a wrong oracle.

    agree oracle + ATen, differ from Triton  -> the kernel is wrong, refusal VALID
    agree Triton + ATen, differ from oracle  -> the ORACLE is wrong
    all three differ                         -> the operands are not what any
                                                of them thinks they are

torch is used here deliberately: R33 forbids it in the Triton branch and
allows it in oracles and diagnostics under `tools/`, which is exactly what
this is.

    tools/who_is_wrong_on_addmm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_measurement_environment import enforce_owned_cache   # noqa: E402

M, N, K = 8, 768, 128


def main() -> int:
    enforce_owned_cache("the addmm arbitration")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels.autotune_certify import (
        f32_to_bf16_bits, bf16_bits_to_f32)

    rng = np.random.default_rng(20260912)
    a32 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    b32 = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    # The bias is made exactly representable in bf16, so every party reads the
    # same numbers and a difference cannot be a rounding of the fixture.
    bias32 = bf16_bits_to_f32(f32_to_bf16_bits(
        (rng.standard_normal(N) * 0.1).astype(np.float32)))

    oracle = (bias32.astype(np.float64)[None, :]
              + a32.astype(np.float64) @ b32.astype(np.float64))

    a = NBXTensor.from_numpy(a32)
    b = NBXTensor.from_numpy(b32)
    bias = NBXTensor.from_numpy(f32_to_bf16_bits(bias32),
                                dtype=NBXDtype.bfloat16)
    ours = np.asarray(W.addmm(bias, a, b).to_cpu().numpy(), dtype=np.float64)

    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # torch.addmm refuses a bf16 bias against fp32 mats outright, which is a
    # fact worth recording: ATen has no PROMOTE_BIAS. The bias is therefore
    # handed to it already widened -- and the widening is EXACT, since the
    # fixture's values are representable in bf16 by construction, so this is
    # the same number our kernel widens on load and not a second rounding.
    t_bias = torch.from_numpy(bias32).to(dev)
    aten = torch.addmm(t_bias,
                       torch.from_numpy(a32).to(dev),
                       torch.from_numpy(b32).to(dev))
    aten = aten.to("cpu").to(torch.float64).numpy()

    def dev_of(x, y):
        scale = float(np.abs(y).max()) or 1.0
        return float(np.abs(x - y).max() / scale)

    print(f"shape {M}x{N}x{K}, bias bf16, a/b fp32, on {dev}")
    print(f"  ours   vs oracle : {dev_of(ours, oracle):.3e}")
    print(f"  ATen   vs oracle : {dev_of(aten, oracle):.3e}")
    print(f"  ours   vs ATen   : {dev_of(ours, aten):.3e}")
    print()
    tol = 0.04
    o_ok, a_ok = dev_of(ours, oracle) <= tol, dev_of(aten, oracle) <= tol
    if a_ok and not o_ok:
        print("  -> ATen agrees with the oracle and our kernel does not:")
        print("     THE KERNEL IS WRONG and the refusal is VALID.")
    elif o_ok and a_ok:
        print("  -> all three agree: the refusal is about some OTHER key,")
        print("     or about operands this probe does not reproduce.")
    elif not a_ok and not o_ok and dev_of(ours, aten) <= tol:
        print("  -> ours and ATen agree with each other and both differ from")
        print("     the oracle: THE ORACLE IS WRONG.")
    else:
        print("  -> all three differ: the operands are not what any of them")
        print("     thinks they are.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
