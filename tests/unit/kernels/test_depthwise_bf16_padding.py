"""depthwise_conv2d must not lose its answer in bf16 when padding is non-zero.

Red on 2026-09-22 (Apple/Metal): with `pad=1` a bf16 depthwise convolution deviated 0.754
from the fp64 oracle — a relative deviation near 1.0, i.e. an output uncorrelated with the
reference, not an imprecise one. Unpadded bf16 was at the mantissa floor (3.2e-03), and fp32
and fp16 were clean at BOTH paddings, so it was neither the shape nor the oracle.

Cause: the stencil multiplied in the operands' own dtype for every type except fp16, which
alone upcast to fp32 first. A bf16 product of a masked-loaded operand is wrong on the Metal
backend; the same source is exact to the mantissa on CUDA (the rack, Triton 3.8.0, V100
sm_70: bf16 pad0 == bf16 pad1 == 5.252e-03). Upcasting before the product removes the
dependence on that lowering, and is what the fp16 branch already did — the accumulator was
always fp32, so nothing is paid for it and bf16 pad0 improved too (3.17e-03 -> 2.955e-03).

The certifier refused all 28 padded stride-1 depthwise keys over this, every config excluded
at 12x-25x tolerance, and the runtime consensus screen refused to seat any config. Both doors
held; this is the defect they were holding against.
"""
import numpy as np
import pytest

QUAL = "neurobrix.kernels.ops.depthwise_conv2d.depthwise_conv2d_kernel"
SEED = 20260907
#: bf16 carries 8 explicit mantissa bits, so ~4e-3 relative is the floor for a 9-tap
#: accumulation. The defect sat two orders of magnitude above this.
BF16_FLOOR = 0.02


def _reference(x, wt, stride, pad):
    """The definition, in float64, written to be obviously right rather than fast."""
    n, c, h, wd = x.shape
    kh, kw = wt.shape[2], wt.shape[3]
    sh, sw = stride
    ph, pw = pad
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (wd + 2 * pw - kw) // sw + 1
    out = np.zeros((n, c, oh, ow), dtype=np.float64)
    for ch in range(c):
        for i in range(oh):
            for j in range(ow):
                win = xp[0, ch, i * sh:i * sh + kh, j * sw:j * sw + kw]
                out[0, ch, i, j] = float((win * wt[ch, 0]).sum())
    return out


def _deviation(c, h, kh, sh, ph, dt):
    from neurobrix.kernels import autotune_certify as AC
    oh = (h + 2 * ph - kh) // sh + 1
    key = (c, h, h, oh, oh, kh, kh, sh, sh, ph, ph, dt == "fp16", dt, dt, dt)
    call, _oracle, _ = AC.synthesize(QUAL, None, key, np.random.default_rng(SEED))
    got = np.asarray(AC.host_values(call()), dtype=np.float64)
    r = np.random.default_rng(SEED)                 # the same draws synthesize made
    x = AC._arr(r, (1, c, h, h), dt)
    wt = AC._arr(r, (c, 1, kh, kh), dt)
    ref = _reference(np.asarray(x, dtype=np.float64), np.asarray(wt, dtype=np.float64), (sh, sh), (ph, ph))
    scale = np.abs(ref).max()
    return float(np.abs(got - ref).max() / scale) if scale else 0.0


@pytest.mark.parametrize("c,h", [(64, 32), (128, 32), (64, 64)])
def test_bf16_padded_matches_the_reference(c, h):
    """The case that was wrong: padding non-zero, stride 1, bf16."""
    dev = _deviation(c, h, 3, 1, 1, "bf16")
    assert dev < BF16_FLOOR, (
        f"bf16 depthwise with padding deviates {dev:.4g} from the fp64 reference "
        f"(floor {BF16_FLOOR}); a deviation near 1.0 means the output is uncorrelated "
        f"with the reference, not imprecise"
    )


def test_padding_does_not_change_the_error_floor():
    """The discriminator: on a correct kernel, pad 0 and pad 1 sit on the SAME floor.

    This is the shape of the rack's clean CUDA result (bf16 pad0 == pad1 to four significant
    figures) and it is what the defect broke — 3.2e-03 against 0.754 on the same step.
    """
    unpadded = _deviation(64, 32, 3, 1, 0, "bf16")
    padded = _deviation(64, 32, 3, 1, 1, "bf16")
    assert padded < BF16_FLOOR and unpadded < BF16_FLOOR
    assert padded < 4 * unpadded, (
        f"padding moved the error floor: pad0 {unpadded:.4g} -> pad1 {padded:.4g}. "
        "Padding changes which taps are masked, never the accuracy of the in-bounds ones."
    )


@pytest.mark.parametrize("dt", ["fp32", "fp16"])
def test_the_other_dtypes_stay_clean(dt):
    """fp32 and fp16 were never affected; the fix must not regress them."""
    assert _deviation(64, 32, 3, 1, 1, dt) < (1e-5 if dt == "fp32" else 5e-3)
