"""`aten::stft` / `aten::istft` are derived as `stft_wrapper` / `istft_wrapper` launch them: the
stft frames the signal (`unfold`, hop and window defaults the wrapper's) and runs
`fft_r2c_wrapper` over [rows x frames, n_fft] — no autotuned key for a power-of-two n_fft (the
radix-2 butterfly), one `mm` key otherwise; the istft runs `fft_c2r_wrapper` over
[batch x frames, bins] to n_fft samples — one `mm` key at any n_fft.

The reference is the walk: chatterbox's 16 GB census (2026-09-29) records the istft as
matmul_kernel (M, 16, 9, True, False, fp32, fp32, fp32) on `aten.istft::0` and no key for its
stft (n_fft 16). Before (2026-10-04): both were "not yet derived", and the Mac's table named
MiniCPM-o-4_5's vocoder (hift: stft + istft at n_fft 16) as a gap.

Injection, seen RED: the stft/istft branch of `_op_launches` restored to "not yet derived"."""
import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402
from neurobrix.kernels import census as _census  # noqa: E402
from neurobrix.kernels import launch_keys as LK  # noqa: E402
from neurobrix.kernels.nbx_tensor import NBXDtype  # noqa: E402

S = lambda v: {"type": "scalar", "value": v}       # noqa: E731
T = lambda t: {"type": "tensor", "tensor_id": t}   # noqa: E731


@pytest.fixture(autouse=True)
def _the_bound_target_is_restored():
    """The binding is process state (`launcher._TARGET`, the active vendor profile): a test that
    binds a profile leaves the next test file the target it found."""
    from neurobrix.kernels import launcher
    from neurobrix.kernels.ops import _configs
    target, active = getattr(launcher, "_TARGET", None), dict(_configs._ACTIVE_PROFILE)
    yield
    launcher._TARGET = target
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(active)


def _launches(kind, args, x_shape):
    """The derivation's launches for one op (the hift / s3gen vocoder's args as traced)."""
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)   # a committed V100 profile's ladders
    o = {"op_type": kind, "attributes": {"args": args}, "input_tensor_ids": ["x", "w"]}
    unhandled = collections.Counter()
    out = D._op_launches(kind, f"{kind.replace('::', '.')}::0", o, ["x", "w"],
                         lambda t: x_shape if t == "x" else [16], lambda t: NBXDtype.float32, LK,
                         None, "float16", False, 0, 128, 0, 1 << 30, unhandled)
    return out, unhandled


def test_the_istft_is_the_inverse_dft_over_its_frames():
    args = [T("x"), S(16), S(4), S(16), T("w")]
    out, unhandled = _launches("aten::istft", args, [1, 9, 4801])
    assert not unhandled, unhandled
    b = LK.bucket_of
    assert out == [(LK.MATMUL, (b("M", 4801), 16, 9, True, False, "fp32", "fp32", "fp32"))]
    out2, _ = _launches("aten::istft", args, [9, 4801])              # a 2-D spectrum: one row
    assert out2 == out


def test_a_power_of_two_stft_forms_no_key_and_another_forms_one():
    args = lambda n, hop, win: [T("x"), S(n), S(hop), S(win), T("w"), S(False), S(None), S(True)]
    out, unhandled = _launches("aten::stft", args(16, 4, 16), [1, 19216])
    assert (out, dict(unhandled)) == ([], {})                         # the butterfly: no key
    out, unhandled = _launches("aten::stft", args(20, 5, 20), [2, 22096])
    assert not unhandled, unhandled
    frames = (22096 - 20) // 5 + 1
    b = LK.bucket_of
    assert out == [(LK.MATMUL, (b("M", 2 * frames), 11, 20, True, False, "fp32", "fp32", "fp32"))]
