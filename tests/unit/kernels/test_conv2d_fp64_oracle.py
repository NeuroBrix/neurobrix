"""The convolution reference, checked against an implementation it shares nothing with.

A reference validated against itself is worthless, and a reference that is WRONG
is worse than none: it turns healthy work into a loud stop, and a refusal reads
as vigilance. So every geometry here is checked against torch's convolution —
a different implementation, a different library, a different decade of tuning —
at float64, where the two must agree to machine precision rather than to a
tolerance.

torch appears here and not in `src/`: the engine's Triton branch is sealed
against it (R33), and an oracle under `tests/` is exactly where the doctrine puts
it.

The geometries are chosen, not sampled. Each one makes a different branch of the
arithmetic observable:

  stride 1, no padding     the base case, where every index is trivially aligned
  stride 2                 the output grid no longer matches the input grid
  asymmetric padding h≠w   catches a transposed (h, w) pair, which is invisible
                           whenever the two are equal
  dilation 2               the taps stop being contiguous
  groups 2 and 4           the channel blocks must line up on both operands
  a 1x1 kernel             no spatial extent at all, where an off-by-one in the
                           tap loop disappears
  even kernel 2x2          padding is not symmetric around the centre

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_conv2d_fp64_oracle.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.oracles.conv2d_fp64 import (
    conv2d_from_args, conv2d_reference, depthwise_from_args, depthwise_reference,
)

torch = pytest.importorskip("torch")


def _rand(*shape, seed=0):
    return np.asarray(np.random.default_rng(seed).standard_normal(shape),
                      dtype=np.float64)


GEOMETRIES = [
    # (N, Cin, H, W, Cout, KH, KW, stride, padding, dilation, groups, why)
    (2, 3, 7, 9, 4, 3, 3, (1, 1), (0, 0), (1, 1), 1, "base case"),
    (1, 4, 8, 8, 8, 3, 3, (2, 2), (1, 1), (1, 1), 1, "stride 2"),
    (2, 4, 9, 7, 6, 3, 5, (1, 2), (2, 1), (1, 1), 1, "asymmetric stride and padding"),
    (1, 2, 11, 11, 4, 3, 3, (1, 1), (2, 2), (2, 2), 1, "dilation 2"),
    (2, 6, 6, 6, 6, 3, 3, (1, 1), (1, 1), (1, 1), 2, "groups 2"),
    (1, 8, 5, 5, 8, 3, 3, (1, 1), (1, 1), (1, 1), 4, "groups 4"),
    (2, 5, 6, 6, 7, 1, 1, (1, 1), (0, 0), (1, 1), 1, "1x1 kernel"),
    (1, 3, 6, 6, 3, 2, 2, (1, 1), (1, 1), (1, 1), 1, "even kernel"),
    (1, 4, 10, 10, 4, 3, 3, (3, 3), (1, 1), (2, 1), 2, "everything at once"),
]


@pytest.mark.parametrize(
    "n,cin,h,w,cout,kh,kw,stride,padding,dilation,groups,why",
    GEOMETRIES, ids=[g[-1] for g in GEOMETRIES])
def test_conv2d_agrees_with_an_independent_implementation(
        n, cin, h, w, cout, kh, kw, stride, padding, dilation, groups, why):
    x = _rand(n, cin, h, w, seed=1)
    weight = _rand(cout, cin // groups, kh, kw, seed=2)
    mine = conv2d_reference(x, weight, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)
    theirs = torch.nn.functional.conv2d(
        torch.from_numpy(x), torch.from_numpy(weight), bias=None,
        stride=stride, padding=padding, dilation=dilation, groups=groups).numpy()
    assert mine.shape == theirs.shape, why
    assert np.allclose(mine, theirs, rtol=0, atol=1e-12), (
        f"{why}: max |diff| {np.abs(mine - theirs).max():.3e}")


DEPTHWISE = [
    (2, 4, 7, 7, 3, 3, (1, 1), (1, 1), "base case"),
    (1, 8, 9, 11, 3, 3, (2, 2), (1, 1), "stride 2"),
    (2, 3, 8, 6, 5, 3, (1, 2), (2, 1), "asymmetric kernel, stride and padding"),
    (1, 6, 5, 5, 1, 1, (1, 1), (0, 0), "1x1"),
]


@pytest.mark.parametrize("n,c,h,w,kh,kw,stride,padding,why", DEPTHWISE,
                         ids=[g[-1] for g in DEPTHWISE])
def test_depthwise_agrees_with_an_independent_implementation(
        n, c, h, w, kh, kw, stride, padding, why):
    x = _rand(n, c, h, w, seed=3)
    weight = _rand(c, kh, kw, seed=4)
    mine = depthwise_reference(x, weight, stride=stride, padding=padding)
    theirs = torch.nn.functional.conv2d(
        torch.from_numpy(x), torch.from_numpy(weight).reshape(c, 1, kh, kw),
        bias=None, stride=stride, padding=padding, groups=c).numpy()
    assert mine.shape == theirs.shape, why
    assert np.allclose(mine, theirs, rtol=0, atol=1e-12), (
        f"{why}: max |diff| {np.abs(mine - theirs).max():.3e}")


def test_the_depthwise_path_is_not_the_grouped_path():
    """They agree numerically and are kept apart on purpose: a shared path would
    let a defect in one hide behind the other, which is why the engine has two
    kernels in the first place."""
    x, weight = _rand(1, 4, 6, 6, seed=5), _rand(4, 3, 3, seed=6)
    grouped = conv2d_reference(x, weight.reshape(4, 1, 3, 3), stride=(1, 1),
                               padding=(1, 1), dilation=(1, 1), groups=4)
    assert np.allclose(grouped,
                       depthwise_reference(x, weight, stride=(1, 1), padding=(1, 1)),
                       rtol=0, atol=1e-12)


# -- the refusals, which matter as much as the answers ---------------------

def _named(**over):
    base = dict(input_pointer=object(), weight_pointer=object(),
                batch_dim=1, in_feat_dim=2, in_height=5, in_width=5,
                out_feat_dim=3, out_height=5, out_width=5,
                kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
                padding_height=1, padding_width=1, groups=1,
                dilation_height=1, dilation_width=1)
    base.update(over)
    return base


def test_an_operand_that_cannot_be_read_is_silence():
    assert conv2d_from_args(_named(), lambda t: None) is None


def test_a_declared_extent_that_contradicts_the_array_is_silence():
    """Reshaping anyway would produce a confident wrong reference."""
    arrays = {"in": _rand(1, 2, 5, 5, seed=7), "w": _rand(3, 2, 3, 3, seed=8)}
    reader = lambda t: arrays["in"] if t is nm["input_pointer"] else arrays["w"]
    nm = _named(in_height=6)          # says 6, the array has 5
    assert conv2d_from_args(nm, reader) is None


def test_a_sound_call_returns_the_reference():
    """Without this the refusals above prove nothing."""
    x, weight = _rand(1, 2, 5, 5, seed=7), _rand(3, 2, 3, 3, seed=8)
    nm = _named()
    reader = lambda t: x if t is nm["input_pointer"] else weight
    got = conv2d_from_args(nm, reader)
    assert got is not None and got.shape == (1, 3, 5, 5)
    assert np.allclose(got, torch.nn.functional.conv2d(
        torch.from_numpy(x), torch.from_numpy(weight), padding=1).numpy(),
        rtol=0, atol=1e-12)


def test_a_missing_key_is_silence_not_an_exception():
    """The screen must never turn a missing argument into a launch failure."""
    nm = _named(); nm.pop("stride_height")
    assert conv2d_from_args(nm, lambda t: _rand(1, 2, 5, 5)) is None


def test_a_geometry_the_kernel_and_the_formula_disagree_on_is_silence():
    """The kernel says the output is 7 tall; the formula says 5. That is a
    finding, not something to paper over with a reshape."""
    x, weight = _rand(1, 2, 5, 5, seed=7), _rand(3, 2, 3, 3, seed=8)
    nm = _named(out_height=7)
    reader = lambda t: x if t is nm["input_pointer"] else weight
    assert conv2d_from_args(nm, reader) is None
