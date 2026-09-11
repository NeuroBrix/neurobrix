"""The screen's oracle goes on the OUTPUT buffer, found by address.

The first version of this provider matched the output by BYTE LENGTH. For a
matmul at (19, 2048, 2048) the operand `a` and the output `c` are both
19x2048 fp32 — the same 155648 bytes — so the oracle landed on `a`, an INPUT.
Every candidate then "disagreed" with it and the screen refused all ten
candidates of a shape whose certified deviation is 1.7e-06.

**An oracle that refuses correct configurations is worse than no oracle.** It
turns a silent wrong into a loud stop on healthy work, and it would have been
believed, because a refusal reads as vigilance.

Both directions are pinned here, because a guard that stops refusing
everything can also stop refusing what is truly wrong.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_screen_oracle_matches_by_address.py -v
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from neurobrix.kernels import screen_oracle as SO


class _Tensor:
    """A live operand: an address, a shape, a dtype."""

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.shape = self._arr.shape

    def data_ptr(self):
        return self._arr.ctypes.data

    @property
    def _nbytes(self):
        return self._arr.nbytes

    def numpy(self):
        return self._arr


class _Tuner:
    def __init__(self, name, nargs):
        # `type("fn", (), {"__name__": name})` makes a class NAMED "fn" —
        # Python overrides the attribute with the class's own name. The
        # provider then read "fn" and found no oracle, and three tests failed
        # on the fixture rather than on the code.
        self.base_fn = type(name, (), {})
        self.nargs = nargs


def _buffers(*tensors):
    return [(t.data_ptr(), t._nbytes, "fp32") for t in tensors]


def test_the_oracle_lands_on_the_output_not_on_a_same_sized_input():
    """The exact shape that broke it: a and c are both 19x2048 fp32."""
    a = _Tensor(np.random.default_rng(0).standard_normal((19, 2048)) * 0.1)
    b = _Tensor(np.random.default_rng(1).standard_normal((2048, 2048)) * 0.1)
    c = _Tensor(np.zeros((19, 2048)))
    assert a._nbytes == c._nbytes, "the premise of the defect"

    tuner = _Tuner("matmul_kernel", {"a_ptr": a, "b_ptr": b, "c_ptr": c})
    out = SO.provider(tuner, ("k",), _buffers(a, b, c))
    assert out is not None

    want = (a.numpy().astype(np.float64) @ b.numpy().astype(np.float64)).astype(np.float32)
    # the THIRD buffer is c: it must carry the oracle
    assert np.frombuffer(out[2], dtype=np.float32).reshape(19, 2048) == pytest.approx(
        want, rel=1e-5), "the oracle is not on the output"
    # the FIRST buffer is a: it must carry a unchanged, NOT the oracle
    assert np.frombuffer(out[0], dtype=np.float32).reshape(19, 2048) == pytest.approx(
        a.numpy()), "an input buffer was given the oracle"


def test_an_unknown_kernel_has_no_oracle_and_says_so(capsys):
    SO._ANNOUNCED.clear()
    tuner = _Tuner("some_kernel_we_cannot_oracle", {"x": 1})
    assert SO.provider(tuner, ("k",), []) is None
    said = capsys.readouterr().out
    assert "CONSENSUS ALONE" in said and "some_kernel_we_cannot_oracle" in said


def test_the_announcement_is_made_once_per_key(capsys):
    SO._ANNOUNCED.clear()
    tuner = _Tuner("unknown_kernel", {"x": 1})
    for _ in range(5):
        SO.provider(tuner, ("same-key",), [])
    assert capsys.readouterr().out.count("CONSENSUS ALONE") == 1, (
        "a warning repeated per launch is a warning nobody reads")


def test_a_size_disagreement_refuses_instead_of_guessing(capsys):
    """If the reference and the output buffer disagree on length, something is
    wrong with the mapping and guessing is how the first version failed."""
    SO._ANNOUNCED.clear()
    a = _Tensor(np.zeros((4, 8)))
    b = _Tensor(np.zeros((8, 8)))
    c = _Tensor(np.zeros((4, 8)))
    tuner = _Tuner("matmul_kernel", {"a_ptr": a, "b_ptr": b, "c_ptr": c})
    # a buffer claiming c's address but the wrong length
    bad = [(c.data_ptr(), c._nbytes + 4, "fp32")]
    assert SO.provider(tuner, ("k",), bad) is None
    assert "bytes" in capsys.readouterr().out


def test_every_registered_kernel_names_its_output_argument():
    for name, entry in SO.ORACLES.items():
        fn, out_name = entry
        assert callable(fn) and isinstance(out_name, str) and out_name, name
