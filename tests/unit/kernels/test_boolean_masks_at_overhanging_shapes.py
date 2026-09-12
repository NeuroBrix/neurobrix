"""Cells C/D/E: the nineteen mask edits against the oracle, at shapes that can
actually see a wrong mask.

WHY THIS FILE IS NOT A ROUTINE ORACLE TEST

A mask expression is only exercised where the block OVERHANGS the tensor. Run
these kernels at `M = 2*BLOCK_M`, `N = 4*BLOCK_N` and every lane of every block
is valid, the mask is true everywhere, and a mask that is true everywhere cannot
be wrong: the botched form passes and the green means nothing. Round numbers are
what one types, so the vacuous version of this test is the one that writes
itself.

So every dimension below carries the reason it has the value it has, and — the
part that matters — **the test asserts its own shapes still overhang before it
uses them**. Block sizes are read from the wrapper module, not copied here, so
if `_RED_BM` moves from 8 to 16 the assertion speaks instead of the test
quietly becoming a decoration. That is the shape rule made mechanical:
`docs/reference/what-a-green-test-proves.md`.

WHAT IS PROVED HERE THAT THE IR HARNESS CANNOT PROVE

`tools/kernel_boolean_ir_equality.py` proves the nineteen edits emit the same
instructions as the code before them — that they are NEUTRAL. It says nothing
about whether those instructions were right in the first place. A `tril` wrong
since the day it was written stays wrong and the IR comparison stays green.
This file is the other half: the kernels against `torch`, at the shapes where a
mask decides something.

COVERAGE, MEASURED RATHER THAN HOPED (`neurobrix coverage`, 2026-09-10)

Three of these kernels are reached by NO container installed on this machine —
`aten::argmin`, `aten::min`, `aten::var`, 0 of 56 — so no model run can
exercise them and this direct test is the only instrument they have. The
repository already paid for that gap once: `min_wrapper` passed four arguments
to a five-argument kernel for an unknown period, "a latent defect on a path no
model of the zoo exercised", found by the kernel reference bank on 2026-09-05
rather than by any campaign.

SEEN FAILING

The shape guard was injected on 2026-09-10 by setting `M_ROWS = 16` — an exact
multiple of the reduction row block — and turned red with the reason: *"every
lane is valid and the mask under test decides nothing. This test would pass on
the botched kernel."* A guard nobody has watched bite is a claim.

STATUS

The CPU half runs today and is what keeps this file honest: it proves the
shapes are non-degenerate and the block constants have not drifted. The GPU
half is skipped without a device and is queued as cells C/D/E behind the
certified-directory campaign — it has NOT run yet, and this docstring says so
rather than letting a green in CI imply it did.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_boolean_masks_at_overhanging_shapes.py -v
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

# A rig-less checkout must still collect and run the CPU half -- but the guard
# covers the PACKAGE, not the names. A catch-all around a name list turns the
# absence of the function under test into the same skip as the absence of the
# machine, and a skip is invisible in a count: five red tests became five skips
# in another file this way, on the day the rule against it was written.
_IMPORTED = False
try:  # pragma: no cover - import-time only
    import neurobrix.kernels.wrappers as w
    _IMPORTED = True
except Exception:
    w = None

if _IMPORTED:
    # Imported OUTSIDE the guard: if one of these names is gone, that is a
    # failure about this repository and must read as one.
    from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch


def _rig_reason() -> str:
    """Why the oracle half may not run, or "" if it may.

    A DOOR, not a check afterwards. The oracle half launches real kernels, and
    this repository runs timed campaigns on the same four cards — a stray
    context on a locked-clock measurement is exactly the perturbation that cost
    a cell on 2026-09-10.

    Deliberately NOT the repository's older idiom, which decides by allocating a
    one-element tensor on `cuda:0`: that creates the very context it is asking
    permission for, and it does so at COLLECTION time, so merely listing the
    tests touches the rig. `nvidia-smi` reads NVML and opens no context.
    """
    if not _IMPORTED:
        return "the kernel wrappers did not import"
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return "no nvidia-smi: cannot establish that the rig is free"
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi did not answer ({exc.__class__.__name__})"
    if out.returncode != 0:
        return "nvidia-smi reported no device"
    busy = [line for line in out.stdout.splitlines() if line.strip()]
    if busy:
        return (f"{len(busy)} compute process(es) hold the rig — the oracle "
                f"half would perturb a measurement in flight")
    return ""


_REASON = _rig_reason()
needs_a_free_rig = pytest.mark.skipif(bool(_REASON), reason=_REASON or "rig free")


# ---------------------------------------------------------------------------
# The shapes, each with the reason it has the value it has.
# ---------------------------------------------------------------------------
#
# `_RED_BM` (8) is the row block of every 2-D reduction; `_RED_BN` (1024) the
# column block of prod/min/max/var. `all`/`any` size their column block as
# `min(4096, next_power_of_2(N))`, so any N that is not a power of two
# overhangs there too.

M_ROWS = 17
"""Rows. 17 = 2*8 + 1 — the last row block holds ONE valid row and seven
invalid ones. A row mask that is dropped reads seven rows past the tensor; a
row mask that is over-applied loses the seventeenth row's result. Both are
visible here and neither is visible at M = 16."""

N_COLS = 67
"""Columns. Prime, and 67 < 1024, so the single column block is 94 % mask: the
column mask decides almost every lane. Also not a power of two, so `all`/`any`
round it to 128 and overhang by 61."""

TRI_M, TRI_N = 37, 23
"""Triangular kernels. Coprime and neither a multiple of its block, so no
alignment can make the two masks agree by accident. Small enough that the
whole matrix is inspected rather than sampled."""

WN_M, WN_N = 13, 29
"""weight_norm. Both prime and both overhang. Both variants are exercised:
`first` walks N (its BLOCK_N must overhang) and `last` walks M. The assertion
is read from the SECOND pass's output, because pass 1 only computes the norm —
a wrong mask there is invisible until pass 2 divides by it."""

WHERE_N = 3000
"""`aten::where`. Not a multiple of the elementwise block, so the tail block is
partial. The condition is built with values OTHER than 0/1 as well, because the
kernel receives a bool through a uint8 container and the question is what it
does with an integer."""


def _block_sizes():
    """Read from the wrapper module, never copied: a constant duplicated into a
    test is a constant that stops matching the code without saying so."""
    return w._RED_BM, w._RED_BN


def _overhangs(extent: int, block: int, what: str) -> None:
    assert extent % block != 0, (
        f"{what}: {extent} is an exact multiple of the block {block}, so every "
        f"lane is valid and the mask under test decides nothing. This test "
        f"would pass on the botched kernel. Choose an extent that overhangs — "
        f"and if the block size moved, move the extent with it.")


# ---------------------------------------------------------------------------
# CPU half — runs today, and is what stops this file becoming a decoration.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IMPORTED, reason="the kernel wrappers did not import")
def test_the_chosen_shapes_still_overhang_their_blocks():
    """The control on this test's own honesty.

    Every assertion below is worthless at an aligned shape. If a block constant
    changes and nobody revisits the shapes, this speaks — instead of the suite
    going green over a mask nothing exercises."""
    bm, bn = _block_sizes()
    _overhangs(M_ROWS, bm, "M_ROWS against the reduction row block")
    _overhangs(N_COLS, bn, "N_COLS against the reduction column block")
    _overhangs(TRI_M, bm, "TRI_M against the triangular row block")
    _overhangs(WN_M, bm, "WN_M against the weight_norm row block")

    assert N_COLS & (N_COLS - 1), (
        "N_COLS is a power of two, so all/any round it to itself and their "
        "column mask covers every lane — the very thing this shape is for")
    assert M_ROWS % bm == 1, (
        f"M_ROWS should leave exactly ONE valid row in the last block "
        f"({bm}k + 1); it leaves {M_ROWS % bm}. That is the worst case for a "
        f"row mask and the reason this number is not merely odd.")


# ---------------------------------------------------------------------------
# GPU half — queued as cells C/D/E. Skipped without a device.
# ---------------------------------------------------------------------------

def _nbx(arr):
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _rows(seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M_ROWS, N_COLS), dtype=np.float32)


@needs_a_free_rig
@pytest.mark.parametrize("op", ["prod", "min", "max", "var", "argmin"])
def test_row_reductions_against_torch_at_an_overhanging_shape(op):
    """prod.py:73, min_reduce.py:78, max_reduce.py:78, var.py:120, argmin.py:80.

    All five share the same edited expression, `(m < M) & (n < N)`, and three of
    the five are reached by no installed container at all."""
    import torch
    a = _rows()
    x, t = _nbx(a), torch.from_numpy(a)

    if op == "prod":
        got, want = w.prod_wrapper(x, dim=1), torch.prod(t, dim=1)
    elif op == "min":
        got, want = w.min_wrapper(x, dim=1)[0], torch.min(t, dim=1).values
    elif op == "max":
        got, want = w.max_wrapper(x, dim=1)[0], torch.max(t, dim=1).values
    elif op == "var":
        got, want = w.var_wrapper(x, dim=1, correction=1), torch.var(t, dim=1, correction=1)
    else:
        got, want = w.argmin_wrapper(x, dim=1), torch.argmin(t, dim=1)

    got = nbx_to_torch(got).cpu()
    assert got.shape == want.shape, f"{op}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    if op == "argmin":
        assert torch.equal(got.long(), want.long()), f"{op}: index mismatch"
    else:
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)


@needs_a_free_rig
@pytest.mark.parametrize("op", ["all", "any"])
def test_all_and_any_with_the_single_disagreeing_element_in_the_tail(op):
    """all_reduce.py:14/78/81 and any_reduce.py:14/78/81.

    THE PLACEMENT IS THE TEST. A uniformly-true tensor returns the right answer
    under ANY mask, including no mask at all, so the disagreeing element goes in
    the LAST column of the LAST valid row — the one position a dropped mask, an
    over-applied mask and a wrong combiner each get differently."""
    import torch
    if op == "all":
        a = np.ones((M_ROWS, N_COLS), dtype=np.float32)
        a[M_ROWS - 1, N_COLS - 1] = 0.0
        got, want = w.all_wrapper(_nbx(a), dim=1), torch.all(torch.from_numpy(a) != 0, dim=1)
    else:
        a = np.zeros((M_ROWS, N_COLS), dtype=np.float32)
        a[M_ROWS - 1, N_COLS - 1] = 1.0
        got, want = w.any_wrapper(_nbx(a), dim=1), torch.any(torch.from_numpy(a) != 0, dim=1)

    got = nbx_to_torch(got).cpu().bool()
    assert torch.equal(got, want), (
        f"{op}: {got.tolist()} vs {want.tolist()} — the last row is the one the "
        f"row mask decides, and the last column the one the column mask decides")


@needs_a_free_rig
@pytest.mark.parametrize("diagonal", [-3, 0, 5])
@pytest.mark.parametrize("op", ["tril", "triu"])
def test_triangular_kernels_at_three_diagonals(op, diagonal):
    """tril.py:32, triu.py:30.

    Three diagonals because the mask and the triangular predicate are different
    conditions on the same indices: at `diagonal = 0` a mask defect can hide
    behind the predicate zeroing the same lanes. Negative and positive move
    them apart."""
    import torch
    rng = np.random.default_rng(11)
    a = rng.standard_normal((TRI_M, TRI_N), dtype=np.float32)
    t = torch.from_numpy(a)
    fn = w.tril_wrapper if op == "tril" else w.triu_wrapper
    ref = torch.tril if op == "tril" else torch.triu
    got = nbx_to_torch(fn(_nbx(a), diagonal=diagonal)).cpu()
    torch.testing.assert_close(got, ref(t, diagonal=diagonal))


@needs_a_free_rig
@pytest.mark.parametrize("dim", [0, 1])
def test_weight_norm_reads_the_second_pass(dim):
    """weight_norm.py:43, 54, 90, 101 — the two passes of both variants.

    The output is asserted, not the norm, because pass 1 computes the norm and
    pass 2 divides by it: a wrong mask in pass 1 shows up only through what
    pass 2 does with it. `dim=0` exercises the `first` variant, `dim=1` the
    `last` one."""
    import torch
    rng = np.random.default_rng(19)
    v = rng.standard_normal((WN_M, WN_N), dtype=np.float32)
    g = rng.standard_normal(WN_M if dim == 0 else WN_N, dtype=np.float32)

    out, _norm = w.weight_norm_interface_wrapper(_nbx(v), _nbx(g), dim=dim)
    tv, tg = torch.from_numpy(v), torch.from_numpy(g)
    want = torch._weight_norm_interface(tv, tg, dim)[0]
    torch.testing.assert_close(nbx_to_torch(out).cpu(), want, rtol=2e-5, atol=2e-5)


@needs_a_free_rig
@pytest.mark.parametrize("truthy", [1, 2, 255])
def test_where_treats_any_nonzero_byte_as_true(truthy):
    """where.py:26.

    The kernel receives a bool through a uint8 container, so the loaded value is
    an integer. `1` is the canonical case; `2` and `255` are here because the
    two ways to silence the deprecation are NOT the same function on those
    inputs — a bitcast to int1 would read the low bit and call 2 false, while a
    cast reads it as true. torch is the oracle for which is right."""
    import torch
    rng = np.random.default_rng(23)
    cond = np.zeros(WHERE_N, dtype=np.uint8)
    cond[::3] = truthy
    x = rng.standard_normal(WHERE_N, dtype=np.float32)
    y = rng.standard_normal(WHERE_N, dtype=np.float32)

    got = nbx_to_torch(w.where_wrapper(_nbx(cond), _nbx(x), _nbx(y))).cpu()
    want = torch.where(torch.from_numpy(cond) != 0,
                       torch.from_numpy(x), torch.from_numpy(y))
    torch.testing.assert_close(got, want)
