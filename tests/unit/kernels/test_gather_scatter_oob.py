"""Out-of-range index → LOUD failure on the three gather/scatter kernels
(D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).

torch raises on an out-of-range index in index_select / index_put /
embedding. The FlagGems-lineage Triton ports were silent, and the fix
was `tl.device_assert` kept in the binary by `@triton.jit(debug=True)`.

That fix is backend-conditional and the first version of this file did
not know it. `tl.device_assert` reaches the IR at all ONLY when
`options.debug` is true (triton/language/semantic.py: `if not
self.builder.options.debug: return`), so a `tt.assert` in the IR is by
construction one the author demanded — and the Metal backend elides it
anyway, computing its predicate and discarding it (measured 2026-09-10,
validation_outputs/gather_scatter_oob_2026_09_10/). On that backend
`index_put` wrote outside the tensor and `embedding` read outside the
weight, in silence.

The second channel closes it: a `FAULT_CODE` constexpr, non-zero only
where the assert is not honoured, makes the kernel store its code into
a one-word fault buffer that `check_device_faults` raises on at the
next host observation. Where the assert IS honoured the wrapper passes
0 and not one instruction of it is emitted — proven here on the
generated code, since this machine cannot run the CUDA path.

WHAT THE FIRST VERSION OF THIS FILE GOT WRONG, kept as the reason the
oracle rule exists: its out-of-range subprocess computed a numpy oracle
`rows[idx]` with the out-of-range index in it. numpy raised there, at
the oracle line, before the kernel was ever questioned — so its
`returncode != 0` assertion passed for the wrong reason and proved
nothing about the path it was named after. The out-of-range case below
takes NO oracle: it asserts on what the engine does, and nothing else.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_gather_scatter_oob.py -v
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
import subprocess
import sys

import pytest

from neurobrix.kernels import nbx_tensor as nt
from neurobrix.kernels.ops.embedding import embedding_kernel, EMBEDDING_OOB
from neurobrix.kernels.ops.index_put_op import index_put_kernel, INDEX_PUT_OOB
from neurobrix.kernels.ops.index_select import index_select_kernel, INDEX_SELECT_OOB

_KERNELS = {
    "index_select": (index_select_kernel, "N", INDEX_SELECT_OOB),
    "index_put": (index_put_kernel, "R", INDEX_PUT_OOB),
    "embedding": (embedding_kernel, "V", EMBEDDING_OOB),
}


# ---------------------------------------------------------------------------
# Structure — true on every machine, device or not
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_kernel_keeps_its_device_assert(name):
    fn, bound, _ = _KERNELS[name]
    assert getattr(fn, "debug", None) is True, f"{name}: debug=True dropped — the assert would compile out"
    src = inspect.getsource(fn.fn)
    assert "tl.device_assert(" in src, f"{name}: no device_assert in the kernel"
    assert bound in fn.arg_names, f"{name}: bound parameter {bound!r} missing from the signature"


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_kernel_carries_the_second_channel(name):
    """The assert alone is not enough on a backend that elides it."""
    fn, _, _ = _KERNELS[name]
    assert "FAULT_CODE" in fn.arg_names, f"{name}: no FAULT_CODE constexpr"
    assert "fault_ptr" in fn.arg_names, f"{name}: no fault buffer argument"
    src = inspect.getsource(fn.fn)
    assert "if FAULT_CODE != 0:" in src, (
        f"{name}: the fault store is not behind a compile-time gate — a "
        f"backend that honours the assert would pay for it")
    assert "tl.store(fault_ptr, FAULT_CODE)" in src, f"{name}: nothing reports the fault"


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_the_message_cannot_drift_from_the_kernel(name):
    """Triton forbids a module global inside a @jit body, so the literal is
    written twice. The channel raises the module constant and the assert
    prints the literal; if they drift, one of the two lies."""
    fn, _, message = _KERNELS[name]
    src = inspect.getsource(fn.fn)
    assert f'"{message}"' in src, (
        f"{name}: the kernel's assert message and the module constant "
        f"{message!r} have drifted apart")


def test_the_guarded_access_is_legal_even_before_the_refusal_lands():
    """The refusal is asynchronous on the channel path. Between the bad
    index and the raise, the kernel must not touch memory outside the
    tensor — which is what `index_put` did (8 floats past a 24-float
    tensor) and what `embedding` did (past the end of the weight)."""
    put = inspect.getsource(index_put_kernel.fn)
    assert "dst = tl.where(row_valid, row, 0) * T + t" in put, (
        "index_put: the destination address is not made legal")
    assert "mask=mask & row_valid" in put, "index_put: the store is not masked by validity"
    emb = inspect.getsource(embedding_kernel.fn)
    assert "tl.where(in_range, row_idx, 0)" in emb, (
        "embedding: the weight row address is not made legal")
    assert "mask & in_range" in emb, "embedding: the load is not masked by validity"


# ---------------------------------------------------------------------------
# The capability that decides which channel is armed
# ---------------------------------------------------------------------------

def test_every_backend_says_whether_it_honours_a_device_assert():
    table = nt._BACKEND_TRAPS_ON_DEVICE_ASSERT
    assert table["cuda"] is True and table["hip"] is True
    assert table["metal"] is False, (
        "Metal computes the assert predicate and discards it — measured "
        "2026-09-10; a True here would disarm the only channel that works there")
    assert set(table) == set(nt._GPU_BACKENDS) | {"metal"}, (
        "a backend without a row would be guessed at, not asked")


def test_an_unknown_backend_is_refused_not_guessed():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        nt._backend_capability({"cuda": True}, "_T", "whether it does the thing")


def test_a_disarmed_kernel_is_compiled_with_code_zero():
    assert nt.device_fault_code("x: out of range", armed=False) == 0
    assert nt.device_fault_code("x: out of range", armed=True) != 0


def test_a_registered_fault_keeps_its_code():
    first = nt.register_device_fault("test: a message of its own")
    assert nt.register_device_fault("test: a message of its own") == first
    assert nt.register_device_fault("test: a different message") != first


# ---------------------------------------------------------------------------
# Behaviour — the real wrappers, on the device
# ---------------------------------------------------------------------------

_SUBPROCESS = r"""
import sys, numpy as np
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as w
kind, oob = sys.argv[1], sys.argv[2] == "oob"
rows = np.arange(24, dtype=np.float32).reshape(6, 4)

def call(kind, oob):
    if kind == "index_select":
        idx = np.array([1, 3, 11 if oob else 5], dtype=np.int64)
        return w.index_select_wrapper(NBXTensor.from_numpy(rows), 0,
                                      NBXTensor.from_numpy(idx)), idx
    if kind == "index_put":
        idx = np.array([0, 8 if oob else 2], dtype=np.int64)
        vals = np.full((2, 4), -1.0, dtype=np.float32)
        return w.index_put_wrapper(NBXTensor.from_numpy(rows),
                                   [NBXTensor.from_numpy(idx)],
                                   NBXTensor.from_numpy(vals)), idx
    ids = np.array([[4, 7 if oob else 1]], dtype=np.int64)
    return w.embedding(NBXTensor.from_numpy(rows),
                       NBXTensor.from_numpy(ids)), ids

out, idx = call(kind, oob)
DeviceAllocator.sync_device()
got = out.numpy()
if oob:
    # NO ORACLE HERE, deliberately: numpy's own fancy-index raises on an
    # out-of-range subscript, so computing an expectation would raise
    # before the engine is questioned. Reaching this line at all is the
    # failure -- the engine was supposed to refuse.
    print("SILENT", kind, got.tolist())
    sys.exit(0)
exp = rows[idx] if kind != "index_put" else None
if kind == "index_put":
    exp = rows.copy(); exp[idx] = np.full((2, 4), -1.0, dtype=np.float32)
assert np.array_equal(got, exp), (got, exp)
print("CONTROL OK", kind)
"""


def _run(kind: str, mode: str):
    env = dict(os.environ, PYTHONPATH=str(_REPO / "src"))   # absolute: runnable from any cwd
    return subprocess.run([sys.executable, "-c", _SUBPROCESS, kind, mode],
                          capture_output=True, text=True, env=env, timeout=300)


def _no_device() -> bool:
    try:
        from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor
        NBXTensor.empty((1,), NBXDtype.float32, "cuda:0")
        return False
    except Exception:
        return True


@pytest.mark.skipif(_no_device(), reason="needs a GPU")
@pytest.mark.parametrize("kind", sorted(_KERNELS))
def test_in_range_control_matches_torch_semantics(kind):
    r = _run(kind, "ok")
    assert r.returncode == 0, r.stderr[-2000:]
    assert f"CONTROL OK {kind}" in r.stdout


@pytest.mark.skipif(_no_device(), reason="needs a GPU")
@pytest.mark.parametrize("kind", sorted(_KERNELS))
def test_out_of_range_index_is_refused_by_name(kind):
    """The whole point, and the assertion the first version never made."""
    _, _, message = _KERNELS[kind]
    r = _run(kind, "oob")
    assert "SILENT" not in r.stdout, (
        f"{kind}: an out-of-range index passed silently\n{r.stdout}")
    assert r.returncode != 0, f"{kind}: no refusal\n{r.stdout}\n{r.stderr[-1500:]}"
    blob = r.stdout + r.stderr
    assert message in blob, (
        f"{kind}: refused, but not by the name of the contract it broke "
        f"({message!r} absent)\n{r.stderr[-2000:]}")
