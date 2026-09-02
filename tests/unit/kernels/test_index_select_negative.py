"""index_select / aten::index: negative indices wrap like torch (2026-09-02).

HAT's `relative_position_index_OCA` buffer carries negative entries; torch's
advanced indexing counts them from the end. The Triton kernel treated them
as invalid and skipped the store, so those output elements kept whatever
the allocation held — zeros on fresh device memory, stale data under the
allocator pool: the pool-on/off byte gate caught the two HAT upscaler
PNGs, the per-op fingerprint named `aten.index::13`, and the op dump
showed rows of zeros (off) vs stale values (on) at the negative indices.
Pin: the wrapped gather equals the numpy reference, byte-exact, on a
buffer pre-filled with NaN (no element may stay unwritten).

Runnable: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_index_select_negative.py -v
"""
from __future__ import annotations

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.kernels import wrappers as w


def test_negative_indices_wrap_and_every_output_element_is_written():
    rng = np.random.default_rng(7)
    table = rng.standard_normal((1521, 6)).astype(np.float32)
    idx = rng.integers(-1521, 1521, size=147456).astype(np.int64)
    ref = table[idx]                                   # numpy wraps negatives like torch
    x = NBXTensor.from_numpy(table)
    i = NBXTensor.from_numpy(idx)
    out = w.index_select_wrapper(x, 0, i)
    got = out.numpy()
    assert got.shape == ref.shape
    assert not np.isnan(got).any(), "unwritten output elements"
    assert np.array_equal(got, ref), "gather differs from the torch/numpy semantics"


def test_meta_index_single_negative_index_matches_reference():
    from neurobrix.kernels.dispatch import _meta_index
    rng = np.random.default_rng(3)
    table = rng.standard_normal((64, 8)).astype(np.float32)
    idx = rng.integers(-64, 64, size=(16, 32)).astype(np.int64)
    ref = table[idx]
    got = _meta_index(NBXTensor.from_numpy(table), [NBXTensor.from_numpy(idx)]).numpy()
    assert np.array_equal(got, ref)


def test_meta_index_joint_negative_indices_match_reference():
    from neurobrix.kernels.dispatch import _meta_index
    rng = np.random.default_rng(11)
    x = rng.standard_normal((5, 7, 3)).astype(np.float32)
    i0 = rng.integers(-5, 5, size=40).astype(np.int64)
    i1 = rng.integers(-7, 7, size=40).astype(np.int64)
    ref = x[i0, i1]
    got = _meta_index(NBXTensor.from_numpy(x), [NBXTensor.from_numpy(i0), NBXTensor.from_numpy(i1)]).numpy()
    assert np.array_equal(got, ref)
