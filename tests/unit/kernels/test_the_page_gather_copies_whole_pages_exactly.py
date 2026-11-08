"""The whole-page KV gather (`paged_kv_gather`, ops/paged_decode_attn.py `paged_kv_gather_kernel`): a prefill chunk over
a session's cache reads every cached key through it (docs/design/prism_kv.md §5bis). A copy is judged by equality, not
a tolerance: the gathered bytes are the pool's bytes, page by page, in the table's order.

What each test would do if the code were wrong:
  1. a SHUFFLED table: a kernel reading pages in pool order, or mis-addressing a position inside a page, copies other
     positions — the bytes differ, red; pages the table does not name are NaN, so a read through them shows;
  2. heads re-expanded to the query heads (GQA groups 4): a kernel taking head h instead of h // groups reads another
     KV head or past the pool — red;
  3. a widening to fp32 is exact (every fp16 value is an fp32 value): the gathered fp32 equals the host's widening;
  4. positions that are not whole pages, more pages than the table names, or heads that are not a multiple of the KV
     heads are refused by name before any launch.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.ops import attention_wrappers as AW

_NP = {NBXDtype.float16: (np.float16, 2), NBXDtype.float32: (np.float32, 4)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.sync_device()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).reshape(tuple(t.shape)).copy()


def _case(seed=0, batch=2, h_kv=4, pt=64, d=64, pages=3, spare=4):
    rng = np.random.default_rng(seed)
    num_pages = batch * pages + spare
    pool = np.full((num_pages, h_kv, pt, d), np.nan, dtype=np.float16)
    order = rng.permutation(num_pages)
    table = np.full((batch, pages + 2), -1, dtype=np.int32)          # wider than the gather, -1 past it
    for b in range(batch):
        for p in range(pages):
            pid = int(order[b * pages + p])
            table[b, p] = pid
            pool[pid] = rng.standard_normal((h_kv, pt, d)).astype(np.float16)
    return pool, table


def _expected(pool, table, positions, groups):
    pt = pool.shape[2]
    out = []
    for b in range(table.shape[0]):
        seq = np.concatenate([pool[table[b, p]] for p in range(positions // pt)], axis=1)   # [H_kv, T, D]
        out.append(np.repeat(seq, groups, axis=0))
    return np.stack(out)


@pytest.mark.parametrize("groups", [1, 4])
def test_the_gather_is_the_pools_pages_in_the_tables_order(groups):
    pool, table = _case()
    positions = 3 * 64
    got = _d2h(AW.paged_kv_gather(NBXTensor.from_numpy(pool), NBXTensor.from_numpy(table), positions,
                                  pool.shape[1] * groups, NBXDtype.float16))
    want = _expected(pool, table, positions, groups)
    assert got.shape == want.shape
    assert not np.isnan(got).any(), "a page the table does not name was read"
    assert got.tobytes() == want.tobytes()


def test_a_widening_to_fp32_is_exact():
    pool, table = _case(seed=1)
    got = _d2h(AW.paged_kv_gather(NBXTensor.from_numpy(pool), NBXTensor.from_numpy(table), 128, 4,
                                  NBXDtype.float32))
    assert got.tobytes() == _expected(pool, table, 128, 1).astype(np.float32).tobytes()


def test_what_the_gather_cannot_copy_is_refused_by_name():
    pool, table = _case(seed=2)
    p, t = NBXTensor.from_numpy(pool), NBXTensor.from_numpy(table)
    with pytest.raises(ValueError, match="whole number"):
        AW.paged_kv_gather(p, t, 100, 4, NBXDtype.float16)
    with pytest.raises(ValueError, match="does not name"):
        AW.paged_kv_gather(p, t, 64 * 6, 4, NBXDtype.float16)
    with pytest.raises(ValueError, match="multiple"):
        AW.paged_kv_gather(p, t, 64, 6, NBXDtype.float16)
