"""Device-address tables of a list of tensors: the fused MoE GEMMs read their experts through them (one [E] int64
table of absolute addresses per projection and device, no offset math), their fingerprinted LRU cache, and on Metal the
pinned scopes that keep those addresses valid for the table's whole cache life (moved from the engine's triton/moe.py,
M2). Torch-free.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict

import numpy as np

import triton
import triton.language as tl

from nbxtensor.tensor import NBXTensor
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.backend.runtime import gpu_backend


# ============================================================================
# POINTER TABLE BUILDER — zero-copy, cached
# ============================================================================

# LRU cache of pointer tables keyed by a data_ptr fingerprint of every
# expert weight. Under zero3 pipelining weights may be freed and
# reallocated at the same virtual address, so a static cache_key
# (e.g. first expert's data_ptr) can resolve to a stale PtrTables and
# cause silent garbage or illegal memory accesses. Hashing ALL
# expert data_ptrs detects any change to the arena layout.
#
# Bounded by _PTR_CACHE_MAXSIZE entries — eviction is LRU (OrderedDict
# move_to_end on hit, popitem(last=False) on overflow). The cap keeps
# memory flat when running many blocks across devices during
# pipelining; 256 entries × 3 int64 tables × few hundred experts is
# still sub-megabyte.
_PTR_CACHE_MAXSIZE = 256
_ptr_cache: "OrderedDict[int, PtrTables]" = OrderedDict()


def _ptr_cache_fingerprint(
    gate_weights, up_weights, down_weights, num_experts: int
) -> int:
    """Stable hash over every expert's data_ptr for all three projections.

    Any weight swap (H2D, D2D, free-and-remalloc at the same address)
    changes at least one data_ptr — and in the rare adversarial case
    where the driver hands out identical addresses for a pair of
    identically-sized buffers, we still hash all 3 × num_experts ptrs
    so a single match isn't enough for a false cache hit.
    """
    vals = []
    vals.append(num_experts)
    for i in range(num_experts):
        vals.append(gate_weights[i].data_ptr())
        vals.append(up_weights[i].data_ptr())
        vals.append(down_weights[i].data_ptr())
    return hash(tuple(vals))


def _ptr_cache_get(fp: int):
    """LRU lookup. Promotes the hit to the MRU end."""
    tables = _ptr_cache.get(fp)
    if tables is not None:
        _ptr_cache.move_to_end(fp)
    return tables


def _ptr_cache_put(fp: int, tables):
    """Insert + evict the oldest entry if we're over the cap."""
    _ptr_cache[fp] = tables
    _ptr_cache.move_to_end(fp)
    while len(_ptr_cache) > _PTR_CACHE_MAXSIZE:
        _fp, evicted = _ptr_cache.popitem(last=False)
        evicted.release_pins()


def release_pinned_tables() -> int:
    """Drop every cached table that PINS (Metal), releasing its pins; returns how many. Called at the unload
    boundary (`MemoryManager.unload_weights`), after the devices were synchronised.

    A pinning table holds its expert tensors and one whole-allocation wrap per arena for as long as it lives,
    and it lived until LRU eviction, 256 tables later: an unloaded component stayed pinned, its tensors not
    freed and its arena still mapped. Under `layer_streaming` each unloaded segment stayed beside the next
    (deepseek-moe-16b-chat on the Mac, 2026-10-04: swap 3.4 -> 10.2 GB in 30 s while the second of four
    segments loaded, the guard killed the run). All of them go, not only the unloading component's: a table is
    three small host arrays, rebuilt at the next call that misses, and a table that outlives ANY weight it
    pins is the defect. A table that pins nothing (CUDA: integers only) is left where it is."""
    pinned = [fp for fp, tables in _ptr_cache.items() if tables.pins]
    for fp in pinned:
        _ptr_cache.pop(fp).release_pins()
    return len(pinned)


class PtrTables:
    """Per-projection absolute pointer tables for zero-copy expert access.

    Each table is [E] int64 on GPU: absolute data_ptr() of each expert.
    The fused kernel loads ptrs[expert_id] and uses it directly as a pointer.
    """
    __slots__ = ('gate_ptrs', 'gate_stride_bk', 'gate_stride_bn',
                 'up_ptrs', 'up_stride_bk', 'up_stride_bn',
                 'down_ptrs', 'down_stride_bk', 'down_stride_bn',
                 'device_experts', 'quantized', 'q_tables', 'pins')

    def __init__(self):
        self.gate_ptrs = {}       # {device → NBXTensor[E] int64}
        self.gate_stride_bk = {}
        self.gate_stride_bn = {}
        self.up_ptrs = {}
        self.up_stride_bk = {}
        self.up_stride_bn = {}
        self.down_ptrs = {}
        self.down_stride_bk = {}
        self.down_stride_bn = {}
        self.device_experts = {}
        # int4-g128-asym experts: *_ptrs holds the qweight table and
        # q_tables[proj][dev] = (sc_ptrs, mn_ptrs, stride_sg, stride_sn);
        # the *_stride_bk/bn slots hold the PACKED int32 strides.
        self.quantized = False
        self.q_tables = {}
        # Metal only: entered pinned_addresses scopes, one per device — the
        # addresses in these tables are valid EXACTLY while these are held,
        # so their lifetime is the table's cache lifetime (released on LRU
        # eviction below), never a call's.
        self.pins = []

    def release_pins(self):
        for pin in self.pins:
            try:
                pin.__exit__(None, None, None)
            except Exception:                          # noqa: BLE001
                pass
        self.pins = []


@triton.jit
def _addr_capture_kernel(src_ptr, table_ptr, idx):
    """Store the DEVICE address of src_ptr into table[idx].

    The bitcast is the whole point: the launcher hands the kernel the address
    a shader dereferences, and capturing it ON DEVICE is the one spelling that
    is correct on every backend (verified bit-equal to the driver's own
    address on triton_ext, #130, 2026-09-19). A host-side data_ptr() is the
    CPU mapping on Metal, and a table of those reads as silent zeros.
    """
    tl.store(table_ptr + idx, tl.cast(src_ptr, tl.int64, bitcast=True))


def _build_ptr_tables(gate_weights, up_weights, down_weights):
    """Build per-projection absolute pointer tables.

    Stores each expert's data_ptr() as int64 in a GPU tensor. No offset math.
    Total GPU allocation: 3 × E × 8 bytes per device (~3KB for 128 experts).

    On Metal the tables are built by DEVICE CAPTURE under a pinned scope
    instead: the addresses a shader dereferences, kept alive for the table's
    whole cache lifetime (PtrTables.pins, released on LRU eviction).
    """
    from nbxtensor.quant.quantized_tensor import QuantizedTensor

    num_experts = len(gate_weights)
    tables = PtrTables()
    tables.quantized = isinstance(gate_weights[0], QuantizedTensor)
    if tables.quantized:
        # Uniformity contract: mixed dense/quantized expert lists are a
        # broken build — refuse loudly (ZERO FALLBACK).
        for proj_name, ws in (("gate", gate_weights), ("up", up_weights),
                              ("down", down_weights)):
            if not all(isinstance(x, QuantizedTensor) for x in ws):
                raise RuntimeError(
                    f"ZERO FALLBACK: mixed dense/quantized experts in "
                    f"'{proj_name}' projection — re-build the variant.")
    else:
        # The kernel takes ONE (stride_bk, stride_bn) pair per projection, read
        # from the first expert: every expert of a projection must share it. A
        # stacked slab's views do by construction (one slab, one geometry); a
        # list mixing layouts would be read through the wrong strides in
        # silence — refuse it by name.
        for proj_name, ws in (("gate", gate_weights), ("up", up_weights),
                              ("down", down_weights)):
            s0 = (tuple(ws[0].shape), ws[0].stride(0), ws[0].stride(1))
            for eid in range(1, num_experts):
                se = (tuple(ws[eid].shape), ws[eid].stride(0), ws[eid].stride(1))
                if se != s0:
                    raise RuntimeError(
                        f"ZERO FALLBACK: MoE '{proj_name}' expert {eid} has "
                        f"(shape, strides) {se}, expert 0 {s0} — the grouped "
                        "GEMM reads every expert through one stride pair.")

    by_device = defaultdict(list)
    for i in range(num_experts):
        by_device[gate_weights[i]._device_idx].append(i)
    tables.device_experts = dict(by_device)

    _metal = gpu_backend() == "metal"
    if _metal and tables.quantized:
        raise RuntimeError(
            "ZERO FALLBACK: quantized (int4) expert tables are not proven on "
            "Metal — the pinned-table contract was measured for the dense "
            "bf16 grouped GEMM only. Adding the quantized path is proving "
            "its triplet tables the same way, not assuming them.")

    for dev, expert_ids in by_device.items():
        DeviceAllocator.set_device(dev)

        if _metal:
            # Pin FIRST — the capture launches below store their wraps under
            # the pin, which is what keeps the captured addresses mapped.
            # Released when this table leaves the cache, never per call.
            from nbxtensor.backend.triton_ext_driver import pinned_addresses
            pin = pinned_addresses(*(
                ws[eid]
                for ws in (gate_weights, up_weights, down_weights)
                for eid in expert_ids))
            pin.__enter__()
            tables.pins.append(pin)

            # The pin's whole-allocation kept wrap is the ONE address
            # authority: every MTLBuffer has its own GPU VA, so an address
            # captured through a transient launch wrap dies with that wrap
            # (zeros, nothing raised — measured 2026-09-21), and per-expert
            # kept wraps blow Metal's alias budget (declined at ~480,
            # granite layer 15). Host-side arithmetic on the pinned wrap's
            # gpu_address needs no launch and one wrap per allocation.
            from nbxtensor.backend.triton_ext_driver import pinned_gpu_address

            def _capture_proj(weights):
                ptrs = np.array(
                    [pinned_gpu_address(weights[eid].data_ptr())
                     for eid in expert_ids],
                    dtype=np.int64)
                tab = NBXTensor.from_numpy(ptrs)
                base = weights[expert_ids[0]]
                return tab, base.stride(1), base.stride(0)

            gp, gbk, gbn = _capture_proj(gate_weights)
            tables.gate_ptrs[dev] = gp
            tables.gate_stride_bk[dev] = gbk
            tables.gate_stride_bn[dev] = gbn
            up, ubk, ubn = _capture_proj(up_weights)
            tables.up_ptrs[dev] = up
            tables.up_stride_bk[dev] = ubk
            tables.up_stride_bn[dev] = ubn
            dp, dbk, dbn = _capture_proj(down_weights)
            tables.down_ptrs[dev] = dp
            tables.down_stride_bk[dev] = dbk
            tables.down_stride_bn[dev] = dbn
            continue

        def _build_proj(weights):
            # Absolute pointers as int64
            if tables.quantized:
                qw = np.array(
                    [weights[eid].qweight.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                sc = np.array(
                    [weights[eid].scales.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                mn = np.array(
                    [weights[eid].qmins.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                base = weights[expert_ids[0]]
                return ((NBXTensor.from_numpy(qw),
                         NBXTensor.from_numpy(sc),
                         NBXTensor.from_numpy(mn)),
                        base.qweight.stride(0), base.qweight.stride(1),
                        base.scales.stride(0), base.scales.stride(1))
            ptrs = np.array(
                [weights[eid].data_ptr() for eid in expert_ids],
                dtype=np.int64)
            ptr_table = NBXTensor.from_numpy(ptrs)
            base = weights[expert_ids[0]]
            return ptr_table, base.stride(1), base.stride(0)

        if tables.quantized:
            for name, ws in (("gate", gate_weights), ("up", up_weights),
                             ("down", down_weights)):
                (qw_t, sc_t, mn_t), qk, qn, sg, sn = _build_proj(ws)
                getattr(tables, f"{name}_ptrs")[dev] = qw_t
                getattr(tables, f"{name}_stride_bk")[dev] = qk
                getattr(tables, f"{name}_stride_bn")[dev] = qn
                tables.q_tables.setdefault(name, {})[dev] = (
                    sc_t, mn_t, sg, sn)
            continue

        gp, gbk, gbn = _build_proj(gate_weights)
        tables.gate_ptrs[dev] = gp
        tables.gate_stride_bk[dev] = gbk
        tables.gate_stride_bn[dev] = gbn

        up, ubk, ubn = _build_proj(up_weights)
        tables.up_ptrs[dev] = up
        tables.up_stride_bk[dev] = ubk
        tables.up_stride_bn[dev] = ubn

        dp, dbk, dbn = _build_proj(down_weights)
        tables.down_ptrs[dev] = dp
        tables.down_stride_bk[dev] = dbk
        tables.down_stride_bn[dev] = dbn

    return tables


def _tables_for_call(gate_weights, up_weights, down_weights, num_experts: int,
                     per_call: bool, act_dev: int):
    """The pointer tables of one call. Weights that live for this call only
    (promoted from the host here or by the dispatcher) get a fresh table that
    never enters the cache — their addresses are freed when the call returns,
    and a later allocation reusing them would meet a stale entry. Resident
    weights go through the fingerprinted LRU cache."""
    if per_call:
        tables = _build_ptr_tables(gate_weights, up_weights, down_weights)
        DeviceAllocator.set_device(act_dev)
        DeviceAllocator.ensure_triton_device(act_dev)
        return tables
    fp = _ptr_cache_fingerprint(gate_weights, up_weights, down_weights, num_experts)
    tables = _ptr_cache_get(fp)
    if tables is None:
        tables = _build_ptr_tables(gate_weights, up_weights, down_weights)
        _ptr_cache_put(fp, tables)
        DeviceAllocator.set_device(act_dev)
        DeviceAllocator.ensure_triton_device(act_dev)
    return tables
