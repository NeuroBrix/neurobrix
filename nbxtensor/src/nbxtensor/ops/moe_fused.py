"""The fused mixture-of-experts layer of one call, read through the experts' pointer tables.

  1. the routing, on the activations' device: the router's top-k (renormalised when the layer says so), or the indices
     and weights a multi-gate layer already blended in its graph, consumed as they are;
  2. the experts on the device: host experts promoted for this call (zero3), an expert store's slots acquired for the
     routed experts (`memory.expert_slots.ExpertSlotStore`), or the store's own residency on the card for one token;
  3. their pointer tables (`memory.pointer_tables.tables_for_call`: cached for resident weights, built per call
     otherwise);
  4. one path: the one-token SIMT band (int4-g128 or half-precision experts on one device), the grouped GEMMs on one
     device (the routing aligned by expert, gate and up, SwiGLU, down weighted by the scores, the sum over top_k), the
     same in runs of experts when a store holds fewer slots than the call routes to, or the layer split over several
     devices (`ops.moe_multi_device`).

Every path reaches its experts through an [E] int64 table of device addresses; a backend that cannot read through a
pointer loaded from memory is refused before any of them. The caller decides two switches (whether the int4 band is
taken and whether its down projection is split over the grid) and what a diagnostic prints; an expert store hears of
the call's routing through its own methods (`on_routed`, `on_enqueued`, `decodes_on_card`). The current device is the
activations' again on return. Moved from the NeuroBrix engine's triton/moe.py (2026-11-12).
"""
from __future__ import annotations

import logging

import numpy as np

from nbxtensor.backend.capabilities import backend_loads_pointers_from_memory
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory import pointer_tables as _pt
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.device_transfer import transfer_tensor
from nbxtensor.ops import elementwise_wrappers as _elementwise
from nbxtensor.ops import moe_wrappers as _moe
from nbxtensor.ops import reduction_wrappers as _reduction
from nbxtensor.ops.moe_multi_device import fused_moe_across_devices
from nbxtensor.tensor import NBXTensor

_log = logging.getLogger(__name__)

_HALF = (NBXDtype.float16, NBXDtype.bfloat16)


class StackedSlabPromotion:
    """A `fetch` for a stacked-slab expert reader that brings a host-resident stacked expert slab to the activation's
    device ONCE and WHOLE — one dense H2D per slab — before the per-expert views are cut from it. Promoting the views
    instead (`fused_moe`'s per-weight promotion) would copy each expert's non-dense half on its own: 2E host-side
    strided copies and 3E transfers per call.

    `per_call` turns True when a slab was promoted: the views then point into memory that lives for THIS call only,
    and the caller hands that to `fused_moe(weights_per_call=...)` so its zero3 rules stay in charge (no pointer-table
    cache entry for per-call addresses, the promoted lists released in the call's frame) — a reused address must never
    meet a stale table."""

    def __init__(self, fetch, device_idx: int):
        self._fetch = fetch
        self._device_idx = device_idx
        self.per_call = False

    def __call__(self, tid):
        t = self._fetch(tid)
        if t is not None and getattr(t, "_device", "cuda") == "cpu":
            self.per_call = True
            return t.to_cuda(self._device_idx)
        return t


class _Call:
    """One call's routing, experts and placement, filled in by `fused_moe`'s steps."""

    def __init__(self, top_k: int, num_experts: int, store, layer, dump, label: str):
        self.top_k, self.num_experts, self.store, self.layer = top_k, num_experts, store, layer
        self.dump, self.label, self.memory_log = dump, label, False
        self.act_dev = self.orig_shape = self.experts = self.host = self.tables = None
        self.hidden = self.flat_indices = self.flat_scores = self.M = self.K = self.n_gate = None
        self.per_call = False


def fused_moe(gate_scores, hidden_states, gate_weights, up_weights, down_weights, top_k: int, num_experts: int, *,
              norm_topk_prob: bool = True, topk_indices=None, topk_weights=None, weights_per_call: bool = False,
              store=None, layer=None, int4_band: bool = True, split_down: bool = True, dump=None, label: str = "",
              log_memory: bool = False):
    """The routed SwiGLU sum of one mixture-of-experts layer: [M, K] (or the activations' [B, S, K]) on the
    activations' device.

    gate_scores: the router's logits [tokens, E], None when the routing is supplied (`topk_indices` and
    `topk_weights` together, [tokens, top_k]: a multi-gate layer's blend, consumed verbatim — `norm_topk_prob` is then
    not applied). hidden_states: [tokens, K] or [B, S, K]. gate/up/down_weights: one weight per expert ([N, K] /
    [K, N] linear layout, or int4-g128 `QuantizedTensor`s), on devices or on the host (promoted for this call).
    weights_per_call: the caller promoted them for this call (`StackedSlabPromotion`): no cached table, released
    before return. store, layer: an `ExpertSlotStore` serving the layer named `layer` (its experts stay on the host).
    int4_band: one int4-g128 token takes the SIMT band (else the grouped GEMMs); split_down: that band's down
    projection puts the experts on the grid (else its fused combine). dump(label, tensor): called with the call's
    intermediate values when given (a diagnostic; `label` names the call in the placement and stride lines it also
    logs then). log_memory: log the device's live bytes around a host promotion."""
    _refuse_unless_routable(hidden_states, gate_scores, topk_indices, topk_weights)
    c = _Call(top_k, num_experts, store, layer, dump, label)
    c.act_dev = hidden_states.device_idx
    _use(c.act_dev)
    blended = topk_indices is not None or topk_weights is not None
    if blended:
        topk_indices, topk_weights = _xfer(topk_indices, c.act_dev), _xfer(topk_weights, c.act_dev)
    else:
        gate_scores = _xfer(gate_scores, c.act_dev)
    c.experts = (gate_weights, up_weights, down_weights)
    any_host = store is None and any(getattr(w, '_device', 'cuda') == 'cpu' for lst in c.experts for w in lst)
    c.per_call, c.memory_log = weights_per_call or any_host or store is not None, log_memory and any_host
    if any_host:
        _promote(c)
    c.orig_shape = hidden_states.shape
    _shape_and_route(c, hidden_states, gate_scores, topk_indices, topk_weights, blended, norm_topk_prob)
    return _dispatch(c, int4_band, split_down)


def _refuse_unless_routable(hidden_states, gate_scores, topk_indices, topk_weights) -> None:
    if hidden_states is None:
        raise RuntimeError("MoE fused: hidden_states is None")
    # EVERY path reaches an expert's weights through the [E] int64 pointer table (`memory.pointer_tables`): the SIMT
    # decode bands and the grouped GEMM load an address and bitcast it to a pointer. A backend that cannot read through
    # such a pointer returns ZEROS with no exception and no warning — measured on triton-ext, where the band's gateup
    # stage wrote nothing and the two stages after it faithfully combined nothing. The gate sits above the path choice
    # because every path shares the defect.
    if not backend_loads_pointers_from_memory():
        raise RuntimeError(
            "NeuroBrix MoE: this backend cannot read through a pointer loaded "
            "from a tensor, and every MoE path addresses its experts that way "
            "(an [E] int64 table of device addresses). Refusing rather than "
            "returning the zeros it would otherwise produce in silence. "
            "Measured with a one-variable control: an address passed AS a "
            "pointer reads correctly, while the same address loaded from a "
            "table reads all zeros in both Triton spellings and raises "
            "nothing.")
    if (topk_indices is None) != (topk_weights is None):
        raise RuntimeError(
            "MoE fused (multi-gate): topk_indices and topk_weights must be "
            "supplied together — got "
            f"indices={'None' if topk_indices is None else 'OK'}, "
            f"weights={'None' if topk_weights is None else 'OK'}")
    if topk_indices is None and gate_scores is None:
        raise RuntimeError("MoE fused: gate_scores is None and no blended routing")


def _use(dev: int) -> None:
    DeviceAllocator.set_device(dev)
    DeviceAllocator.ensure_triton_device(dev)


def _xfer(tensor, target_dev: int):
    """`tensor` on `target_dev`: itself when it is there, else through the cross-device transfer, which waits for the
    source device's stream before the peer copy and materialises a view that is not one dense span (a copy enqueued on
    the target's stream alone could read routing or activations still being written: illegal addresses at the next
    fused launch on that device)."""
    if tensor.device_idx == target_dev:
        return tensor
    return transfer_tensor(tensor, target_dev)


def _onto(c: _Call, dev: int, *tensors):
    """`tensors` on `dev`, made the current device, when it is not the activations' own."""
    if dev == c.act_dev:
        return tensors
    _use(dev)
    return tuple(_xfer(t, dev) for t in tensors)


def _dump(c: _Call, label: str, tensor) -> None:
    if c.dump is not None:
        c.dump(label, tensor)


def _log_memory(c: _Call, stage: str) -> None:
    _log.debug(f"[Z3_MOE] {c.label} {stage} cuda_live={DeviceAllocator.memory_allocated(c.act_dev)/1e6:.1f}MB")


def _promote(c: _Call) -> None:
    """Host experts brought to the activations' device for THIS call (zero3): they must be there before a table bakes
    their addresses, and their table is built per call — the promoted copies are freed at the call's end and a later
    allocation may reuse their addresses, which a cached table would then serve stale."""
    if c.memory_log:
        _log_memory(c, "pre-promote")
    c.experts = tuple([w.to_cuda(c.act_dev) if getattr(w, '_device', 'cuda') == 'cpu' else w for w in lst]
                      for lst in c.experts)
    if c.memory_log:
        _log_memory(c, "post-promote")


def _shape_and_route(c: _Call, hidden, gate_scores, topk_indices, topk_weights, blended: bool, norm: bool) -> None:
    """The activations as [tokens, K] in the experts' dtype, and the flat routing (`flat_indices` int64 and
    `flat_scores` in the experts' dtype, [tokens * top_k])."""
    if hidden.ndim == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    if blended:
        topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1]) if topk_indices.ndim > 2 else topk_indices
        topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1]) if topk_weights.ndim > 2 else topk_weights
    elif gate_scores.ndim == 3:
        gate_scores = gate_scores.reshape(-1, gate_scores.shape[-1])
    w_dtype = c.experts[0][0].nbx_dtype
    if hidden.nbx_dtype != w_dtype:
        hidden = hidden.to(w_dtype)
    c.hidden, (c.M, c.K), c.n_gate = hidden, hidden.shape, c.experts[0][0].shape[0]
    if not blended:
        _dump(c, "gate_scores_in", gate_scores)
    _dump(c, "hidden_states_in", hidden)
    scores, indices = (_blended_routing(c, topk_indices, topk_weights) if blended
                       else _topk_routing(c, gate_scores, norm))
    c.flat_indices = indices.reshape(-1)
    c.flat_scores = scores.reshape(-1).to(w_dtype)
    _dump(c, "flat_scores", c.flat_scores)


def _blended_routing(c: _Call, topk_indices, topk_weights):
    """A multi-gate layer's routing as the graph blended it (each gate's softmax, top-k and normalisation already run,
    blended by per-token modality masks): taken verbatim — a top-k here would discard the modality selection, a
    normalisation would normalise twice. The indices as int64, which the alignment and the split op read."""
    indices = topk_indices
    if indices.nbx_dtype != NBXDtype.int64:
        indices = indices.to(NBXDtype.int64)
    scores = topk_weights.to(NBXDtype.float32)
    _dump(c, "blend_indices", indices)
    _dump(c, "blend_scores", scores)
    return scores, indices


def _topk_routing(c: _Call, gate_scores, norm: bool):
    gate_fp32 = gate_scores.to(NBXDtype.float32)
    scores, indices = _elementwise.topk_wrapper(gate_fp32, c.top_k, dim=-1)
    _dump(c, "topk_scores", scores)
    _dump(c, "topk_indices", indices)
    if norm:
        denom = _reduction.sum_wrapper(scores, dim=-1, keepdim=True)
        scores = _elementwise.div(scores, denom)
        _dump(c, "scores_norm", scores)
    return scores, indices


def _dispatch(c: _Call, int4_band: bool, split_down: bool):
    """The call's path, from its shape and its experts' placement."""
    if c.store is not None and c.M == 1 and c.store.decodes_on_card(c.layer):
        return _decode_on_card(c)
    routed = None
    if c.store is not None:
        routed = _routed_experts(c)
        if len(routed) > c.store.residency.n_slots:
            return _store_grouped(c)
        c.experts = c.store.acquire(c.layer, routed, c.num_experts, c.host)
    c.tables = _pt.tables_for_call(*c.experts, c.num_experts, c.per_call, c.act_dev)
    band = _one_token_band(c, int4_band)
    if band is not None:
        return _band(c, band, split_down)
    return _finish(c, _grouped(c), routed)


def _shaped_back(c: _Call, output):
    return output.reshape(c.orig_shape) if len(c.orig_shape) == 3 else output


def _pins_of(store):
    return None if store is None else store.table_pins


def _host_reader(experts):
    def host(e):
        return (experts[0][e], experts[1][e], experts[2][e])
    return host


def _decode_on_card(c: _Call):
    """One token through a store-served layer: the residency runs on the card (the store's `decode_on_card`) — no
    routing read back, no host decision, the same launches at every token (the step replays)."""
    sdev = c.store.device_idx
    hidden, scores, indices = _onto(c, sdev, c.hidden, c.flat_scores, c.flat_indices)
    output = c.store.decode_on_card(c.layer, indices, scores, hidden, _host_reader(c.experts), c.top_k, c.K,
                                    c.n_gate)
    if sdev != c.act_dev:
        output = _xfer(output, c.act_dev)
    output = _shaped_back(c, output)
    _use(c.act_dev)
    return output


def _routed_experts(c: _Call) -> list:
    """The distinct experts this call routes to, read once on the host; the store hears of them with the host reader
    its copies take an expert from."""
    c.host = _host_reader(c.experts)
    routed = sorted({int(e) for e in c.flat_indices.numpy().ravel().tolist()})
    c.store.on_routed(c.layer, routed, c.host)
    return routed


def _group_size(weight, width: int) -> int:
    scales = getattr(weight, "scales", None)
    return width // scales.shape[0] if scales is not None and scales.shape[0] > 0 else 0


def _one_token_band(c: _Call, int4_band: bool):
    """Which one-token SIMT band serves the call: "int4" (int4 experts in groups of `MOE_INT4_GROUP`, verified from
    the weights), "half" (fp16 / bf16 experts), or None (the grouped GEMMs). One token on one device only: at one token
    the grouped GEMM's every 16-row tile carries ONE real row, and the band replaces alignment, three grouped launches,
    the SwiGLU pass and the combine by a fixed-order in-kernel loop (ops/moe_decode_vec.py)."""
    one = c.M == 1 and len(c.tables.device_experts) == 1
    if (one and c.tables.quantized and int4_band and _group_size(c.experts[0][0], c.K) == _moe.MOE_INT4_GROUP
            and _group_size(c.experts[2][0], c.n_gate) == _moe.MOE_INT4_GROUP):
        return "int4"
    if one and not c.tables.quantized and c.experts[0][0].nbx_dtype in _HALF:
        return "half"
    return None


def _band(c: _Call, band: str, split_down: bool):
    dev = next(iter(c.tables.device_experts))
    hidden, scores, indices = _onto(c, dev, c.hidden, c.flat_scores, c.flat_indices)
    if band == "int4":
        output = _moe.moe_decode_vec_int4_pass(hidden, c.tables, dev, indices, scores, c.top_k, c.K, c.n_gate,
                                               split_down)
    else:
        output = _moe.moe_decode_vec_half_pass(hidden, c.tables, dev, indices, scores, c.top_k, c.K, c.n_gate,
                                               c.experts[0][0].nbx_dtype == NBXDtype.bfloat16)
    if dev != c.act_dev:
        output = _xfer(output, c.act_dev)
        if band == "int4":          # as the int4 band always did; the half band's device comes back below
            _use(c.act_dev)
    output = _shaped_back(c, output)
    if c.per_call:
        _pt.end_call_pins(c.tables, _pins_of(c.store), sync=True)
    _use(c.act_dev)
    return output


def _grouped(c: _Call):
    """The grouped GEMMs: the routing aligned by expert on the activations' device, then the three passes where the
    experts are (one device), or the layer handed to the split op (several)."""
    aligned = _moe.moe_align_block_size(c.flat_indices, _moe.MOE_BLOCK_M, c.num_experts, c.act_dev)
    if len(c.tables.device_experts) != 1:
        # a layer whose experts sit on several devices: one share per device, summed (the placement is the caller's,
        # read from the tables)
        return fused_moe_across_devices(c.hidden, c.flat_indices, c.flat_scores, c.tables, c.top_k, c.num_experts,
                                        c.n_gate, _moe.MOE_BLOCK_M, c.act_dev)
    dev = next(iter(c.tables.device_experts))
    hidden, scores, *aligned = _onto(c, dev, c.hidden, c.flat_scores, *aligned)
    if c.dump is not None:
        _log_placement(c, dev, hidden, scores, aligned)
    indices = c.flat_indices if dev == c.act_dev else _xfer(c.flat_indices, dev)
    result = _three_passes(c, dev, hidden, (scores, tuple(aligned), indices, {}))
    if dev != c.act_dev:
        result = _xfer(result, c.act_dev)
        _use(c.act_dev)
    return result


def _finish(c: _Call, output, routed):
    """The grouped path's end: per-call tables' pins ended and the per-call weights and tables released IN THIS FRAME
    (their int64 addresses are stale once the weights are freed; kept one tick longer, a 16 GB card held ~800 MB per
    MoE op and ran out after 7-8 blocks), the activations' device current, and the store told the launches are
    enqueued."""
    output = _shaped_back(c, output)
    if c.per_call:
        # NBXTensor frees its device buffer at refcount zero: no gc.collect() (once per layer per token, it cost a
        # deepseek-moe request its 900 s budget). A per-call table's pins (Metal) end with the call, or with its store.
        _pt.end_call_pins(c.tables, _pins_of(c.store))
        c.experts = c.tables = None
    _use(c.act_dev)
    if c.memory_log:
        _log_memory(c, "post-return")
    if c.store is not None:
        c.store.on_enqueued(c.layer, routed)
    return output


def _log_placement(c: _Call, dev: int, hidden, scores, aligned) -> None:
    s, e, n = aligned
    g = c.tables.gate_ptrs[dev]
    _log.debug(f"[MOE_DIAG] {c.label} dev={dev} act_dev={c.act_dev} "
               f"h=(d{hidden.device_idx},{hidden.shape},{hidden.data_ptr():#x}) "
               f"scores=(d{scores.device_idx},{scores.shape}) "
               f"sorted=(d{s.device_idx},{s.shape}) "
               f"expids=(d{e.device_idx},{e.shape}) "
               f"npad=(d{n.device_idx}) "
               f"tbl_gate=(d{g.device_idx},{g.shape},{g.data_ptr():#x}) "
               f"M={c.M} topk={c.top_k}")


def _log_strides(c: _Call, dev: int, aligned) -> None:
    t = c.tables
    _log.debug(f"[MOE_DIAG] strides gate: bk={t.gate_stride_bk[dev]} bn={t.gate_stride_bn[dev]}")
    _log.debug(f"[MOE_DIAG] strides up:   bk={t.up_stride_bk[dev]} bn={t.up_stride_bn[dev]}")
    _log.debug(f"[MOE_DIAG] strides down: bk={t.down_stride_bk[dev]} bn={t.down_stride_bn[dev]}")
    _log.debug(f"[MOE_DIAG] sorted_token_ids.numel={aligned[0].numel()} expert_ids.numel={aligned[1].numel()} "
               f"M={c.M} top_k={c.top_k} N_gate={c.n_gate} K={c.K}")


def _three_passes(c: _Call, dev: int, hidden, route):
    """Gate and up, SwiGLU, down weighted by the scores, the sum over top_k: [M, K] on `dev`. `route`: (scores, the
    alignment (sorted ids, block experts, padded count), the flat indices on `dev`, the alignments made at a unit
    tile's rows so far)."""
    dt = hidden.nbx_dtype
    padded = route[1][0].shape[0]
    if c.dump is not None:
        _log_strides(c, dev, route[1])
    gate_out = NBXTensor.zeros((padded, c.n_gate), dtype=dt, device=f"cuda:{dev}")
    _projection(c, dev, "gate", hidden, gate_out, (c.n_gate, c.K, False, True), route)
    _dump(c, "gate_out", gate_out)
    up_out = NBXTensor.zeros((padded, c.n_gate), dtype=dt, device=f"cuda:{dev}")
    _projection(c, dev, "up", hidden, up_out, (c.n_gate, c.K, False, True), route)
    _dump(c, "up_out", up_out)
    gate_silu = _elementwise.silu(gate_out)
    activated = _elementwise.mul(gate_silu, up_out)
    _dump(c, "activated", activated)
    down_out = NBXTensor.zeros((padded, c.K), dtype=dt, device=f"cuda:{dev}")
    _projection(c, dev, "down", activated, down_out, (c.K, c.n_gate, True, False), route)
    _dump(c, "down_out", down_out)
    result = down_out.narrow(0, 0, c.M * c.top_k).reshape(c.M, c.top_k, c.K)
    result = _reduction.sum_wrapper(result, dim=1)
    _dump(c, "result_final", result)
    return result


def _projection(c: _Call, dev: int, name: str, x, out, dims, route) -> None:
    """One grouped-GEMM pass: W4 when the tables are quantized; else on the matrix unit where the projection's
    certificate chose it (`moe_wrappers.fused_moe_unit_tile`, the routing aligned again at the unit tile's rows, once
    per block size), on the FMA kernel otherwise. `dims`: (N, K, the routing weight multiplied, divided by top_k)."""
    n, k, mul_routed, topk_div = dims
    scores, aligned, indices, unit_aligned = route
    t = c.tables
    ptrs, sbk, sbn = (getattr(t, f"{name}_ptrs")[dev], getattr(t, f"{name}_stride_bk")[dev],
                      getattr(t, f"{name}_stride_bn")[dev])
    if t.quantized:
        sc_t, mn_t, sg, sn = t.q_tables[name][dev]
        _moe.invoke_fused_moe_wna16(x, ptrs, sc_t, mn_t, out, scores, *aligned, n, k, sbk, sbn, sg, sn, c.top_k,
                                    mul_routed_weight=mul_routed, topk_divide=topk_div)
        return
    launch = (x, ptrs, out, scores, *aligned, n, k, sbk, sbn, c.top_k)
    tile = _moe.fused_moe_unit_tile(*launch, mul_routed_weight=mul_routed, topk_divide=topk_div)
    if tile is None:
        _moe.invoke_fused_moe(*launch, mul_routed_weight=mul_routed, topk_divide=topk_div)
        return
    bm = int(tile["block_m"])
    if bm not in unit_aligned:
        unit_aligned[bm] = _moe.moe_align_block_size(indices, bm, c.num_experts, dev)
    sids, eids, npad = unit_aligned[bm]
    _moe.invoke_fused_moe_m8n8k4(x, ptrs, out, scores, sids, eids, npad, n, k, sbk, sbn, c.top_k, tile,
                                 mul_routed_weight=mul_routed, topk_divide=topk_div)


def _runs_of_experts(block_experts, n_slots: int, n_blocks: int) -> list:
    """The aligned blocks (each block's expert, sorted by expert) cut into runs of consecutive experts of at most
    `n_slots` distinct ones: [(first block, end block, [experts])]."""
    groups = []
    lo, experts = 0, []
    for b, e in enumerate(block_experts):
        if experts and e == experts[-1]:
            continue
        if len(experts) == n_slots:
            groups.append((lo, b, experts))
            lo, experts = b, []
        experts.append(e)
    if experts:
        groups.append((lo, n_blocks, experts))
    return groups


def _store_grouped(c: _Call):
    """One call whose routed experts outnumber the store's slots (a prefill on a cache below one layer's expert set):
    the aligned blocks, sorted by expert, are cut into runs of consecutive experts of at most `n_slots` distinct ones;
    each run acquires its experts and launches the three grouped GEMMs on its own range of blocks (a view of the one
    alignment).

    Byte for byte the ungrouped pass: every block computes the same rows against the same expert with the same launch
    constants, the kernels write rows keyed by token, never by block, the SwiGLU is elementwise, and the top_k reduce
    runs once over the whole result."""
    bm, dev = _moe.MOE_BLOCK_M, c.act_dev
    sorted_ids, expert_ids, ntpp = _moe.moe_align_block_size(c.flat_indices, bm, c.num_experts, dev)
    n_blocks = int(ntpp.numpy().ravel()[0]) // bm
    blk = [int(e) for e in expert_ids.narrow(0, 0, n_blocks).numpy().ravel().tolist()]
    groups = _runs_of_experts(blk, c.store.residency.n_slots, n_blocks)
    dt = c.hidden.nbx_dtype
    padded = sorted_ids.shape[0]
    DeviceAllocator.set_device(dev)
    outs = (NBXTensor.zeros((padded, c.n_gate), dtype=dt, device=f"cuda:{dev}"),
            NBXTensor.zeros((padded, c.n_gate), dtype=dt, device=f"cuda:{dev}"),
            NBXTensor.zeros((padded, c.K), dtype=dt, device=f"cuda:{dev}"))
    for lo, hi, experts in groups:
        _store_run(c, (sorted_ids, expert_ids), (lo, hi), experts, outs)
    result = outs[2].narrow(0, 0, c.M * c.top_k).reshape(c.M, c.top_k, c.K)
    output = _shaped_back(c, _reduction.sum_wrapper(result, dim=1))
    _use(c.act_dev)
    return output


def _store_run(c: _Call, aligned, blocks, experts, outs) -> None:
    """One run of `_store_grouped`: its experts acquired into the slots, the three grouped GEMMs over blocks [lo, hi)."""
    (sorted_ids, expert_ids), (lo, hi), bm, dev = aligned, blocks, _moe.MOE_BLOCK_M, c.act_dev
    gate_out, up_out, down_out = outs
    g, u, d = c.store.acquire(c.layer, experts, c.num_experts, c.host)
    tables = _pt.build_ptr_tables(g, u, d)
    _use(dev)
    s_g = sorted_ids.narrow(0, lo * bm, (hi - lo) * bm)
    e_g = expert_ids.narrow(0, lo, hi - lo)
    n_g = NBXTensor.from_numpy(np.array([(hi - lo) * bm], dtype=np.int64))
    for name, out in (("gate", gate_out), ("up", up_out)):
        _moe.invoke_fused_moe(
            c.hidden, getattr(tables, f"{name}_ptrs")[dev], out, c.flat_scores, s_g, e_g, n_g,
            c.n_gate, c.K, getattr(tables, f"{name}_stride_bk")[dev],
            getattr(tables, f"{name}_stride_bn")[dev], c.top_k, mul_routed_weight=False)
    activated = _elementwise.mul(_elementwise.silu(gate_out), up_out)
    _moe.invoke_fused_moe(
        activated, tables.down_ptrs[dev], down_out, c.flat_scores, s_g, e_g, n_g,
        c.K, c.n_gate, tables.down_stride_bk[dev], tables.down_stride_bn[dev], c.top_k,
        mul_routed_weight=True, topk_divide=False)
    _pt.end_call_pins(tables, c.store.table_pins)
