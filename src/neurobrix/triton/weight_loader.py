"""Triton Weight Loader — safetensors → NBXTensor via arena + cudaMemcpy.

Zero torch dependency. Uses ComponentArena for one-shot GPU allocation
per device (zero fragmentation). Loads weights via raw byte parsing.

Zero3 CPU offload support: when shard_map maps a weight key to "cpu"
(Prism zero3 strategy), the weight is loaded as a CPU-backed NBXTensor
(pinned host memory via cudaMallocHost) instead of going through the
GPU arena. Non-block weights (embeddings, final norm, lm_head) are
kept GPU-resident regardless of the shard_map value, because they are
called by flow handlers (GraphLMSession.prefill → w.embedding) that
bypass the execution sequence and expect GPU pointers. The block-vs-
non-block distinction is driven by the same _BLOCK_RE regex that
CompiledSequence.get_op_blocks uses, so the partitioning is consistent
across native and triton modes.
"""

import json
import math
import re
import struct
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from neurobrix.kernels.nbx_tensor import (
    NBXTensor, NBXDtype, DeviceAllocator, _contiguous_strides, dtype_size,
    _set_device, float32_to_bf16_bits,
)
from .memory_pool import ComponentArena


# Same pattern as core/runtime/graph/compiled_sequence.py _BLOCK_RE.
# Matches NeuroTax singular (block.N.) and vendor plurals (blocks.N.,
# layers.N., model.layers.N., encoder.layers.N., decoder.layers.N.).
# Weights that do NOT match are treated as non-block and stay GPU-
# resident under zero3.
_BLOCK_RE = re.compile(
    r'(?:blocks?|layers|model\.layers|encoder\.layers|decoder\.layers)\.(\d+)\.')


def is_block_key(key: str) -> bool:
    """Whether a weight belongs to a numbered block. Its complement — the token embedding, a
    head, the norms, the embedders — is what a flow may read by name outside the graph; the ONE
    definition the loader's filter, a streamed base's resident set and Prism's reserve for it
    all read (`GraphExecutor.consumed_in_loader_space`, `load_flow_read_weights`,
    `PrismSolver._try_layer_streaming`)."""
    return bool(_BLOCK_RE.search(key))


# safetensors dtype → (numpy dtype for reading, NBXDtype, bytes per element)
_SF_DTYPE_INFO = {
    "F16":  (np.float16,  NBXDtype.float16,  2),
    "BF16": (None,        NBXDtype.bfloat16, 2),
    "F32":  (np.float32,  NBXDtype.float32,  4),
    "F64":  (np.float64,  NBXDtype.float64,  8),
    "I8":   (np.int8,     NBXDtype.int8,     1),
    "I16":  (np.int16,    NBXDtype.int16,    2),
    "I32":  (np.int32,    NBXDtype.int32,    4),
    "I64":  (np.int64,    NBXDtype.int64,    8),
    "U8":   (np.uint8,    NBXDtype.uint8,    1),
    "BOOL": (np.bool_,    NBXDtype.bool_,    1),
}


class AbsentWeightError(RuntimeError):
    """A weight a load needs, or the graph binds, is absent from the container — refused by
    name (component, key, the shard the index expects it in) before any op runs.

    2026-10-04, Allegro on a staged container whose directory was replaced under the run: the
    loader globbed the shards it found, never compared them with the index, and handed back
    nothing for `post_quant_conv.weight`/`.bias`; the first reader was `aten.convolution::0`,
    which met None (`'NoneType' object has no attribute 'ndim'`). A weight is never absent
    silently: the index is the container's statement of what it holds, and a load that cannot
    deliver what the index says refuses here, in both engines (one function, R30)."""


def read_weights_index(component: str, comp_dir) -> Optional[dict]:
    """The component's index tensor table (`weights_index.json` → `tensors`): key → entry, each
    entry naming its `shard`. None for a component that stores no weights (no index AND no
    shard on disk). Shards without an index, or an index that cannot be read, are refused: no
    weight could then be verified present. Torch-free; both engines' loaders read it."""
    comp_dir = Path(comp_dir)
    index_path = comp_dir / "weights_index.json"
    if not index_path.exists():
        weights_dir = comp_dir / "weights"
        shards = sorted(p.name for p in weights_dir.glob("*.safetensors")) \
            if weights_dir.is_dir() else []
        if shards:
            raise AbsentWeightError(
                f"component '{component}': {len(shards)} weight shard(s) in {weights_dir} "
                f"(e.g. {shards[0]}) but no index at {index_path} — refused: without the index "
                f"no weight the graph binds can be verified present.")
        return None
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        raise AbsentWeightError(
            f"component '{component}': its weights index {index_path} cannot be read ({e}) — "
            f"refused: without it no weight can be verified present.") from e
    tensors = index.get("tensors") if isinstance(index, dict) else None
    if not isinstance(tensors, dict):
        raise AbsentWeightError(
            f"component '{component}': its weights index {index_path} carries no `tensors` "
            f"table — refused: without it no weight can be verified present.")
    return tensors


def absent_weights(index_tensors: dict, wanted, header_of) -> list:
    """The wanted keys a load cannot deliver, as (key, shard, reason), in sorted key order.

    `header_of(shard)` returns that shard file's safetensors header, or None when the file is
    absent, or True when the file is present but has no header to read before loading (a
    `.bin`: its keys are judged by the post-load door); an exception it raises (a truncated or
    unreadable file) is the reason. The three
    absences, each named:

      * the index lists no such key (shard None) — the graph or the flow asks for a weight the
        container does not declare;
      * the index places the key in a shard that is absent on disk;
      * the shard is there but its header holds no such key.
    """
    headers: dict = {}
    out = []
    for key in sorted(wanted):
        entry = index_tensors.get(key)
        shard = entry.get("shard") if isinstance(entry, dict) else None
        if not shard:
            out.append((key, None, "the weights index lists no such key"))
            continue
        if shard not in headers:
            try:
                headers[shard] = header_of(shard)
            except Exception as e:      # named below, never swallowed
                headers[shard] = e
        h = headers[shard]
        if h is None:
            out.append((key, shard, "the shard file is absent on disk"))
        elif isinstance(h, Exception):
            out.append((key, shard, f"the shard's header cannot be read ({h})"))
        elif h is True:
            continue        # present; its keys are checked after the load
        elif key not in h:
            out.append((key, shard, "the shard's header holds no such key"))
    return out


def refuse_absent_weights(component: str, weights_dir: str, index_path: str,
                          absent: list, limit: int = 20) -> None:
    """Raise AbsentWeightError naming each absent weight (`absent_weights`' rows) — the
    component, the key, and the shard path the index expects it in. No-op when nothing is
    absent."""
    if not absent:
        return
    lines = []
    for key, shard, reason in absent[:limit]:
        where = f"{weights_dir.rstrip('/')}/{shard}" if shard else index_path
        lines.append(f"  {key}: expected in {where} — {reason}")
    more = f"\n  ... and {len(absent) - limit} more" if len(absent) > limit else ""
    shards = sorted({s for _, s, _ in absent if s})
    raise AbsentWeightError(
        f"component '{component}': {len(absent)} weight(s) the load needs are absent from the "
        f"container — refused before execution (index: {index_path}):\n"
        + "\n".join(lines) + more
        + (f"\nShard(s) involved: {', '.join(shards)}. A container whose files changed under "
           f"the run is restaged, never run partially." if shards else ""))


def loader_weight_consumers(dag: dict) -> Dict[str, str]:
    """Every `param::`/`buffer::` tensor an op in the graph's execution order reads whose value
    comes from the container's weights → the first op that reads it.

    A graph constant (its bytes are in graph.json), a computed buffer and a folded constant
    are filled by their own paths, each with its own refusal, and are left out. What remains
    must be bound when the sequence binds its weights: a slot left empty there is a weight the
    graph binds and the load did not deliver, and the first op to read it would meet None.
    Both engines' sequences compute it once at compile, before their elimination passes
    rewire consumers (an eliminated `detach`/`t` hands its weight to the op after it, which
    still reads it)."""
    tensors = dag.get("tensors") or {}
    ops = dag.get("ops") or {}
    if isinstance(ops, list):
        ops = {op.get("op_uid"): op for op in ops}
    out: Dict[str, str] = {}
    for uid in dag.get("execution_order") or []:
        for tid in (ops.get(uid) or {}).get("input_tensor_ids") or []:
            if tid in out or not (tid.startswith("param::") or tid.startswith("buffer::")):
                continue
            meta = tensors.get(tid) or {}
            if meta.get("constant") or meta.get("is_computable") or meta.get("folded_const"):
                continue
            out[tid] = uid
    return out


def refuse_unbound_weights(component, unbound: list, tensors: dict, limit: int = 20) -> None:
    """Raise AbsentWeightError for weights the graph binds that reached the bind empty —
    `unbound` is [(tid, first consumer op uid)]. No-op when empty. Shared by both engines'
    sequences (R30)."""
    if not unbound:
        return
    lines = []
    for tid, uid in unbound[:limit]:
        name = (tensors.get(tid) or {}).get("weight_name") or tid.split("::", 1)[-1]
        lines.append(f"  {tid} (read by {uid}): no loaded tensor under '{name}' nor under "
                     f"any trailing suffix of it")
    more = f"\n  ... and {len(unbound) - limit} more" if len(unbound) > limit else ""
    raise AbsentWeightError(
        f"component '{component}': {len(unbound)} weight(s) the graph binds are absent at the "
        f"weight bind — refused before the first op reads None:\n" + "\n".join(lines) + more
        + "\nThe container's weights_index.json does not provide them, or their load did not "
          "deliver them.")


def _resolve_shard_device(shard_path: str, weight_device: Dict[str, int]):
    """Device for a filesystem shard FILE from a shard-PATH-keyed shard_map.

    Prism's weight_sharding produces a shard_map keyed by .nbx rel-paths
    (e.g. "components/transformer/weights/shard_000.safetensors") with one
    "cuda:N" per shard file — every weight in that file goes to that device.
    The loader iterates filesystem shard paths, so match the shard_map key as
    a suffix of the filesystem path. Returns None when no shard-path key
    matches (weight-name-keyed FGP style, or unsharded). Without this the
    loader looked up by WEIGHT name against a shard-PATH-keyed map, missed
    every time, and collapsed the whole component onto device_idx — the 14B
    transformer landed entirely on one GPU and OOMed instead of sharding.
    """
    p = shard_path.replace("\\", "/")
    for k, dv in weight_device.items():
        if isinstance(k, str) and k.endswith(".safetensors") and p.endswith(k):
            return dv
    return None


def _weight_target_dev(key: str, shard_dev, weight_device: Dict[str, int],
                       device_idx: int) -> int:
    """Per-weight device: weight-name key (FGP) first, else the shard file's
    device (weight_sharding), else the component's default device."""
    dv = weight_device.get(key)
    if dv is not None:
        return dv
    if shard_dev is not None:
        return shard_dev
    return device_idx


def load_component_weights(
    cache_path: str,
    component: str,
    device_idx: int,
    compute_dtype: NBXDtype = NBXDtype.float16,
    shard_map: Optional[Dict[str, str]] = None,
    upcast_fp16_to_fp32: bool = False,
    per_device_vram_budget: Optional[Dict[int, int]] = None,
    only: Optional[Set[str]] = None,
) -> Dict[str, NBXTensor]:
    """Load weights for a component as NBXTensor. Zero torch.

    `only` restricts the load to the named weights. It is the capability two
    separate things need and neither had:

      * a parameter that NO op in the graph consumes is never read during
        execution, so loading it is pure cost. DeepSeek-Coder-V2-Lite carries
        11781 MB of such weights — MoE experts the trace never routed to — out
        of 30638 MB, 38% of the component.
      * streaming a component at layer granularity means holding one segment's
        weights at a time, which is this same call with a smaller set.

    None means every weight in the shards, which is what every caller did
    before and still does unless it says otherwise.

    Uses ComponentArena: one cudaMalloc per device, sub-allocate inside.
    Zero fragmentation vs 18866 individual cudaMalloc calls.

    Args:
        cache_path: Model cache path
        component: Component name (e.g. "model")
        device_idx: Default GPU device
        compute_dtype: Target dtype from Prism (bf16↔fp16 remap)
        shard_map: weight_name → "cuda:N" from Prism strategy
        upcast_fp16_to_fp32: If True AND the fp32 footprint fits the
            per-device budget, store fp16-target weights as fp32 at load
            time. Used on pre-Ampere hardware (no native bf16) to avoid
            per-call weight upcast in mm/bmm/addmm.
        per_device_vram_budget: Per-device byte budget reserved for
            weights (typically ~50% of device memory, leaving room for
            activations + KV cache). If the doubled fp32 footprint for a
            given device exceeds its budget, upcast is globally disabled
            to keep dtype consistency across the model.
    """
    comp_dir = Path(cache_path) / "components" / component
    weights_dir = comp_dir / "weights"

    # The index is the container's statement of what it holds. Read it first: a component that
    # stores no weights has neither index nor shard; one whose `weights/` directory is missing
    # while its index lists tensors is refused below by name, never returned as `{}`.
    index_tensors = read_weights_index(component, comp_dir)
    if index_tensors is None:
        return {}

    # Phase 0: every shard header, read once — the pre-flight below and the sizing and loading
    # phases all walk these same headers, in this same (sorted) order.
    shard_files = sorted(weights_dir.glob("*.safetensors")) if weights_dir.is_dir() else []
    shard_headers = []
    for shard_path in shard_files:
        header, data_offset = _read_header(str(shard_path))
        shard_headers.append((str(shard_path), header, data_offset))

    # Pre-flight, before any device is touched: every weight this load is asked for (`only`,
    # else every key the index lists) is in the shard the index places it in, and that shard
    # is among the files this load reads. A key outside both is a None waiting for its first
    # reader. Judged on the very headers the load walks below, so what passes is what loads.
    by_name = {Path(p).name: h for p, h, _ in shard_headers}
    refuse_absent_weights(
        component, str(weights_dir), str(comp_dir / "weights_index.json"),
        absent_weights(index_tensors, index_tensors.keys() if only is None else only,
                       by_name.get))
    if not weights_dir.is_dir():
        return {}           # no shard directory, and the index lists nothing this load wants

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)

    # Parse shard_map. Prism uses two styles:
    #
    #   1. Per-shard-path (standard): keys are zip paths like
    #      "components/model/weights/shard_000.safetensors". The
    #      value is the target device string for every weight in
    #      that shard file. zero3 produces this form with all values
    #      equal to "cpu" (whole component → host offload); multi-GPU
    #      strategies (pipeline_parallel, component_placement) produce
    #      "cuda:N" per shard.
    #
    #   2. Per-weight-name (FGP): keys are weight names like
    #      "block.0.attn.q.weight" with "cuda:N" values. The triton path
    #      DOES support this (see the per-weight loop below populating
    #      `weight_device` from every "cuda:N" value) — block_scatter /
    #      component_placement / pipeline_parallel all land here and place
    #      each weight on its owner GPU. (Comment corrected 2026-07-20,
    #      D7 — it previously claimed triton had no FGP, contradicting the
    #      code right below it.)
    #
    # In the triton path we detect style (1) and route accordingly.
    # Under zero3 we partition block vs non-block via _BLOCK_RE and
    # only honor "cpu" for block weights — non-block weights
    # (embeddings, final norm, lm_head) must stay GPU-resident
    # because flow handlers (GraphLMSession.prefill) call
    # w.embedding(embed_weight, ...) directly with raw pointers, and
    # a CPU pointer passed to a Triton kernel would segfault.
    weight_device: Dict[str, int] = {}
    cpu_weights: set = set()
    all_cpu_component = False
    if shard_map:
        # Detect "whole component → CPU" pattern: every shard_map
        # value is "cpu". This is zero3's calling convention.
        vals = [str(v).lower() for v in shard_map.values()]
        if vals and all(v == "cpu" for v in vals):
            all_cpu_component = True
        else:
            # Multi-GPU style (FGP): values are "cuda:N" per shard path or
            # per weight name. Each key that resolves to a device index is
            # placed on that owner GPU (the triton FGP path — proven with
            # Wan-I2V-14B sharded across cuda:2+cuda:3, commit 1d00037).
            for wname, dev_str in shard_map.items():
                if isinstance(dev_str, str) and ':' in dev_str:
                    weight_device[wname] = int(dev_str.split(':')[-1])
                elif isinstance(dev_str, int):
                    weight_device[wname] = dev_str

    # Phase 1: Scan all shards to compute per-device bytes for both the
    # regular (fp16/bf16-targeted) path and the hypothetical fp32 upcast
    # path. We then decide globally whether to upcast.
    dev_bytes_regular: Dict[int, int] = {}
    dev_bytes_upcast: Dict[int, int] = {}

    for shard_path, header, _ in shard_headers:
        # All weights in a shard FILE share the file's target device under
        # weight_sharding (shard-path-keyed map); FGP weight-name keys override
        # per weight inside _weight_target_dev.
        shard_dev = _resolve_shard_device(str(shard_path), weight_device)

        for key, info in header.items():
            if key == '__metadata__':
                continue
            if only is not None and key not in only:
                continue        # not asked for: not sized, not loaded
            # Zero3 whole-component CPU offload: block weights (those
            # matching _BLOCK_RE) go to pinned host memory; non-block
            # weights (embeddings, norms, lm_head) stay GPU-resident.
            if all_cpu_component:
                if _BLOCK_RE.search(key):
                    cpu_weights.add(key)
                    continue
                # Non-block → GPU sizing continues below.
                target_dev = device_idx
            else:
                target_dev = _weight_target_dev(key, shard_dev, weight_device, device_idx)
            regular = _target_nbytes(info, compute_dtype)
            upcast = _target_nbytes(info, compute_dtype,
                                    upcast_fp16_to_fp32=True)
            reg_aligned = (regular + ComponentArena.ALIGNMENT - 1) & ~(
                ComponentArena.ALIGNMENT - 1)
            up_aligned = (upcast + ComponentArena.ALIGNMENT - 1) & ~(
                ComponentArena.ALIGNMENT - 1)
            dev_bytes_regular[target_dev] = (
                dev_bytes_regular.get(target_dev, 0) + reg_aligned)
            dev_bytes_upcast[target_dev] = (
                dev_bytes_upcast.get(target_dev, 0) + up_aligned)

    # Decide: upcast only if every device's fp32 footprint fits its budget.
    upcast_effective = upcast_fp16_to_fp32
    if upcast_effective and per_device_vram_budget is not None:
        for dev, up in dev_bytes_upcast.items():
            budget = per_device_vram_budget.get(dev, 0)
            if budget <= 0 or up > budget:
                upcast_effective = False
                print(f"[weight_loader] bind-time fp16→fp32 upcast skipped "
                      f"for {component}: device {dev} needs {up/1e9:.1f}GB "
                      f"fp32, budget {budget/1e9:.1f}GB — per-call fallback "
                      f"will handle overflow protection")
                break
    elif upcast_effective and per_device_vram_budget is None:
        # No budget means we can't verify fit — be conservative.
        upcast_effective = False

    dev_bytes = dev_bytes_upcast if upcast_effective else dev_bytes_regular

    import os as _os
    if _os.environ.get("NBX_DIAG_TRITON_PRELOOP") == "1":
        _dist = {f"cuda:{d}": f"{b/1e9:.2f}GB" for d, b in sorted(dev_bytes.items())}
        print(f"   [NBX-DIAG-SHARD] {component}: shard_map_entries={len(shard_map or {})} "
              f"per-device={_dist}", flush=True)

    if upcast_effective:
        total_up = sum(dev_bytes_upcast.values())
        total_reg = sum(dev_bytes_regular.values())
        print(f"[weight_loader] bind-time fp16→fp32 upcast ENABLED for "
              f"{component}: {total_reg/1e9:.2f}GB → {total_up/1e9:.2f}GB")

    # Phase 2: Create one arena per device
    arenas: Dict[int, ComponentArena] = {}
    for dev, total in dev_bytes.items():
        arenas[dev] = ComponentArena(total, dev)

    # Phase 3: Load tensors into arenas (GPU) or CPU pinned buffers.
    weights: Dict[str, NBXTensor] = {}

    for shard_path, header, data_offset in shard_headers:
        _load_shard_into_arenas(
            shard_path, header, data_offset,
            weights, device_idx, compute_dtype,
            weight_device, arenas, cpu_weights,
            upcast_effective=upcast_effective, only=only)

    # Store arenas on the dict so they stay alive (prevent GC of GPU memory)
    weights['_arenas'] = arenas  # type: ignore

    if cpu_weights:
        total_cpu_mb = DeviceAllocator.host_pinned_allocated() / (1024 * 1024)
        print(f"[weight_loader] zero3 CPU partition for {component}: "
              f"{len(cpu_weights)} block weights on pinned host "
              f"(~{total_cpu_mb:.0f}MB), "
              f"non-block on cuda:{device_idx}")

    return weights


def _read_header(path: str):
    """Read safetensors header without loading data."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size
    return header, data_offset


def _target_nbytes(info: dict, compute_dtype: NBXDtype,
                   upcast_fp16_to_fp32: bool = False) -> int:
    """Compute target byte count for a tensor after dtype remap.

    When upcast_fp16_to_fp32 is set, weights whose EFFECTIVE target
    dtype is fp16 (either native fp16 or bf16 remapped to fp16 via the
    standard bf16→fp16 path) are treated as fp32 for sizing purposes.
    """
    from neurobrix.kernels.nbx_tensor import dtype_size as _dsize
    shape = info["shape"]
    sf_dtype = info["dtype"]
    dtype_info = _SF_DTYPE_INFO.get(sf_dtype)
    if dtype_info is None:
        return 0
    _, nbx_dtype, _ = dtype_info

    numel = math.prod(shape) if shape else 1

    # Effective target dtype — must mirror the remap logic in the loader
    # below so arena sizing matches the bytes we actually write.  The
    # fp32 → half downcast (fp32-on-disk + half-precision compute) was
    # missing here and caused triton to allocate full fp32-sized arenas
    # for fp32-shipping weights (T5 text_encoder in PixArt: 19 GB vs the
    # 9.5 GB the load loop actually writes → OOM on a 32 GB V100 even
    # though the data fits).
    target = stored_dtype_in_compute(nbx_dtype, compute_dtype)

    if upcast_fp16_to_fp32 and target == NBXDtype.float16:
        return numel * 4  # pre-Ampere overflow protection path

    return numel * _dsize(target)


def _bf16_to_fp16_inplace(ptr: int, numel: int, device_idx: int):
    """Run bf16→fp16 kernel in-place on a GPU buffer.

    Both bf16 and fp16 are 2 bytes/element, so the kernel reads uint16
    (bf16 raw bits) and writes fp16 to the same memory locations.

    Creates typed NBXTensor wrappers around the raw pointer for Triton.
    """
    import triton
    from neurobrix.kernels.ops.dtype_convert import bf16_to_fp16_kernel

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)

    # Wrap raw pointer as typed NBXTensor for Triton kernel.
    # int16 and uint16 are same 2-byte memory layout — Triton kernel
    # treats bits as unsigned via .to(tl.uint32) anyway.
    flat = (numel,)
    strides = _contiguous_strides(flat)
    src = NBXTensor(ptr, flat, strides, NBXDtype.int16, 'cuda',
                    owns_data=False, device_idx=device_idx)
    dst = NBXTensor(ptr, flat, strides, NBXDtype.float16, 'cuda',
                    owns_data=False, device_idx=device_idx)

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _set_device(src)
    bf16_to_fp16_kernel[grid](src, dst, numel, BLOCK=BLOCK, num_warps=4)


def _load_to_pinned_cpu(
    raw: bytes,
    shape: tuple,
    source_dtype: NBXDtype,
    target_dtype: NBXDtype,
) -> NBXTensor:
    """Decode safetensors raw bytes into a pinned-host NBXTensor.

    Applies the standard dtype remap chain (bf16→fp16, fp16→bf16) via
    numpy before copying to pinned memory. bf16→fp16 uses the same
    uint16-bit reinterpret path the GPU arena loader uses, so the
    resulting values match exactly.
    """
    # Decode source bytes to a numpy array matching target_dtype.
    if source_dtype == NBXDtype.bfloat16:
        # Raw bf16 bits stored as uint16. Stays uint16 if target is
        # bf16, or gets bit-shifted to fp32 then cast to fp16 if target
        # is fp16.
        raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        if target_dtype == NBXDtype.float16:
            u32 = raw_u16.astype(np.uint32) << 16
            arr = np.ascontiguousarray(u32.view(np.float32)).astype(np.float16)
        else:
            arr = np.ascontiguousarray(raw_u16)
    else:
        np_dtype = {
            NBXDtype.float16: np.float16,
            NBXDtype.float32: np.float32,
            NBXDtype.float64: np.float64,
            NBXDtype.int64: np.int64,
            NBXDtype.int32: np.int32,
            NBXDtype.int16: np.int16,
            NBXDtype.int8: np.int8,
            NBXDtype.uint8: np.uint8,
            NBXDtype.bool_: np.bool_,
        }[source_dtype]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        arr = np.ascontiguousarray(arr)
        if target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float16:
            # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact)
            arr = float32_to_bf16_bits(arr.astype(np.float32))
        elif target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float32:
            # fp32 → bf16 rounded to nearest even, the SAME transform the
            # GPU arena loader uses (_load_shard_into_arenas), so the zero3
            # host-staged copy of a weight is bit-identical to its arena copy.
            # An fp32 encoder (PixArt / CogVideoX T5) staged to a bf16 compute
            # target reached here still 4 bytes wide and overflowed the 2-byte
            # pinned buffer — _NUMPY_FOR has no bf16 entry (numpy cannot
            # represent it). The target itself is decided by
            # stored_dtype_in_compute, the one rule shared with the census
            # shadow, so completing this pair keeps the shadow's dtypes and the
            # real run's identical rather than diverging.
            arr = float32_to_bf16_bits(arr)

    # Any remaining source->target pair: cast through numpy.
    #
    # The two bit-level paths above cover bf16, which numpy cannot represent.
    # Everything else is an ordinary cast, and it MUST happen: before
    # 2026-09-03 only the two bf16 pairs were converted, so a float32 weight
    # with a float16 target reached the copy still 4 bytes wide and wrote
    # `arr.nbytes` into a buffer sized for the TARGET dtype — 1024 bytes into
    # 512. Kokoro-82M's decoder is exactly that case (a (256,) fp32 tensor),
    # and the full-zoo gate recorded it as a 240 s TIMEOUT rather than as the
    # overflow it is, because the run never got far enough to fail.
    _NUMPY_FOR = {
        NBXDtype.float16: np.float16, NBXDtype.float32: np.float32,
        NBXDtype.float64: np.float64, NBXDtype.int64: np.int64,
        NBXDtype.int32: np.int32, NBXDtype.int16: np.int16,
        NBXDtype.int8: np.int8, NBXDtype.uint8: np.uint8,
        NBXDtype.bool_: np.bool_,
    }
    want = _NUMPY_FOR.get(target_dtype)
    if want is not None and arr.dtype != want:
        arr = np.ascontiguousarray(arr.astype(want))

    # Allocate pinned host memory and copy in (kind=0 = H2H).
    dst = NBXTensor.empty_cpu(shape, target_dtype, pinned=True)

    # The copy is sized from the SOURCE and the buffer from the TARGET, so a
    # dtype pair nobody converted is a write past the end of a cudaMallocHost
    # region. That is only caught at all because a pinned range is registered
    # with the driver; the same defect on ordinary host memory corrupts the
    # heap in silence. Refuse here, naming the pair, rather than rely on the
    # driver noticing.
    if arr.nbytes != dst.nbytes():
        raise RuntimeError(
            f"weight staging would overflow: {source_dtype!r} -> "
            f"{target_dtype!r} for shape {tuple(shape)} produced "
            f"{arr.nbytes} bytes for a {dst.nbytes()}-byte buffer. "
            f"This dtype pair has no conversion in _load_to_pinned_cpu — add "
            f"one; never widen the buffer to match."
        )
    DeviceAllocator.memcpy(dst.data_ptr(), arr.ctypes.data,
                           arr.nbytes, kind=0)
    return dst


def _load_shard_into_arenas(
    path: str,
    header: dict,
    data_offset: int,
    weights: Dict[str, NBXTensor],
    device_idx: int,
    compute_dtype: NBXDtype,
    weight_device: Dict[str, int],
    arenas: Dict[int, ComponentArena],
    cpu_weights: set,
    upcast_effective: bool = False,
    only: Optional[Set[str]] = None,
) -> None:
    """Load tensors from one shard, sub-allocating from arenas.

    When upcast_effective is True, weights whose target dtype resolves
    to fp16 (via the standard remap chain) are stored as fp32 instead.
    This is decided upstream based on per-device VRAM budget.

    Weights listed in cpu_weights skip the GPU arena and land on pinned
    host memory as CPU-backed NBXTensors (zero3 offload). They are not
    subject to upcast_effective — CPU weights are stored in their
    native dtype (post-remap) to save host RAM. The runtime slow path
    handles CPU→GPU transfer per op.
    """
    # Device for every weight in THIS shard file (weight_sharding); FGP
    # weight-name keys still override per weight in _weight_target_dev.
    shard_dev = _resolve_shard_device(path, weight_device)
    with open(path, 'rb') as f:
        for key, info in header.items():
            if key == '__metadata__':
                continue
            if only is not None and key not in only:
                continue        # sized out above; skip the read too

            sf_dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            nbytes = end - start

            dtype_info = _SF_DTYPE_INFO.get(sf_dtype)
            if dtype_info is None:
                raise RuntimeError(f"Unknown safetensors dtype: {sf_dtype}")

            np_dtype, nbx_dtype, elem_size = dtype_info

            # Read raw bytes from file
            f.seek(data_offset + start)
            raw = f.read(nbytes)

            # Determine target dtype after remap — the one rule, `stored_dtype_in_compute`
            target_dtype = stored_dtype_in_compute(nbx_dtype, compute_dtype)

            # Zero3 CPU offload — skip the GPU arena entirely and
            # allocate pinned host memory for this weight. The numpy
            # decode step still applies (bf16→fp16 via uint16 remap,
            # etc.) but the final NBXTensor is CPU-backed.
            if key in cpu_weights:
                weights[key] = _load_to_pinned_cpu(
                    raw, shape, nbx_dtype, target_dtype)
                continue

            # Bind-time fp16→fp32 upcast for pre-Ampere overflow protection.
            # Only promote weights whose final target is fp16; bf16 targets,
            # integer types, and already-fp32 weights pass through unchanged.
            upcast_this = upcast_effective and target_dtype == NBXDtype.float16

            target_dev = _weight_target_dev(key, shard_dev, weight_device, device_idx)
            arena = arenas[target_dev]
            DeviceAllocator.set_device(target_dev)

            # GPU-accelerated bf16→fp16 path (same buffer, 2 bytes/elem).
            # Only used when we're NOT also upcasting to fp32 — otherwise
            # the staged bf16→fp32 conversion happens via numpy below.
            if (nbx_dtype == NBXDtype.bfloat16
                    and target_dtype == NBXDtype.float16
                    and not upcast_this):
                ptr = arena.alloc(nbytes)
                arr_u16 = np.frombuffer(raw, dtype=np.uint16)
                DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
                _bf16_to_fp16_inplace(ptr, arr_u16.shape[0], target_dev)
                strides = _contiguous_strides(shape)
                nbx = NBXTensor(ptr, shape, strides, NBXDtype.float16, 'cuda',
                                owns_data=False, device_idx=target_dev)
                weights[key] = nbx
                continue

            # Standard path: numpy → H2D, with optional fp32 upcast.
            if nbx_dtype == NBXDtype.bfloat16:
                # bf16 bits in uint16 containers. If upcast_this, expand
                # them to fp32 via the standard bit-shift: bf16 is the top
                # 16 bits of a fp32, so (bf16_bits << 16) reinterpret-cast
                # as fp32 reproduces the value exactly (no approximation).
                raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                if upcast_this:
                    u32 = raw_u16.astype(np.uint32) << 16
                    arr = np.ascontiguousarray(u32.view(np.float32))
                    final_dtype = NBXDtype.float32
                else:
                    arr = np.ascontiguousarray(raw_u16)
                    final_dtype = target_dtype  # bf16 (kept as uint16)
            else:
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                arr = np.ascontiguousarray(arr)
                if upcast_this:
                    # Native fp16 → fp32 widening (exact, no precision loss).
                    arr = np.ascontiguousarray(arr.astype(np.float32))
                    final_dtype = NBXDtype.float32
                elif (nbx_dtype == NBXDtype.float32
                      and target_dtype == NBXDtype.float16):
                    # fp32 → fp16 downcast. Required when an encoder ships
                    # fp32 on disk (PixArt T5, SDXL text_encoder_2, ...)
                    # but the compute path wants fp16. Native's
                    # WeightLoader(torch_dtype=fp16) does this implicitly;
                    # triton needs to do it explicitly so arena sizing
                    # and the kernel-side dtype agree.
                    arr = np.ascontiguousarray(arr.astype(np.float16))
                    final_dtype = NBXDtype.float16
                elif (nbx_dtype == NBXDtype.float32
                      and target_dtype == NBXDtype.bfloat16):
                    # fp32 → bf16, rounded to nearest even.
                    arr = float32_to_bf16_bits(arr)
                    final_dtype = NBXDtype.bfloat16
                elif target_dtype != nbx_dtype:
                    # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact).
                    arr = float32_to_bf16_bits(arr.astype(np.float32))
                    final_dtype = target_dtype
                else:
                    final_dtype = target_dtype

            arr_bytes = arr.nbytes
            ptr = arena.alloc(arr_bytes)
            DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

            # Create NBXTensor wrapping the arena sub-allocation
            strides = _contiguous_strides(shape)
            nbx = NBXTensor(ptr, shape, strides, final_dtype, 'cuda',
                            owns_data=False, device_idx=target_dev)
            weights[key] = nbx

def stored_dtype_in_compute(nbx_dtype, compute_dtype):
    """The dtype a stored weight takes in memory under a compute dtype — THE rule, in one place.

    A half weight follows the compute half (bf16 on disk runs fp16 on Volta and the other way
    round); fp32 on disk under a half compute is downcast at load (mirrors the ATen branch's
    WeightLoader(torch_dtype=fp16); T5 text encoders ship fp32 and run fp16); anything else
    keeps its stored dtype. The loader applies it to every weight of a shard, param or buffer
    alike; the certification census (kernels/census.py) applies the same rule to shape its
    shadow weights, so the keys it records are the keys the loaded weights form — a second
    copy of this rule drifted once (2026-09-21: the shadow kept fp32 weights fp32 and recorded
    addmm keys the live launcher never forms).
    """
    if nbx_dtype == NBXDtype.bfloat16 and compute_dtype == NBXDtype.float16:
        return NBXDtype.float16
    if nbx_dtype == NBXDtype.float16 and compute_dtype == NBXDtype.bfloat16:
        return NBXDtype.bfloat16
    if nbx_dtype == NBXDtype.float32 and compute_dtype in (NBXDtype.float16, NBXDtype.bfloat16):
        return compute_dtype
    return nbx_dtype

