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

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from neurobrix.kernels.nbx_tensor import (
    NBXTensor, NBXDtype, DeviceAllocator, _contiguous_strides, dtype_size,
    _set_device, float32_to_bf16_bits,
)
from .memory_pool import ComponentArena, HostPinnedArena
# The loader's worker count — one figure for every loader and for Prism's host estimate (torch-free).
from neurobrix.core.workspace import io_workers, apply_host_mmap_threshold, apply_host_huge_pages
# The absent-weight rule lives beside the container format it guards (torch-free), one rule for
# both engines' loaders and binds and for the validator.
from neurobrix.nbx.weight_presence import (   # noqa: F401 — re-exported for this engine's callers
    AbsentWeightError, read_weights_index, absent_weights, refuse_absent_weights,
    loader_weight_consumers, refuse_unbound_weights, read_safetensors_header,
)


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


# One tensor at a time, read and cast by `io_workers()` threads: a shard read by one `f.read` and
# cast by one numpy loop ran at 0.40 GB/s (cold) / 0.52 GB/s (warm) on Wan2.2-I2V-A14B's fp32
# shards over the RDMA cache; split across 8 workers, 1.4-1.7 / 2.7 GB/s, the same bytes
# (V100 rack, 2026-10-21). Both `os.preadv` and numpy's casting loop release the GIL. The tensor
# in flight is still ONE, as Prism's triton host estimate prices it (`host_footprint`:
# TRITON_COPIES_PER_TENSOR x the largest tensor) — the work on it is what is shared.
_IO_POOL: Optional[tuple] = None      # (pid, workers, pool): a forked child builds its own
_PAGE = os.sysconf("SC_PAGE_SIZE")


def _io_pool():
    """(workers, pool) of this process — read once from `io_workers()`."""
    global _IO_POOL
    if _IO_POOL is None or _IO_POOL[0] != os.getpid():
        workers = io_workers()
        _IO_POOL = (os.getpid(), workers,
                    ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nbx-weight-io"))
    return _IO_POOL[1], _IO_POOL[2]


def _spans(n: int, unit: int, align: int):
    """[start, end) element spans of an n-element run, one per worker, each a multiple of `align`
    elements; a run under one page per worker stays whole (a split costs more than it shares)."""
    workers = _io_pool()[0]
    if workers < 2 or n * unit < workers * _PAGE:
        return [(0, n)]
    step = -(-n // workers)
    step = -(-step // align) * align
    return [(a, min(a + step, n)) for a in range(0, n, step)]


def _read_bytes(fd: int, offset: int, nbytes: int) -> np.ndarray:
    """nbytes of the file at offset, read by the loader's workers into one buffer (page-aligned
    spans; `preadv` may return short, so each span reads until it is full)."""
    buf = np.empty(nbytes, dtype=np.uint8)
    view = memoryview(buf)

    def _one(span):
        a, b = span
        while a < b:
            got = os.preadv(fd, [view[a:b]], offset + a)
            if got <= 0:
                raise RuntimeError(f"weight read: {nbytes} bytes at offset {offset} of fd {fd} ended "
                                   f"{b - a} bytes short (file truncated?)")
            a += got

    list(_io_pool()[1].map(_one, _spans(nbytes, 1, _PAGE)))
    return buf


def _bf16_bits_to_half(u16: np.ndarray) -> np.ndarray:
    """bf16 bits (uint16) to float16, exactly as `(bits << 16).view(float32).astype(float16)`, the elements split
    across the loader's workers — the widening ran on one thread for every host-held bf16 weight (a third of
    Qwen3-30B-A3B's 447 s load, py-spy 2026-10-27)."""
    src = np.ascontiguousarray(u16).reshape(-1)
    out = np.empty(u16.shape, dtype=np.float16)
    flat = out.reshape(-1)

    def _one(span):
        a, b = span
        np.copyto(flat[a:b], (src[a:b].astype(np.uint32) << 16).view(np.float32), casting="unsafe")

    list(_io_pool()[1].map(_one, _spans(src.size, src.itemsize, 1)))
    return out


def _cast(arr: np.ndarray, want) -> np.ndarray:
    """`arr.astype(want)`, C-contiguous, the elements split across the loader's workers."""
    src = np.ascontiguousarray(arr).reshape(-1)
    out = np.empty(arr.shape, dtype=want)
    flat = out.reshape(-1)

    def _one(span):
        np.copyto(flat[span[0]:span[1]], src[span[0]:span[1]], casting="unsafe")

    list(_io_pool()[1].map(_one, _spans(src.size, src.itemsize, 1)))
    return out


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
    krsc: Optional[Set[str]] = None,
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
        krsc: loader keys of the convolution weights stored KRSC (`host_weight_layout`), from
            the hardware profile's `conv.weight_layout` and the graph's readers
            (`launch_keys.krsc_conv_weights`). None stores every weight in its file order.
    """
    # The host read and cast buffers: given back to the OS when freed, and never a huge-page request that a
    # fragmented host answers by compacting memory (config/system.yml `io`; the compiled loader does the same).
    apply_host_mmap_threshold()
    apply_host_huge_pages()
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
    host_bytes = 0          # the host-held weights' bytes, in their target dtype: ONE pinned block (HostPinnedArena)

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
                    host_bytes += (_target_nbytes(info, compute_dtype) + HostPinnedArena.ALIGNMENT - 1) \
                        & ~(HostPinnedArena.ALIGNMENT - 1)
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

    # Phase 3: Load tensors into arenas (GPU) or the one pinned host block.
    weights: Dict[str, NBXTensor] = {}
    host_arena = HostPinnedArena(host_bytes) if host_bytes else None

    for shard_path, header, data_offset in shard_headers:
        _load_shard_into_arenas(
            shard_path, header, data_offset,
            weights, device_idx, compute_dtype,
            weight_device, arenas, cpu_weights,
            upcast_effective=upcast_effective, only=only, krsc=krsc, host_arena=host_arena)

    # Store arenas on the dict so they stay alive (prevent GC of GPU memory)
    weights['_arenas'] = arenas  # type: ignore

    if cpu_weights:
        total_cpu_mb = DeviceAllocator.host_pinned_allocated() / (1024 * 1024)
        print(f"[weight_loader] zero3 CPU partition for {component}: "
              f"{len(cpu_weights)} block weights on pinned host "
              f"(~{total_cpu_mb:.0f}MB), "
              f"non-block on cuda:{device_idx}")

    return weights


# The safetensors header reader is the container layer's, one reader for every engine.
_read_header = read_safetensors_header


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
    from nbxtensor.ops.dtype_convert import bf16_to_fp16_kernel

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

    _set_device(src)
    from neurobrix.kernels.ops.elementwise_launch import ew_meta, ew_grid
    m = ew_meta(numel)          # the certified tile of the count's class, as every element-wise launch
    bf16_to_fp16_kernel[ew_grid(numel, m)](src, dst, numel, BLOCK=m['BLOCK_SIZE'], num_warps=m['num_warps'])


def _load_to_pinned_cpu(
    raw: bytes,
    shape: tuple,
    source_dtype: NBXDtype,
    target_dtype: NBXDtype,
    host_arena: Optional[HostPinnedArena] = None,
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
            arr = _bf16_bits_to_half(raw_u16)
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
        arr = _cast(arr, want)

    # Pinned host memory: a slice of the component's one block when the load sized it, else its own block.
    if host_arena is not None:
        from neurobrix.kernels.nbx_tensor import dtype_size as _dsize
        ptr = host_arena.alloc((math.prod(shape) if shape else 1) * _dsize(target_dtype))
        dst = NBXTensor(ptr, tuple(shape), _contiguous_strides(tuple(shape)), target_dtype, 'cpu',
                        owns_data=False, device_idx=0, base=host_arena, pinned=True)
    else:
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


def host_weight_layout(arr: np.ndarray, krsc: bool):
    """The host array to copy to the device and the element strides of the logical tensor over
    it. A KRSC weight (`launch_keys.conv_weight_krsc`) is written channels-innermost, once, here,
    and stays a logical (K, C, R, S) tensor: the original order is never on the device."""
    if not krsc:
        return arr, _contiguous_strides(arr.shape)
    K, C, R, S = arr.shape
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1)), (R * S * C, 1, S * C, C)


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
    krsc: Optional[Set[str]] = None,
    host_arena: Optional[HostPinnedArena] = None,
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
            raw = _read_bytes(f.fileno(), data_offset + start, nbytes)

            # Determine target dtype after remap — the one rule, `stored_dtype_in_compute`
            target_dtype = stored_dtype_in_compute(nbx_dtype, compute_dtype)

            # Zero3 CPU offload — skip the GPU arena entirely and
            # allocate pinned host memory for this weight. The numpy
            # decode step still applies (bf16→fp16 via uint16 remap,
            # etc.) but the final NBXTensor is CPU-backed.
            if key in cpu_weights:
                weights[key] = _load_to_pinned_cpu(
                    raw, shape, nbx_dtype, target_dtype, host_arena)
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
                arr_u16, strides = host_weight_layout(
                    np.frombuffer(raw, dtype=np.uint16).reshape(shape), bool(krsc) and key in krsc)
                DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
                _bf16_to_fp16_inplace(ptr, arr_u16.size, target_dev)
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
                    arr = _cast(arr, np.float32)
                    final_dtype = NBXDtype.float32
                elif (nbx_dtype == NBXDtype.float32
                      and target_dtype == NBXDtype.float16):
                    # fp32 → fp16 downcast. Required when an encoder ships
                    # fp32 on disk (PixArt T5, SDXL text_encoder_2, ...)
                    # but the compute path wants fp16. Native's
                    # WeightLoader(torch_dtype=fp16) does this implicitly;
                    # triton needs to do it explicitly so arena sizing
                    # and the kernel-side dtype agree.
                    arr = _cast(arr, np.float16)
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

            arr, strides = host_weight_layout(arr, bool(krsc) and key in krsc)
            arr_bytes = arr.nbytes
            ptr = arena.alloc(arr_bytes)
            DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

            # Create NBXTensor wrapping the arena sub-allocation (KRSC: the logical shape over
            # the channels-innermost memory)
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

