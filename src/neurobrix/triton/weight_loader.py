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

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.backend.runtime import _set_device
from nbxtensor.dtype.itemsize import itemsize
from nbxtensor.shape import numel
from nbxtensor.memory.memory_pool import ComponentArena, HostPinnedArena
from nbxtensor.io import safetensors as _sf
from nbxtensor.memory.layout import aligned
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


def _io_pool():
    """(workers, pool) of this process — read once from `io_workers()`."""
    global _IO_POOL
    if _IO_POOL is None or _IO_POOL[0] != os.getpid():
        workers = io_workers()
        _IO_POOL = (os.getpid(), workers,
                    ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nbx-weight-io"))
    return _IO_POOL[1], _IO_POOL[2]


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
                    host_bytes += aligned(_stored_bytes(info, compute_dtype), HostPinnedArena.ALIGNMENT)
                    continue
                # Non-block → GPU sizing continues below.
                target_dev = device_idx
            else:
                target_dev = _weight_target_dev(key, shard_dev, weight_device, device_idx)
            regular = _stored_bytes(info, compute_dtype)
            upcast = _stored_bytes(info, compute_dtype,
                                    upcast_fp16_to_fp32=True)
            reg_aligned = aligned(regular, ComponentArena.ALIGNMENT)
            up_aligned = aligned(upcast, ComponentArena.ALIGNMENT)
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


def _stored_bytes(info: dict, compute_dtype: NBXDtype,
                   upcast_fp16_to_fp32: bool = False) -> int:
    """The bytes a tensor occupies once the loader stores it in its compute dtype (the remap below), from
    nbxtensor's itemsize and numel.

    When upcast_fp16_to_fp32 is set, weights whose EFFECTIVE target
    dtype is fp16 (either native fp16 or bf16 remapped to fp16 via the
    standard bf16→fp16 path) are treated as fp32 for sizing purposes.
    """
    shape = info["shape"]
    sf_dtype = info["dtype"]
    dtype_info = _sf.SF_DTYPE_INFO.get(sf_dtype)
    if dtype_info is None:
        return 0
    _, nbx_dtype, _ = dtype_info

    n = numel(shape)

    # Effective target dtype — must mirror the remap logic in the loader
    # below so arena sizing matches the bytes we actually write.  The
    # fp32 → half downcast (fp32-on-disk + half-precision compute) was
    # missing here and caused triton to allocate full fp32-sized arenas
    # for fp32-shipping weights (T5 text_encoder in PixArt: 19 GB vs the
    # 9.5 GB the load loop actually writes → OOM on a 32 GB V100 even
    # though the data fits).
    target = stored_dtype_in_compute(nbx_dtype, compute_dtype)

    if upcast_fp16_to_fp32 and target == NBXDtype.float16:
        return n * itemsize("float32")  # pre-Ampere overflow protection path

    return n * itemsize(target)


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

            dtype_info = _sf.SF_DTYPE_INFO.get(sf_dtype)
            if dtype_info is None:
                raise RuntimeError(f"Unknown safetensors dtype: {sf_dtype}")

            np_dtype, nbx_dtype, elem_size = dtype_info

            # Read raw bytes from file
            raw = _sf.read_bytes(f.fileno(), data_offset + start, nbytes, _io_pool())

            # Determine target dtype after remap — the one rule, `stored_dtype_in_compute`
            target_dtype = stored_dtype_in_compute(nbx_dtype, compute_dtype)

            # Zero3 CPU offload — skip the GPU arena entirely and
            # allocate pinned host memory for this weight. The numpy
            # decode step still applies (bf16→fp16 via uint16 remap,
            # etc.) but the final NBXTensor is CPU-backed.
            if key in cpu_weights:
                weights[key] = _sf.load_to_pinned_host(
                    raw, shape, nbx_dtype, target_dtype, _io_pool(), host_arena)
                continue

            # Bind-time fp16→fp32 upcast for pre-Ampere overflow protection.
            # Only promote weights whose final target is fp16; bf16 targets,
            # integer types, and already-fp32 weights pass through unchanged.
            upcast_this = upcast_effective and target_dtype == NBXDtype.float16

            target_dev = _weight_target_dev(key, shard_dev, weight_device, device_idx)
            arena = arenas[target_dev]
            DeviceAllocator.set_device(target_dev)

            weights[key] = _sf.place_in_arena(raw, shape, nbx_dtype, np_dtype, target_dtype, upcast_this,
                                              bool(krsc) and key in krsc, arena, target_dev, _io_pool())

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

