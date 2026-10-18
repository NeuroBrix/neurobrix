"""The compiled engine's caching-allocator settings, declared by the vendor profile.

Mode 1 computes in torch tensors, so its device memory is torch's caching allocator. Prism admits a
plan by BYTES (`PrismSolver._place_component`: an activation figure against the card's usable
megabytes) and so assumes the card's free bytes serve any request that fits in them. Under the
allocator's default fixed segments that assumption is false: a freed block stays inside its segment,
and a request larger than every free block fails while the free bytes add up to more than it asks.
Wan2.1-I2V-14B compiled on a 16 GB V100 (2026-10-05, the transformer on zero3 at its derived
request): Prism priced 9 020 MB of activations + 2 015 MB of overhead; at the death, torch held
8.52 GiB allocated and asked 2.50 GiB more at aten.view_as_complex::4 — 11.0 GiB of bytes, the
price — while 6.76 GiB sat reserved-but-unallocated. Allegro-TI2V compiled died the same way with
5.93 GiB reserved-unallocated.

The vendor's remedy for exactly this case is the allocator's `expandable_segments` mode ("allocation
sizes that change", https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf):
one segment per stream that grows by mapping pages, whose free pages go back to the driver when an
allocation would fail. It makes the bytes Prism prices the bytes the card can hand out. Which
setting a vendor's cards run is a fact of that hardware stack, so it is DATA: the vendor profile's
`memory.compiled_allocator_settings` (config/vendors/<vendor>/<arch>.yml), declared wherever the
vendor's allocator implements the setting: every NVIDIA profile, and every AMD one — ROCm's build of
the same allocator maps expandable segments with hipMemCreate / hipMemMap from ROCm 7.0
(https://github.com/pytorch/pytorch/blob/v2.14.0/c10/cuda/CUDAAllocatorConfig.h). Apple's MPS
allocator has no such setting (aten/src/ATen/mps/MPSAllocator.mm reads only its watermark ratios),
so no Apple profile declares one. A profile that declares none runs torch's own default.

Apple's MPS allocator has one setting of its own: how far its pool may grow, a fraction of the device's
recommended working set (`torch.mps.set_per_process_memory_fraction`, its high watermark; torch's default 1.7).
Past that fraction the allocator first releases the freed blocks it caches, then refuses. At 1.7 the pool kept
the freed buffers of Allegro's cut attention and grew past the M4 Pro's 17.8 GB working set into swap (23.7 GB
footprint at the first feed-forward, the Mac, 2026-10-15 08:30) while the live set was ~15.5 GiB. A vendor file
states it as `memory.compiled_allocator_memory_fraction`, applied through the executing device's torch backend
(`torch.<type>.set_per_process_memory_fraction`); an operator's `PYTORCH_MPS_HIGH_WATERMARK_RATIO` wins.

Applied once per process, before the executors load a weight (`RuntimeExecutor.setup`), under the
compiled engine only: the Triton engine never imports torch (R33) and allocates through its own
runtime. An operator's explicit `PYTORCH_ALLOC_CONF` (or a legacy spelling, `OPERATOR_ENV`) is
theirs and wins; the engine says so instead of overriding it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

#: The allocator environment variables torch reads: the unified name and the CUDA and HIP legacy
#: spellings (https://github.com/pytorch/pytorch/blob/v2.14.0/c10/core/AllocatorConfig.cpp).
OPERATOR_ENV = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")

#: The vendor-profile key, under `memory:`.
PROFILE_KEY = "compiled_allocator_settings"
#: The vendor-profile key of the pool's growth bound, a fraction of the device's memory, and the operator's own
#: spelling of it (https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/mps/MPSAllocator.mm).
FRACTION_KEY = "compiled_allocator_memory_fraction"
FRACTION_OPERATOR_ENV = ("PYTORCH_MPS_HIGH_WATERMARK_RATIO",)
#: The vendor-profile key of the MPS pool's LOW watermark: above it the allocator returns its cached blocks to the system.
#: torch reads it once, when the MPS allocator is created (PYTORCH_MPS_LOW_WATERMARK_RATIO, MPSAllocator.mm), so it is set
#: in the environment before the first pool call below. Default 1.4 x the working set on unified memory: the cache can
#: hold the whole device beside the plan (measured 2026-10-18, the Mac: VACE's encoder stream alone 31.3 GB at the
#: default, 18.3 GB at 0.5; the encoder completes in the full run at 0.2).
LOW_WATERMARK_KEY = "compiled_allocator_low_watermark_ratio"
LOW_WATERMARK_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"

_applied: Optional[str] = None


def _declared(plan: Any, key: str):
    """{value: [(vendor/arch, device)]} that the vendor profiles of `plan`'s accelerators declare under `memory.<key>`
    (None for a profile declaring none)."""
    from neurobrix.core.config.loader import get_vendor_config
    seen = {}
    for name, alloc in (getattr(plan, "components", None) or {}).items():
        devices = [str(d) for d in (getattr(alloc, "devices", None) or []) if not str(d).startswith("cpu")]
        if not devices:
            continue
        vendor, arch = getattr(alloc, "vendor", None), getattr(alloc, "architecture", None)
        if not vendor or not arch:
            raise RuntimeError(
                f"ZERO FALLBACK: the plan's allocation of {name!r} names no vendor/architecture; "
                f"the compiled engine cannot read its allocator settings")
        value = (get_vendor_config(vendor, arch).get("memory", {}) or {}).get(key)
        seen.setdefault(value, []).extend((f"{vendor}/{arch}", d) for d in devices)
    return seen


def declared_fraction(plan: Any):
    """(fraction, the accelerator devices it bounds) the plan's vendor profiles declare, or None. One process, one
    pool per device type: two fractions in one plan are refused by name."""
    seen = _declared(plan, FRACTION_KEY)
    declared = {v: who for v, who in seen.items() if v is not None}
    if len(declared) > 1 or (declared and None in seen):
        raise RuntimeError(
            f"ZERO FALLBACK: this plan's vendor profiles declare different memory.{FRACTION_KEY}: "
            + "; ".join(f"{v} ({', '.join(sorted({w for w, _ in who}))})" for v, who in seen.items()))
    if not declared:
        return None
    value, who = next(iter(declared.items()))
    return float(value), sorted({d for _, d in who})


def configure_memory_fraction(plan: Any):
    """Bound each accelerator's torch pool at the declared fraction of its memory, through its backend's
    `set_per_process_memory_fraction`. Returns the fraction in force by the engine's doing, or None."""
    found = declared_fraction(plan)
    if found is None:
        return None
    fraction, devices = found
    operator = {k: os.environ[k] for k in FRACTION_OPERATOR_ENV if os.environ.get(k)}
    if operator:
        logging.getLogger(__name__).warning(
            "compiled allocator: the vendor profile declares a pool fraction of %s; the operator's %s wins", fraction,
            ", ".join(f"{k}={v!r}" for k, v in operator.items()))
        return None
    low = {v for v in _declared(plan, LOW_WATERMARK_KEY) if v is not None}
    if len(low) > 1:
        raise RuntimeError(f"ZERO FALLBACK: this plan's vendor profiles declare different memory.{LOW_WATERMARK_KEY}: {sorted(low)}")
    if low and not os.environ.get(LOW_WATERMARK_ENV):
        os.environ[LOW_WATERMARK_ENV] = str(next(iter(low)))     # before the allocator exists: torch reads it at creation
    import inspect
    import torch
    for dev in devices:
        d = torch.device(dev)
        backend = getattr(torch, d.type, None)
        fn = getattr(backend, "set_per_process_memory_fraction", None)
        if fn is None:
            raise RuntimeError(f"ZERO FALLBACK: memory.{FRACTION_KEY} is declared for {dev} and torch.{d.type} has no "
                               f"set_per_process_memory_fraction")
        if "device" in inspect.signature(fn).parameters:
            fn(fraction, device=d)
        else:
            fn(fraction)
    print(f"   [Allocator] compiled engine: torch pool bounded at {fraction} of the device's memory on "
          f"{', '.join(devices)} (vendor profile memory.{FRACTION_KEY})")
    return fraction


def declared_settings(plan: Any) -> Optional[str]:
    """The allocator settings the vendor profiles of `plan`'s accelerators declare, or None when
    they declare none. The setting is process-wide: two accelerators of one plan declaring two
    different settings is refused by name rather than one silently chosen."""
    from neurobrix.core.config.loader import get_vendor_config
    seen = {}
    for name, alloc in (getattr(plan, "components", None) or {}).items():
        devices = list(getattr(alloc, "devices", None) or [])
        if not devices or all(str(d).startswith("cpu") for d in devices):
            continue
        vendor, arch = getattr(alloc, "vendor", None), getattr(alloc, "architecture", None)
        if not vendor or not arch:
            raise RuntimeError(
                f"ZERO FALLBACK: the plan's allocation of {name!r} names no vendor/architecture; "
                f"the compiled engine cannot read its allocator settings")
        value = (get_vendor_config(vendor, arch).get("memory", {}) or {}).get(PROFILE_KEY)
        seen.setdefault(value, []).append(f"{vendor}/{arch}")
    declared = {v: who for v, who in seen.items() if v}
    if len(declared) > 1 or (declared and None in seen):
        raise RuntimeError(
            f"ZERO FALLBACK: the caching allocator is one per process and this plan's vendor "
            f"profiles declare different memory.{PROFILE_KEY}: "
            + "; ".join(f"{v or 'none'} ({', '.join(sorted(set(w)))})" for v, w in seen.items()))
    return next(iter(declared), None)


def configure_torch_allocator(plan: Any) -> Optional[str]:
    """Apply the declared allocator settings to torch's caching allocator. Returns the settings
    in force by the engine's doing, or None (none declared, or the operator's own)."""
    global _applied
    value = declared_settings(plan)
    if value is None:
        return None
    log = logging.getLogger(__name__)
    operator = {k: os.environ[k] for k in OPERATOR_ENV if os.environ.get(k)}
    if operator:
        log.warning("compiled allocator: the vendor profile declares %r; the operator's %s wins",
                    value, ", ".join(f"{k}={v!r}" for k, v in operator.items()))
        return None
    if _applied == value:
        return value
    import torch
    torch._C._accelerator_setAllocatorSettings(value)
    _applied = value
    print(f"   [Allocator] compiled engine: torch caching allocator {value} "
          f"(vendor profile memory.{PROFILE_KEY})")
    return value
