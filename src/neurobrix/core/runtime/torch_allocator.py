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
`memory.compiled_allocator_settings` (config/vendors/<vendor>/<arch>.yml). A profile that declares
none runs torch's own default.

Applied once per process, before the executors load a weight (`RuntimeExecutor.setup`), under the
compiled engine only: the Triton engine never imports torch (R33) and allocates through its own
runtime. An operator's explicit `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` is theirs and wins;
the engine says so instead of overriding it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

#: The allocator environment variables torch reads (the second is its deprecated spelling).
OPERATOR_ENV = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")

#: The vendor-profile key, under `memory:`.
PROFILE_KEY = "compiled_allocator_settings"

_applied: Optional[str] = None


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
