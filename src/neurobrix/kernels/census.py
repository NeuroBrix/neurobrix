"""The certification census — a shadow execution that records every kernel key.

Doctrine (2026-09-21): certification happens in three stages on every GPU we will ever
certify. CENSUS: the machine enumerates, for its own hardware profile, the kernel × shape ×
dtype keys the catalogue demands, reading the containers' graphs and never their weights.
CERTIFICATION: those keys are swept and oracle-proven on synthetic tensors. VERIFICATION: a
judged run per mode, served at zero miss — a miss is a census defect to fix here.

The key is computed at launch from what the dispatch hands the kernel, so reading op shapes
is not enough: the census derives keys the way the launcher does. This module runs the
engine's own Triton dispatch — the flows, the dtype engine, the wrappers, the autotuners —
over metadata-only tensors: allocations hand out addresses that address nothing, every
launch is recorded and skipped, every value read answers a neutral value, weights are never
opened (their shapes and dtypes come from the graph), and the plan is solved against the
profile's own capacity, not the room's. Two doors:

  NBX_KEY_RECORD=<path>   every key an autotuner forms, live or shadow, one per line as the
                          certifier reads it (`<kernel qualname>::<key tuple>`); the live
                          form is the proof target a shadow census is measured against.
  NBX_CENSUS=1            the shadow mode itself (with NBX_CENSUS_DEVICES=<n> for the
                          profile's card count).

Vendor-neutral: nothing here names CUDA, Metal or ROCm; the seams are the library's own.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Set

_RECORD_LOCK = threading.Lock()
_RECORDED: Set[str] = set()
_ACTIVE = {"census": False}
_SHADOW_PTR = [1 << 40]                 # addresses that address nothing, distinct and aligned


def key_line(tuned, key: tuple) -> Optional[str]:
    from neurobrix.triton import autotune_cache as _atc
    from neurobrix.kernels.autotune_certified import key_repr
    qual = _atc._qual_of(tuned) or getattr(getattr(tuned, "base_fn", None), "__name__", None)
    if not qual:
        return None
    return f"{qual}::{key_repr(tuple(key))}"


def record(tuned, key: tuple) -> None:
    """Called where the launch-time key is formed (kernels/ops/_configs.run_with_notice)."""
    path = os.environ.get("NBX_KEY_RECORD")
    if not path:
        return
    line = key_line(tuned, key)
    if line is None:
        return
    with _RECORD_LOCK:
        if line in _RECORDED:
            return
        _RECORDED.add(line)
        with open(path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()


def active() -> bool:
    return _ACTIVE["census"]


# --------------------------------------------------------------------------------------------
# The shadow mode: installed once per process by the CLI when NBX_CENSUS=1.
# --------------------------------------------------------------------------------------------

def _shadow_malloc(nbytes: int, dev_idx: Optional[int] = None) -> int:
    ptr = _SHADOW_PTR[0]
    _SHADOW_PTR[0] += (int(nbytes) + 255) // 256 * 256 + 256
    return ptr


def _noop(*_a, **_k):
    return None


def shadow_run(*_a, **_k):
    """What stands behind an autotuned kernel's `run` in the shadow: nothing."""
    return None


def _shadow_item_value(dtype):
    """The benign scalar a value-read (`.item()`) answers in shadow mode.

    In a census shadow no VALUE means anything — a `.item()` is only ever a
    guard or a token index. A guard's `all(isfinite(x))` (a bool tensor) must
    read healthy, or a loop's step-boundary NaN gate refuses the shadow (Sana
    at step 1, and PixArt/CogVideoX once Prism stopped mis-placing on the
    host); an integer token id answers 0 (the same shapes run for any id —
    UNVERIFIED where a model's eos id is 0); anything else answers 0.0.

    Read the dtype by NAME. NBXDtype is an IntEnum, so `str(<NBXDtype.bool_: 9>)`
    is "9", not "bool_": the previous `"bool" in str(dtype)` never matched, so
    every diffusion shadow died at step 1 on a finite gate reading falsy, and
    every integer read answered 0.0 (float) instead of 0 (int). Same defect
    class as the input-synth dtype read (ac10eddf)."""
    name = getattr(dtype, "name", str(dtype)).lower()
    if "bool" in name:
        return True                     # a guard's `all(isfinite(x))`: healthy
    return 0 if "int" in name else 0.0


def _shadow_params_for(executor, nbx_path, component) -> Dict[str, Any]:
    """The weights a component would load, as metadata-only tensors: the graph's param
    and buffer shapes, each in the dtype the loader would give it under the component's
    compute dtype (`weight_loader.stored_dtype_in_compute`, the loader's own rule).
    Nothing is opened."""
    from neurobrix.kernels.nbx_tensor import NBXTensor, parse_dtype
    from neurobrix.triton.weight_loader import stored_dtype_in_compute
    tensors = (executor._dag or {}).get("tensors", {})
    compute = parse_dtype(executor.dtype)
    # The device the executor loads on, read the way the loader reads it ("cuda:N").
    dev_s = str(getattr(executor, "device", "") or "")
    dev = int(dev_s.split(":", 1)[1]) if ":" in dev_s else 0
    out: Dict[str, Any] = {}
    for tid, spec in tensors.items():
        if not (tid.startswith("param::") or tid.startswith("buffer::")):
            continue
        name = spec.get("weight_name") or tid.split("::", 1)[1]
        shape = tuple(int(d) for d in spec.get("shape", []))
        if not spec.get("dtype"):
            raise RuntimeError(
                f"ZERO FALLBACK: {component}: the graph states no dtype for {tid} — a container "
                "defect; the census does not guess a weight's dtype.")
        dt = stored_dtype_in_compute(parse_dtype(spec["dtype"]), compute)
        out[name] = NBXTensor.empty(shape, dtype=dt, device=f"cuda:{dev}")
    return out


def install() -> None:
    """Turn this process into a shadow: no device memory, no launch, no value, no weight file."""
    if _ACTIVE["census"]:
        return
    from neurobrix.kernels import nbx_tensor as _nt
    from neurobrix.kernels import launcher as _launcher
    DA = _nt.DeviceAllocator
    # THE DOOR: a shadow is proven unable to touch a card only when no card can be seen.
    # With a device visible, a path this module does not cover would succeed on real
    # hardware in silence; behind the door it fails loudly (error 100). Refused at entry,
    # with the command that satisfies it. `--allow-visible-devices` does not exist: the
    # census has no honest use for a card.
    visible = DA.device_count()
    if visible > 0:
        raise RuntimeError(
            f"census shadow refused: {visible} device(s) visible. Run it with no card in "
            "sight — CUDA_VISIBLE_DEVICES= NBX_CENSUS=1 NBX_CENSUS_DEVICES=<the profile's "
            "count> neurobrix run ... --hardware <profile>")
    _ACTIVE["census"] = True
    n_dev = int(os.environ.get("NBX_CENSUS_DEVICES", "1") or 1)

    DA.malloc_cuda = staticmethod(_shadow_malloc)
    for name in ("free_cuda", "memset_cuda", "memcpy", "memcpy_async", "sync_device",
                 "set_device", "ensure_triton_device", "empty_cache_pool"):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(_noop))
    DA.device_count = staticmethod(lambda: n_dev)
    # The whole driver surface answers without a driver: syncs, streams, events, peer
    # access, host-pinned memory, device queries. A shadow that reaches one of these
    # through a path not listed here fails LOUDLY (error 100, no device) — never silently.
    _cur = {"dev": 0}
    DA.set_device = staticmethod(lambda device_id: _cur.__setitem__("dev", int(device_id)))
    DA.get_device = staticmethod(lambda: _cur["dev"])
    for name in ("device_synchronize", "stream_synchronize", "event_synchronize",
                 "record_event", "stream_wait_event", "destroy_stream", "destroy_event",
                 "reset_peak_memory", "print_alloc_stats", "_maybe_init_pool"):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(_noop))
    _handles = {"n": 0}
    def _handle(*_a, **_k):
        _handles["n"] += 1
        return _handles["n"]
    for name in ("create_stream", "create_event"):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(_handle))
    for name, value in (("clear_last_error", 0), ("event_elapsed_ms", 0.0),
                        ("can_access_peer", False), ("ensure_peer_access", False),
                        ("enable_peer_access", False), ("most_free_device", 0),
                        ("device_free_bytes", -1), ("device_total_bytes", -1),
                        ("mem_get_info", (0, 0)), ("visible_device_memory", []),
                        ("holds", False), ("empty_cache_pool", 0), ("_pool_flush", 0)):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(lambda *a, _v=value, **k: _v))
    if hasattr(DA, "malloc_host_pinned"):
        DA.malloc_host_pinned = staticmethod(_shadow_malloc)
    if hasattr(DA, "free_host_pinned"):
        DA.free_host_pinned = staticmethod(_noop)
    if hasattr(DA, "free_memory_mb"):
        # The plan is solved against the profile's own capacity: a census carries the machine's
        # plan, not the room's ambient (the ladder still rounds it onto its rungs).
        DA.free_memory_mb = staticmethod(lambda index=0: None)

    # Every launch is skipped: the launcher's entry answers None, and each autotuned
    # kernel's `run` — wrapped at its definition by `_configs._announce_first_sweep`,
    # AFTER this install — keeps the wrapper and runs `shadow_run` behind it. The key is
    # recorded by that ONE seam, the same door the live record uses; a shadow that never
    # reaches it records nothing, and says so by an empty file. Install before the
    # engine imports (the CLI does, in main()): a kernel defined earlier keeps its bench.
    _launcher.launch = lambda kernel, grid, *a, **k: None

    # Value reads answer ZERO — float or int — and a bool reads TRUE. Shapes are what the
    # census is made of; a value read is a guard or a token: a guard's `all(isfinite(.))`
    # must read healthy (Sana's loop refused the shadow at step 1 on a bool that read 0),
    # and a token id 0 runs the same shapes as any other — UNVERIFIED for a model whose
    # end-of-sequence id is 0: its decode would stop at step 1 and every later decode-step
    # key would be lost in silence. What verifies it: a census of such a model against its
    # replay set (none in the catalogue today; TinyLlama's eos is 2).
    T = _nt.NBXTensor

    def _item(self):
        return _shadow_item_value(self._dtype)

    def _numpy(self):
        import numpy as np
        return np.zeros(tuple(self._shape), dtype=np.float32)

    def _tolist(self):
        import numpy as np
        return np.zeros(tuple(self._shape), dtype=np.int64).tolist()

    T.item = _item
    T.numpy = _numpy
    if hasattr(T, "tolist"):
        T.tolist = _tolist

    # A diffusion scheduler binds its step index to the TIMESTEP'S VALUE (`_init_step_index`
    # reads `timestep.item()` and takes the nearest table entry); with every value read at
    # zero it would bind to the last entry and run off the table at the next step (Sana:
    # "index 21 is out of bounds for axis 0 with size 21"). In a shape-only pass the loop
    # counter IS the step index: the first bind answers 0 and the scheduler's own `+= 1`
    # walks the table from there.
    import importlib as _il
    import pkgutil as _pu
    from neurobrix.triton import scheduler as _sched_pkg

    def _init_step_index_shadow(self, timestep):
        self._step_index = 0

    for _mi in _pu.iter_modules(_sched_pkg.__path__):
        _mod = _il.import_module(f"{_sched_pkg.__name__}.{_mi.name}")
        for _name in dir(_mod):
            _cls = getattr(_mod, _name)
            if isinstance(_cls, type) and "_init_step_index" in vars(_cls):
                _cls._init_step_index = _init_step_index_shadow

    # Weights: shapes and dtypes from the graph, nothing opened.
    from neurobrix.core.runtime import graph_executor as _ge
    GE = _ge.GraphExecutor if hasattr(_ge, "GraphExecutor") else None
    if GE is not None:
        def _load_weights_triton(self, nbx_path, component, shard_map, only=None):
            self._weights = _shadow_params_for(self, nbx_path, component)
            return self._weights
        GE._load_weights_triton = _load_weights_triton
