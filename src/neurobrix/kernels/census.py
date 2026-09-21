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


_SAID: Set[str] = set()


def _say_once(msg: str) -> None:
    if msg not in _SAID:
        _SAID.add(msg)
        print(msg, flush=True)


def record(tuned, key: tuple) -> None:
    """Called where the launch-time key is formed (kernels/ops/_configs.run_with_notice)."""
    path = os.environ.get("NBX_KEY_RECORD")
    if not path:
        return
    if any(isinstance(k, int) and not isinstance(k, bool) and k < 0 for k in key):
        # A negative extent is a shadow artefact, never a request (Wan 2.2's image encoder
        # reached a batch of -2 after a frame expression went negative, 2026-09-21); such a key
        # is refused here, said once, and the run's failure names the op.
        _say_once(f"[census] key with a negative extent refused: {key_line(tuned, key)}")
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


def install(hardware: Optional[str] = None, hardware_profile: Optional[dict] = None) -> None:
    """Turn this process into a shadow: no device memory, no launch, no value, no weight file.

    `hardware` names the hardware profile the census is taken for (the run's `--hardware`);
    `hardware_profile` is its loaded YAML (tests). One of them is how the shadow learns which
    VENDOR profile its keys are composed under — see `_bind_target`.
    """
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

    # A value born on the HOST is known to the shadow: `from_numpy` keeps the array it was
    # given beside the tensor, and every host read answers from it. The flows compose their
    # requests from such tensors (a frame count, a token id list, a timestep table); a shadow
    # answering ones there built Wan 2.2's image encoder for a negative frame count
    # (2026-09-21). Values computed on the device stay unknown and answer as below.
    _from_numpy = T.from_numpy

    def _from_numpy_shadow(arr, dtype=None):
        t = _from_numpy(arr, dtype)
        try:
            import numpy as np
            t._shadow_host = np.array(arr, copy=True)
        except Exception:  # noqa: BLE001 — a host copy that cannot be kept is a device value
            pass
        return t
    T.from_numpy = staticmethod(_from_numpy_shadow)

    def _item(self):
        h = getattr(self, "_shadow_host", None)
        if h is not None and h.size == 1:
            v = h.reshape(-1)[0]
            return bool(v) if "bool" in str(h.dtype) else (int(v) if "int" in str(h.dtype) else float(v))
        d = str(self._dtype).lower()
        if "bool" in d:
            return True                 # a guard's `all(isfinite(x))`: the shadow is healthy
        # An integer read answers ONE, not zero: a read that is a SIZE (a duration, a frame
        # count, a length derived from data) shaped an empty tensor at zero — Kokoro's
        # index_select divided by zero, Allegro's group norm met (0, …), Wan 2.2's div met a
        # negative extent (2026-09-21) — while a token id 1 runs the same shapes as 0.
        return 1 if "int" in d else 0.0

    def _numpy(self):
        h = getattr(self, "_shadow_host", None)
        if h is not None and tuple(h.shape) == tuple(self._shape):
            return h
        # Host reads of an integer tensor answer ONES for the same reason `_item` does: a
        # zero read as a count shaped an empty tensor (Allegro-TI2V's group norm met batch 0
        # after a frame count read 0, 2026-09-21). Float reads stay zero.
        import numpy as np
        if "int" in str(self._dtype).lower():
            return np.ones(tuple(self._shape), dtype=np.int64)
        return np.zeros(tuple(self._shape), dtype=np.float32)

    def _tolist(self):
        h = getattr(self, "_shadow_host", None)
        if h is not None and tuple(h.shape) == tuple(self._shape):
            return h.tolist()
        import numpy as np
        return np.ones(tuple(self._shape), dtype=np.int64).tolist()

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
            # A CPU-staged lazy component's plan device is a staging location; the live
            # loader resolves the GPU and persists it on the executor. The shadow does the
            # same, or the strategy refuses "no GPU execution device" (CogVideoX on the
            # 16 GB profile, 2026-09-21).
            dev_s = str(getattr(self, "device", "") or "")
            if not dev_s.startswith(("cuda", "hip", "mps")):
                self.device = f"cuda:{_cur['dev']}"
            self._weights = _shadow_params_for(self, nbx_path, component)
            return self._weights
        GE._load_weights_triton = _load_weights_triton

    # The torch-side device utilities a shared flow path reaches (orpheus: device_sync →
    # torch.cuda.synchronize → "No CUDA GPUs are available"): no-ops in the shadow.
    # Rebound in every module already loaded that imported them BY NAME (the memory manager,
    # the strategies, the serving engine import before the shadow installs — the CLI's own
    # imports run first): orpheus's cleanup reached the original through the manager's name.
    import sys as _sys
    from neurobrix.core import device_utils as _du
    for _name in ("device_sync", "device_empty_cache", "device_seed"):
        _orig = getattr(_du, _name, None)
        if _orig is None:
            continue
        setattr(_du, _name, _noop)
        for _mod in list(_sys.modules.values()):
            if _mod is not None and getattr(_mod, _name, None) is _orig:
                setattr(_mod, _name, _noop)
    # A sampler draws from probabilities the shadow cannot make (zero logits, NaN after the
    # filters — openaudio, 2026-09-21): under the shadow every draw is token 0.
    try:
        from neurobrix.triton.flow import dual_ar as _dual_ar
        if hasattr(_dual_ar, "_sample_token_np"):
            _dual_ar._sample_token_np = lambda *a, **k: 0
    except Exception:  # noqa: BLE001 — a tree without that flow has nothing to shadow
        pass

    _bind_target(hardware, hardware_profile)


def _bind_target(hardware: Optional[str], profile: Optional[dict]) -> None:
    """The shadow's launcher target, from the hardware profile the census names.

    Behind the door (`CUDA_VISIBLE_DEVICES=`) no driver answers which vendor profile applies:
    `arch_smem_budget` resolved EMPTY, the bucket ladder went unread and every recorded key was
    composed in the exact form (2026-09-21: the catalogue censuses recorded matmul M = 226 and
    3 136 where the served launcher keys 240 and 3 200), the SMEM budget and the config spaces
    unread with it. The hardware profile names its device's brand and compute capability; the
    target is bound from them, so the keys a census records are the keys the launcher forms when
    it serves. A profile that names no device is refused: a census under no vendor profile is a
    census of nothing. A device whose capability is not a number (Apple: the arch is a device
    name) keeps its own driver's answer.
    """
    if profile is None and hardware:
        import yaml
        from neurobrix.core.prism.loader import HARDWARE_DIR
        path = HARDWARE_DIR / f"{hardware}.yml"
        if not path.exists():
            raise RuntimeError(f"census: the hardware profile {hardware!r} is not at {path}; the shadow cannot choose a vendor profile")
        profile = yaml.safe_load(path.read_text())
    if not profile:
        return
    devices = profile.get("devices") or []
    if not devices:
        raise RuntimeError("census: the hardware profile names no device; the shadow cannot choose a vendor profile")
    dev = devices[0]
    brand = str(dev.get("brand") or "").strip().lower()
    cc = str(dev.get("compute_capability") or "").strip()
    if brand not in ("nvidia", "amd") or not cc.replace(".", "").isdigit():
        return
    major, minor = (cc.split(".") + ["0"])[:2]
    from triton.backends.compiler import GPUTarget
    from neurobrix.kernels import launcher as _launcher
    _launcher._TARGET = GPUTarget("cuda" if brand == "nvidia" else "hip", int(major) * 10 + int(minor), 32 if brand == "nvidia" else 64)
    from neurobrix.kernels.ops import _configs
    _configs._ACTIVE_PROFILE.clear()     # a profile resolved empty before the bind is read again
