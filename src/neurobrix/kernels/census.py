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
from typing import List, Any, Dict, Optional, Set
import logging
_log = logging.getLogger(__name__)

_RECORD_LOCK = threading.Lock()
_RECORDED: Set[str] = set()
_RECORDED_OPS: Set[tuple] = set()      # (op uid, key line) already written to <record>.ops
_OP: List[Optional[str]] = [None]      # the graph op whose dispatch is forming keys now (set_op)
_RECORDING: List[Optional[bool]] = [None]


def recording() -> bool:
    """Whether this process records the keys it forms (NBX_KEY_RECORD) — read once: the dispatch
    loops ask it per op, so it must cost a list read, not an environment lookup."""
    if _RECORDING[0] is None:
        _RECORDING[0] = bool(os.environ.get("NBX_KEY_RECORD"))
    return _RECORDING[0]


def set_op(op_uid: Optional[str]) -> None:
    """The dispatch loops name the graph op they are about to run, so every key it forms is recorded
    with the op that formed it (`<record>.ops`, one `op<TAB>key line` pair per line) — the census
    table's `op` column. Called only while recording."""
    _OP[0] = op_uid
_OBSERVERS: List[Set[str]] = []         # key lines formed during a `walk_extent` run, dedup or not
_ACTIVE = {"census": False}


def key_line(tuned, key: tuple) -> Optional[str]:
    from nbxtensor.ops import autotune_cache as _atc
    from nbxtensor.ops.autotune_certified import key_repr
    qual = _atc._qual_of(tuned) or getattr(getattr(tuned, "base_fn", None), "__name__", None)
    if not qual:
        return None
    return f"{qual}::{key_repr(tuple(key))}"


from nbxtensor.ops.autotune_certified import say_once as _say_once  # noqa: E402  the library's once-said line


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
    if line is not None:
        from nbxtensor.ops.autotune_certified import degenerate_extent
        bad = degenerate_extent(line.split("::", 1)[0], tuple(key))
        if bad is not None:
            # An extent of zero names a launch that cannot exist and a tensor the certifier
            # cannot synthesise; the shadow reaches them at the bottom of a value-derived
            # extent (a four-token speech), and one is already in the served directory.
            _say_once(f"[census] key with a zero extent refused ({bad}): {line}")
            return
    if line is None:
        return
    with _RECORD_LOCK:
        for obs in _OBSERVERS:
            obs.add(line)
        op = _OP[0]
        if op is not None and (op, line) not in _RECORDED_OPS:
            _RECORDED_OPS.add((op, line))
            with open(path + ".ops", "a") as fh:
                fh.write(f"{op}\t{line}\n")
                fh.flush()
        if line in _RECORDED:
            return
        _RECORDED.add(line)
        with open(path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()


def pace(length: int) -> int:
    """How many extra positions a request loop may skip after `length` under the shadow: up to
    the last position of the current bucket, so the next step lands on a new bucketed key.

    A census costs shapes, and under the bucketed keys a decode's shapes change only at the
    bucket tops of the context (or cache) length: the steps in between produce the keys the
    step before them produced. Walking every one of them cost chatterbox 390 steps for the
    keys of ~110 lengths (9 min on one thread, 2026-09-21). A flow calls this with the length
    it has and appends that many copies of the token it sampled: the keys of every bucket are
    still met once, and the walk is enumerated by the key classes it produces. 0 outside the
    shadow — a live run never skips a position."""
    if not _ACTIVE["census"] or length < 1:
        return 0
    try:
        from nbxtensor.ops.autotune_bucket import bucket_of
        nxt = int(length) + 1
        top = max(int(bucket_of(dim, nxt)) for dim in ("M", "N", "K"))
        return max(0, top - nxt)
    except Exception:  # noqa: BLE001 — no ladder: every position is its own key, nothing to skip
        return 0


def walks_extents() -> bool:
    """Whether this shadow enumerates the classes of a value-derived extent (the census tool
    sets `NBX_CENSUS_EXTENTS=1` on the rung it walks them at)."""
    return _ACTIVE["census"] and os.environ.get("NBX_CENSUS_EXTENTS", "") not in ("", "0")


def extent_ids(ids: list, n: int) -> list:
    """`ids` cut or padded to `n` positions (the last id repeated: a shadow has no values)."""
    ids = list(ids)
    if not ids:
        ids = [0]
    return (ids + [ids[-1]] * max(0, n - len(ids)))[:n]


def bisect_extent(lo: int, hi: int, at) -> None:
    """Every key class of an extent in [lo, hi], met once: `at(n)` answers the key set at n (and
    remembers it); between two extents whose sets differ, the midpoint, until the differing pair
    is adjacent. Keys are monotone step functions of the extent (a bucket top, an exact extent),
    so two extents with one set bracket a range with that set. The ONE bisection: the shadow's
    `walk_extent` runs the flow at n; the derived census derives the keys at n."""
    at(lo)          # both ends, always: a range of one or two extents has no midpoint to reach them
    at(hi)
    stack = [(lo, hi)]
    while stack:
        a, b = stack.pop()
        if b - a <= 1 or at(a) == at(b):
            continue
        m = (a + b) // 2
        at(m)
        stack.append((m, b))
        stack.append((a, m))


def walk_extent(lo: int, hi: int, run, name: str = ""):
    """Run `run(n)` at every KEY CLASS of an extent the flow learns only from values — the
    number of speech tokens a vocoder receives, the frames a codec decodes — instead of at the
    one value the shadow happened to produce: `run(lo)` and `run(hi)`, then between any two
    extents whose recorded key sets differ, the midpoint, until the differing pair is
    adjacent. Every kernel's key is a monotone step function of the extent (a bucket top, an
    exact extent), so two extents with one key set bracket a range with that key set, and the
    walk meets every class once at the cost of one run per class and a log of the class's
    width. Returns the last run's result. Said in clear: classes met, runs, wall."""
    import time as _t
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi < lo:
        lo, hi = hi, lo
    seen = {}
    last = [None]
    t0 = _t.perf_counter()

    def at(n):
        if n not in seen:
            obs: Set[str] = set()
            _OBSERVERS.append(obs)
            try:
                last[0] = run(n)
                seen[n] = frozenset(obs)
            except Exception as exc:  # noqa: BLE001 — an extent the graph refuses is its own class, said
                head = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
                _say_once(f"[census] extent {name or 'walk'} refused at {n}: {head}")
                seen[n] = frozenset({f"<refused> {head}"})
            finally:
                _OBSERVERS.remove(obs)
        return seen[n]

    bisect_extent(lo, hi, at)
    refused = sorted(n for n, v in seen.items() if any(k.startswith("<refused>") for k in v))
    if len(refused) == len(seen):
        # Every extent refused: the walk recorded nothing and the model would read as censused
        # (instrumentation that lies by construction). The run fails with the graph's own words.
        raise RuntimeError(f"census extent walk {name or ''} {lo}..{hi}: every extent refused; "
                           f"first: {sorted(seen[refused[0]])[0]}")
    keyed = {v for n, v in seen.items() if v and n not in refused}
    if not keyed:
        # Every extent the graph took launched nothing: a stage that never ran at any length (the
        # flow's own filter kept it from running) would otherwise print a walk and certify no key.
        raise RuntimeError(f"census extent walk {name or ''} {lo}..{hi}: no extent recorded a key "
                           f"in {len(seen)} run(s) — the stage never ran")
    classes = len(keyed)
    _log.debug("%s", f"[census] extent {name or 'walk'} {lo}..{hi}: {classes} key class(es) in {len(seen)} run(s), "
                     f"{_t.perf_counter() - t0:.1f} s" + (f"; refused at {len(refused)} extent(s) up to {refused[-1]}" if refused else ""))
    return last[0]


def active() -> bool:
    return _ACTIVE["census"]


# --------------------------------------------------------------------------------------------
# The shadow mode: installed once per process by the CLI when NBX_CENSUS=1.
# --------------------------------------------------------------------------------------------

def _shadow_params_for(executor, nbx_path, component) -> Dict[str, Any]:
    """The weights a component would load, as metadata-only tensors: the graph's param
    and buffer shapes, each in the dtype the loader would give it under the component's
    compute dtype (`weight_loader.stored_dtype_in_compute`, the loader's own rule).
    Nothing is opened."""
    from nbxtensor.ops import launch_keys as _lk
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    from neurobrix.triton.weight_loader import stored_dtype_in_compute
    tensors = (executor._dag or {}).get("tensors", {})
    # The convolution weights the loader stores KRSC on this arch (`conv.weight_layout`): the
    # shadow binds the view the run binds, the logical (K, C, R, S) over channels-innermost memory.
    krsc = _lk.krsc_conv_weights(executor._dag or {}, _lk.conv_weight_layout())
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
        # The loader reads a weight's shape from its FILE. A weight whose aten::t was eliminated
        # carries the transposed shape in the graph (`pretransposed`, stamped by the compile), and
        # the bind transposes the loaded tensor itself; shaped from the stamped metadata, the
        # shadow's weight was transposed twice on every pass after the first compile.
        if spec.get("pretransposed") and len(shape) == 2:
            shape = (shape[1], shape[0])
        if not spec.get("dtype"):
            raise RuntimeError(
                f"ZERO FALLBACK: {component}: the graph states no dtype for {tid} — a container "
                "defect; the census does not guess a weight's dtype.")
        dt = stored_dtype_in_compute(parse_dtype(spec["dtype"]), compute)
        if tid in krsc:
            K, Cg, R, S = shape
            out[name] = NBXTensor.empty((K, R, S, Cg), dtype=dt, device=f"cuda:{dev}").permute(0, 3, 1, 2)
        else:
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
    from nbxtensor.memory import shadow as _shadow
    # THE DOOR: the library's shadow refuses while a device is visible (a path it does not cover would succeed on real
    # hardware in silence). `--allow-visible-devices` does not exist: the census has no honest use for a card.
    try:
        _shadow.install_shadow(int(os.environ.get("NBX_CENSUS_DEVICES", "1") or 1))
    except _shadow.DevicesVisible as e:
        raise RuntimeError(
            f"census shadow refused: {e}. Run it with no card in sight — CUDA_VISIBLE_DEVICES= NBX_CENSUS=1 "
            "NBX_CENSUS_DEVICES=<the profile's count> neurobrix run ... --hardware <profile>") from None
    _ACTIVE["census"] = True

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
                self.device = f"cuda:{_shadow.shadow_device()}"
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
    def _multinomial_zero(probs, num_samples=1):
        # A draw from probabilities the shadow cannot make (NaN after the filters on zero
        # logits — chatterbox, 2026-09-21): token 0, the same shapes as any other.
        import numpy as _np
        return _np.zeros((num_samples,), dtype=_np.int64)
    _noop = lambda *_a, **_k: None  # noqa: E731
    for _name, _rep in (("device_sync", _noop), ("device_empty_cache", _noop), ("device_seed", _noop),
                        ("device_multinomial", _multinomial_zero)):
        _orig = getattr(_du, _name, None)
        if _orig is None:
            continue
        setattr(_du, _name, _rep)
        for _mod in list(_sys.modules.values()):
            if _mod is not None and getattr(_mod, _name, None) is _orig:
                setattr(_mod, _name, _rep)
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
        from neurobrix.core.prism.loader import profile_path
        path = profile_path(hardware)
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
        # A brand whose capability is not a CUDA-style number (Apple: the arch is a DEVICE
        # NAME) still needs its vendor profile bound, and used to get nothing here — the
        # early return was written for the target, and the profile was left to the driver
        # that the door has just silenced. Measured 2026-09-22 on the first bucketed Apple
        # census: 825 of 2 955 harvested keys carried a first dimension off the ladder
        # (matmul M_BUCKET 8 664 where the launcher keys 8 704, addmm 158 400 where it keys
        # 163 840) — the exact defect this function was written to close, still open for
        # every non-CUDA brand. The profile is resolved WITHOUT a target
        # (`vendor_profile_for_arch`), which is why that door exists.
        from nbxtensor.backend.device_names import arch_string
        # the capability when it NAMES the compile target (Intel: "lnl", its device file's), else the model's arch string
        model = cc.lower() if cc and not cc.replace(".", "").isdigit() else arch_string(dev.get("model"))
        if not model:
            raise RuntimeError(
                "census: the hardware profile's device names no model, so no vendor profile "
                "can be resolved for it; a census under no vendor profile is a census of "
                "nothing (the ladder goes unread and every key is composed exact)")
        from nbxtensor.backend import device_files
        device_files.forget()
        if not device_files.for_arch(model):
            raise RuntimeError(
                f"census: no vendor profile matches the hardware profile's device {model!r}; "
                f"refusing rather than recording a catalogue of exact keys a bucketed "
                f"launcher will never ask for")
        return
    major, minor = (cc.split(".") + ["0"])[:2]
    from triton.backends.compiler import GPUTarget
    from nbxtensor.backend import launcher as _launcher
    _launcher.bind_target(GPUTarget("cuda" if brand == "nvidia" else "hip", int(major) * 10 + int(minor), 32 if brand == "nvidia" else 64))
    from nbxtensor.backend import device_files
    device_files.forget()     # a profile resolved empty before the bind is read again
