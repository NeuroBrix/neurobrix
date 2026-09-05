"""Shared autotune configurations for Triton kernels.

V100 safety: _safe_num_stages() clamps num_stages to 2 on pre-Ampere GPUs.
V100 (sm_70) lacks cp.async — Triton's software pipelining with num_stages > 2
emits async copy instructions that cause CUDA_ERROR_MISALIGNED_ADDRESS on Volta.
"""

import inspect
import sys
import warnings
from typing import Dict, List, Optional

from pathlib import Path

import triton
from triton import next_power_of_2


# ---------------------------------------------------------------------------
# Autotune compatibility across Triton versions
# ---------------------------------------------------------------------------

_AUTOTUNE_DROPPED: set = set()


def nbx_autotune(*args, **kwargs):
    """`triton.autotune`, minus any keyword the installed Triton lacks.

    NeuroBrix passes `cache_results=True` so autotune results persist to
    disk instead of being re-measured every process. That keyword arrived
    in a later Triton than the one `pip install torch --index-url .../cu121`
    resolves (Triton 3.1 for torch 2.5), and passing it there raises

        TypeError: autotune() got an unexpected keyword argument 'cache_results'

    at the FIRST kernel launch — which made a clean install of NeuroBrix
    unable to run a single model on any GPU whose torch build ships an
    older Triton. That is what this shim exists to stop.

    The degradation is real but bounded, and it is announced rather than
    hidden: `cache_results` is a caching optimisation, so dropping it costs
    one autotune sweep per process and changes no result. Anything that
    would change a RESULT must never be dropped this way — it would be a
    silent fallback, which this engine does not do.
    """
    import triton

    try:
        supported = set(inspect.signature(triton.autotune).parameters)
    except (TypeError, ValueError):
        return triton.autotune(*args, **kwargs)

    unsupported = {k for k in kwargs if k not in supported}
    if unsupported:
        for name in sorted(unsupported - _AUTOTUNE_DROPPED):
            _AUTOTUNE_DROPPED.add(name)
            warnings.warn(
                f"Triton {getattr(triton, '__version__', '?')} does not support "
                f"@triton.autotune({name}=...); continuing without it. "
                f"Autotune results will be re-measured once per process instead "
                f"of persisted to disk (slower first launch, identical results). "
                f"Triton 3.6+ restores it.",
                RuntimeWarning,
                stacklevel=2,
            )
        kwargs = {k: v for k, v in kwargs.items() if k not in unsupported}

    decorator = triton.autotune(*args, **kwargs)

    def announce_then_decorate(fn):
        return _announce_first_sweep(decorator(fn))

    return announce_then_decorate


# Said once per process, the first time the autotuner actually measures.
_SWEEP_ANNOUNCED = [False]

_SWEEP_NOTICE = (
    "[neurobrix] First run for these tensor shapes on this machine: measuring "
    "kernel configurations.\n"
    "[neurobrix] This happens ONCE per shape per machine and is cached to disk "
    "(~/.triton/cache).\n"
    "[neurobrix] Later runs skip it entirely — measured 254.9 s -> 7.9 s on "
    "whisper-large-v3-turbo, 15.5 s -> 7.4 s on TinyLlama-1.1B.\n"
)


def _announce_first_sweep(tuned):
    """Say, once, that the engine is tuning rather than hung.

    A first `whisper-large-v3-turbo` transcription spends **247 seconds**
    autotuning before it produces a word, and said nothing at all while it did
    — measured 2026-09-03,
    `validation_outputs/audio_launch_census_2026_09_03/VERDICT.md`. The second
    run of the same command takes 7.9 s. A user whose first run looks hung for
    four minutes concludes the engine is broken, and they are not being
    unreasonable: nothing on screen distinguished it from a hang.

    The cost is not a defect — it buys ~12 % throughput per the autotune policy
    and it is paid once per shape per machine. Being silent about it was.

    The wrapper REMOVES ITSELF once it has spoken, restoring the class method,
    so the warm path (the overwhelming majority of launches) pays nothing for
    an announcement that has already happened.
    """
    cache = getattr(tuned, "cache", None)
    if cache is None:                 # not an Autotuner — nothing to watch
        return tuned
    original = tuned.run

    def run_with_notice(*args, **kwargs):
        if _SWEEP_ANNOUNCED[0]:
            # Someone else announced: drop the instance override and get out
            # of the hot path for good.
            try:
                del tuned.run
            except AttributeError:                        # pragma: no cover
                pass
            return original(*args, **kwargs)
        before = len(cache)
        result = original(*args, **kwargs)
        if len(cache) > before and not _SWEEP_ANNOUNCED[0]:
            _SWEEP_ANNOUNCED[0] = True
            print(_SWEEP_NOTICE, file=sys.stderr, end="")
        return result

    tuned.run = run_with_notice
    return tuned


def _safe_num_stages(n: int) -> int:
    """Clamp num_stages to the executing architecture's pipelining budget.

    The budget is a HARDWARE CAPABILITY and is read from
    `config/vendors/<vendor>/<arch>.yml` `pipelining.max_num_stages`
    (R23/R24) — never decided in code, never queried from the driver in a
    hot path. Volta's 2 is measured: sm_70 has no cp.async, and Triton's
    pipelining above 2 stages emits async copies that fault with
    CUDA_ERROR_MISALIGNED_ADDRESS in any kernel using tl.dot. Ampere and
    Hopper carry a cap above the widest config space, so it never binds.
    The CDNA profiles carry a conservative 2 pending first light.

    Fallback path — a bare process with no hardware profile (unit tests):
    the driver is the only signal available. Note that
    `get_device_capability()` on ROCm returns the *gfx* version
    (gfx90a -> (9, 0)), which is not an NVIDIA compute capability and must
    never be compared against 8 — doing so silently cleared unvalidated
    pipelining on every CDNA card. ROCm is clamped explicitly instead.
    """
    try:
        from neurobrix.kernels.wrappers import _arch_param, get_hardware_profile
        if get_hardware_profile() is not None:
            cap = _arch_param("pipelining", "max_num_stages", None)
            if cap:
                return min(n, int(cap))
    except Exception:
        pass

    # Ask the TRITON target, not torch. This module is imported by every
    # triton kernel (rmsnorm, softmax, conv2d, matmul, baddbmm,
    # depthwise_conv2d), so a module-level `import torch` here pulled torch
    # into the process on the branch that exists to contain none — the widest
    # torch surface under kernels/ (R33 inventory, 2026-09-03). The same
    # question is answerable from the Triton runtime, which is the documented
    # bootstrap mechanism already used by `matmul._detect_arch_configs`.
    try:
        from neurobrix.kernels.launcher import target as _nbx_target   # engine data, no driver probe (R33)
        arch = _nbx_target().arch
    except Exception:
        return n                       # no runtime to ask: change nothing
    if isinstance(arch, int):
        # NVIDIA: Triton reports compute capability x 10 (sm_70 -> 70).
        return n if arch >= 80 else min(n, 2)
    # AMD ("gfx90a") and any other string-arch target: conservative, since
    # cp.async does not exist there and their pipelining is unvalidated.
    return min(n, 2)


# ---------------------------------------------------------------------------
# SHARED-MEMORY BUDGET — the physical limit on which autotune tiles can run
# ---------------------------------------------------------------------------
#
# Every vendor profile has declared `memory.max_shared_memory_per_block` since
# the profiles were written, and until 2026-09-03 NOTHING read it. The autotune
# config space was instead chosen by an architecture NAME, with the reasoning
# that anything unrecognised should take "the Volta subset, which fits the
# smallest budget". That reasoning is false on Apple: Volta declares 96 KB and
# an Apple GPU has 32 KB, so a BM=64/BN=128/BK=64 tile at 3 stages wants 72 KB
# and cannot run there at all. The same silence sent every CDNA card into the
# Volta space by coincidence rather than by decision, which the arch-selection
# docstring already flagged as a first-light task.
#
# The budget is a hardware parameter, so it comes from the profile (R23/R24).
# Resolution is data, not a mapping table: each profile declares a
# `compute_capability` in exactly the form the Triton target reports it
# ("7.0" / "gfx90a" / "apple9"), so the right file is found by matching, and a
# new architecture is a new YAML rather than a new branch.


def smem_bytes_for_config(config, dtype_bytes: int = 2) -> int:
    """Shared memory one matmul-class tile needs, in bytes.

    A blocked matmul stages an `[BLOCK_M, BLOCK_K]` slab of A and a
    `[BLOCK_K, BLOCK_N]` slab of B per pipeline stage. That working set is what
    saturates and spills — the measured Phase 1.5 collapse to 98-145 ms on
    Volta was this quantity exceeding 96 KB.

    Returns 0 for a config that names no tile (element-wise spaces), which the
    filter reads as "no constraint to check".
    """
    kw = getattr(config, "kwargs", {}) or {}
    m, n, k = kw.get("BLOCK_M"), kw.get("BLOCK_N"), kw.get("BLOCK_K")
    if not (m and n and k):
        return 0
    stages = max(1, int(getattr(config, "num_stages", 1) or 1))
    return (m * k + k * n) * dtype_bytes * stages


def arch_smem_budget() -> Optional[int]:
    """`memory.max_shared_memory_per_block` for the executing hardware.

    Read from the vendor YAML, never from the driver — the driver is asked
    only WHICH profile applies, which is identification and not a hardware
    parameter. Returns None when no profile matches, and the caller must then
    leave the config space alone rather than guess a budget: filtering on an
    invented number would silently delete working configs.
    """
    try:
        from neurobrix.kernels.launcher import target as _nbx_target   # engine data, no driver probe (R33)
        arch = _nbx_target().arch
    except Exception:
        return None

    # NVIDIA reports capability x 10 (sm_70 -> 70); the profiles spell it
    # "7.0". AMD and Apple report the profile's string form directly.
    wanted = (f"{arch // 10}.{arch % 10}" if isinstance(arch, int)
              else str(arch).strip().lower())

    vendors = Path(__file__).resolve().parents[2] / "config" / "vendors"
    try:
        import yaml
    except ImportError:                                   # pragma: no cover
        return None

    exact, same_family = None, None
    for path in sorted(vendors.glob("*/*.yml")):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:                                 # pragma: no cover
            continue
        declared = str(cfg.get("compute_capability", "")).strip().lower()
        if not declared:
            continue
        budget = (cfg.get("memory") or {}).get("max_shared_memory_per_block")
        if not budget:
            continue
        if declared == wanted:
            exact = int(budget)
        elif (declared.split(".")[0] == wanted.split(".")[0]
                and "." in declared and "." in wanted):
            # Same NVIDIA capability MAJOR. Only three NVIDIA profiles exist
            # — 7.0, 8.0, 9.0 — so an exact match covers V100, A100 and H100
            # and nothing else: a T4 (7.5), an A10 or 3090 (8.6) and an L4 or
            # 4090 (8.9) would resolve to no profile and get no filtering at
            # all. Falling back to the major mirrors what Prism's runtime
            # detection already does (`_NVIDIA_ARCH_MAP`: "8" -> ampere), and
            # the two MUST agree — a card whose runtime profile is ampere
            # while its autotune space was chosen as if no profile existed is
            # a disagreement inside the engine about its own hardware.
            #
            # That map is not imported here on purpose: `autodetect` pulls
            # torch, and this module is imported by every Triton kernel (R33).
            # Matching majors reads the same fact out of the profiles instead
            # of copying the table.
            same_family = int(budget)
    return exact if exact is not None else same_family


def configs_within_smem_budget(configs, budget: Optional[int],
                               dtype_bytes: int = 2):
    """Drop the tiles that cannot physically fit in `budget` bytes.

    `budget=None` (no profile matched) returns the space untouched: removing
    configs on a guessed budget is worse than exploring a few that spill,
    because the autotuner measures spilling configs and rejects them, whereas
    a config that was never offered can never be chosen.

    The smallest tile is always kept even if it exceeds the budget, so the
    space is never emptied — an empty autotune list is an import-time crash,
    and a wrong-but-present config is a slow kernel.
    """
    if not budget:
        return configs
    fitting = [c for c in configs
               if smem_bytes_for_config(c, dtype_bytes) <= budget]
    if fitting:
        return fitting
    return [min(configs, key=lambda c: smem_bytes_for_config(c, dtype_bytes))]


def element_wise_configs() -> List[triton.Config]:
    """Autotune configs for element-wise (1D) kernels."""
    return [
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ]


def warps_configs() -> List[triton.Config]:
    """Autotune configs sweeping warp counts (for row-wise kernels)."""
    return [triton.Config({}, num_warps=2**i) for i in range(6)]


def batch_block_heuristic(args: Dict) -> int:
    """Heuristic for batch block size in softmax/norm kernels.

    For small feature dims (< 64) and large batch, processes multiple rows
    per program for efficiency.
    """
    return (min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128)
            if args['feat_dim'] < 64 else 1)


def reduction_configs() -> List[triton.Config]:
    """Autotune configs for row-wise reduction kernels (BLOCK_M x BLOCK_N)."""
    return [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512}, num_warps=4),
    ]


def matmul_configs() -> List[triton.Config]:
    """Autotune configs for matmul kernels (CUDA).

    num_stages clamped to 2 on V100 via _safe_num_stages().
    """
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(3), num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(5), num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(5), num_warps=2),
    ]
