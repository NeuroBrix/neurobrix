"""
Triton-CPU backend detection and gate.

NeuroBrix triton modes (`--triton`, `--triton-sequential`) on a CPU-only
profile (no visible GPU) require the upstream `triton-cpu` package from
``triton-lang/triton-cpu``. We do NOT bundle, auto-fetch, or vendor it.
When it is missing we raise a clean error naming the install command and
a viable alternative (`--compiled` runs on CPU without any extra
install).

Rationale (per project doctrine, P-PRISM-NEVER-REFUSE v2 / S3
2026-05-12):
- Security: auto-fetching a wheel at runtime is a supply-chain surface
  the user did not consent to.
- Trust: the user controls their environment; surprising side-effects
  on `neurobrix run` damage that trust.
- Air-gapped deployments: enterprise ML datacenters run on isolated
  networks where auto-fetch fails with a confusing network error.
- Alignment: llama.cpp, vLLM, transformers, torch itself do not
  auto-install optional backends.

Numerical coverage gaps that remain blocked by upstream issues are
tracked in `triton/triton_cpu_coverage_gaps.md`. The
``TRITON_CPU_FP16_UPSTREAM_BLOCKED`` marker below is read by the
dispatcher to decide whether to refuse a fp16 launch with a clear
message or fall through to a documented escalation path.

R34 (model-agnostic) preserved — nothing in this module is keyed on
model name. R33 (zero-torch in triton) preserved — this module does
not import torch, even at boundary.
"""

from __future__ import annotations

import os


# Marker constants — read by the triton-CPU dispatcher path to gate
# launches that are known to fail or drift on the current upstream.
# These exist so a future P-TRITON-CPU-FP16-UPSTREAM-FOLLOWUP can flip
# them in one place when the upstream issues close.
TRITON_CPU_FP16_UPSTREAM_BLOCKED: bool = True  # triton-cpu #147 (Dot3D), #144
TRITON_CPU_MASKED_BLOCKPTR_GEMM_BLOCKED: bool = True  # triton-cpu #222


class TritonCPUNotInstalledError(ImportError):
    """Raised when `--triton` is invoked on a CPU-only profile and the
    upstream `triton-cpu` package is missing.

    The message is intentionally actionable: install command, docs
    pointer, and a viable already-working alternative.
    """


def _is_cpu_only_profile() -> bool:
    """True when no GPU is visible (forces CPU triton path)."""
    # `CUDA_VISIBLE_DEVICES=""` is the standard way to hide GPUs.
    # If the user set a Prism profile name explicitly to a cpu profile
    # we can also detect it via NBX_HARDWARE_PROFILE, but the env-var
    # path is the source of truth for "no GPU on this run".
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible == "":
        return True
    # No NVIDIA driver loaded → also CPU-only.
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
        return False
    except OSError:
        return True


def _triton_cpu_available() -> bool:
    """Probe the upstream `triton-cpu` install without importing it."""
    try:
        # Upstream exposes either `triton_cpu` (older releases) or
        # registers itself under `triton.backends.cpu` (newer). We
        # accept either; both indicate the user installed the package.
        import importlib.util
        if importlib.util.find_spec("triton_cpu") is not None:
            return True
        # Newer integration path: triton with CPU backend registered.
        spec = importlib.util.find_spec("triton.backends.cpu")
        if spec is not None:
            return True
    except Exception:
        return False
    return False


def ensure_triton_cpu_or_raise() -> None:
    """Verify `triton-cpu` is installed when we're on a CPU-only profile.

    Call sites: the entry points of `--triton` / `--triton-sequential`
    on `RuntimeExecutor` and the CLI before any triton import that would
    otherwise produce a cryptic `triton.runtime.driver.active` error.

    Raises:
        TritonCPUNotInstalledError: with an actionable message naming
        the install command, docs path, and the `--compiled` fallback
        that already works on CPU without any extra install.
    """
    if not _is_cpu_only_profile():
        return  # GPU visible — standard triton path.
    if _triton_cpu_available():
        # Pre-set the env var so the triton runtime picks the CPU
        # backend at first driver query. Must happen BEFORE any
        # `import triton.runtime.driver` in this process.
        os.environ.setdefault("TRITON_CPU_BACKEND", "1")
        return
    raise TritonCPUNotInstalledError(
        "neurobrix run --triton (or --triton-sequential) on a CPU-only "
        "device requires the upstream triton-cpu package, which is NOT "
        "installed in this environment.\n"
        "\n"
        "Upstream does NOT publish a PyPI wheel today — install is "
        "build-from-source:\n"
        "    git clone https://github.com/triton-lang/triton-cpu.git\n"
        "    cd triton-cpu\n"
        "    pip install -r python/requirements.txt\n"
        "    pip install -e .\n"
        "(prerequisites: LLVM, CMake, Ninja, C++17 compiler. "
        "Expect ~30 min build + ~15 GB peak disk.)\n"
        "\n"
        "See docs/triton_cpu_install.md for full instructions and "
        "verification command.\n"
        "\n"
        "Alternative (no extra install required): run with --compiled. "
        "The PyTorch CPU backend is mature and is the recommended "
        "production path for CPU-only inference today. Numerical "
        "fp16 paths on triton-cpu have known upstream gaps "
        "(triton-cpu issues #147, #222); --compiled has no such gaps."
    )


def is_cpu_triton_active() -> bool:
    """True iff triton-cpu has been selected for this process.

    Set after `ensure_triton_cpu_or_raise()` succeeds on a CPU-only
    profile. Downstream code (DeviceAllocator, wrappers, kv_cache)
    uses this to branch between CUDA pointers and host pointers.
    """
    return os.environ.get("TRITON_CPU_BACKEND") == "1"


def get_install_instructions() -> str:
    """Return the install instructions string. Useful for CLI `--help`
    text and error consolidation. Single source of truth for the
    message so it cannot drift between sites."""
    return (
        "pip install triton-cpu   "
        "(see docs/triton_cpu_install.md for build-from-source)"
    )
