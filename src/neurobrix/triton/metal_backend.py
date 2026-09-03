"""Triton-on-Metal backend detection and gate (Apple Silicon).

`--triton` / `--triton-sequential` on an Apple GPU require an out-of-tree
Triton backend that targets Metal. We do NOT bundle, auto-fetch or vendor
it — the same doctrine `cpu_backend.py` applies to `triton-cpu`, and for
the same reasons: auto-fetching a wheel is a supply-chain surface the user
did not consent to, and air-gapped installs fail with a confusing network
error instead of a clear one.

Why an external backend rather than our own (2026-09-03 decision, sourced
in ``docs/internal/metal_scoping_2026_09_03.md``):

* Upstream Triton has no Metal target and nobody is building one there —
  issue #4824 has sat open since 2024-09-28 with no maintainer engagement.
* Triton 3.7 (2026-07) shipped a plugin-extension system, so a new target
  is an out-of-tree backend loaded at runtime, **not a fork**. R25's
  "no internal fork" reading is satisfied by construction.
* `bledden/triton-msl` already implements that backend under MIT, aligned
  to Triton 3.7, lowering TTGIR to Metal Shading Language. It reports zero
  silent-wrong results across upstream's own `test_core.py`, and refuses
  what it cannot lower rather than computing something wrong — which is
  this engine's Zero-Fallback doctrine arrived at independently.

Our own 424 `@triton.jit` kernels are portable **as source** because R33
kept `torch` out of the triton tree; what is not portable is the CUDA
runtime binding in `kernels/nbx_tensor.py`. That is the one real port, and
it is tracked as the coupling point in the adoption plan.

R33 preserved — nothing here imports torch, even at the boundary.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import importlib.util
import os
import platform


# --- Known coverage gaps of the Metal backend -------------------------------
# Marker constants read by the dispatcher, so a future chantier flips them in
# ONE place when the upstream gap closes. Same pattern as the triton-cpu
# markers. Sourced from the backend's published refusal list, 2026-09-03;
# none of them is measured by us — the compile census in the adoption plan is
# what replaces a README with a fact.

# Batched 3-D matmul is not implemented upstream. This is the gap that blocks
# us: `baddbmm` is the largest single item in our own prefill split (6.2 s of
# 21 s), so attention prefill does not run without it. It is also the kernel
# class this project has spent the year on — closing it upstream is the
# contribution named in the adoption plan.
TRITON_METAL_BATCHED_MATMUL_BLOCKED: bool = True

# bf16 inputs are refused by the backend's FlashAttention path.
TRITON_METAL_BF16_ATTENTION_BLOCKED: bool = True

# Metal GPUs have NO double precision. `tl.float64` cannot be lowered at all,
# which reaches us through the device-scalar kernels in
# kernels/ops/{add,mul,div}.py that widen through f64 to stay bit-exact with
# the host path. On Apple those must take the host-sync route instead.
TRITON_METAL_FP64_UNAVAILABLE: bool = True


class TritonMetalNotInstalledError(ImportError):
    """Raised when `--triton` is invoked on an Apple GPU and no Metal Triton
    backend is installed.

    The message is deliberately actionable: what is missing, the install
    command, and the alternative that already works today.
    """


def is_apple_silicon() -> bool:
    """True on an Apple-Silicon Mac."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def triton_metal_available() -> bool:
    """Probe for a Metal Triton backend without importing it.

    Accepts either the standalone package or a backend registered into
    Triton's plugin system, because the plugin path (Triton 3.7+) is how a
    third-party target is expected to arrive and the package name is not
    guaranteed to be the import name.
    """
    for module in ("triton_msl", "triton.backends.metal"):
        try:
            if importlib.util.find_spec(module) is not None:
                return True
        except (ImportError, ValueError):
            continue
    # A plugin can also be pointed at by the upstream env var.
    plugins = os.environ.get("TRITON_PLUGIN_PATHS", "")
    return any("metal" in p.lower() or "msl" in p.lower()
               for p in plugins.split(os.pathsep) if p)


def ensure_triton_metal_or_raise() -> None:
    """Verify a Metal Triton backend is present before the triton path starts.

    Call sites mirror `ensure_triton_cpu_or_raise`: the `--triton` /
    `--triton-sequential` entry points, before any triton import that would
    otherwise produce a cryptic `triton.runtime.driver.active` error.

    A no-op off Apple Silicon.

    Raises:
        TritonMetalNotInstalledError: naming the install, the doctrine
        (we never auto-fetch), and the `--compiled` path that already runs
        on Apple GPUs through MPS today.
    """
    if not is_apple_silicon():
        return
    if triton_metal_available():
        return

    raise TritonMetalNotInstalledError(
        "The --triton engine needs a Triton backend that targets Metal, and "
        "none is installed.\n"
        "\n"
        "Upstream Triton has no Apple GPU target. An out-of-tree backend "
        "exists:\n"
        "    pip install triton-msl        (MIT; needs Triton 3.7+ built "
        "with TRITON_EXT_ENABLED=1)\n"
        "\n"
        "NeuroBrix does not install it for you: fetching a wheel at runtime "
        "is a\n"
        "supply-chain surface you did not consent to, and it fails "
        "confusingly on\n"
        "air-gapped machines.\n"
        "\n"
        "Working alternative today: drop --triton. The default compiled "
        "engine runs\n"
        "on Apple GPUs through PyTorch MPS with no extra install.\n"
        "\n"
        "Status and known gaps: docs/internal/metal_adoption_plan_2026_09_03.md"
    )
