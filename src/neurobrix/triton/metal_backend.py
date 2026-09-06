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

Our own `@triton.jit` kernels are portable **as source** because R33 kept
`torch` out of the triton tree; what is not portable is the CUDA runtime
binding in `kernels/nbx_tensor.py`. That is the one real port, and it is
tracked as the coupling point in the adoption plan.

The corpus is **280 kernels across 160 modules**, counted by parsing
`kernels/ops/` on 2026-09-05. The 424/161 figure carried by the adoption
plan and the scoping study is a `grep` count: 144 of those occurrences are
module docstrings saying "pure `@triton.jit` kernel", and one is a comment
saying a helper is *never* `@triton.jit`.

R33 preserved — nothing here imports torch, even at the boundary.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess


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
#
# MEASURED 2026-09-05 on an M4 Pro, and the shape is narrower and wider than
# the README said (validation_outputs/metal_first_light_2026_09_05/):
#   * narrower — plain `matmul.py::matmul_kernel` LOWERS cleanly. It is not
#     "matmul on Metal" that fails.
#   * wider — the refusal is the fused-epilogue template declining a `tt.dot`
#     inside an `scf.for` K-loop that carries a trailing compute epilogue, so
#     it takes `matmul.py::addmm_kernel` as well as `baddbmm_op.py`. addmm was
#     on nobody's gap list.
# Both refuse LOUDLY (`MetalNonRecoverableError`), which is the contract
# working.
TRITON_METAL_BATCHED_MATMUL_BLOCKED: bool = True

# bf16 inputs are refused by the backend's FlashAttention path.
TRITON_METAL_BF16_ATTENTION_BLOCKED: bool = True

# Metal GPUs have NO double precision. The prediction written here was that
# `tl.float64` "cannot be lowered at all" and that the device-scalar kernels
# in kernels/ops/{add,mul,div}.py — which widen through f64 to stay bit-exact
# with the host path — would therefore be refused.
#
# MEASURED 2026-09-05, and the prediction was wrong in the direction that
# matters. All three kernels compile through the MSL stage with NO refusal.
# The f64 survives Triton's own middle end intact (`arith.extf f32 to f64`,
# `arith.mulf f64`, `arith.truncf f64 to f32` are all present in the TTGIR)
# and it is the backend's MSL lowering that drops it, emitting
# `float val = static_cast<float>(s_ptr[0]); float r = val * alpha;`.
# Silent narrowing, not a loud refusal.
#
# For these three kernels that costs nothing: a product of two f32 values is
# exactly representable in f64, so widening then narrowing round-trips. Checked
# rather than argued — 20,000,000 random pairs plus subnormals and overflow
# cases, every one bit-identical. `mul`/`div` are a bare `.to(f64).to(f32)`
# round trip and are no-ops by inspection.
#
# The hardware fact stands and so does this flag. What changed is the failure
# MODE to expect: any FUTURE f64 expression with more than one rounding —
# an accumulation, a division, a sum of three terms — would be narrowed just
# as silently, and there would be no refusal to catch it.
TRITON_METAL_FP64_UNAVAILABLE: bool = True


# The Metal backend's compile pipeline ends in `xcrun metal` + `xcrun metallib`
# (MSL -> AIR -> metallib), so it needs Apple's OFFLINE shader compiler. That
# compiler is not part of the Command Line Tools: on macOS 26 / Xcode 26 it is
# an on-demand Xcode component installed with
# `sudo xcodebuild -downloadComponent MetalToolchain`, and `xcodebuild` itself
# refuses to run under a Command-Line-Tools-only developer directory.
#
# MEASURED 2026-09-05 on an M4 Pro with CLT and no Xcode:
#   * the Metal FRAMEWORK compiles MSL at runtime perfectly well —
#     `newLibraryWithSource:` builds a pipeline that dispatches and returns
#     the right numbers. So this is NOT a platform limitation.
#   * triton-msl's driver `is_active()` probes `xcrun --find metal` and
#     returns False without it, so Triton reports ZERO active drivers and
#     cannot name a target at all. Nothing compiles and nothing runs.
#   * compilation with an EXPLICIT target still reaches the `msl` stage and
#     fails only at `metallib` — which is what let the compile census run.
TRITON_METAL_NEEDS_OFFLINE_SHADER_COMPILER: bool = True


class TritonMetalNotInstalledError(ImportError):
    """Raised when `--triton` is invoked on an Apple GPU and no Metal Triton
    backend is installed.

    The message is deliberately actionable: what is missing, the install
    command, and the alternative that already works today.
    """


class TritonMetalShaderCompilerMissingError(TritonMetalNotInstalledError):
    """Raised when the Metal backend IS installed but Apple's offline shader
    compiler is not, so nothing it compiles can be built.

    A subclass, so every existing call site that catches
    `TritonMetalNotInstalledError` keeps working while the message stays
    specific. It is a separate condition because the remedy is completely
    different — a ~700 MB Xcode component, not a pip install — and because
    a package check alone reports "ready" on a machine where nothing runs.
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


def metal_shader_compiler_available() -> bool:
    """True when Apple's offline shader compiler (`xcrun metal`) can be run.

    The SAME probe the backend's own driver uses for `is_active()`, so this
    answers the question that actually decides whether anything runs, rather
    than a question that merely correlates with it.

    Why `xcrun --find metal` and not `shutil.which("metal")`: the compiler is
    not on `PATH`, it is resolved by `xcrun` inside the active developer
    directory. And why not trust `xcode-select -p`: Command Line Tools give a
    valid developer directory and still have no `metal` in it, which is
    exactly the machine this was written on.

    False on any non-Darwin host, and False rather than raising if `xcrun` is
    missing entirely — a probe that takes the run down is not a probe.
    """
    if platform.system() != "Darwin":
        return False
    if shutil.which("xcrun") is None:
        return False
    try:
        return subprocess.run(
            ["xcrun", "--find", "metal"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=20,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_triton_metal_or_raise() -> None:
    """Verify a Metal Triton backend is present before the triton path starts.

    Call sites mirror `ensure_triton_cpu_or_raise`: the `--triton` /
    `--triton-sequential` entry points, before any triton import that would
    otherwise produce a cryptic `triton.runtime.driver.active` error.

    A no-op off Apple Silicon.

    Two conditions, not one. The package being importable was the whole gate
    until 2026-09-05, when the first Apple machine showed that it says "ready"
    on a Mac where nothing can run: with Command Line Tools but no Metal
    Toolchain the backend imports fine, this gate passed, and the run then
    died several steps later inside Triton's driver with "0 active drivers".
    That is precisely the `cpu_backend.py` defect this file was written to
    stop, reappearing one layer down.

    Raises:
        TritonMetalNotInstalledError: no Metal backend at all — naming the
        install, the doctrine (we never auto-fetch), and the `--compiled`
        path that already runs on Apple GPUs through MPS today.
        TritonMetalShaderCompilerMissingError: the backend is installed but
        Apple's offline shader compiler is not, so it can compile nothing.
    """
    if not is_apple_silicon():
        return

    if not triton_metal_available():
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

    if not metal_shader_compiler_available():
        raise TritonMetalShaderCompilerMissingError(
            "A Triton Metal backend is installed, but Apple's offline shader "
            "compiler is not,\n"
            "so nothing it compiles can be built and the Metal target will "
            "not even activate.\n"
            "\n"
            "Measured on an M4 Pro, 2026-09-05: without `xcrun metal` the "
            "backend's driver\n"
            "reports itself inactive, Triton then finds ZERO active drivers, "
            "and the run dies\n"
            "with 'Backend device metal is not active' well after this "
            "point. That late,\n"
            "cryptic death is the exact failure this gate exists to replace.\n"
            "\n"
            "The compiler is NOT part of the Command Line Tools. On macOS 26 "
            "/ Xcode 26 it is\n"
            "a separate on-demand component:\n"
            "    sudo xcodebuild -downloadComponent MetalToolchain   "
            "(~700 MB)\n"
            "and `xcodebuild` requires a full Xcode: under a Command-Line-"
            "Tools-only developer\n"
            "directory it refuses to run. Apple publishes a standalone Metal "
            "toolchain for\n"
            "Windows only, so on macOS there is no route that avoids Xcode.\n"
            "Verify with `xcrun metal --version`.\n"
            "\n"
            "Note this is a property of THIS backend's compile path, not of "
            "Metal: the Metal\n"
            "framework compiles shader source at runtime with no Xcode "
            "present.\n"
            "\n"
            "Working alternative today: drop --triton. The default compiled "
            "engine runs\n"
            "on Apple GPUs through PyTorch MPS with no extra install.\n"
            "\n"
            "Status and known gaps: docs/internal/metal_adoption_plan_2026_09_03.md"
        )
