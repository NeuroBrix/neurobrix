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
    guaranteed to be the import name. Recognises BOTH selectable Metal backends:
    the bledden fork (`triton_msl`) and the triton-ext AppleGPU plugin
    (registered as the `apple` backend). No engine file names a vendor; this one
    seam does.
    """
    for module in ("triton_msl", "triton.backends.metal"):
        try:
            if importlib.util.find_spec(module) is not None:
                return True
        except (ImportError, ValueError):
            continue
    # triton-ext's AppleGPU backend registers into Triton's plugin registry
    # under the name `apple` rather than as an importable `triton.backends.metal`.
    try:
        from triton.backends import backends as _tb
        if "apple" in (_tb.keys() if hasattr(_tb, "keys") else _tb):
            return True
    except Exception:
        pass
    # A plugin can also be pointed at by the upstream env var.
    plugins = os.environ.get("TRITON_PLUGIN_PATHS", "")
    return any("metal" in p.lower() or "msl" in p.lower() or "apple" in p.lower()
               for p in plugins.split(os.pathsep) if p)


def backend_refusal_types() -> tuple:
    """The exception types a Metal backend raises to say 'I cannot compile this
    config correctly' — collected here so the SHARED refusal module
    (`autotune_refusals.py`, which the Dell also runs) names no vendor.

    Both selectable backends contribute their type if present: the bledden fork
    (`triton_msl.errors.MetalNonRecoverableError`) and, when triton-ext exposes a
    named refusal type, that too. Empty off Apple / when no Metal backend is
    installed — so on CUDA the shared check is exactly as inert as before this
    seam existed. A backend absent contributes nothing rather than raising.
    """
    types: list = []
    try:
        from triton_msl.errors import MetalNonRecoverableError
        types.append(MetalNonRecoverableError)
    except Exception:
        pass
    # triton-ext (AppleGPU) refusal type, if/when it exposes one by name.
    try:
        from triton_apple_backend.errors import AppleGPUNonRecoverableError  # type: ignore
        types.append(AppleGPUNonRecoverableError)
    except Exception:
        pass
    return tuple(types)


def is_backend_refusal(exc: BaseException) -> bool:
    """True when `exc` is a Metal backend saying it cannot compile this config —
    asked by class against whichever backend(s) are present, naming none in the
    caller. False (never raising) when no Metal backend is installed."""
    rt = backend_refusal_types()
    return bool(rt) and isinstance(exc, rt)


#: The Metal backends NeuroBrix can target. The engine targets Triton; WHICH
#: Metal backend runs is a selection, not a branch, and this table is the only
#: place either is named. `probe` is the import that proves it is installed;
#: `compiler` is the dotted path to its Triton backend class (the thing a driver
#: needs to patch a stage).
METAL_BACKENDS = {
    "triton_msl": {
        "probe": "triton_msl",
        "compiler": ("triton_msl.backend.compiler", "MetalBackend"),
        "target": "metal",
        # The NeuroBrix launcher driver that implements THIS backend's launch
        # ABI. Our Metal driver derives its argument binding from the emitted
        # MSL's own conventions (a scalar the emitter passes through a pointer
        # is named `<param>_buf`), so it is specific to this emitter.
        "nbx_driver": "neurobrix.triton.metal_driver",
        "what": "the bledden triton-msl fork (text MSL emitter)",
    },
    "triton_ext": {
        "probe": "triton_apple_backend",
        "compiler": ("triton_apple_backend.compiler", "MetalBackend"),
        "target": "mps",
        # NONE YET. triton-ext emits MSL from C++ MLIR passes with its own
        # argument conventions and ships its own driver; our fork-shaped driver
        # mis-binds its kernels (measured 2026-09-16: scalars after the first
        # arrive as 0, so every mask is false and the output keeps its zeros).
        # Until an adapter exists, selecting this backend REFUSES at the driver
        # rather than launching through an ABI that is not its own.
        "nbx_driver": None,
        "what": "triton-lang/triton-ext AppleGPU (C++ MLIR -> MSL)",
    },
}

_PROFILE_KEY = "metal_backend"
_ENV_KEY = "NEUROBRIX_METAL_BACKEND"


def _installed(name: str) -> bool:
    spec = METAL_BACKENDS.get(name)
    if not spec:
        return False
    try:
        return importlib.util.find_spec(spec["probe"]) is not None
    except (ImportError, ValueError):
        return False


def selected_metal_backend() -> str:
    """WHICH Metal backend this machine runs, decided by the PROFILE.

    The profile may declare `metal_backend: triton_msl | triton_ext`. A declared
    backend that is not installed is REFUSED BY NAME — silently falling back to
    the other one would make a measurement attribute itself to the wrong
    implementation, which is the whole reason this is a selection and not a
    branch. When the profile declares nothing, the single installed backend is
    used; if both are installed and none is declared, that ambiguity is refused
    too — a machine that can run either must say which.
    """
    # An explicit operator override, above the profile and named in the run's
    # own environment. It exists for one real case: evaluating a SECOND backend
    # from a different virtualenv that shares this source tree, so the profile
    # file cannot hold both answers at once. It is still a SELECTION — declared,
    # refused by name when absent — never a silent fallback.
    declared = os.environ.get(_ENV_KEY)
    if not declared:
        try:
            from neurobrix.kernels.ops._configs import active_vendor_profile
            declared = (active_vendor_profile() or {}).get(_PROFILE_KEY)
        except Exception:
            declared = None

    if declared:
        if declared not in METAL_BACKENDS:
            raise RuntimeError(
                f"the hardware profile declares `{_PROFILE_KEY}: {declared}`, which is "
                f"not a Metal backend this engine knows. Known: "
                f"{', '.join(sorted(METAL_BACKENDS))}.")
        if not _installed(declared):
            raise RuntimeError(
                f"the hardware profile declares `{_PROFILE_KEY}: {declared}` "
                f"({METAL_BACKENDS[declared]['what']}) and it is NOT installed "
                f"(no module {METAL_BACKENDS[declared]['probe']!r}). Refusing rather "
                f"than running on the other backend and attributing the numbers to "
                f"the declared one. Install it, or change the profile.")
        return declared

    present = [n for n in METAL_BACKENDS if _installed(n)]
    if not present:
        raise RuntimeError(
            "no Metal backend is installed (looked for "
            + ", ".join(f"{n} ({METAL_BACKENDS[n]['probe']})" for n in METAL_BACKENDS)
            + "), and the profile declares none.")
    if len(present) > 1:
        raise RuntimeError(
            f"both Metal backends are installed ({', '.join(present)}) and the "
            f"profile declares no `{_PROFILE_KEY}`. A machine that can run either "
            f"must say which, or its measurements cannot name the backend that "
            f"produced them.")
    return present[0]


def backend_target_name() -> str:
    """The Triton GPUTarget backend NAME the selected implementation answers to.

    Triton resolves a backend by asking each registered one `supports_target`,
    which compares this string. The two Metal backends do not use the same one —
    the fork answers to `metal`, triton-ext's AppleGPU to `mps` — so a driver
    that hardcodes either can only ever reach one of them ("0 compatible
    backends for target (metal)"). The name is a property of the selected
    backend, so it lives in the table with it.
    """
    return METAL_BACKENDS[selected_metal_backend()]["target"]


def backend_compiler_class():
    """The selected backend's Triton backend class — the object a driver patches.

    Raises with the backend NAMED when it is selected but its compiler cannot be
    imported, so a driver never silently does nothing.
    """
    name = selected_metal_backend()
    mod_name, attr = METAL_BACKENDS[name]["compiler"]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            f"the Metal backend in force is {name!r} ({METAL_BACKENDS[name]['what']}) "
            f"but its compiler module {mod_name!r} could not be imported ({exc}).")
    cls = getattr(mod, attr, None)
    if cls is None:
        raise RuntimeError(
            f"the Metal backend in force is {name!r} but {mod_name}.{attr} does not "
            f"exist — this engine cannot patch a stage it cannot name.")
    return cls


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
    # EXECUTE `metal --version`, not just `--find`: the Xcode default-toolchain
    # shim is FOUND even when it cannot run (Xcode 26/27 delegation defect), so
    # `--find` said yes on a machine where nothing compiled. `_metal_runs`
    # answers the question the compile pipeline actually asks. (Callers that
    # need the delegation fix applied first call `ensure_metal_toolchain_selected`.)
    return _metal_runs()


_DELEGATION_ANNOUNCED = False


def _metal_runs(env_extra: "dict | None" = None) -> bool:
    """True when `xcrun metal` can actually EXECUTE (not merely be found).

    `xcrun --find metal` resolves the Xcode default-toolchain SHIM, which is
    present even when it cannot run — so the find-probe says yes on a machine
    where nothing compiles. This runs `metal --version`, which is what the
    compile pipeline needs."""
    import os as _os
    env = dict(_os.environ, **(env_extra or {}))
    try:
        return subprocess.run(["xcrun", "metal", "--version"], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_metal_toolchain_selected() -> None:
    """Work around Xcode 26/27's broken default-toolchain delegation, once.

    macOS 26 / Xcode 26+ ship the Metal compiler as a separate on-demand
    toolchain (`com.apple.dt.toolchain.Metal`). Once downloaded it is mounted
    and its `metal` binary runs — but Xcode's DEFAULT-toolchain shim does not
    delegate to it, so a bare `xcrun metal` fails with "missing Metal Toolchain"
    even though the toolchain is installed. Selecting it explicitly
    (`TOOLCHAINS=Metal`) resolves it. This is a recognised Apple defect since
    Xcode 26; every user on 26 or 27 meets it, so the backend detects and fixes
    it here rather than leaving each user to discover `TOOLCHAINS` alone.

    A no-op off Apple Silicon, when a TOOLCHAINS is already set, or when the
    bare compiler already runs. Announced ONCE per process — a fix repeated on
    every kernel is noise nobody reads."""
    global _DELEGATION_ANNOUNCED
    if not is_apple_silicon():
        return
    import os as _os
    if _os.environ.get("TOOLCHAINS"):
        return                                        # the caller already chose
    if _metal_runs():
        return                                        # delegation is fine here
    if _metal_runs({"TOOLCHAINS": "Metal"}):
        _os.environ["TOOLCHAINS"] = "Metal"
        if not _DELEGATION_ANNOUNCED:
            _DELEGATION_ANNOUNCED = True
            print("[metal] Xcode's default toolchain does not delegate to the "
                  "installed Metal compiler (a known Apple defect since Xcode "
                  "26); selecting it with TOOLCHAINS=Metal for this process.",
                  flush=True)


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

    # Fix Xcode 26/27's broken toolchain delegation BEFORE probing the compiler,
    # so the probe sees the compiler that will actually run.
    ensure_metal_toolchain_selected()

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


def nbx_driver_module() -> str:
    """The NeuroBrix launcher driver module implementing the SELECTED backend's
    launch ABI, or a refusal naming why there is none.

    A driver is not interchangeable between Metal backends: ours reads the
    fork's MSL emission conventions to decide how each scalar is bound. Handing
    it a kernel another emitter produced binds the wrong things silently —
    measured, and the failure is not loud: scalars after the first arrive as 0,
    every mask is false, and the kernel writes nothing, so the output keeps the
    zeros it was allocated with.
    """
    name = selected_metal_backend()
    mod = METAL_BACKENDS[name].get("nbx_driver")
    if mod:
        return mod
    raise RuntimeError(
        f"the Metal backend in force is {name!r} ({METAL_BACKENDS[name]['what']}), "
        f"and NeuroBrix has no launcher driver implementing its launch ABI. "
        f"Refusing to launch through another backend's driver: ours derives its "
        f"argument binding from the fork's MSL conventions, and on a kernel this "
        f"backend compiled that binds scalars to the wrong places WITHOUT failing "
        f"— the kernel reads zeros and writes nothing. Correct-or-refuse: an "
        f"adapter for {name!r} is owed before it can be launched."
    )
