"""The `--triton` backend gates must refuse loudly, and must be WIRED.

Two modules existed to stop `--triton` from dying inside the Triton driver
on hardware whose backend is not installed: `cpu_backend.py` (triton-cpu)
and, since 2026-09-03, `metal_backend.py` (Apple GPUs). The first had been
written, documented — and never imported by anything. An unwired gate is
not a gate, so these pins cover the wiring as much as the logic.

What the gates replace at the call site:

* Apple: a hardcoded "this will be supported in a future version" message
  that also returned None instead of a non-zero code. It was out of date —
  an out-of-tree Metal backend exists.
* CPU-only: `TRITON_CPU_BACKEND=1` was set without checking triton-cpu was
  installed at all, so the run died one step later in the driver.
"""

from __future__ import annotations

import pytest

from neurobrix.triton import cpu_backend, metal_backend


# --- the gate is reachable from the command that needs it -------------------

def test_run_command_imports_both_gates():
    """The regression that motivated this file: a gate nothing calls.

    Read as source rather than executed, because importing the run command
    pulls the whole runtime; the point is only that the wiring exists.
    """
    import inspect

    from neurobrix.cli.commands import run as run_cmd

    source = inspect.getsource(run_cmd)
    assert "ensure_triton_metal_or_raise" in source
    assert "ensure_triton_cpu_or_raise" in source, (
        "cpu_backend was orphan code from its creation until 2026-09-03"
    )


# --- Apple detection --------------------------------------------------------

def test_not_apple_silicon_is_a_no_op(monkeypatch):
    """Every non-Apple machine must pass straight through the gate."""
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(metal_backend.platform, "machine", lambda: "x86_64")
    assert metal_backend.is_apple_silicon() is False
    metal_backend.ensure_triton_metal_or_raise()   # must not raise


def test_intel_mac_is_not_apple_silicon(monkeypatch):
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_backend.platform, "machine", lambda: "x86_64")
    assert metal_backend.is_apple_silicon() is False


def test_apple_silicon_is_detected(monkeypatch):
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_backend.platform, "machine", lambda: "arm64")
    assert metal_backend.is_apple_silicon() is True


# --- the refusal ------------------------------------------------------------

def test_apple_without_backend_refuses_with_an_actionable_message(monkeypatch):
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(metal_backend, "triton_metal_available", lambda: False)

    with pytest.raises(metal_backend.TritonMetalNotInstalledError) as excinfo:
        metal_backend.ensure_triton_metal_or_raise()

    message = str(excinfo.value)
    assert "pip install triton-msl" in message, "a refusal without the install command is half a refusal"
    assert "--triton" in message
    assert "compiled" in message, "must name the path that works today"
    assert "does not install it for you" in message, "we never auto-fetch"


def test_apple_with_backend_and_compiler_passes(monkeypatch):
    """Both conditions satisfied is what "ready" means.

    This pin used to stub only `triton_metal_available`. Written without a
    Mac, it encoded a premise the first Apple machine refuted on 2026-09-05:
    the package being importable is not enough, because the backend's compile
    path ends in Apple's offline shader compiler. Stubbing one probe and
    calling that "passes" is how the gate came to report ready on a machine
    where nothing runs.
    """
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(metal_backend, "triton_metal_available", lambda: True)
    monkeypatch.setattr(metal_backend, "metal_shader_compiler_available",
                        lambda: True)
    metal_backend.ensure_triton_metal_or_raise()   # must not raise


# --- the offline shader compiler -------------------------------------------

def test_backend_installed_without_shader_compiler_refuses(monkeypatch):
    """The condition this whole file exists for, one layer down.

    Measured on an M4 Pro with Command Line Tools and no Xcode: triton-msl
    imports, the old gate passed, and the run died later inside Triton with
    "0 active drivers". The gate must refuse HERE, and say what to install.
    """
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(metal_backend, "triton_metal_available", lambda: True)
    monkeypatch.setattr(metal_backend, "metal_shader_compiler_available",
                        lambda: False)

    with pytest.raises(
            metal_backend.TritonMetalShaderCompilerMissingError) as excinfo:
        metal_backend.ensure_triton_metal_or_raise()

    message = str(excinfo.value)
    assert "xcodebuild -downloadComponent MetalToolchain" in message, (
        "a refusal without the install command is half a refusal"
    )
    assert "Command Line Tools" in message, (
        "the trap is that CLT look sufficient and are not"
    )
    assert "compiled" in message, "must name the path that works today"


def test_shader_compiler_refusal_is_catchable_as_the_old_error(monkeypatch):
    """Call sites already catch `TritonMetalNotInstalledError`; the new
    condition must not slip past them."""
    assert issubclass(metal_backend.TritonMetalShaderCompilerMissingError,
                      metal_backend.TritonMetalNotInstalledError)


def test_shader_compiler_probe_is_false_off_darwin(monkeypatch):
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Linux")
    assert metal_backend.metal_shader_compiler_available() is False


def test_shader_compiler_probe_survives_a_missing_xcrun(monkeypatch):
    """A probe that takes the run down is not a probe."""
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_backend.shutil, "which", lambda _n: None)
    assert metal_backend.metal_shader_compiler_available() is False


def test_shader_compiler_probe_survives_a_raising_subprocess(monkeypatch):
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_backend.shutil, "which", lambda _n: "/usr/bin/xcrun")

    def boom(*_a, **_k):
        raise OSError("no fork for you")

    monkeypatch.setattr(metal_backend.subprocess, "run", boom)
    assert metal_backend.metal_shader_compiler_available() is False


def test_shader_compiler_probe_reads_the_return_code(monkeypatch):
    """`xcrun` exists even when the `metal` sub-tool does not — which is the
    Command-Line-Tools case — so the probe must read the exit status, not the
    presence of xcrun."""
    monkeypatch.setattr(metal_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_backend.shutil, "which", lambda _n: "/usr/bin/xcrun")

    class _Result:
        def __init__(self, code):
            self.returncode = code

    monkeypatch.setattr(metal_backend.subprocess, "run",
                        lambda *_a, **_k: _Result(1))
    assert metal_backend.metal_shader_compiler_available() is False
    monkeypatch.setattr(metal_backend.subprocess, "run",
                        lambda *_a, **_k: _Result(0))
    assert metal_backend.metal_shader_compiler_available() is True


# --- backend probing --------------------------------------------------------

def test_backend_probe_accepts_the_plugin_env_var(monkeypatch):
    """Triton 3.7 loads out-of-tree backends via TRITON_PLUGIN_PATHS, so a
    backend can be present without an importable package name we know."""
    monkeypatch.setenv("TRITON_PLUGIN_PATHS", "/opt/somewhere/libtriton_msl.so")
    assert metal_backend.triton_metal_available() is True


def test_backend_probe_is_negative_on_an_unrelated_plugin(monkeypatch):
    monkeypatch.setenv("TRITON_PLUGIN_PATHS", "/opt/somewhere/libutlx.so")
    monkeypatch.setattr(metal_backend.importlib.util, "find_spec", lambda _n: None)
    assert metal_backend.triton_metal_available() is False


def test_backend_probe_survives_a_broken_module_name(monkeypatch):
    """`find_spec` raises for some malformed names; the probe must answer
    False rather than take the run down with it."""
    def boom(_name):
        raise ValueError("bad module name")

    monkeypatch.setattr(metal_backend.importlib.util, "find_spec", boom)
    monkeypatch.delenv("TRITON_PLUGIN_PATHS", raising=False)
    assert metal_backend.triton_metal_available() is False


# --- the coverage markers ---------------------------------------------------

def test_known_gaps_are_declared_in_one_place():
    """Marker constants exist so a future chantier flips them once, the same
    pattern triton-cpu uses for its upstream gaps."""
    assert metal_backend.TRITON_METAL_BATCHED_MATMUL_BLOCKED is True
    assert metal_backend.TRITON_METAL_BF16_ATTENTION_BLOCKED is True
    assert metal_backend.TRITON_METAL_FP64_UNAVAILABLE is True
    assert metal_backend.TRITON_METAL_NEEDS_OFFLINE_SHADER_COMPILER is True


def test_metal_backend_imports_no_torch():
    """R33: the triton tree stays sealed against torch, boundary included."""
    import inspect

    source = inspect.getsource(metal_backend)
    assert "import torch" not in source
    assert "torch." not in source


def test_cpu_gate_still_refuses_with_its_install_path():
    """Unchanged behaviour, now actually reachable."""
    assert issubclass(cpu_backend.TritonCPUNotInstalledError, ImportError)
