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


def test_apple_with_backend_passes(monkeypatch):
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(metal_backend, "triton_metal_available", lambda: True)
    metal_backend.ensure_triton_metal_or_raise()   # must not raise


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


def test_metal_backend_imports_no_torch():
    """R33: the triton tree stays sealed against torch, boundary included."""
    import inspect

    source = inspect.getsource(metal_backend)
    assert "import torch" not in source
    assert "torch." not in source


def test_cpu_gate_still_refuses_with_its_install_path():
    """Unchanged behaviour, now actually reachable."""
    assert issubclass(cpu_backend.TritonCPUNotInstalledError, ImportError)
