"""The environment a CHILD process needs to compile anything on this machine.

Several tests spawn a subprocess with a hand-built `env={...}` rather than
inheriting `os.environ`. That is deliberate and worth keeping: a bare
environment is what proves the launch path does not quietly depend on something
the developer happens to have exported — R33's "no torch" checks are exactly
that shape.

What it must not strip is what the MACHINE needs to compile at all. Measured
2026-09-17 on this Mac, after the Metal backend became triton-ext:

  * without `TOOLCHAINS=Metal`, `xcrun metal` does not resolve here —
    "cannot execute tool 'metal' due to missing Metal Toolchain" — and
    triton-ext's compiler shells out to it, so every child compile died;
  * without the backend selection the child's profile lookup could land
    somewhere else entirely.

On the archived fork neither mattered: it was installed in the engine's own venv
and its compile path did not go through `xcrun metal` the same way. So these
tests passed for a reason that stopped being true, which is why this lives in
one place with the reason written down rather than as a line copied into each
file.

Nothing here adds torch, a GPU, or a model path. It adds the compiler's address.
"""
from __future__ import annotations

import os

#: Variables that say WHERE the toolchain and WHICH backend, and nothing else.
#: A child that inherits these can compile; one that does not, cannot.
_MACHINE_PREREQUISITES = ("TOOLCHAINS", "NEUROBRIX_METAL_BACKEND")


def child_env(base: dict | None = None, **extra) -> dict:
    """`base` (default: a minimal env) plus this machine's compile prerequisites.

    Pass the same minimal dict the test built before; the prerequisites are
    added only when they are set in THIS process, so a machine that needs none
    of them gets exactly what it had.
    """
    env = dict(base) if base is not None else {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
    }
    for name in _MACHINE_PREREQUISITES:
        value = os.environ.get(name)
        if value:
            env.setdefault(name, value)
    env.update(extra)
    return env


def the_callers_door(default: str) -> str:
    """The cards a child process may see: the caller's `CUDA_VISIBLE_DEVICES` when it set one,
    `default` only when it set none.

    A child told its own ordinal escapes the door the workshop runs the suite under. Measured
    2026-09-27: a gate told `CUDA_VISIBLE_DEVICES=3` put a TinyLlama child on physical card 2
    (the lazy-bind boundary test's "2") and a launch-path child on card 0 (the launcher's "0"),
    twice meeting a retrace that owned card 2. An empty door holds no card, and the test says so
    instead of finding one.
    """
    import pytest

    door = os.environ.get("CUDA_VISIBLE_DEVICES")
    if door is None:
        return default
    if not door.strip():
        pytest.skip("the caller's door holds no card (CUDA_VISIBLE_DEVICES is empty)")
    return door


def missing_prerequisites() -> list:
    """Which of them this process does not have — for a test that wants to say
    'this machine cannot compile in a child' rather than fail obscurely."""
    return [n for n in _MACHINE_PREREQUISITES if not os.environ.get(n)]
