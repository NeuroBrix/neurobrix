"""One door for every cell that MEASURES on the shared rig.

A cell that times a kernel, or runs a subprocess which autotunes one, is making a
measurement. Another process holding the cards perturbs it, and a perturbed
measurement is not a weaker measurement — it is a different one, reported in the
same words.

The door existed already, in `test_boolean_masks_at_overhanging_shapes.py`, as a
module-level `needs_a_free_rig`. It was written once and needed twice: on
2026-09-18 the release commit's suite came back `1 failed, 2279 passed` where the
same commit had given `2280 passed, 0 failed` an hour before, and the one red was
`test_launcher.py::test_an_autotuned_kernel_launches_and_benchmarks_without_torch`
— an R33 cell whose subprocess autotunes a matmul and asserts `returncode == 0`.
It passes alone, passes twice in a row on demand, and failed while three compute
processes held the rig. The same bug written twice is a missing brick, so here is
the brick.

**It asks when the test runs, not when the file is imported.** The original form
computed its reason at module scope, which is the register-74 shape: a decorator
argument describes collection time and is read as though it described the test. A
suite that collects on a quiet rack and runs on a busy one would decide the wrong
way round, and so would the reverse.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def rig_reason() -> str:
    """Why this machine cannot be measured on right now, or "" if it can."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return ""                       # no NVIDIA rig: nothing of ours is holding it
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi did not answer ({exc.__class__.__name__})"
    if out.returncode != 0:
        return "nvidia-smi reported no device"
    busy = [line for line in out.stdout.splitlines() if line.strip()]
    if busy:
        return (f"{len(busy)} compute process(es) hold the rig — a measurement "
                f"taken beside them is a different measurement, not a weaker one")
    return ""


@pytest.fixture
def a_free_rig():
    """Skip the calling test if anything else is computing on the cards.

    Used as a fixture rather than a mark so the question is asked at the moment
    the measurement is about to be taken.
    """
    reason = rig_reason()
    if reason:
        pytest.skip(reason)


def running_backend(triton_version=None) -> dict:
    """A proof's `backend` stamp for THIS machine, for fixtures that must be served.

    A fixture that hardcodes `{"name": "cuda"}` and then asserts its entry IS
    SERVED can only pass on a CUDA box. That was invisible while
    `running_generator()` read `nbx_tensor.BACKEND_NAME` — a symbol that does not
    exist — and silently defaulted to cuda, so every machine looked like a CUDA
    box to the gate. The moment that was fixed (2026-09-19; Apple reports
    `triton 3.8.0 mps`), three such fixtures turned red at once.

    `triton_version` is left settable because the STALE-generator cells vary it
    deliberately; the NAME is the part that must follow the machine.

    Delegates to `generator_identity()` — the one door — rather than carrying a
    third spelling of the identity. This fixture WAS a second spelling reading
    `triton.__version__`, and it turned red the moment the door moved to the
    distribution metadata (2026-09-20), which is the behaviour a copy always
    buys.
    """
    from neurobrix.kernels.autotune_certified import generator_identity
    ident = dict(generator_identity())
    if triton_version is not None:
        ident["triton"] = str(triton_version)
    return ident
