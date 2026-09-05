"""A refused launch must not cost the command queue a slot.

A Metal command buffer counts against its queue's in-flight limit — 64 — from
the moment it is created until it COMPLETES. A buffer abandoned because
encoding raised never completes, so its slot is gone for the life of the
process. After 64 refusals the queue blocks in `commandBuffer()` and nothing
runs on that device again.

Found 2026-09-05 the expensive way. The kernels suite reached ~90% and
stopped dead: 25 seconds of CPU across an hour of wall clock, the main thread
parked in `_dispatch_semaphore_wait_slow` under
`[AGXG16XFamilyCommandQueue commandBuffer]`. Not a slow test — a queue that
could not hand out another buffer. That suite has ~219 failing tests and most
fail inside a launch; each one leaked a slot. It surfaced only when the
launcher started routing every kernel launch through this driver.

The refusals here are the real thing — a device address the allocator never
handed out — not a simulated raise, because the point is that the ordinary
refusal path returns its slot.

This is a *timeout* test, deliberately: the failure mode is a hang.
"""

from __future__ import annotations

import sys
import threading

import pytest

pytest.importorskip("neurobrix.triton.metal_driver")


def _has_metal():
    try:
        from neurobrix.kernels import nbx_tensor
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no Apple GPU here")

#: Comfortably past the 64-buffer limit.
_REFUSALS = 90
_TIMEOUT_SECONDS = 60

#: An address the Metal allocator cannot have handed out.
_FOREIGN_ADDRESS = 0x1000


def test_a_refused_launch_returns_its_command_buffer_slot():
    import numpy as np

    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import add
    from neurobrix.triton.metal_driver import MetalKernelError, compile_kernel

    from ._kernels.touch import touch

    kernel = compile_kernel(touch, {"p": "*fp32", "BLOCK": "constexpr"},
                            {"BLOCK": 64}, num_warps=1)

    # The WHOLE sequence runs in a worker, because the failure mode is a
    # block inside `commandBuffer()` — on the main thread that would hang
    # pytest itself with no report. Here the main thread stays free to say
    # what happened.
    done = threading.Event()
    outcome = {}

    def sequence():
        # Retained on purpose: pytest keeps a traceback per failure, and a
        # traceback pins every local in every frame — including the command
        # buffer. Releasing them here would hide the leak this test is for.
        retained = []
        try:
            for index in range(_REFUSALS):
                try:
                    kernel.launch((1,), [_FOREIGN_ADDRESS])
                except MetalKernelError:
                    retained.append(sys.exc_info())
                else:
                    outcome["error"] = (
                        f"launch {index} with a foreign address was accepted; "
                        f"the allocator ownership rule is not being applied")
                    return
                outcome["refusals"] = len(retained)

            x = NBXTensor.from_numpy(np.ones(256, dtype=np.float32))
            y = NBXTensor.from_numpy(np.ones(256, dtype=np.float32))
            outcome["result"] = add(x, y).numpy()
        except Exception as exc:                        # pragma: no cover
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    threading.Thread(target=sequence, daemon=True).start()
    finished = done.wait(_TIMEOUT_SECONDS)

    assert finished, (
        f"blocked after {outcome.get('refusals', 0)} of {_REFUSALS} refused "
        f"launches: the queue stopped handing out command buffers. Every "
        f"abandoned buffer must be committed — with its encoder closed first "
        f"— so its in-flight slot comes back.")
    assert "error" not in outcome, outcome["error"]
    assert outcome["refusals"] == _REFUSALS
    assert np.array_equal(outcome["result"],
                          np.full(256, 2.0, dtype=np.float32))
