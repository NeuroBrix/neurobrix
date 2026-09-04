"""A resumed render must produce the SAME result as an uninterrupted one.

The rack has one breaker and no UPS, so a long render can be cut at any
step. Resuming is only worth having if it is exact: a resume that produces
a subtly different image is worse than restarting, because nothing reports
it. Multistep solvers (UniPC, DPM++) are where this bites — they compute
the next sample from the last `solver_order` model outputs, so restoring
the latent alone silently changes the trajectory.

Each test here runs a scheduler straight through, then runs it again with a
simulated power cut in the middle, and requires the two final samples to be
**bit-identical**. Not close — identical. Anything less means the
checkpoint is changing the render.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobrix.core.module.scheduler.diffusion.ddim import DDIMScheduler
from neurobrix.core.module.scheduler.diffusion.dpm_solver_pp import DPMSolverPPScheduler
from neurobrix.core.module.scheduler.diffusion.euler import EulerDiscreteScheduler
from neurobrix.core.module.scheduler.diffusion.unipc_multistep import UniPCMultistepScheduler
from neurobrix.core.module.scheduler.flow.flow_euler import FlowEulerScheduler

STEPS = 8
CUT_AT = 3          # steps completed before the "power cut"
SHAPE = (1, 4, 8, 8)


def _config(cls=None):
    """Scheduler config. NeuroBrix validation is ZERO-FALLBACK — every value a
    scheduler needs must be present, because in production they come from the
    container rather than from defaults. So each family gets exactly its keys."""
    base = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": "epsilon",
        "timestep_spacing": "leading",
        "steps_offset": 1,
    }
    name = getattr(cls, "__name__", "")
    if name in ("UniPCMultistepScheduler", "DPMSolverPPScheduler"):
        base.update({"solver_order": 2, "solver_type": "bh2" if "UniPC" in name else "midpoint",
                     "lower_order_final": True})
    if name in ("FlowEulerScheduler", "RectifiedFlowScheduler"):
        base.update({"shift": 3.0})
    return base


def _make(cls):
    scheduler = cls(_config(cls))
    scheduler.set_timesteps(STEPS)
    return scheduler


def _fake_model_output(step: int, sample: torch.Tensor) -> torch.Tensor:
    """Deterministic stand-in for the denoiser: the identity of the run must
    come from the scheduler, not from randomness."""
    g = torch.Generator().manual_seed(1234 + step)
    return torch.randn(sample.shape, generator=g, dtype=sample.dtype)


def _run(cls, cut_at: int | None = None) -> torch.Tensor:
    """Run STEPS steps; when `cut_at` is set, checkpoint there, rebuild the
    scheduler from scratch, restore, and finish — the power-cut path."""
    torch.manual_seed(0)
    sample = torch.randn(SHAPE)
    scheduler = _make(cls)

    for step, t in enumerate(scheduler.timesteps):
        if cut_at is not None and step == cut_at:
            state = scheduler.checkpoint_state()
            assert state is not None, f"{cls.__name__} declared itself unresumable"
            # the cut: everything in memory is lost, latent included except
            # for what was written to disk
            latent = sample.numpy().copy()
            scheduler = _make(cls)
            scheduler.restore_state(state)
            sample = torch.from_numpy(latent)

        out = _fake_model_output(step, sample)
        result = scheduler.step(out, t, sample, return_dict=False)
        sample = result[0] if isinstance(result, tuple) else result
    return sample


@pytest.mark.parametrize("cls", [
    EulerDiscreteScheduler, DDIMScheduler, FlowEulerScheduler,
    UniPCMultistepScheduler, DPMSolverPPScheduler,
])
def test_resume_is_bit_identical(cls):
    straight = _run(cls)
    resumed = _run(cls, cut_at=CUT_AT)
    assert straight.shape == resumed.shape
    assert torch.equal(straight, resumed), (
        f"{cls.__name__}: a resumed render diverged from the uninterrupted one "
        f"(max |diff| {(straight - resumed).abs().max().item():.3e}). A resume "
        f"that is not exact must refuse rather than differ."
    )


@pytest.mark.parametrize("cls", [
    EulerDiscreteScheduler, DDIMScheduler, FlowEulerScheduler,
    UniPCMultistepScheduler, DPMSolverPPScheduler,
])
def test_state_is_serialisable_without_torch(cls):
    """The triton engine needs the same checkpoint and R33 keeps torch out of
    that tree, so the state must survive a numpy round-trip."""
    scheduler = _make(cls)
    sample = torch.randn(SHAPE)
    for step, t in enumerate(scheduler.timesteps[:CUT_AT]):
        out = _fake_model_output(step, sample)
        result = scheduler.step(out, t, sample, return_dict=False)
        sample = result[0] if isinstance(result, tuple) else result

    state = scheduler.checkpoint_state()
    assert state is not None
    for key, value in state.items():
        if isinstance(value, list):
            for item in value:
                assert item is None or isinstance(item, (np.ndarray, int, float)), (
                    f"{cls.__name__}.{key} carries a {type(item).__name__}, "
                    f"which will not serialise torch-free"
                )
        else:
            assert value is None or isinstance(value, (np.ndarray, int, float)), (
                f"{cls.__name__}.{key} is a {type(value).__name__}"
            )


def test_a_scheduler_that_opts_out_is_refused():
    """The base default is 'not resumable'. A scheduler that has not thought
    about its state must not be resumed by accident."""
    from neurobrix.core.module.scheduler.base import (
        DiffusionSchedulerBase, FlowSchedulerBase,
    )

    assert DiffusionSchedulerBase.checkpoint_state(object()) is None  # type: ignore[arg-type]
    assert FlowSchedulerBase.checkpoint_state(object()) is None       # type: ignore[arg-type]


def test_ancestral_sampling_refuses_to_resume():
    """Ancestral sampling draws fresh noise every step. Resuming without the
    generator state would re-draw different noise and produce a different
    image while reporting success — so it declines."""
    from neurobrix.core.module.scheduler.diffusion.euler import (
        EulerAncestralDiscreteScheduler,
    )

    scheduler = _make(EulerAncestralDiscreteScheduler)
    assert scheduler.checkpoint_state() is None
