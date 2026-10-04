"""FlowEuler follows the pipeline sigma schedule the container declares
(extracted_values.scheduler.sigma_schedule, carried by Forge from the registry), in both
engines; absent, it keeps its own linspace byte for byte.

mochi-1-preview's MochiPipeline builds its sigmas with genmo's linear-quadratic schedule
(diffusers pipeline_mochi.py L61-75, threshold_noise 0.025 at L668) — the engine ran the
scheduler's own linspace, and the vendor's run at the two schedules from the same noise
differs at final-latent cos 0.488 (measured 2026-10-04). Reference values: diffusers
0.37.0's `linear_quadratic_schedule` and the vendor run's scheduler sigmas (inverted:
0, .0125, .025, .275, 1 at 4 steps). What this test would do if the code were wrong:
the declared schedule would give the linspace values (seen failing on injection).
"""
import numpy as np
import pytest
import torch

from neurobrix.core.module.scheduler.flow.flow_euler import FlowEulerScheduler
from neurobrix.core.module.scheduler.config import SchedulerConfigError

MOCHI = {"_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.32.0.dev0",
         "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": True, "max_image_seq_len": 4096,
         "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False}
LQ = {"family": "linear_quadratic", "threshold_noise": 0.025, "linear_steps": None}

# diffusers 0.37.0 pipeline_mochi.linear_quadratic_schedule(n, 0.025): first three, last two
VENDOR = {4: ([1.0, 0.9875, 0.9750000000000001], [0.9750000000000001, 0.7250000000000003]),
          64: ([1.0, 0.99921875, 0.9984375], [0.11660156250000031, 0.05922851562500031])}


def _torch(decl):
    return FlowEulerScheduler(dict(MOCHI, **({"sigma_schedule": decl} if decl is not None else {})))


def _triton(decl):
    pytest.importorskip("triton")
    from neurobrix.triton.scheduler.flow_euler import TritonFlowEulerScheduler
    return TritonFlowEulerScheduler(dict(MOCHI, **({"sigma_schedule": decl} if decl is not None else {})))


def _ts(s):
    return [float(t) for t in (s._ts_np if hasattr(s, "_ts_np") else s.timesteps)]


@pytest.mark.parametrize("engine", [_torch, _triton])
@pytest.mark.parametrize("steps", [4, 64])
def test_the_declared_linear_quadratic_schedule_is_the_vendor_s(engine, steps):
    s = engine(LQ); s.set_timesteps(steps)
    ts = _ts(s)                                            # inverted: 1 - vendor sigma
    first, last = VENDOR[steps]
    np.testing.assert_allclose(ts[:3], [1 - x for x in first], atol=1e-6)
    np.testing.assert_allclose(ts[-2:], [1 - x for x in last], atol=1e-6)
    assert len(ts) == steps


def test_the_vendor_run_s_four_step_sigmas_are_reproduced():
    s = _torch(LQ); s.set_timesteps(4)
    np.testing.assert_allclose(_ts(s) + [1.0], [0.0, 0.012499988, 0.024999976, 0.27499998, 1.0], atol=1e-6)


def test_absent_the_scheduler_keeps_its_own_linspace_byte_for_byte():
    s = _torch(None); s.set_timesteps(20)
    assert torch.equal(s.timesteps, 1.0 - torch.linspace(1, 0, 21)[:-1])
    s2 = _torch({"family": "linspace"}); s2.set_timesteps(20)
    assert torch.equal(s2.timesteps, s.timesteps)


@pytest.mark.parametrize("decl", [{"family": "cosine"}, {"family": "linear_quadratic"}, "linear_quadratic"])
def test_an_unknown_or_incomplete_declaration_is_refused(decl):
    with pytest.raises(SchedulerConfigError, match="ZERO FALLBACK"):
        _torch(decl)


def test_a_step_count_the_vendor_schedule_cannot_divide_is_refused():
    s = _torch(LQ)
    with pytest.raises(SchedulerConfigError, match="linear_steps"):
        s.set_timesteps(1)                                 # the vendor divides by zero here


def test_the_container_flag_reaches_the_scheduler_config():
    from neurobrix.nbx import component_flags
    from neurobrix.core.runtime.registry_flags import get_component_flag
    component_flags.register("a-model", {"scheduler": {"shift": 1.0, "sigma_schedule": LQ}})
    try:
        assert get_component_flag("a-model", "scheduler", "sigma_schedule") == LQ
    finally:
        component_flags.clear()
