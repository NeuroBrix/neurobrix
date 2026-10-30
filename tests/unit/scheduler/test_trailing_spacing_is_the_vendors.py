"""'trailing' timestep spacing is the vendor's: round(arange(T, 0, -T/N)) - 1, for every scheduler that asks for it.

The shared helper computed the 'linspace' formula under the trailing name — its body was the linspace one, word for
word (the engine-slim scan found the two functions identical, 2026-10-30) — off by one on several timesteps against
diffusers' DPMSolverMultistepScheduler, EulerDiscreteScheduler and DDIMScheduler with timestep_spacing='trailing'
(DDIM carried its own correct formula beside it). No catalogue model reached it (the two trailing configs are
CogVideoX's DDIM). What would this file do on that code? 999, 874, 749, 624, 500 against the vendor's ..., 499: RED."""
import pytest

diffusers = pytest.importorskip("diffusers")

from neurobrix.core.module.scheduler.utils.timestep_utils import get_timesteps, trailing_timesteps


@pytest.mark.parametrize("steps", [2, 8, 20, 25, 50])
@pytest.mark.parametrize("vendor", ["DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "DDIMScheduler"])
def test_trailing_is_the_vendors(vendor, steps):
    s = getattr(diffusers, vendor)(num_train_timesteps=1000, timestep_spacing="trailing")
    s.set_timesteps(steps)
    expected = [int(t) for t in s.timesteps.tolist()]
    assert trailing_timesteps(steps, 1000).tolist() == expected
    assert get_timesteps("trailing", steps, 1000).tolist() == expected
