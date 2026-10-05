"""FlowEuler's dynamic shift reads the LENGTH the pipeline reads, as the container declares it
(extracted_values.scheduler.dynamic_shift_length), in both engines; absent, it keeps the total
packed token count byte for byte.

hpcai-tech Open-Sora v2 (opensora/utils/sampling.py get_schedule) computes its shift over ONE
frame's tokens, `get_res_lin_function(256 -> 1, 4096 -> 3)((h*w)//patch**2)`, then multiplies it by
sqrt(latent frames). The engine computed it over every packed token (13 x 252 = 3276), mu 2.573
against the vendor's 3.598: at 4 steps the engine ran t = 1, .886, .721, .462 where the vendor ran
1, .9152, .7825, .5453 (vendor run from the engine's own noise, 2026-10-04,
validation ladder vendor/osora/run.log). What this test would do if the code were wrong: the
declared family would give the total-token values (seen failing on injection).
"""
import numpy as np
import pytest
import torch

from neurobrix.core.module.scheduler.config import SchedulerConfigError
from neurobrix.core.module.scheduler.flow.flow_euler import FlowEulerScheduler

# Open-Sora-v2's scheduler_config.json, as the container carries it
OSORA = {"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 1.0,
         "max_image_seq_len": 4096, "max_shift": 3.0, "num_train_timesteps": 1000, "shift": 1.0,
         "use_dynamic_shifting": True}
PER_FRAME = {"family": "per_frame_sqrt_frames", "frames": "runtime.latent_frames"}
# the vendor run: z (1, 16, 13, 24, 42), patch 2 -> 252 tokens a frame, 13 latent frames, 4 steps
VENDOR_4 = [1.0, 0.9152119755744934, 0.7825160026550293, 0.5453194975852966]
TOKENS, FRAMES = 13 * 252, 13


def _torch(decl):
    return FlowEulerScheduler(dict(OSORA, **({"dynamic_shift_length": decl} if decl is not None else {})))


def _triton(decl):
    pytest.importorskip("triton")
    from neurobrix.triton.scheduler.flow_euler import TritonFlowEulerScheduler
    return TritonFlowEulerScheduler(dict(OSORA, **({"dynamic_shift_length": decl} if decl is not None else {})))


def _ts(s):
    return [float(t) for t in (s._ts_np if hasattr(s, "_ts_np") else s.timesteps)]


@pytest.mark.parametrize("engine", [_torch, _triton])
def test_the_per_frame_length_gives_the_vendor_s_timesteps(engine):
    s = engine(PER_FRAME)
    assert s.shift_frames_pointer == "runtime.latent_frames"
    s.set_timesteps(4, image_seq_len=TOKENS, frames=FRAMES)
    np.testing.assert_allclose(_ts(s), VENDOR_4, atol=1e-6)


@pytest.mark.parametrize("engine", [_torch, _triton])
def test_the_per_frame_length_moves_with_the_request_s_frames(engine):
    """A second length, far from the trace: 33 latent frames at 252 tokens each."""
    s = engine(PER_FRAME)
    s.set_timesteps(4, image_seq_len=33 * 252, frames=33)
    alpha = (1.0 + (3.0 - 1.0) / (4096 - 256) * (252 - 256)) * np.sqrt(33)
    t = np.linspace(1, 0, 5)[:-1]
    np.testing.assert_allclose(_ts(s), alpha * t / (1 + (alpha - 1) * t), atol=1e-6)


@pytest.mark.parametrize("engine", [_torch, _triton])
@pytest.mark.parametrize("decl", [None, {"family": "total"}])
def test_absent_the_shift_reads_the_total_token_count(engine, decl):
    s = engine(decl)
    assert s.shift_frames_pointer is None
    s.set_timesteps(4, image_seq_len=TOKENS)
    mu = 1.0 + (3.0 - 1.0) / (4096 - 256) * (TOKENS - 256)
    t = np.linspace(1, 0, 5)[:-1]
    np.testing.assert_allclose(_ts(s), mu * t / (1 + (mu - 1) * t), atol=1e-6)


def test_absent_the_core_schedule_is_byte_for_byte_the_previous_one():
    s, ref = _torch(None), _torch(None)
    s.set_timesteps(20, image_seq_len=TOKENS)
    mu = ref._calculate_mu(TOKENS)
    assert torch.equal(s.timesteps, ref._apply_shift(torch.linspace(1, 0, 21)[:-1], mu))


@pytest.mark.parametrize("engine", [_torch, _triton])
def test_a_missing_frame_count_is_refused_by_name(engine):
    s = engine(PER_FRAME)
    with pytest.raises((SchedulerConfigError, RuntimeError), match="runtime.latent_frames"):
        s.set_timesteps(4, image_seq_len=TOKENS)


@pytest.mark.parametrize("engine", [_torch, _triton])
def test_tokens_that_do_not_split_into_the_frames_are_refused(engine):
    s = engine(PER_FRAME)
    with pytest.raises((SchedulerConfigError, RuntimeError), match="do not split"):
        s.set_timesteps(4, image_seq_len=TOKENS + 1, frames=FRAMES)


@pytest.mark.parametrize("engine", [_torch, _triton])
@pytest.mark.parametrize("decl", [{"family": "sqrt"}, {"family": "per_frame_sqrt_frames"},
                                  {"family": "per_frame_sqrt_frames", "frames": 13}, "total"])
def test_an_unknown_or_incomplete_declaration_is_refused(engine, decl):
    with pytest.raises((SchedulerConfigError, RuntimeError), match="ZERO FALLBACK"):
        engine(decl)


def test_the_container_flag_reaches_the_scheduler_config():
    from neurobrix.nbx import component_flags
    from neurobrix.core.runtime.registry_flags import get_component_flag
    component_flags.register("a-model", {"scheduler": {"shift": 1.0, "dynamic_shift_length": PER_FRAME}})
    try:
        assert get_component_flag("a-model", "scheduler", "dynamic_shift_length") == PER_FRAME
    finally:
        component_flags.clear()


def test_the_runtime_pointer_resolves_to_the_request_s_latent_frames():
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    vr = VariableResolver.__new__(VariableResolver)
    vr.defaults = {"latent_frames": 33}
    assert vr.resolve_pointer("runtime.latent_frames") == 33
