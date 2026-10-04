"""Triton FlowEuler — zero-torch mirror of core/module/scheduler/flow/flow_euler.py.

Flow-matching Euler: timesteps run 1 -> 0 (with optional time-shift), sigma == t,
and each step is  prev = sample + (t_next - t) * model_output. numpy for the
timestep schedule, Python float for the dt scalar, NBXTensor for the latent.
No torch. Used by Sana/Flex/SD3-class flow-matching diffusion in triton mode.
"""
import numpy as np
from neurobrix.kernels.nbx_tensor import NBXTensor


class TritonFlowEulerScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.shift = float(config.get("shift", config.get("flow_shift", 1.0)))
        self.base_shift = float(config.get("base_shift", 0.5))
        self.max_shift = float(config.get("max_shift", 1.15))
        self.base_image_seq_len = int(config.get("base_image_seq_len", 256))
        self.max_image_seq_len = int(config.get("max_image_seq_len", 4096))
        # Inverted-sigma flow (Mochi: invert_sigmas=True) — R30 mirror of core
        # flow_euler.py (commit 3770103). Diffusers FlowMatchEulerDiscrete flips
        # the schedule `sigmas = 1 - sigmas` with a TERMINAL sigma of 1 (not 0).
        # Without it the denoising direction is reversed and the latents never
        # leave the noise distribution (divergent / uniform-noise output). Gated
        # on invert_sigmas, so non-inverted flow (Flux/SD3/Wan) is untouched.
        self.invert_sigmas = bool(config.get("invert_sigmas", False))
        # R30 mirror of core flow_euler.py: the pipeline's declared sigma schedule.
        self.sigma_schedule = _validate_sigma_schedule(config.get("sigma_schedule"))
        # Video flow-match denoisers (Mochi) consume RAW [0, num_train_timesteps]
        # timesteps; the loop scales the [0,1] sigma by this (R30 mirror of core
        # _get_component_timestep_scale, commit 3770103 part 2).
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.num_inference_steps = None
        self._ts_np = None          # np.float64 [N] — internal
        self.timesteps = None       # list[NBXTensor [1]] — exposed to loop/CFG
        self.sigmas = None
        self._step_index = 0

    def _calculate_mu(self, image_seq_len: int) -> float:
        m = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        b = self.base_shift - m * self.base_image_seq_len
        return image_seq_len * m + b

    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        image_seq_len = kwargs.get("image_seq_len", None)
        mu = self._calculate_mu(image_seq_len) if image_seq_len is not None else self.shift
        ts = _flow_sigma_schedule(self.sigma_schedule, num_inference_steps)
        if mu != 1.0:
            ts = mu * ts / (1 + (mu - 1) * ts)
        if self.invert_sigmas:
            ts = 1.0 - ts
        self._ts_np = ts.copy()
        self.sigmas = ts.copy()
        # Exposed as NBXTensor [1] floats (tensor-like timesteps for loop/CFG).
        self.timesteps = [NBXTensor.from_numpy(np.array([float(t)], dtype=np.float32))
                          for t in self._ts_np]
        self._step_index = 0

    @property
    def step_index(self):
        return self._step_index

    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        t = float(timestep.item()) if isinstance(timestep, NBXTensor) else float(timestep)
        if self._step_index < len(self._ts_np) - 1:
            t_next = float(self._ts_np[self._step_index + 1])
        else:
            # Terminal sigma: 1.0 for inverted flow (Mochi), 0.0 otherwise —
            # mirrors core flow_euler.py + diffusers' torch.cat([sigmas, 1/0]).
            t_next = 1.0 if self.invert_sigmas else 0.0
        dt = t_next - t
        prev = sample + model_output * dt
        self._step_index += 1
        return {"prev_sample": prev} if return_dict else prev

    def scale_model_input(self, sample: NBXTensor, timestep) -> NBXTensor:
        return sample

    @property
    def init_noise_sigma(self) -> float:
        return 1.0


# R30 mirror of core/module/scheduler/flow/flow_euler.py (numpy only, R33; no shared
# compute code between the engines). See flow_sigma_schedule there for the sources.
_SIGMA_SCHEDULE_FAMILIES = ("linspace", "linear_quadratic")


def _validate_sigma_schedule(decl):
    if decl is None:
        return None
    if not isinstance(decl, dict) or decl.get("family") not in _SIGMA_SCHEDULE_FAMILIES:
        raise RuntimeError(
            f"ZERO FALLBACK: unknown sigma_schedule {decl!r}; families: {_SIGMA_SCHEDULE_FAMILIES}.")
    if decl["family"] == "linear_quadratic" and not isinstance(decl.get("threshold_noise"), (int, float)):
        raise RuntimeError(
            f"ZERO FALLBACK: sigma_schedule linear_quadratic needs a numeric threshold_noise, got {decl!r}.")
    return dict(decl)


def _flow_sigma_schedule(decl, num_steps: int):
    if decl is None or decl.get("family") == "linspace":
        return np.linspace(1, 0, num_steps + 1, dtype=np.float64)[:-1]
    thr = float(decl["threshold_noise"])
    linear_steps = decl.get("linear_steps")
    linear_steps = num_steps // 2 if linear_steps is None else int(linear_steps)
    if linear_steps < 1 or linear_steps >= num_steps:
        raise RuntimeError(
            f"ZERO FALLBACK: sigma_schedule linear_quadratic needs 1 <= linear_steps < steps; "
            f"got linear_steps={linear_steps} at {num_steps} step(s).")
    lin = [i * thr / linear_steps for i in range(linear_steps)]
    diff = linear_steps - thr * num_steps
    q_steps = num_steps - linear_steps
    q_coef = diff / (linear_steps * q_steps ** 2)
    l_coef = thr / linear_steps - 2 * diff / (q_steps ** 2)
    const = q_coef * (linear_steps ** 2)
    quad = [q_coef * (i ** 2) + l_coef * i + const for i in range(linear_steps, num_steps)]
    return np.array([1.0 - x for x in lin + quad], dtype=np.float64)
