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
        self.num_inference_steps = None
        self.timesteps = None       # np.float64 [N]
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
        ts = np.linspace(1, 0, num_inference_steps + 1, dtype=np.float64)[:-1]
        if mu != 1.0:
            ts = mu * ts / (1 + (mu - 1) * ts)
        self.timesteps = ts.copy()
        self.sigmas = ts.copy()
        self._step_index = 0

    @property
    def step_index(self):
        return self._step_index

    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        t = float(timestep.item()) if isinstance(timestep, NBXTensor) else float(timestep)
        if self._step_index < len(self.timesteps) - 1:
            t_next = float(self.timesteps[self._step_index + 1])
        else:
            t_next = 0.0
        dt = t_next - t
        prev = sample + model_output * dt
        self._step_index += 1
        return {"prev_sample": prev} if return_dict else prev

    def scale_model_input(self, sample: NBXTensor, timestep) -> NBXTensor:
        return sample

    @property
    def init_noise_sigma(self) -> float:
        return 1.0
