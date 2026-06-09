# core/module/scheduler/diffusion/unipc_multistep.py
"""
UniPC Multistep Scheduler - ZERO FALLBACK, RUNTIME-PURE.

UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion
Models (Zhao et al., 2023) - https://huggingface.co/papers/2302.04867
Reference implementation: https://github.com/wl-zhao/UniPC

Used by: Wan2.1-T2V (flow_prediction), and any diffusion model whose
scheduler_config declares UniPCMultistepScheduler.
Matches diffusers UniPCMultistepScheduler numerically (max|diff| < 1e-4).

RUNTIME-PURE (R34): NO `import diffusers` anywhere. The predictor / corrector
B(h) math, convert_model_output, and flow-sigma schedule are ported to pure
torch from the vendored diffusers 0.36.0 algorithm oracle. The runtime engine
never imports a vendor scheduler package.

ZERO FALLBACK: all required config values MUST come from the NBX container.
Missing required key = explicit crash with a helpful message.
"""

import torch
from typing import Dict, Any, Union, Optional, List

from ..base import DiffusionSchedulerBase, PredictionType
from ..config import SchedulerConfig, SchedulerConfigError


class UniPCMultistepScheduler(DiffusionSchedulerBase):
    """
    UniPC Multistep Scheduler.

    A training-free predictor-corrector framework for fast diffusion sampling.
    Faithful pure-torch port of diffusers UniPCMultistepScheduler.

    Key features:
    - Predictor (UniP) + Corrector (UniC) B(h) updates, bh1 / bh2 variants
    - Arbitrary solver_order; closed-form for order<=2, torch.linalg.solve for
      order>=3
    - x0-prediction (`predict_x0=True`) or noise-prediction algorithm
    - Flow matching support (`use_flow_sigmas`, `flow_shift`,
      `prediction_type='flow_prediction'`) for rectified-flow models (Wan2.1)
    - Rank-agnostic: works for 4D image latents and 5D video latents via the
      `"k,bkc...->bc..."` einsum contraction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        ZERO FALLBACK: required keys must be present in config (validated by
        SchedulerConfig.validate). Use from_config() for NBX integration.
        """
        # ZERO FALLBACK: validate config - crash if required keys missing.
        validated = SchedulerConfig.validate(config, "UniPCMultistepScheduler")

        # REQUIRED values - guaranteed present after validation.
        self.num_train_timesteps = validated["num_train_timesteps"]
        beta_start = validated["beta_start"]
        beta_end = validated["beta_end"]
        beta_schedule = validated["beta_schedule"]
        self.prediction_type = PredictionType(validated["prediction_type"])
        self.timestep_spacing = validated["timestep_spacing"]
        self.solver_order = validated["solver_order"]
        self.solver_type = validated["solver_type"]
        self.lower_order_final = validated["lower_order_final"]

        # UniPC-specific values (OPTIONAL_SAFE_DEFAULTS, diffusers-matching).
        self.predict_x0 = validated.get("predict_x0", True)
        self.disable_corrector = validated.get("disable_corrector", [])

        # OPTIONAL values - safe defaults applied by validator.
        self.thresholding = validated["thresholding"]
        self.sample_max_value = validated["sample_max_value"]
        self.dynamic_thresholding_ratio = validated.get("dynamic_thresholding_ratio", 0.995)
        # final_sigmas_type: diffusers UniPC defaults to "zero" (NOT the shared
        # validator's "sigma_min", which is DPM++'s default). Read from the RAW
        # config so the UniPC default is correct standalone — Wan2.1's config may
        # omit this key, and "zero" is what its trace expects.
        self.final_sigmas_type = config.get("final_sigmas_type", "zero")

        # Flow matching support (Wan2.1 and newer flow models).
        self.use_flow_sigmas = validated.get("use_flow_sigmas", False)
        self.flow_shift = validated.get("flow_shift", 1.0)

        # ZERO FALLBACK: solver_type must be bh1 or bh2 (UniPC has no other
        # variants; legacy midpoint/heun/logrho map to bh2 in diffusers, but we
        # require an explicit valid value here rather than silently coercing).
        if self.solver_type not in ("bh1", "bh2"):
            raise SchedulerConfigError(
                f"ZERO FALLBACK: UniPCMultistepScheduler solver_type must be "
                f"'bh1' or 'bh2', got '{self.solver_type}'."
            )

        # Compute the training noise schedule (used by the non-flow sigma path
        # and add_noise). Mirrors diffusers: betas -> alphas -> alphas_cumprod.
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            betas = (
                torch.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    self.num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            betas = self._betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Unknown beta_schedule '{beta_schedule}'"
            )

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # State.
        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

        # Multistep buffers (diffusers order: oldest at [0], newest at [-1]).
        self.model_outputs: List[Optional[torch.Tensor]] = [None] * self.solver_order
        self.timestep_list: List[Optional[Union[int, torch.Tensor]]] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: Optional[torch.Tensor] = None
        self.this_order: int = 0

    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
        """Cosine (Glide) beta schedule. Pure-torch port of diffusers helper."""
        import math

        def alpha_bar_fn(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        """Set the discrete timesteps and sigmas used for the diffusion chain."""
        import numpy as np

        self.num_inference_steps = num_inference_steps

        if self.use_flow_sigmas:
            # Flow matching: compute sigmas FIRST, then derive timesteps.
            # Mirrors diffusers UniPC/DPM flow-sigma block EXACTLY:
            #   alphas = linspace(1, 1/num_train_timesteps, num_steps + 1)
            #   sigmas = 1 - alphas
            #   sigmas = flip(flow_shift*sigmas/(1+(flow_shift-1)*sigmas))[:-1]
            #   timesteps = sigmas * num_train_timesteps
            alphas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)
            sigmas = 1.0 - alphas
            sigmas = np.flip(
                self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
            )[:-1].copy()
            timesteps = (sigmas * self.num_train_timesteps).copy()

            if self.final_sigmas_type == "sigma_min":
                sigma_last = sigmas[-1]
            else:  # "zero" (diffusers UniPC default)
                sigma_last = 0.0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

            self.sigmas = torch.from_numpy(sigmas)
            self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
        else:
            # Standard diffusion: timestep spacing (diffusers linspace branch),
            # then sigmas via np.interp over the training sigma schedule.
            if self.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
                    .round()[::-1][:-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.timestep_spacing == "leading":
                step_ratio = self.num_train_timesteps // (num_inference_steps + 1)
                timesteps = (
                    (np.arange(0, num_inference_steps + 1) * step_ratio)
                    .round()[::-1][:-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.timestep_spacing == "trailing":
                step_ratio = self.num_train_timesteps / num_inference_steps
                timesteps = np.arange(
                    self.num_train_timesteps, 0, -step_ratio
                ).round().copy().astype(np.int64)
                timesteps -= 1
            else:
                raise SchedulerConfigError(
                    f"ZERO FALLBACK: Unknown timestep_spacing '{self.timestep_spacing}'"
                )

            alphas_cumprod = self.alphas_cumprod.cpu().numpy()
            train_sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            sigmas = np.interp(timesteps, np.arange(0, len(train_sigmas)), train_sigmas)

            if self.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - alphas_cumprod[0]) / alphas_cumprod[0]) ** 0.5
            else:  # "zero" (diffusers UniPC default)
                sigma_last = 0.0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

            self.sigmas = torch.from_numpy(sigmas)
            self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)

        # num_inference_steps reflects the actual number of discrete timesteps.
        self.num_inference_steps = len(self.timesteps)

        if device is not None:
            self.timesteps = self.timesteps.to(device)

        # diffusers keeps sigmas on CPU to avoid CPU/GPU chatter; the step math
        # extracts scalars from them. Mirror that (sigmas stay on CPU).

        # Reset multistep state.
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0
        self._step_index = None

    @property
    def step_index(self):
        return self._step_index

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        """Initialize step index from a timestep (diffusers index_for_timestep)."""
        assert self.timesteps is not None, "timesteps must be set"
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        elif len(index_candidates) > 1:
            # Second occurrence (diffusers convention for mid-schedule starts).
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        self._step_index = int(step_index)

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor):
        """Convert a sigma to (alpha_t, sigma_t). Flow branch: alpha=1-s, sigma=s."""
        if self.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding from Imagen (pure-torch port of diffusers)."""
        import numpy as np

        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))
        abs_sample = sample.abs()

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)
        return sample

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert the raw model output to the type UniPC needs.
        Faithful port of diffusers UniPC convert_model_output.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"
        sigma = self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.prediction_type == PredictionType.EPSILON:
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == PredictionType.SAMPLE:
                x0_pred = model_output
            elif self.prediction_type == PredictionType.V_PREDICTION:
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.prediction_type in (
                PredictionType.FLOW,
                PredictionType.FLOW_PREDICTION,
            ):
                # Flow matching: sigma_t IS the raw sigma at step_index.
                sigma_t = self.sigmas[self._step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise SchedulerConfigError(
                    f"ZERO FALLBACK: Unknown prediction type '{self.prediction_type}' "
                    f"for UniPCMultistepScheduler."
                )

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred
        else:
            if self.prediction_type == PredictionType.EPSILON:
                return model_output
            elif self.prediction_type == PredictionType.SAMPLE:
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.prediction_type == PredictionType.V_PREDICTION:
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise SchedulerConfigError(
                    f"ZERO FALLBACK: Unknown prediction type '{self.prediction_type}' "
                    f"for UniPCMultistepScheduler (noise-prediction mode supports "
                    f"epsilon / sample / v_prediction only)."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """
        One UniP (predictor, B(h) version) step.
        Faithful port of diffusers multistep_uni_p_bh_update.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self._step_index + 1], self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks: List[Union[float, torch.Tensor]] = []
        D1s: List[torch.Tensor] = []
        for i in range(1, order):
            si = self._step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R: List[torch.Tensor] = []
        b: List[torch.Tensor] = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h * phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Unknown solver_type '{self.solver_type}'"
            )

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s_t = torch.stack(D1s, dim=1)  # (B, K, ...)
            # Closed form for order 2; linear solve for order >= 3.
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s_t = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s_t is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s_t)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s_t is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s_t)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """
        One UniC (corrector, B(h) version) step.
        Faithful port of diffusers multistep_uni_c_bh_update.
        """
        assert self._step_index is not None, "step_index must be initialized"
        assert self.sigmas is not None, "sigmas must be set"

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self._step_index], self.sigmas[self._step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks: List[Union[float, torch.Tensor]] = []
        D1s: List[torch.Tensor] = []
        for i in range(1, order):
            si = self._step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R: List[torch.Tensor] = []
        b: List[torch.Tensor] = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h * phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise SchedulerConfigError(
                f"ZERO FALLBACK: Unknown solver_type '{self.solver_type}'"
            )

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s_t = torch.stack(D1s, dim=1)
        else:
            D1s_t = None

        # Closed form for order 1; linear solve otherwise.
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s_t is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s_t)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s_t is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s_t)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        x_t = x_t.to(x.dtype)
        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict the sample at the previous timestep via the multistep UniPC.
        Faithful port of diffusers UniPCMultistepScheduler.step.

        CRITICAL: statement order matches diffusers exactly. The corrector runs
        on the un-shifted multistep buffer (consuming the previous step's
        converted output) BEFORE the buffer is shifted and the current converted
        output is stored. self.this_order / self.last_sample / self.lower_order_nums
        persist across steps and gate the warmup order ramp 1 -> solver_order.
        """
        if self.num_inference_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: set_timesteps() must be called before step()"
            )

        assert self.timesteps is not None, "timesteps must be set"

        # Initialize step index on first call.
        if self._step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self._step_index > 0
            and self._step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Shift multistep buffers left, store current converted output at the end.
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self.solver_order

        # Warmup ramp for multistep (order grows 1 -> solver_order).
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # original (non-converted) output
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # Increment step index for the next call.
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion). Port of diffusers add_noise."""
        assert self.sigmas is not None, "set_timesteps() must be called before add_noise()"
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        step_indices = []
        for t in timesteps:
            index_candidates = (schedule_timesteps == t).nonzero()
            if len(index_candidates) == 0:
                idx = len(schedule_timesteps) - 1
            elif len(index_candidates) > 1:
                idx = index_candidates[1].item()
            else:
                idx = index_candidates[0].item()
            step_indices.append(int(idx))

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scale input sample. UniPC does not scale the input (returns as-is)."""
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma. UniPC uses unit variance noise."""
        return 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UniPCMultistepScheduler":
        """Create scheduler from NBX config."""
        return cls(config)
