"""Triton UniPC multistep — zero-torch mirror of
core/module/scheduler/diffusion/unipc_multistep.py.

Same predictor (UniP) / corrector (UniC) B(h) algorithm and the same flow-sigma
schedule, but: the noise schedule + every per-step coefficient (sigma, alpha,
lambda, h, h_phi, B_h, rks, rhos) is computed in numpy/Python float (CPU,
init/step time), and the only tensor operands (sample, model_output, the
multistep buffer) are NBXTensor — whose +,-,*,/ dispatch to triton kernels. No
torch anywhere (R33).

Scope: solver_order <= 2 (the closed-form rhos: predictor 2nd-order uses
rhos_p=[0.5], corrector 1st-order uses rhos_c=[0.5]), which is what Wan2.1 (and
every diffusers UniPC default) uses. Order >= 3 needs a linear solve over a
coefficient matrix (torch.linalg.solve in the PyTorch path); NBXTensor has no
solver, so order >= 3 raises rather than silently degrade.
"""
import math
import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor


class TritonUniPCMultistepScheduler:
    """Zero-torch UniPC multistep scheduler (NBXTensor latents), order <= 2."""

    def __init__(self, config: dict):
        self.config = config
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.prediction_type = config.get("prediction_type", "flow_prediction")
        self.solver_order = int(config.get("solver_order", 2))
        self.solver_type = config.get("solver_type", "bh2")
        self.predict_x0 = bool(config.get("predict_x0", True))
        self.lower_order_final = bool(config.get("lower_order_final", True))
        self.disable_corrector = list(config.get("disable_corrector", []) or [])
        self.use_flow_sigmas = bool(config.get("use_flow_sigmas", False))
        self.flow_shift = float(config.get("flow_shift", 1.0))
        self.final_sigmas_type = config.get("final_sigmas_type", "zero")

        if self.solver_type not in ("bh1", "bh2"):
            raise RuntimeError(
                f"ZERO FALLBACK: TritonUniPC solver_type must be 'bh1'/'bh2', "
                f"got '{self.solver_type}'.")
        if self.solver_order > 2:
            raise RuntimeError(
                f"ZERO FALLBACK: TritonUniPC supports solver_order <= 2 "
                f"(NBXTensor has no linear solver for the order>=3 rhos); got "
                f"{self.solver_order}.")

        # Non-flow schedule support (alphas_cumprod) only if needed.
        self.alphas_cumprod = None
        if not self.use_flow_sigmas:
            beta_start = float(config.get("beta_start", 0.0001))
            beta_end = float(config.get("beta_end", 0.02))
            betas = np.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=np.float64)
            self.alphas_cumprod = np.cumprod(1.0 - betas)

        self.num_inference_steps = None
        self._ts_np = None
        self.timesteps = None
        self.sigmas = None
        self._step_index = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0

    # ---- schedule (numpy; mirrors the PyTorch UniPC flow/standard blocks) ----
    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self.num_inference_steps = num_inference_steps
        if self.use_flow_sigmas:
            alphas = np.linspace(1, 1 / self.num_train_timesteps,
                                 num_inference_steps + 1, dtype=np.float64)
            sigmas = 1.0 - alphas
            sigmas = np.flip(
                self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
            )[:-1].copy()
            timesteps = (sigmas * self.num_train_timesteps).copy()
            sigma_last = sigmas[-1] if self.final_sigmas_type == "sigma_min" else 0.0
            self.sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float64)
            self._ts_np = timesteps.astype(np.int64)
        else:
            ts = (np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
                  .round()[::-1][:-1].copy().astype(np.int64))
            ac = self.alphas_cumprod
            train_sigmas = ((1 - ac) / ac) ** 0.5
            sigmas = np.interp(ts, np.arange(len(train_sigmas)), train_sigmas)
            sigma_last = (((1 - ac[0]) / ac[0]) ** 0.5
                          if self.final_sigmas_type == "sigma_min" else 0.0)
            self.sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float64)
            self._ts_np = ts
        self.timesteps = [NBXTensor.from_numpy(np.array([int(t)], dtype=np.int64))
                          for t in self._ts_np]
        self._step_index = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0

    def _init_step_index(self, timestep):
        ts = int(timestep.item()) if isinstance(timestep, NBXTensor) else int(timestep)
        self._step_index = int(np.abs(self._ts_np - ts).argmin())

    @property
    def step_index(self):
        return self._step_index

    def _sigma_to_alpha_sigma_t(self, sigma: float):
        if self.use_flow_sigmas:
            return (1.0 - sigma), sigma
        alpha_t = 1.0 / (sigma ** 2 + 1.0) ** 0.5
        return alpha_t, sigma * alpha_t

    # ---- model-output conversion (NBXTensor) ----
    def convert_model_output(self, model_output: NBXTensor, sample: NBXTensor) -> NBXTensor:
        sigma = float(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        pt = self.prediction_type
        if self.predict_x0:
            if pt == "epsilon":
                return (sample - model_output * sigma_t) * (1.0 / alpha_t)
            if pt == "sample":
                return model_output
            if pt == "v_prediction":
                return sample * alpha_t - model_output * sigma_t
            if pt in ("flow", "flow_prediction"):
                return sample - model_output * float(self.sigmas[self._step_index])
            raise RuntimeError(f"ZERO FALLBACK: unknown prediction_type '{pt}'")
        # noise-prediction mode
        if pt == "epsilon":
            return model_output
        if pt == "v_prediction":
            return model_output * alpha_t + sample * sigma_t
        if pt == "sample":
            return (sample - model_output * alpha_t) * (1.0 / sigma_t)
        raise RuntimeError(f"ZERO FALLBACK: unknown prediction_type '{pt}'")

    def _bh_scalars(self, h: float):
        """B(h) scalar coefficients shared by predictor/corrector."""
        hh = -h if self.predict_x0 else h
        h_phi_1 = math.expm1(hh)               # e^hh - 1
        B_h = math.expm1(hh) if self.solver_type == "bh2" else hh
        return hh, h_phi_1, B_h

    # ---- predictor (UniP, B(h)); order <= 2 closed form ----
    def multistep_uni_p_bh_update(self, model_output: NBXTensor, sample: NBXTensor,
                                  order: int) -> NBXTensor:
        m0 = self.model_outputs[-1]
        x = sample
        s_t = float(self.sigmas[self._step_index + 1])
        s_s0 = float(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(s_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(s_s0)
        lambda_t = math.log(alpha_t) - math.log(sigma_t)
        lambda_s0 = math.log(alpha_s0) - math.log(sigma_s0)
        h = lambda_t - lambda_s0
        _, h_phi_1, B_h = self._bh_scalars(h)

        # order-2 second model output (previous converted output) + rk
        D1 = None
        if order == 2:
            mi = self.model_outputs[-2]
            si = self._step_index - 1
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(float(self.sigmas[si]))
            lambda_si = math.log(alpha_si) - math.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            D1 = (mi - m0) * (1.0 / rk)         # NBXTensor

        if self.predict_x0:
            x_t = x * (sigma_t / sigma_s0) - m0 * (alpha_t * h_phi_1)
            if D1 is not None:
                x_t = x_t - D1 * (0.5 * alpha_t * B_h)   # rhos_p=[0.5]
        else:
            x_t = x * (alpha_t / alpha_s0) - m0 * (sigma_t * h_phi_1)
            if D1 is not None:
                x_t = x_t - D1 * (0.5 * sigma_t * B_h)
        return x_t

    # ---- corrector (UniC, B(h)); order <= 2 closed form ----
    def multistep_uni_c_bh_update(self, this_model_output: NBXTensor,
                                  last_sample: NBXTensor, this_sample: NBXTensor,
                                  order: int) -> NBXTensor:
        m0 = self.model_outputs[-1]
        x = last_sample
        model_t = this_model_output
        s_t = float(self.sigmas[self._step_index])
        s_s0 = float(self.sigmas[self._step_index - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(s_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(s_s0)
        lambda_t = math.log(alpha_t) - math.log(sigma_t)
        lambda_s0 = math.log(alpha_s0) - math.log(sigma_s0)
        h = lambda_t - lambda_s0
        _, h_phi_1, B_h = self._bh_scalars(h)

        D1 = None
        if order == 2:
            mi = self.model_outputs[-2]
            si = self._step_index - 2
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(float(self.sigmas[si]))
            lambda_si = math.log(alpha_si) - math.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            D1 = (mi - m0) * (1.0 / rk)

        # rhos_c: order 1 -> [0.5]; order 2 -> linalg.solve(R,b) closed form.
        # For order 2 with rks=[rk,1], R=[[1,1],[rk,1]], b=[h_phi_1/B_h,
        # h_phi_2/B_h]; the diffusers closed form gives rhos_c = [r0, r1] with the
        # corrector residual corr_res + rhos_c[-1]*D1_t. For order 1 the corrector
        # uses rhos_c[-1]=0.5 and no corr_res.
        D1_t = model_t - m0
        if self.predict_x0:
            x_t = x * (sigma_t / sigma_s0) - m0 * (alpha_t * h_phi_1)
            if order == 1:
                x_t = x_t - D1_t * (0.5 * alpha_t * B_h)
            else:
                # order 2 closed-form rhos_c (see _rhos_c_order2)
                r_corr, r_last = self._rhos_c_order2(h, rk)
                x_t = x_t - (D1 * r_corr + D1_t * r_last) * (alpha_t * B_h)
        else:
            x_t = x * (alpha_t / alpha_s0) - m0 * (sigma_t * h_phi_1)
            if order == 1:
                x_t = x_t - D1_t * (0.5 * sigma_t * B_h)
            else:
                r_corr, r_last = self._rhos_c_order2(h, rk)
                x_t = x_t - (D1 * r_corr + D1_t * r_last) * (sigma_t * B_h)
        return x_t

    def _rhos_c_order2(self, h: float, rk: float):
        """Closed-form solve of R rhos = b for the order-2 corrector.
        R = [[1, 1], [rk, 1]], b = [h_phi_1/B_h - 1 (=h_phi_k), ...]. Mirrors the
        diffusers 2x2 torch.linalg.solve, done with Cramer's rule (scalars)."""
        hh = -h if self.predict_x0 else h
        h_phi_1 = math.expm1(hh)
        B_h = math.expm1(hh) if self.solver_type == "bh2" else hh
        h_phi_k = h_phi_1 / hh - 1.0
        b0 = h_phi_k * 1.0 / B_h
        h_phi_k = h_phi_k / hh - 1.0 / 2.0
        b1 = h_phi_k * 2.0 / B_h
        # R = [[1,1],[rk,1]] -> det = 1 - rk
        det = 1.0 - rk
        r0 = (b0 * 1.0 - 1.0 * b1) / det
        r1 = (1.0 * b1 - rk * b0) / det
        return r0, r1

    def step(self, model_output: NBXTensor, timestep, sample: NBXTensor,
             return_dict: bool = True, **kwargs):
        if self.num_inference_steps is None:
            raise RuntimeError("ZERO FALLBACK: set_timesteps() before step()")
        if self._step_index is None:
            self._init_step_index(timestep)

        use_corrector = (self._step_index > 0
                         and (self._step_index - 1) not in self.disable_corrector
                         and self.last_sample is not None)
        model_output_convert = self.convert_model_output(model_output, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert, last_sample=self.last_sample,
                this_sample=sample, order=self.this_order)

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output_convert

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self._ts_np) - self._step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output, sample=sample, order=self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def scale_model_input(self, sample: NBXTensor, timestep=None) -> NBXTensor:
        return sample

    @property
    def init_noise_sigma(self) -> float:
        return 1.0
