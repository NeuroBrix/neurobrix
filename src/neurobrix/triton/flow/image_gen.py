"""Generative-image leg (P-OMNI-GEN model 2/3) — triton engines (R33-pure).

Mirror of core/flow/image_gen.py over the SAME flow.image_gen contract
and the SAME component graphs — NBXTensor end-to-end, zero torch (R33).
The two engines share ONE seeded CPU fp32 gaussian frontier
(kernels/seeded_draw.py:seeded_gaussian): identical seed, identical
initial latents, so the engines can only diverge through kernel
numerics (R30 by construction).

Structure is the compiled leg's, section for section:

  1. condition_connector run on the LAST scale's gen hiddens (delivered
     by the vlm splice branch) -> prompt_embeds; negative = zero embeds
     (vendor contract).
  2. FlowMatchEuler loop (TritonSchedulerFactory, shift from the
     contract) with 2-chunk CFG over the denoiser component; timestep
     conditioning through the contract's timestep_scale
     (per-component-scale doctrine: our flow schedulers expose raw
     [0,1] sigmas; the DiT conditions on sigma*1000).
  3. image_vae decode leg (contract scaling/shift factors) ->
     resolved["global.output_image"] (the CLI writes the PNG; the flow
     never writes files).

Every quantity comes from topology.flow.image_gen / pkg.defaults — no
model names, no literals (ZERO FALLBACK on missing contract keys).
"""

import time
from typing import Any, Dict

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels.seeded_draw import seeded_gaussian
from neurobrix.triton.device_transfer import needs_move, transfer_tensor


def _require(block: Dict[str, Any], key: str, where: str):
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.image_gen is missing '{key}' "
            f"({where}) — re-emit the image_gen contract from the registry.")
    return val


def _from_np_on(arr: np.ndarray, dev_idx: int) -> NBXTensor:
    """Upload a host array to a SPECIFIC device (speech-leg idiom: under
    a multi-GPU placement the leg's components sit on different devices,
    so every host-created tensor is pinned explicitly)."""
    prev = DeviceAllocator.get_device()
    try:
        DeviceAllocator.set_device(dev_idx)
        return NBXTensor.from_numpy(arr)
    finally:
        DeviceAllocator.set_device(prev)


class ImageGenLeg:
    """Triton-engine image-gen leg over the flow.image_gen contract."""

    def __init__(self, engine):
        # engine = the triton VLMEngine instance: same ctx, same
        # component plumbing, same lifecycle (compiled-leg idiom).
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved
        self._dev: int = 0   # bound to the gen-hidden tap's device in run()

    # ── component plumbing (dual-write + run, compiled-leg idiom) ──

    def _run(self, comp: str, **inputs) -> NBXTensor:
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        if needs_move(out, self._dev):
            out = transfer_tensor(out, self._dev)
        return out

    def _load_query_tokens_np(self) -> Dict[str, np.ndarray]:
        """Load the query-token constants from the container asset
        (runtime/image_gen_constants.safetensors) as HOST numpy arrays —
        the vlm splice branch pins them onto the LM's device. Keys:
        <prefix>.{s}x{s}."""
        from pathlib import Path
        from safetensors import safe_open
        base = Path(self.ctx.pkg.cache_path
                    if hasattr(self.ctx.pkg, "cache_path")
                    else self.ctx.pkg.path)
        asset = base / "runtime" / "image_gen_constants.safetensors"
        if not asset.exists():
            raise RuntimeError(
                "ZERO FALLBACK: runtime/image_gen_constants.safetensors "
                "absent from the container — rebuild with the image_gen "
                "contract (query-token constants).")
        out: Dict[str, np.ndarray] = {}
        with safe_open(str(asset), framework="np", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out

    # ── the leg ──

    def run(self, state: Dict[str, Any]) -> None:
        ig = self.ctx.pkg.topology.get("flow", {}).get("image_gen")
        if not ig:
            raise RuntimeError(
                "ZERO FALLBACK: image-gen leg invoked without "
                "topology.flow.image_gen.")
        t0 = time.perf_counter()
        gen_hidden: NBXTensor = state["gen_hidden"]          # [1, s², H]
        dtype = state["dtype"]
        self._dev = int(gen_hidden._device_idx)

        comps = _require(ig, "components", "functional component slots")
        c_conn = _require(comps, "connector", "image_gen components")
        c_den = _require(comps, "denoiser", "image_gen components")
        c_vae = _require(comps, "vae", "image_gen components")

        steps = int(self.resolved.get("global.steps")
                    or _require(ig, "steps", "guidance contract"))
        gscale = float(self.resolved.get("global.guidance_scale")
                       or _require(ig, "guidance_scale", "guidance contract"))
        height = int(self.resolved.get("global.height")
                     or _require(ig, "height", "geometry"))
        width = int(self.resolved.get("global.width")
                    or _require(ig, "width", "geometry"))
        sched_c = _require(ig, "scheduler", "scheduler contract")
        # Per-component timestep scale (the
        # _get_component_timestep_scale doctrine): our flow schedulers
        # expose raw [0,1] sigmas; the DiT conditions on sigma*1000.
        ts_scale = float(_require(ig, "timestep_scale",
                                  "timestep conditioning"))

        # Registry-driven seed (generation.seed → defaults.json);
        # CLI global.seed overrides. `is None` checks — seed 0 legal.
        _seed_v = self.resolved.get("global.seed")
        if _seed_v is None:
            _seed_v = self.ctx.pkg.defaults.get("seed")
        if _seed_v is None:
            raise RuntimeError(
                "ZERO FALLBACK: no RNG seed — the build must emit "
                "generation.seed into defaults.json; re-import a current "
                "build or pass --set global.seed.")
        seed = int(_seed_v)
        print(f"   [image_gen] leg start: {width}x{height} steps={steps} "
              f"cfg={gscale} seed={seed}")

        # ── 1. connector → prompt embeds ─────────────────────────────
        prompt_embeds = self._run(c_conn, hidden_states=gen_hidden)
        neg_embeds = prompt_embeds * 0.0                     # vendor contract
        cond = NBXTensor.cat([neg_embeds, prompt_embeds], dim=0)  # [2, N, H]

        # ── 2. FlowMatchEuler loop + 2-chunk CFG ─────────────────────
        from neurobrix.triton.scheduler.factory import TritonSchedulerFactory
        scheduler = TritonSchedulerFactory.create({
            "_class_name": str(_require(sched_c, "type",
                                        "scheduler contract")),
            **{k: v for k, v in sched_c.items() if k != "type"}})
        vae_c = _require(ig, "vae", "vae contract")
        vae_scale = int(_require(vae_c, "vae_scale_factor", "vae contract"))
        lat_ch = int(_require(vae_c, "latent_channels", "vae contract"))
        h_lat, w_lat = height // vae_scale, width // vae_scale
        # Latent init through the SHARED seeded frontier (R30): both
        # engines consume the same CPU fp32 array — RNG provenance can
        # never explain a cross-engine divergence.
        latents = _from_np_on(
            seeded_gaussian(seed, (1, lat_ch, h_lat, w_lat)), self._dev)

        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise RuntimeError(
                "ZERO FALLBACK: scheduler produced no timesteps.")
        # F1 step cache (drift tier, opt-in) — the shared brick, same
        # config channels as the diffusion flow (registry step_cache /
        # --set global.step_cache_* / NBX_STEP_CACHE_*). The signal is
        # observed on the post-guidance prediction; a skip reuses it
        # while the scheduler still advances the timestep. R30 mirror
        # of the compiled leg.
        from neurobrix.triton.flow.step_cache import StepCache
        _sc = StepCache.setup(self.ctx, steps)
        for i, t in enumerate(timesteps):
            t_f = float(t.item()) if isinstance(t, NBXTensor) else float(t)
            if _sc is not None and _sc.should_skip(i):
                guided = _sc.prev_pred
                latents = scheduler.step(
                    guided.to("float32"), t_f, latents.to("float32"),
                    return_dict=False)
                if (i + 1) % 5 == 0:
                    print(f"   [image_gen] step {i + 1}/{steps}")
                continue
            lat_h = latents.to(dtype)
            lat_in = NBXTensor.cat([lat_h, lat_h], dim=0)
            t_in = _from_np_on(
                np.full((2,), t_f * ts_scale, dtype=np.float32),
                self._dev).to(dtype)
            noise_pred = self._run(
                c_den, hidden_states=lat_in,
                timestep=t_in, encoder_hidden_states=cond,
                return_dict=False)
            if isinstance(noise_pred, (tuple, list)):
                noise_pred = noise_pred[0]
            half = int(noise_pred.shape[0]) // 2
            uncond = noise_pred.narrow(0, 0, half).contiguous()
            text = noise_pred.narrow(0, half, half).contiguous()
            guided = uncond + (text - uncond) * gscale
            if _sc is not None:
                _sc.observe(guided)
            latents = scheduler.step(
                guided.to("float32"), t_f, latents.to("float32"),
                return_dict=False)
            if (i + 1) % 5 == 0:
                print(f"   [image_gen] step {i + 1}/{steps}")
        if _sc is not None:
            _sc.report()

        # ── 3. VAE decode ─────────────────────────────────────────────
        sf = float(_require(vae_c, "scaling_factor", "vae contract"))
        sh = float(_require(vae_c, "shift_factor", "vae contract"))
        z = latents.to("float32") / sf + sh
        image = self._run(c_vae, sample=z.to(dtype))
        self.resolved["global.output_image"] = image
        dt = time.perf_counter() - t0
        print(f"   [image_gen] image {list(image.shape)} in {dt:.1f}s")
