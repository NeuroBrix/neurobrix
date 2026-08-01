"""Generative-image leg (P-OMNI-GEN model 2/3) — compiled engine.

Runs INSTEAD of the vlm text decode when the request asks --mode image
and the container declares topology.flow.image_gen (the registry-
emitted contract: condition layout, guidance, scheduler). The leg
mirrors the vendor pipeline op-for-op:

  prompt ids (chat template) + per-scale appended gen spans
  [start, patch x s^2, end] (scales from the contract; gen_mask marks
  patch positions)
    -> words embeds + QUERY-TOKEN CONSTANTS spliced at the gen
       positions (runtime/image_gen_constants.safetensors — static
       learned tensors, an .nbx constant per the conds.pt precedent)
    -> LM ENCODER pass (no AR decode; M-RoPE positions; principal
       hidden out)
    -> LAST scale's gen hiddens -> condition_connector component
       (proj_in -> non-causal qwen2 -> proj_out -> normalize, all
       in-graph) -> prompt_embeds
    -> FlowMatchEuler loop (OUR scheduler factory; shift/steps from
       the contract) with 2-chunk CFG through OUR CFG engine
       (negative = zero embeds, vendor contract) over the
       image_denoiser component (ToClipMLP + pooled IN-GRAPH)
    -> image_vae decode leg -> resolved["global.output_image"]
       (the CLI writes the PNG; the flow never writes files)

Every quantity is read from topology.flow.image_gen / pkg.defaults —
no model names, no literals (ZERO FALLBACK on missing contract keys).
The RNG is the generation.seed chain (CLI global.seed override >
defaults.seed > raise), driving latent init deterministically.
"""

import time
from typing import Any, Dict, List, Optional

import torch


def _require(block: Dict[str, Any], key: str, where: str):
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.image_gen is missing '{key}' "
            f"({where}) — re-emit the image_gen contract from the registry.")
    return val


class ImageGenLeg:
    """Compiled-engine image-gen leg over the flow.image_gen contract."""

    def __init__(self, engine):
        # engine = the VLMEngine instance (same ctx, same component
        # plumbing, same lifecycle — the SpeechLeg idiom).
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved

    def _run(self, comp: str, **inputs) -> torch.Tensor:
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        return out.to(self._device) if hasattr(out, "to") else out

    def _load_query_tokens(self) -> Dict[str, torch.Tensor]:
        """Load the query-token constants from the container asset
        (runtime/image_gen_constants.safetensors — the conds.pt loading
        precedent). Keys: <prefix>.{s}x{s}."""
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
        out: Dict[str, torch.Tensor] = {}
        with safe_open(str(asset), framework="pt", device="cpu") as f:
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
        device = state["device"]
        self._device = device
        dtype = state["dtype"]

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
        # The condition ENCODER pass lives in the vlm splice branch
        # (the gen spans + query-token constants ride the six-input
        # splice contract; the graph splices them) — the leg receives
        # the LAST scale's gen hiddens ready for the bridge.
        gen_hidden = state["gen_hidden"]                     # [1, s², H]
        prompt_embeds = self._run(c_conn, hidden_states=gen_hidden)
        neg_embeds = prompt_embeds * 0                       # vendor contract

        # ── 2. FlowMatchEuler loop + 2-chunk CFG ─────────────────────
        from neurobrix.core.module.scheduler.factory import SchedulerFactory
        scheduler = SchedulerFactory.create({
            "_class_name": str(_require(sched_c, "type",
                                        "scheduler contract")),
            **{k: v for k, v in sched_c.items() if k != "type"}})
        vae_c = _require(ig, "vae", "vae contract")
        vae_scale = int(_require(vae_c, "vae_scale_factor", "vae contract"))
        lat_ch = int(_require(vae_c, "latent_channels", "vae contract"))
        h_lat, w_lat = height // vae_scale, width // vae_scale
        gen = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, lat_ch, h_lat, w_lat), generator=gen,
            dtype=torch.float32).to(device=device, dtype=dtype)

        scheduler.set_timesteps(steps, device=device)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise RuntimeError(
                "ZERO FALLBACK: scheduler produced no timesteps.")
        cond = torch.cat([neg_embeds, prompt_embeds], dim=0)  # [2, N, H]
        for i, t in enumerate(timesteps):
            lat_in = torch.cat([latents, latents], dim=0)
            t_in = torch.full((2,), float(t) * ts_scale,
                              device=device, dtype=dtype)
            noise_pred = self._run(
                c_den, hidden_states=lat_in.to(dtype),
                timestep=t_in, encoder_hidden_states=cond,
                return_dict=False)
            if isinstance(noise_pred, (tuple, list)):
                noise_pred = noise_pred[0]
            uncond, text = noise_pred.chunk(2)
            guided = uncond + gscale * (text - uncond)
            latents = scheduler.step(
                guided.float(), t, latents.float(),
                return_dict=False).to(dtype)
            if (i + 1) % 5 == 0:
                print(f"   [image_gen] step {i + 1}/{steps}")

        # ── 6. VAE decode ─────────────────────────────────────────────
        sf = float(_require(vae_c, "scaling_factor", "vae contract"))
        sh = float(_require(vae_c, "shift_factor", "vae contract"))
        z = latents.float() / sf + sh
        image = self._run(c_vae, sample=z.to(dtype))
        self.resolved["global.output_image"] = image
        dt = time.perf_counter() - t0
        print(f"   [image_gen] image {list(image.shape)} in {dt:.1f}s")
