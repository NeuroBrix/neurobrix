"""
VLMEngine — Vision-Conditioned LLM Flow (image+text → text)

Handles models where an image is patch-encoded and fed as embeddings to an
autoregressive LLM decoder alongside text: GLM-4.1V, Qwen-VL lineage.

Pattern: vision_encoder(pixel_values, grid_thw) → merge with text embeds
         at the image-placeholder span → LLM(inputs_embeds, mrope
         position_ids) → text

Mirror of AudioLLMEngine (audio_llm.py). Vendor-equivalence note that
carries over verbatim: the vendor scatters vision embeddings over
placeholder token positions with masked_scatter; this flow CONCATENATES
prefix-embeds + vision-embeds + suffix-embeds, which is equivalent for a
CONTIGUOUS placeholder block (a single image span is contiguous by
construction of the chat template).

M-RoPE position_ids (the get_rope_index contract, re-implemented
NeuroBrix-internal per R34): positions have 3 planes [3, B, S]
(temporal/height/width). Text segments advance all three planes
sequentially from the running offset; an image segment of llm-grid
t×h×w (grid_thw with h,w divided by spatial_merge_size) gets
t/h/w meshgrid indices + the running offset; each following segment
starts at max(previous positions) + 1.

Second contract handled by the same engine — the STAGED-SPLICE chain
(MiniCPM-o class), detected data-driven from the vision graph's declared
inputs (input::all_pixel_values): vision tower → projection component →
LM whose modality splice is IN-GRAPH (bool-mask masked_scatter over
placeholder-token runs), with plain 1-D positions (no M-RoPE). See
_execute_staged_splice.

ZERO SEMANTIC: no model-name knowledge — everything reads topology.flow.vlm.
ZERO FALLBACK: missing schema fields raise.
"""

import os
import time
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import FlowHandler, FlowContext, register_flow
from neurobrix.core.memory.manager import release_flow_memory


@register_flow("vlm")
class VLMEngine(FlowHandler):
    """Vision-conditioned LLM: encode image → merge embeds → decode text.

    topology.flow.vlm:
        input:  {preprocessing, image_variable, grid_variable, prompt_variable}
        output: {variable}
        vision_component / lm_component / head_component
        image_token_id / image_start_token_id / image_end_token_id
        mrope_section / spatial_merge_size
        stages: [{component, execution: forward},
                 {component, execution: autoregressive, logits_source}]
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    # ─── main ────────────────────────────────────────────────────────────

    def execute(self) -> Dict[str, Any]:
        flow = self.ctx.pkg.topology.get("flow", {})
        vlm = flow.get("vlm", {})
        if not vlm:
            raise RuntimeError("ZERO FALLBACK: vlm flow requires topology.flow.vlm.")
        defaults = self.ctx.pkg.defaults
        resolved = self.ctx.variable_resolver.resolved

        vision_name = vlm.get("vision_component")
        lm_name = vlm.get("lm_component")
        head_name = vlm.get("head_component")

        # ── Staged-splice contract detection (data-driven from the DAG) ──
        # A MiniCPM-o-class vision graph declares the packed NaViT inputs
        # (input::all_pixel_values + patch_attention_mask + tgt_sizes) and
        # chains into a projection component (perceiver resampler); its LM
        # graph splices modality embeds IN-GRAPH via bool-mask
        # masked_scatter. The omni contract (vision graph fed
        # hidden_states + grid_thw) keeps the legacy path below
        # bit-for-bit — its graphs never declare all_pixel_values.
        _vis_exec = self.ctx.executors.get(vision_name) if vision_name else None
        _vis_tensors = (getattr(_vis_exec, "_dag", None) or {}).get(
            "tensors", {})
        if "input::all_pixel_values" in _vis_tensors:
            return self._execute_staged_splice(vlm)

        # ── M-RoPE masked-splice contract detection (data-driven) ──
        # A bailingmm-class LM graph declares TWO modality pairs as
        # inputs — image_pos_masks/image_embeds AND
        # audio_pos_masks/audio_embeds — plus rank-3 M-RoPE
        # position_ids: both the placeholder splice (masked_scatter) and
        # the per-modality MoE router gating live IN-GRAPH, and the flow
        # only supplies the four tensors. Discriminator = the declared
        # input NAME `image_pos_masks`, which is the contract itself:
        # the deepstack omni graphs declare `visual_pos_masks` +
        # `deepstack_visual_embeds.N`, the staged (MiniCPM) graphs
        # declare `visual_pos_masks` + `vision_hidden_states` and are
        # already routed above, and the GLM-class graphs declare neither.
        _lm_exec0 = self.ctx.executors.get(lm_name) if lm_name else None
        _lm_tensors0 = (getattr(_lm_exec0, "_dag", None) or {}).get(
            "tensors", {})
        if "input::image_pos_masks" in _lm_tensors0:
            return self._execute_mrope_masked_splice(vlm)

        image_token_id = vlm.get("image_token_id")
        merge = vlm.get("spatial_merge_size")
        if not vision_name or not lm_name or image_token_id is None or not merge:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.vlm must declare vision_component, "
                "lm_component, image_token_id and spatial_merge_size.")

        dtype = self._get_compute_dtype()
        device = self.ctx.primary_device

        # ── Step 1: request modality (image XOR audio, plus text) ──
        in_cfg = vlm.get("input", {})
        pixel_values = resolved.get(in_cfg.get("image_variable", "global.pixel_values"))
        grid_thw = resolved.get(in_cfg.get("grid_variable", "global.image_grid_thw"))
        pixel_values_videos = resolved.get("global.pixel_values_videos")
        video_grid_thw = resolved.get("global.video_grid_thw")
        video_spg = resolved.get("global.video_second_per_grid")
        audio_path = resolved.get("global.audio_path")
        audio_name = vlm.get("audio_component")
        prompt = resolved.get(in_cfg.get("prompt_variable", "global.prompt"))
        if not prompt:
            raise RuntimeError("ZERO FALLBACK: vlm flow requires a text prompt.")
        has_image = pixel_values is not None
        has_video = pixel_values_videos is not None
        if has_image and has_video:
            raise RuntimeError(
                "ZERO FALLBACK: one visual modality per request — provide "
                "--input-image or --input-video, not both.")
        if has_image and grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: image request missing the grid input "
                f"({in_cfg.get('grid_variable')}).")
        if has_video and video_grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: video request missing the grid input "
                "(global.video_grid_thw).")
        has_visual = has_image or has_video
        if not has_visual and audio_path is None:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires a modality input — "
                "provide --input-image, --input-video or --audio.")
        if not has_visual and not audio_name:
            raise RuntimeError(
                "ZERO FALLBACK: this build declares no audio_component — "
                "audio understanding needs a build traced with the audio "
                "tower.")

        # ── DeepStack detection (data-driven from the LM graph spec) ──
        # A Qwen3-VL/omni-lineage LM graph declares visual_pos_masks +
        # deepstack_visual_embeds.N as inputs: the vision encoder's
        # multi-scale features are injected into the first N decoder
        # layers via bool-mask indexing. Read the DAG input spec — no
        # family/model branch (R15); GLM-class graphs (no such inputs)
        # keep the legacy path bit-for-bit.
        lm_exec = self.ctx.executors.get(lm_name)
        _dag_tensors = (getattr(lm_exec, "_dag", None) or {}).get("tensors", {})
        has_deepstack = "input::visual_pos_masks" in _dag_tensors
        ds_input_names = sorted(
            (tid[len("input::"):] for tid in _dag_tensors
             if tid.startswith("input::deepstack_visual_embeds.")),
            key=lambda n: int(n.rsplit(".", 1)[1]))
        if has_deepstack and not ds_input_names:
            raise RuntimeError(
                "ZERO FALLBACK: LM graph declares visual_pos_masks but no "
                "deepstack_visual_embeds.N inputs — inconsistent trace.")

        deepstack_embeds: List[torch.Tensor] = []
        if has_visual:
            # ── Step 2 (visual): vision encoder forward — the video path
            # rides the SAME tower with its own patch stream and grid
            # (temporal dim > 1); only the placeholder token and the
            # M-RoPE temporal scale differ downstream. ──
            vis_pixels = pixel_values if has_image else pixel_values_videos
            vis_grid = grid_thw if has_image else video_grid_thw
            print(f"   [{vision_name}] Running vision forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(vision_name)
            resolved[f"{vision_name}.hidden_states"] = vis_pixels.to(device=device, dtype=dtype)
            resolved["global.hidden_states"] = resolved[f"{vision_name}.hidden_states"]
            resolved[f"{vision_name}.grid_thw"] = vis_grid.to(device=device)
            resolved["global.grid_thw"] = resolved[f"{vision_name}.grid_thw"]
            self._execute_component(vision_name, "forward", None)
            modal_embeds = self._get_component_output(vision_name)
            if modal_embeds is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision component '{vision_name}' produced no output.")
            if modal_embeds.dim() == 2:
                modal_embeds = modal_embeds.unsqueeze(0)        # [1, n_merged, H]
            modal_embeds = modal_embeds.to(dtype=dtype)
            n_modal = modal_embeds.shape[1]
            if has_deepstack:
                # The tracer names the vision tuple's flattened extras
                # output_1..N ((hidden, [deepstack x3]) return); each is
                # [n_modal, H_text], row-aligned with the merged hidden.
                for k in range(1, len(ds_input_names) + 1):
                    t_ds = resolved.get(f"{vision_name}.output_{k}")
                    if t_ds is None:
                        raise RuntimeError(
                            f"ZERO FALLBACK: LM graph declares "
                            f"{len(ds_input_names)} deepstack inputs but vision "
                            f"component '{vision_name}' produced no output_{k}.")
                    deepstack_embeds.append(t_ds.to(device=device, dtype=dtype))
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{vision_name}] {n_modal} vision tokens in {elapsed:.0f}ms"
                  + (f" (+{len(deepstack_embeds)} deepstack levels)"
                     if deepstack_embeds else ""))
            if not self.ctx.persistent_mode:
                self._unload_component_weights(vision_name)
                release_flow_memory(self.ctx.primary_device)

            # merged-token sanity: placeholder count must equal t*h*w/merge²
            t, h, w = (int(x) for x in vis_grid.reshape(-1)[:3])
            expected = t * (h // merge) * (w // merge)
            if n_modal != expected:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision output has {n_modal} tokens but "
                    f"grid {t}x{h}x{w} / merge {merge} expects {expected}.")
            if has_image:
                span_token_id = image_token_id
                span_content_type = "image"
            else:
                span_token_id = vlm.get("video_token_id")
                if span_token_id is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: topology.flow.vlm carries no "
                        "video_token_id — video understanding needs it.")
                span_content_type = "video"
        else:
            # ── Step 2 (audio): mel features → audio tower forward ──
            print(f"   [{audio_name}] Running audio forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(audio_name)
            aud_exec = self.ctx.executors.get(audio_name)
            _aud_spec = (getattr(aud_exec, "_dag", None) or {}).get(
                "tensors", {}).get("input::input_features", {})
            _in_shape = _aud_spec.get("shape")
            _ap = vlm.get("audio_preprocessing") or {}
            # Registry naming → runtime DSP naming (the flow.audio
            # precedent's map, e.g. whisper_mel → mel_spectrogram).
            _ptype = {"whisper_mel": "mel_spectrogram"}.get(
                _ap.get("type", "whisper_mel"), _ap.get("type"))
            from pathlib import Path as _Path
            from neurobrix.core.module.audio.input_processor import (
                AudioInputProcessor,
            )
            feats = AudioInputProcessor.process(
                _ptype, str(audio_path), _Path(self.ctx.pkg.cache_path),
                device, dtype,
                tuple(_in_shape) if _in_shape else None,
                params=_ap)
            while feats.dim() > 2:
                feats = feats.squeeze(0)                       # [mel, L]
            L = int(feats.shape[1])
            # Windowed-stem length arithmetic (the vendor signature's
            # aftercnn contract): W = 2*n_window; each stride-2 conv
            # keeps (x+1)//2 positions, applied three times per chunk.
            lm_cfg_a = self.ctx.pkg.defaults.get("lm_config", {})
            _n_window = lm_cfg_a.get("audio_n_window")
            if _n_window is None:
                raise RuntimeError(
                    "ZERO FALLBACK: audio_n_window missing from lm_config "
                    "— the registry must map it for audio understanding.")
            _W = 2 * int(_n_window)

            def _aftercnn(x: int) -> int:
                return (((x + 1) // 2 + 1) // 2 + 1) // 2

            _nc = (L + _W - 1) // _W
            _tail = L - (_nc - 1) * _W
            n_modal = _aftercnn(_W) * (_nc - 1) + _aftercnn(_tail)
            resolved[f"{audio_name}.input_features"] = feats
            resolved["global.input_features"] = feats
            _flens = torch.tensor([L], dtype=torch.long, device=device)
            resolved[f"{audio_name}.feature_lens"] = _flens
            resolved["global.feature_lens"] = _flens
            _alens = torch.tensor([n_modal], dtype=torch.long, device=device)
            resolved[f"{audio_name}.aftercnn_lens"] = _alens
            resolved["global.aftercnn_lens"] = _alens
            self._execute_component(audio_name, "forward", None)
            modal_embeds = self._get_component_output(audio_name)
            if modal_embeds is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio component '{audio_name}' produced no output.")
            if modal_embeds.dim() == 2:
                modal_embeds = modal_embeds.unsqueeze(0)        # [1, S, H]
            modal_embeds = modal_embeds.to(dtype=dtype)
            if modal_embeds.shape[1] != n_modal:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio output has {modal_embeds.shape[1]} "
                    f"tokens but aftercnn({L}) expects {n_modal}.")
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{audio_name}] {n_modal} audio tokens "
                  f"(L={L} mel frames) in {elapsed:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(audio_name)
                release_flow_memory(self.ctx.primary_device)
            span_token_id = vlm.get("audio_token_id")
            if span_token_id is None:
                raise RuntimeError(
                    "ZERO FALLBACK: topology.flow.vlm carries no "
                    "audio_token_id — audio understanding needs it.")
            span_content_type = "audio"

        # ── Step 3: tokenize prompt around the modality span ──
        prefix_ids, suffix_ids = self._tokenize_around_span(
            str(prompt), span_token_id, span_content_type)

        # ── Step 4: LM decode with merged embeds + mrope positions ──
        from neurobrix.core.runtime.decode_bound import decode_bound
        # CLI overrides first (mirror of the AR flow's _CLI_OVERRIDES):
        # the request's --max-tokens/--temperature land in the resolver as
        # global.* variables; pkg.defaults holds only the build defaults.
        max_tokens = decode_bound(
            int(resolved["global.max_tokens"])
            if resolved.get("global.max_tokens") is not None
            else defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = (float(resolved["global.temperature"])
                       if resolved.get("global.temperature") is not None
                       else defaults.get("temperature"))
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple))
                      else [eos_token_id])
        repetition_penalty = (
            float(resolved["global.repetition_penalty"])
            if resolved.get("global.repetition_penalty") is not None
            else defaults.get("repetition_penalty", 1.0))

        self._ensure_weights_loaded(lm_name)
        # Declared-MoE fusion (mirror of the AR flow's setup — the Stage-1
        # wall-5 class: without it only the trace-fired experts of a
        # 128-expert MoE run and the logits go near-uniform ⇒ degenerate
        # single-token repetition; proven again on the first vlm-path run).
        lm_cfg = self.ctx.pkg.defaults.get("lm_config", {})
        _n_exp = lm_cfg.get("num_experts")
        if _n_exp is not None and _n_exp > 1 and lm_exec is not None:
            _norm_topk = lm_cfg.get("norm_topk_prob")
            if _norm_topk is None:
                raise RuntimeError(
                    "ZERO FALLBACK: norm_topk_prob missing from lm_config "
                    "for MoE model — add moe.norm_topk_prob to the registry.")
            lm_exec.set_moe_config(norm_topk_prob=_norm_topk)
        embed_weight = self._get_embed_weight(lm_name)
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vlm stage '{lm_name}' requires embed_tokens weight.")

        def _embed(ids: List[int]) -> torch.Tensor:
            tens = torch.tensor([ids], dtype=torch.long, device=embed_weight.device)
            with torch.no_grad():
                return torch.nn.functional.embedding(tens, embed_weight)\
                    .to(device=device, dtype=dtype)

        parts = []
        if prefix_ids:
            parts.append(_embed(prefix_ids))
        parts.append(modal_embeds.to(device=device))
        if suffix_ids:
            parts.append(_embed(suffix_ids))
        context_embeds = torch.cat(parts, dim=1)

        # segment layout for mrope positions: (kind, length,
        # (t, h, w, t_scale) or None). Visual spans get the 3-D grid
        # planes with a temporal scale; audio spans use 1-D text-style
        # positions on all three planes — the vendor get_rope_index's
        # audio-only branch is arange().expand(3, -1).
        # Temporal scale (vendor get_rope_index): omni-lineage configs
        # declare position_id_per_seconds — image t_index scales by it
        # (1 s per grid step), video by second_per_grid × it; legacy
        # mrope configs (no such key) keep the plain +1-per-grid index.
        _t_scale = 1.0
        if has_visual:
            _pids = lm_cfg.get("position_id_per_seconds")
            if _pids is not None:
                _spg = 1.0
                if has_video:
                    if video_spg is None:
                        raise RuntimeError(
                            "ZERO FALLBACK: video request missing "
                            "global.video_second_per_grid.")
                    _spg = float(video_spg.reshape(-1)[0])
                _t_scale = _spg * float(_pids)
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int, float]]]] = []
        if prefix_ids:
            segments.append(("text", len(prefix_ids), None))
        if has_visual:
            segments.append(("visual", n_modal,
                             (t, h // merge, w // merge, _t_scale)))
        else:
            segments.append(("text", n_modal, None))
        if suffix_ids:
            segments.append(("text", len(suffix_ids), None))
        base_positions, next_pos = self._build_mrope_positions(segments, device)

        logits_source = "lm_head" if (head_name and head_name in self.ctx.executors) \
            else "embed_weight_tied"
        print(f"   [{lm_name}] Generating (max={max_tokens}, "
              f"context={context_embeds.shape[1]}, logits={logits_source})...")
        start = time.perf_counter()
        generated_ids: List[int] = []

        img_span_start = len(prefix_ids)
        if has_deepstack and not has_visual:
            # Audio (and any non-visual) requests on a DeepStack graph
            # feed the empty-stub form: all-False mask + zero-length
            # embeds — the injection ops become exact no-ops (the vendor
            # `is None` skip; visual_pos_masks is the IMAGE mask only).
            _h_ds = context_embeds.shape[2]
            deepstack_embeds = [
                torch.zeros(0, _h_ds, dtype=context_embeds.dtype,
                            device=device)
                for _ in ds_input_names]
        for step in range(max_tokens):
            n_gen = len(generated_ids)
            if n_gen:
                gen_pos = torch.arange(next_pos, next_pos + n_gen, dtype=torch.long,
                                       device=device).view(1, 1, -1).expand(3, 1, -1)
                position_ids = torch.cat([base_positions, gen_pos], dim=2)
            else:
                position_ids = base_positions

            resolved["global.inputs_embeds"] = context_embeds
            resolved["inputs_embeds"] = context_embeds
            resolved["global.position_ids"] = position_ids
            resolved["position_ids"] = position_ids
            if has_deepstack:
                # visual_pos_masks travels in the wrapper's expanded
                # [B, S, H] masked_scatter form (the graph reduces it via
                # [..., 0]). True exactly over the IMAGE span (all-False
                # for audio requests); generated tokens extend the mask
                # with False, so the per-step full re-forward matches
                # vendor semantics (deepstack adds only at visual
                # positions).
                _s = context_embeds.shape[1]
                _mask2d = torch.zeros(1, _s, dtype=torch.bool, device=device)
                if has_visual:
                    _mask2d[0, img_span_start:img_span_start + n_modal] = True
                _mask3d = _mask2d.unsqueeze(-1).expand(
                    -1, -1, context_embeds.shape[2]).contiguous()
                resolved["visual_pos_masks"] = _mask3d
                resolved["global.visual_pos_masks"] = _mask3d
                for _i, _t in enumerate(deepstack_embeds):
                    resolved[f"deepstack_visual_embeds.{_i}"] = _t
                    resolved[f"global.deepstack_visual_embeds.{_i}"] = _t

            self._execute_component(lm_name, "forward", None)
            output = self._get_component_output(lm_name)
            if output is None:
                break

            logits = self._compute_logits(output, embed_weight, logits_source, head_name)
            if step == 0 and os.environ.get("NBX_DEBUG"):
                _l = logits.reshape(-1).float()
                _top = torch.topk(_l, 5)
                _ds_norms = [round(float(t.float().norm()), 3)
                             for t in deepstack_embeds]
                print(f"   [vlm-diag] context_norm="
                      f"{float(context_embeds.float().norm()):.4f} "
                      f"hidden_norm={float(output.float().norm()):.4f} "
                      f"ds_norms={_ds_norms} "
                      f"pos_head={position_ids[:, 0, :3].tolist()} "
                      f"top5_ids={_top.indices.tolist()} "
                      f"top5_logits={[round(float(x), 3) for x in _top.values]}")
            from .audio_utils import sample_token
            next_token = sample_token(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            if next_token in eos_ids:
                break
            context_embeds = torch.cat(
                [context_embeds, _embed([next_token]).to(device=device)], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens in {elapsed:.0f}ms")
        resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

        # ── Step 5: decode tokens to text ──
        from .audio_utils import postprocess_text_output
        postprocess_text_output(self.ctx)
        out_var = vlm.get("output", {}).get("variable", "global.generated_text")
        if out_var and resolved.get("global.transcription") is not None:
            resolved[out_var] = resolved["global.transcription"]
        return self.ctx.variable_resolver.resolve_all()

    # ─── staged-splice contract (MiniCPM-o class) ────────────────────────

    def _execute_staged_splice(self, vlm: Dict[str, Any]) -> Dict[str, Any]:
        """Staged VLM contract: vision tower → projection chain, audio
        tower → projection chain, LM with IN-GRAPH bool-mask splice.

        Grounded on the MiniCPM-o-4_5 build (first consumer) — graph
        input contracts read from the traced DAGs:
          vpm(all_pixel_values [1,N,588] , patch_attention_mask [1,1,N]
              bool, tgt_sizes [[gh,gw]] int32) → [1, N, 1152]
          resampler(x [1,N,1152], tgt_sizes) → [1, 64, 4096]
          apm(input_features [1,80,L], attention_mask [1,1,S,S] additive
              chunked, S=(L-1)//2+1) → [1, S, 1024]
          audio_projection_layer(audio_features [1,S,1024]) →
              [1, (S-pool)//pool+1, 4096] (avg-pool in-graph)
          llm.model(inputs_embeds [B,S,4096], position_ids [B,S] int64
              (plain 1-D — no M-RoPE), visual_pos_masks /
              audio_pos_masks [B,S,4096] bool expanded,
              vision_hidden_states / audio_hidden_states [n,4096]
              flattened) → hidden → llm.lm_head.

        Placeholder ids (unk runs between image/audio start-end markers)
        come from topology.flow.vlm.preprocessing /
        .audio_preprocessing — ZERO FALLBACK when absent. v1 request
        contract: one modality per request (image XOR audio) plus
        text-only; requests without a modality feed all-False masks +
        one zero source row (masked_scatter consumes zero rows — inert,
        and keeps the graph's n >= 1 symbol constraint).
        """
        defaults = self.ctx.pkg.defaults
        resolved = self.ctx.variable_resolver.resolved
        dtype = self._get_compute_dtype()
        device = self.ctx.primary_device

        vision_name = vlm.get("vision_component")
        proj_name = vlm.get("vision_projection_component")
        lm_name = vlm.get("lm_component")
        head_name = vlm.get("head_component")
        audio_name = vlm.get("audio_component")
        aproj_name = vlm.get("audio_projection_component")
        pre = vlm.get("preprocessing") or {}
        apre = vlm.get("audio_preprocessing") or {}
        if not lm_name:
            raise RuntimeError(
                "ZERO FALLBACK: staged vlm contract requires lm_component.")

        # Positions discriminator (data-driven, the mandate's gate): the
        # staged builder emits plain [B, S] positions; an M-RoPE LM graph
        # declares rank-3 position_ids and needs its own builder.
        lm_exec = self.ctx.executors.get(lm_name)
        _lm_tensors = (getattr(lm_exec, "_dag", None) or {}).get("tensors", {})
        _pos_rank = len((_lm_tensors.get("input::position_ids", {})
                         .get("shape")) or [])
        if _pos_rank != 2:
            raise RuntimeError(
                "ZERO FALLBACK: staged splice expects plain 1-D positions "
                f"(position_ids rank 2 [B, S]); this LM graph declares rank "
                f"{_pos_rank} — an M-RoPE staged variant needs its own "
                "position builder.")

        in_cfg = vlm.get("input", {})
        prompt = resolved.get(in_cfg.get("prompt_variable", "global.prompt"))
        if not prompt:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires a text prompt.")
        all_pixel_values = resolved.get("global.all_pixel_values")
        patch_mask = resolved.get("global.patch_attention_mask")
        tgt_sizes = resolved.get("global.tgt_sizes")
        audio_path = resolved.get("global.audio_path")
        has_image = all_pixel_values is not None
        has_audio = audio_path is not None
        if has_image and has_audio:
            raise RuntimeError(
                "ZERO FALLBACK: staged vlm v1 accepts one modality per "
                "request — image XOR audio (mixed-modality lands with a "
                "multi-span splice extension).")
        if has_image and (patch_mask is None or tgt_sizes is None):
            raise RuntimeError(
                "ZERO FALLBACK: image request missing patch_attention_mask "
                "/ tgt_sizes — the declared image preprocessing emits all "
                "three vision inputs.")

        # ── Stage 1 (image): vision tower → projection ──
        modal_embeds: Optional[torch.Tensor] = None   # [n_modal, H] flattened
        span_ids: Tuple[Optional[int], Optional[int]] = (None, None)
        span_kind: Optional[str] = None
        if has_image:
            if not proj_name:
                raise RuntimeError(
                    "ZERO FALLBACK: staged vision chain requires "
                    "vision_projection_component in topology.flow.vlm.")
            print(f"   [{vision_name}] Running vision forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(vision_name)
            # Write BOTH the component-scoped and the global slots (the
            # topology connections route global.* — omni precedent).
            _apv = all_pixel_values.to(device=device, dtype=dtype)
            _pam = patch_mask.to(device=device)
            _tgt = tgt_sizes.to(device=device)
            resolved[f"{vision_name}.all_pixel_values"] = _apv
            resolved["global.all_pixel_values"] = _apv
            resolved[f"{vision_name}.patch_attention_mask"] = _pam
            resolved["global.patch_attention_mask"] = _pam
            resolved[f"{vision_name}.tgt_sizes"] = _tgt
            resolved["global.tgt_sizes"] = _tgt
            self._execute_component(vision_name, "forward", None)
            vis_hidden = self._get_component_output(vision_name)
            if vis_hidden is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision component '{vision_name}' "
                    f"produced no output.")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(vision_name)
                release_flow_memory(self.ctx.primary_device)

            self._ensure_weights_loaded(proj_name)
            resolved[f"{proj_name}.x"] = vis_hidden.to(dtype=dtype)
            resolved["global.x"] = resolved[f"{proj_name}.x"]
            resolved[f"{proj_name}.tgt_sizes"] = \
                resolved[f"{vision_name}.tgt_sizes"]
            self._execute_component(proj_name, "forward", None)
            proj_out = self._get_component_output(proj_name)
            if proj_out is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: projection component '{proj_name}' "
                    f"produced no output.")
            modal_embeds = proj_out.reshape(
                -1, proj_out.shape[-1]).to(device=device, dtype=dtype)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{vision_name}->{proj_name}] {modal_embeds.shape[0]} "
                  f"vision tokens in {elapsed:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(proj_name)
                release_flow_memory(self.ctx.primary_device)
            span_ids = (pre.get("image_start_token_id"),
                        pre.get("image_end_token_id"))
            span_kind = "image"

        # ── Stage 1 (audio): mel → audio tower → projection(+pool) ──
        elif has_audio:
            if not audio_name or not aproj_name:
                raise RuntimeError(
                    "ZERO FALLBACK: audio request needs audio_component + "
                    "audio_projection_component in topology.flow.vlm.")
            print(f"   [{audio_name}] Running audio forward...")
            start = time.perf_counter()
            _ptype = {"whisper_mel": "mel_spectrogram"}.get(
                apre.get("type", "whisper_mel"), apre.get("type"))
            from pathlib import Path as _Path
            from neurobrix.core.module.audio.input_processor import (
                AudioInputProcessor,
            )
            feats = AudioInputProcessor.process(
                _ptype, str(audio_path), _Path(self.ctx.pkg.cache_path),
                device, dtype, None, params=apre)
            while feats.dim() > 2:
                feats = feats.squeeze(0)                        # [mel, L]
            L = int(feats.shape[1])
            s_cnn = (L - 1) // 2 + 1        # stride-2 conv stem length
            # Chunked additive encoder mask — the vendor
            # get_audio_embedding contract: the padding mask is all-False
            # at batch 1 (no padding), the chunk mask
            # (subsequent_chunk_mask closed form, full left context:
            # allowed[i, j] = j < ((i // chunk) + 1) * chunk) fills the
            # rest with -inf. chunk = audio_chunk_length seconds of
            # post-stem frames = chunk_len * ((sample_rate/hop_length)//2).
            _lm_cfg_a = defaults.get("lm_config", {})
            chunk_len = _lm_cfg_a.get("audio_chunk_length")
            sr_a = apre.get("sample_rate")
            hop_a = apre.get("hop_length")
            pool_k = _lm_cfg_a.get("audio_pool_step")
            if (chunk_len is None or sr_a is None or hop_a is None
                    or pool_k is None):
                raise RuntimeError(
                    "ZERO FALLBACK: audio chain needs "
                    "lm_config.audio_chunk_length + audio_pool_step and "
                    "audio_preprocessing.sample_rate/hop_length — "
                    "registry-emitted, missing from this build.")
            chunk = int(float(chunk_len) * ((int(sr_a) // int(hop_a)) // 2))
            idx = torch.arange(s_cnn, device=device)
            allowed = (idx.view(1, -1)
                       < ((idx.view(-1, 1) // chunk) + 1) * chunk)
            attn = torch.zeros(1, 1, s_cnn, s_cnn, dtype=dtype, device=device)
            attn = attn.masked_fill(~allowed.view(1, 1, s_cnn, s_cnn),
                                    float("-inf"))
            self._ensure_weights_loaded(audio_name)
            resolved[f"{audio_name}.input_features"] = feats.unsqueeze(0)
            resolved["global.input_features"] = \
                resolved[f"{audio_name}.input_features"]
            resolved[f"{audio_name}.attention_mask"] = attn
            resolved["global.attention_mask"] = attn
            self._execute_component(audio_name, "forward", None)
            aud_hidden = self._get_component_output(audio_name)
            if aud_hidden is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio component '{audio_name}' "
                    f"produced no output.")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(audio_name)
                release_flow_memory(self.ctx.primary_device)

            self._ensure_weights_loaded(aproj_name)
            resolved[f"{aproj_name}.audio_features"] = \
                aud_hidden.to(dtype=dtype)
            resolved["global.audio_features"] = \
                resolved[f"{aproj_name}.audio_features"]
            self._execute_component(aproj_name, "forward", None)
            aproj_out = self._get_component_output(aproj_name)
            if aproj_out is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio projection component "
                    f"'{aproj_name}' produced no output.")
            n_expected = (s_cnn - int(pool_k)) // int(pool_k) + 1
            if aproj_out.shape[1] != n_expected:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio projection produced "
                    f"{aproj_out.shape[1]} tokens but the pooling "
                    f"arithmetic ((({L}-1)//2+1 - {pool_k})//{pool_k}+1) "
                    f"expects {n_expected}.")
            modal_embeds = aproj_out.reshape(
                -1, aproj_out.shape[-1]).to(device=device, dtype=dtype)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{audio_name}->{aproj_name}] {modal_embeds.shape[0]} "
                  f"audio tokens (L={L} mel frames) in {elapsed:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(aproj_name)
                release_flow_memory(self.ctx.primary_device)
            span_ids = (apre.get("audio_start_token_id"),
                        apre.get("audio_end_token_id"))
            span_kind = "audio"

        # ── Stage 2: token ids with the placeholder run ──
        ids, span_lo, span_hi = self._tokenize_with_placeholder_run(
            str(prompt), span_kind, span_ids, pre.get("unk_token_id"),
            0 if modal_embeds is None else int(modal_embeds.shape[0]))

        # ── Stage 3: LM decode (full re-forward, in-graph splice) ──
        from neurobrix.core.runtime.decode_bound import decode_bound
        max_tokens = decode_bound(
            int(resolved["global.max_tokens"])
            if resolved.get("global.max_tokens") is not None
            else defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError(
                "ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = (float(resolved["global.temperature"])
                       if resolved.get("global.temperature") is not None
                       else defaults.get("temperature"))
        if temperature is None:
            raise RuntimeError(
                "ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError(
                "ZERO FALLBACK: eos_token_id missing from defaults.json.")
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple))
                      else [eos_token_id])
        repetition_penalty = (
            float(resolved["global.repetition_penalty"])
            if resolved.get("global.repetition_penalty") is not None
            else defaults.get("repetition_penalty", 1.0))

        self._ensure_weights_loaded(lm_name)
        # Declared-MoE fusion mirror (data-driven; no-op on dense builds).
        lm_cfg = defaults.get("lm_config", {})
        _n_exp = lm_cfg.get("num_experts")
        if _n_exp is not None and _n_exp > 1 and lm_exec is not None:
            _norm_topk = lm_cfg.get("norm_topk_prob")
            if _norm_topk is None:
                raise RuntimeError(
                    "ZERO FALLBACK: norm_topk_prob missing from lm_config "
                    "for MoE model — add moe.norm_topk_prob to the registry.")
            lm_exec.set_moe_config(norm_topk_prob=_norm_topk)
        embed_weight = self._get_embed_weight(lm_name)
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vlm stage '{lm_name}' requires embed "
                "weight.")

        def _embed(token_ids: List[int]) -> torch.Tensor:
            tens = torch.tensor([token_ids], dtype=torch.long,
                                device=embed_weight.device)
            with torch.no_grad():
                return torch.nn.functional.embedding(tens, embed_weight)\
                    .to(device=device, dtype=dtype)

        # The placeholder rows of the context embedding are overwritten
        # IN-GRAPH by masked_scatter — the vendor embeds the full ids
        # (unk placeholders included) and scatters over them.
        context_embeds = _embed(ids)
        hidden_size = context_embeds.shape[2]
        _zero_stub = torch.zeros(1, hidden_size, dtype=dtype, device=device)
        vis_src = modal_embeds if span_kind == "image" else _zero_stub
        aud_src = modal_embeds if span_kind == "audio" else _zero_stub

        logits_source = ("lm_head"
                         if (head_name and head_name in self.ctx.executors)
                         else "embed_weight_tied")
        print(f"   [{lm_name}] Generating (max={max_tokens}, "
              f"context={context_embeds.shape[1]}, logits={logits_source})...")
        start = time.perf_counter()
        generated_ids: List[int] = []
        for step in range(max_tokens):
            S = context_embeds.shape[1]
            vis_mask = torch.zeros(1, S, hidden_size, dtype=torch.bool,
                                   device=device)
            aud_mask = torch.zeros(1, S, hidden_size, dtype=torch.bool,
                                   device=device)
            if span_kind == "image":
                vis_mask[0, span_lo:span_hi, :] = True
            elif span_kind == "audio":
                aud_mask[0, span_lo:span_hi, :] = True
            position_ids = torch.arange(
                S, dtype=torch.long, device=device).view(1, -1)
            for _key, _value in (
                    ("inputs_embeds", context_embeds),
                    ("position_ids", position_ids),
                    ("visual_pos_masks", vis_mask),
                    ("audio_pos_masks", aud_mask),
                    ("vision_hidden_states", vis_src),
                    ("audio_hidden_states", aud_src)):
                resolved[_key] = _value
                resolved[f"global.{_key}"] = _value

            self._execute_component(lm_name, "forward", None)
            output = self._get_component_output(lm_name)
            if output is None:
                break
            logits = self._compute_logits(output, embed_weight,
                                          logits_source, head_name)
            from .audio_utils import sample_token
            next_token = sample_token(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            if next_token in eos_ids:
                break
            context_embeds = torch.cat(
                [context_embeds, _embed([next_token])], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens "
              f"in {elapsed:.0f}ms")
        resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

        from .audio_utils import postprocess_text_output
        postprocess_text_output(self.ctx)
        out_var = vlm.get("output", {}).get("variable",
                                            "global.generated_text")
        if out_var and resolved.get("global.transcription") is not None:
            resolved[out_var] = resolved["global.transcription"]
        return self.ctx.variable_resolver.resolve_all()

    def _tokenize_with_placeholder_run(
            self, prompt: str, span_kind: Optional[str],
            span_ids: Tuple[Optional[int], Optional[int]],
            unk_id: Optional[int], n_modal: int
    ) -> Tuple[List[int], int, int]:
        """Build the id sequence carrying the vendor placeholder run.

        Vendor (_convert_omni_to_inputs): the modality marker inside the
        user content is replaced by start + unk*n + end BEFORE encoding;
        the splice bound is (start_idx + 1, end_idx). Runtime mirror:
        the placeholder STRING is rebuilt from the topology's token ids
        via single-id decode (added tokens round-trip exactly), placed
        modality-first + "\\n" + prompt (the vendor chat() content
        join), then chat-templated and encoded in one pass. The vendor
        default adds an <image_id>N</image_id> text prefix
        (use_image_id); the registry emits no image_id token ids, so
        this flow runs the vendor's use_image_id=False mode.

        Returns (ids, span_lo, span_hi) with [span_lo, span_hi) covering
        the unk run; (ids, 0, 0) for text-only requests."""
        tokenizer = self._get_tokenizer()
        if span_kind is None:
            content = prompt
        else:
            if unk_id is None or span_ids[0] is None or span_ids[1] is None:
                raise RuntimeError(
                    "ZERO FALLBACK: staged splice needs unk_token_id + "
                    f"{span_kind} start/end token ids in the topology "
                    "preprocessing blocks (registry-emitted).")
            if n_modal <= 0:
                raise RuntimeError(
                    f"ZERO FALLBACK: {span_kind} chain produced no tokens.")
            # skip_special_tokens=False is MANDATORY: the engine-internal
            # tokenizer runner (HFTokenizer/sp_tokenizer) defaults to
            # skipping special ids on decode — the placeholder markers ARE
            # special ids and would decode to "" (proven on the first
            # audio probe: '0 unk positions, expected 95').
            s_start = tokenizer.decode([int(span_ids[0])],
                                       skip_special_tokens=False)
            s_unk = tokenizer.decode([int(unk_id)],
                                     skip_special_tokens=False)
            s_end = tokenizer.decode([int(span_ids[1])],
                                     skip_special_tokens=False)
            content = s_start + s_unk * n_modal + s_end + "\n" + prompt
        messages = [{"role": "user", "content": content}]
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            raise RuntimeError(
                "ZERO FALLBACK: the embedded tokenizer could not apply its "
                "chat template — the staged vlm flow requires a text chat "
                "template.") from e
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = list(ids[0] if ids and isinstance(ids[0], (list, tuple))
                   else ids)
        if span_kind is None:
            return ids, 0, 0
        positions = [i for i, tid in enumerate(ids) if tid == int(unk_id)]
        first, last = (positions[0], positions[-1]) if positions else (0, -1)
        if (len(positions) != n_modal
                or positions != list(range(first, last + 1))):
            raise RuntimeError(
                "ZERO FALLBACK: placeholder run not recovered after "
                f"templating ({len(positions)} unk positions, expected a "
                f"contiguous run of {n_modal}) — the embedded tokenizer "
                "did not round-trip the placeholder tokens.")
        return ids, first, last + 1

    # ─── M-RoPE masked-splice contract (bailingmm class) ────────────────

    def _execute_mrope_masked_splice(self, vlm: Dict[str, Any]) -> Dict[str, Any]:
        """Third staged contract: two projection chains, TWO in-graph
        modality mask/embed pairs, 3-D M-RoPE positions.

        Grounded on the Ming-Lite-Omni-1.5 build (first consumer) — the
        traced DAG contracts, quoted from the graphs:

          vision(hidden_states [n_patch, C*Tp*P*P], grid_thw [1, 3])
              -> [t*(h/2)*(w/2), vision_out_hidden]
          linear_proj(image_embeds [n, vision_out_hidden])
              -> [n, H]                     (in-graph F.normalize)
          audio(x [1, T_mel, n_mels])
              -> [1, (T_mel-1)//2+1, d_audio]
          linear_proj_audio(audio_feats [1, L1, d_audio])
              -> [1, (L1-1)//2+1, H]        (ds conv k3 s2 + in-graph
                                             F.normalize)
          model.model(inputs_embeds [B, S, H],
                      position_ids [3, B, S] int64,
                      image_pos_masks / audio_pos_masks [B, S, H] bool,
                      image_embeds / audio_embeds [n, H])
              -> hidden -> lm_head

        The LM graph's first two ops are
        `masked_scatter(inputs_embeds, image_pos_masks, image_embeds)`
        then the same for audio, and `select(mask, dim=2, index=0)`
        rebuilds the [B, S] router masks that gate the per-modality MoE
        gates — so the expanded [B, S, H] mask form is the contract, not
        a convenience. Absent modalities feed an all-False mask plus ONE
        zero source row (masked_scatter consumes zero rows — inert, and
        it keeps the graph's n >= 1 symbol constraint), exactly the
        deepstack/staged stub pattern.

        Positions follow the vendor get_rope_index: text runs advance the
        three planes together from the running offset, a visual run gets
        the t/h/w meshgrid of the llm grid (h, w divided by
        spatial_merge_size, no temporal scaling — the vendor calls
        get_rope_index without second_per_grid_ts), audio runs are
        text-like on all three planes, and every run starts at
        max(previous) + 1. The [16, 24, 24] section split is applied
        IN-GRAPH by the traced rotary; the flow only supplies positions.

        ZERO FALLBACK: every id, token count and template comes from the
        topology / the request; a missing one raises by name.
        """
        defaults = self.ctx.pkg.defaults
        resolved = self.ctx.variable_resolver.resolved
        dtype = self._get_compute_dtype()
        device = self.ctx.primary_device

        vision_name = vlm.get("vision_component")
        vproj_name = vlm.get("vision_projection_component")
        audio_name = vlm.get("audio_component")
        aproj_name = vlm.get("audio_projection_component")
        lm_name = vlm.get("lm_component")
        head_name = vlm.get("head_component")
        pre = vlm.get("preprocessing") or {}
        apre = vlm.get("audio_preprocessing") or {}
        merge = vlm.get("spatial_merge_size")
        if not lm_name or not merge:
            raise RuntimeError(
                "ZERO FALLBACK: the M-RoPE masked-splice contract requires "
                "topology.flow.vlm.lm_component and spatial_merge_size.")

        lm_exec = self.ctx.executors.get(lm_name)
        lm_tensors = (getattr(lm_exec, "_dag", None) or {}).get("tensors", {})
        _missing = [n for n in ("inputs_embeds", "position_ids",
                                "image_pos_masks", "image_embeds",
                                "audio_pos_masks", "audio_embeds")
                    if f"input::{n}" not in lm_tensors]
        if _missing:
            raise RuntimeError(
                "ZERO FALLBACK: the M-RoPE masked-splice LM graph must "
                f"declare all six splice inputs — missing {_missing}.")
        _pos_rank = len((lm_tensors["input::position_ids"].get("shape")) or [])
        if _pos_rank != 3:
            raise RuntimeError(
                "ZERO FALLBACK: this contract expects M-RoPE positions "
                f"(position_ids rank 3 [3, B, S]); the graph declares rank "
                f"{_pos_rank}.")

        in_cfg = vlm.get("input", {})
        prompt = resolved.get(in_cfg.get("prompt_variable", "global.prompt"))
        if not prompt:
            raise RuntimeError("ZERO FALLBACK: vlm flow requires a text prompt.")
        pixel_values = resolved.get(
            in_cfg.get("image_variable", "global.pixel_values"))
        grid_thw = resolved.get(
            in_cfg.get("grid_variable", "global.image_grid_thw"))
        pixel_values_videos = resolved.get("global.pixel_values_videos")
        video_grid_thw = resolved.get("global.video_grid_thw")
        audio_path = resolved.get("global.audio_path")
        has_image = pixel_values is not None
        has_video = pixel_values_videos is not None
        has_audio = audio_path is not None
        if has_image and has_video:
            raise RuntimeError(
                "ZERO FALLBACK: one visual modality per request — provide "
                "--input-image or --input-video, not both.")
        if has_image and grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: image request missing the grid input "
                f"({in_cfg.get('grid_variable')}).")
        if has_video and video_grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: video request missing the grid input "
                "(global.video_grid_thw).")
        has_visual = has_image or has_video

        img_embeds: Optional[torch.Tensor] = None      # [n_img, H]
        aud_embeds: Optional[torch.Tensor] = None      # [n_aud, H]
        vis_grid_llm: Optional[Tuple[int, int, int]] = None

        # ── Stage 1 (visual): vision tower → projection ──
        if has_visual:
            if not vision_name or not vproj_name:
                raise RuntimeError(
                    "ZERO FALLBACK: the visual chain needs vision_component "
                    "+ vision_projection_component in topology.flow.vlm.")
            vis_pixels = pixel_values if has_image else pixel_values_videos
            vis_grid = grid_thw if has_image else video_grid_thw
            print(f"   [{vision_name}] Running vision forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(vision_name)
            _px = vis_pixels.to(device=device, dtype=dtype)
            _gr = vis_grid.to(device=device)
            resolved[f"{vision_name}.hidden_states"] = _px
            resolved["global.hidden_states"] = _px
            resolved[f"{vision_name}.grid_thw"] = _gr
            resolved["global.grid_thw"] = _gr
            self._execute_component(vision_name, "forward", None)
            vis_out = self._get_component_output(vision_name)
            if vis_out is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision component '{vision_name}' "
                    f"produced no output.")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(vision_name)
                release_flow_memory(self.ctx.primary_device)

            self._ensure_weights_loaded(vproj_name)
            # NOTE: `global.image_embeds` is the topology's connection
            # source for BOTH the projection input (raw tower output) and
            # the LM input (projected rows). The slot carries the tower
            # output here and is rewritten with the projected rows before
            # every LM forward below.
            _vin = vis_out.reshape(-1, vis_out.shape[-1]).to(dtype=dtype)
            resolved[f"{vproj_name}.image_embeds"] = _vin
            resolved["global.image_embeds"] = _vin
            self._execute_component(vproj_name, "forward", None)
            proj_out = self._get_component_output(vproj_name)
            if proj_out is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision projection component "
                    f"'{vproj_name}' produced no output.")
            img_embeds = proj_out.reshape(
                -1, proj_out.shape[-1]).to(device=device, dtype=dtype)
            t, h, w = (int(x) for x in vis_grid.reshape(-1)[:3])
            vis_grid_llm = (t, h // merge, w // merge)
            expected = t * (h // merge) * (w // merge)
            if int(img_embeds.shape[0]) != expected:
                raise RuntimeError(
                    f"ZERO FALLBACK: the visual chain produced "
                    f"{int(img_embeds.shape[0])} tokens but grid {t}x{h}x{w} "
                    f"/ merge {merge} expects {expected}.")
            print(f"   [{vision_name}->{vproj_name}] {expected} vision "
                  f"tokens in {(time.perf_counter() - start) * 1000:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(vproj_name)
                release_flow_memory(self.ctx.primary_device)

        # ── Stage 1 (audio): mel → audio tower → projection ──
        if has_audio:
            if not audio_name or not aproj_name:
                raise RuntimeError(
                    "ZERO FALLBACK: audio request needs audio_component + "
                    "audio_projection_component in topology.flow.vlm.")
            print(f"   [{audio_name}] Running audio forward...")
            start = time.perf_counter()
            _ptype = {"whisper_mel": "mel_spectrogram"}.get(
                apre.get("type", "whisper_mel"), apre.get("type"))
            from pathlib import Path as _Path
            from neurobrix.core.module.audio.input_processor import (
                AudioInputProcessor,
            )
            feats = AudioInputProcessor.process(
                _ptype, str(audio_path), _Path(self.ctx.pkg.cache_path),
                device, dtype, None, params=apre)
            while feats.dim() > 2:
                feats = feats.squeeze(0)                        # [n_mels, L]
            L = int(feats.shape[1])
            # Vendor encode_audio_segments length arithmetic, verbatim:
            # whisper stem conv (k3 s2 p1) then the projection ds conv
            # (kernel/stride declared by the registry-emitted lm_config).
            _lm_cfg_a = defaults.get("lm_config", {})
            _k = _lm_cfg_a.get("audio_ds_kernel_size")
            _s = _lm_cfg_a.get("audio_ds_stride")
            if _k is None or _s is None:
                raise RuntimeError(
                    "ZERO FALLBACK: audio_ds_kernel_size / audio_ds_stride "
                    "missing from lm_config — the registry must map them "
                    "for audio understanding.")
            _l1 = (L - 1) // 2 + 1
            _l2 = (_l1 - int(_k) + 2 * (int(_k) // 2)) // int(_s) + 1
            self._ensure_weights_loaded(audio_name)
            _x = feats.transpose(0, 1).unsqueeze(0).contiguous()  # [1, L, mel]
            resolved[f"{audio_name}.x"] = _x
            resolved["global.x"] = _x
            self._execute_component(audio_name, "forward", None)
            aud_hidden = self._get_component_output(audio_name)
            if aud_hidden is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio component '{audio_name}' "
                    f"produced no output.")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(audio_name)
                release_flow_memory(self.ctx.primary_device)

            self._ensure_weights_loaded(aproj_name)
            _ain = aud_hidden.to(dtype=dtype)
            resolved[f"{aproj_name}.audio_feats"] = _ain
            resolved["global.audio_feats"] = _ain
            self._execute_component(aproj_name, "forward", None)
            aproj_out = self._get_component_output(aproj_name)
            if aproj_out is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio projection component "
                    f"'{aproj_name}' produced no output.")
            aud_embeds = aproj_out.reshape(
                -1, aproj_out.shape[-1]).to(device=device, dtype=dtype)
            if int(aud_embeds.shape[0]) != _l2:
                raise RuntimeError(
                    f"ZERO FALLBACK: the audio chain produced "
                    f"{int(aud_embeds.shape[0])} tokens but the vendor "
                    f"length arithmetic on {L} mel frames "
                    f"(({L}-1)//2+1 then k={_k} s={_s}) expects {_l2}.")
            print(f"   [{audio_name}->{aproj_name}] {_l2} audio tokens "
                  f"(L={L} mel frames) in "
                  f"{(time.perf_counter() - start) * 1000:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(aproj_name)
                release_flow_memory(self.ctx.primary_device)

        # ── Stage 2: prompt ids with both placeholder runs ──
        ids, img_span, aud_span = self._build_masked_splice_ids(
            str(prompt), vlm, pre, apre,
            0 if img_embeds is None else int(img_embeds.shape[0]),
            0 if aud_embeds is None else int(aud_embeds.shape[0]),
            is_video=has_video)

        # ── Stage 3: LM decode (full re-forward, in-graph splice) ──
        from neurobrix.core.runtime.decode_bound import decode_bound
        max_tokens = decode_bound(
            int(resolved["global.max_tokens"])
            if resolved.get("global.max_tokens") is not None
            else defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = (float(resolved["global.temperature"])
                       if resolved.get("global.temperature") is not None
                       else defaults.get("temperature"))
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple))
                      else [eos_token_id])
        repetition_penalty = (
            float(resolved["global.repetition_penalty"])
            if resolved.get("global.repetition_penalty") is not None
            else defaults.get("repetition_penalty", 1.0))

        self._ensure_weights_loaded(lm_name)
        # Declared-MoE fusion (data-driven; no-op on dense builds) — same
        # call as every other branch of this engine.
        lm_cfg = defaults.get("lm_config", {})
        _n_exp = lm_cfg.get("num_experts")
        if _n_exp is not None and _n_exp > 1 and lm_exec is not None:
            _norm_topk = lm_cfg.get("norm_topk_prob")
            if _norm_topk is None:
                raise RuntimeError(
                    "ZERO FALLBACK: norm_topk_prob missing from lm_config "
                    "for MoE model — add moe.norm_topk_prob to the registry.")
            lm_exec.set_moe_config(norm_topk_prob=_norm_topk)
        embed_weight = self._get_embed_weight(lm_name)
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vlm stage '{lm_name}' requires the token "
                "embedding weight.")

        def _embed(token_ids: List[int]) -> torch.Tensor:
            tens = torch.tensor([token_ids], dtype=torch.long,
                                device=embed_weight.device)
            with torch.no_grad():
                return torch.nn.functional.embedding(tens, embed_weight)\
                    .to(device=device, dtype=dtype)

        # Placeholder rows are overwritten IN-GRAPH by masked_scatter —
        # the context embeds the full ids, placeholders included.
        context_embeds = _embed(ids)
        hidden_size = int(context_embeds.shape[2])
        _zero_stub = torch.zeros(1, hidden_size, dtype=dtype, device=device)
        img_src = img_embeds if img_embeds is not None else _zero_stub
        aud_src = aud_embeds if aud_embeds is not None else _zero_stub

        base_positions, next_pos = self._build_mrope_positions(
            self._mrope_segments(len(ids), img_span, aud_span, vis_grid_llm),
            device)

        logits_source = ("lm_head"
                         if (head_name and head_name in self.ctx.executors)
                         else "embed_weight_tied")
        print(f"   [{lm_name}] Generating (max={max_tokens}, "
              f"context={context_embeds.shape[1]}, logits={logits_source})...")
        start = time.perf_counter()
        generated_ids: List[int] = []
        for _step in range(max_tokens):
            S = int(context_embeds.shape[1])
            n_gen = len(generated_ids)
            if n_gen:
                gen_pos = torch.arange(next_pos, next_pos + n_gen,
                                       dtype=torch.long, device=device)\
                    .view(1, 1, -1).expand(3, 1, -1)
                position_ids = torch.cat([base_positions, gen_pos], dim=2)
            else:
                position_ids = base_positions
            img_mask = torch.zeros(1, S, hidden_size, dtype=torch.bool,
                                   device=device)
            aud_mask = torch.zeros(1, S, hidden_size, dtype=torch.bool,
                                   device=device)
            if img_span[1] > img_span[0]:
                img_mask[0, img_span[0]:img_span[1], :] = True
            if aud_span[1] > aud_span[0]:
                aud_mask[0, aud_span[0]:aud_span[1], :] = True
            for _key, _value in (
                    ("inputs_embeds", context_embeds),
                    ("position_ids", position_ids),
                    ("image_pos_masks", img_mask),
                    ("audio_pos_masks", aud_mask),
                    ("image_embeds", img_src),
                    ("audio_embeds", aud_src)):
                resolved[_key] = _value
                resolved[f"global.{_key}"] = _value

            self._execute_component(lm_name, "forward", None)
            output = self._get_component_output(lm_name)
            if output is None:
                break
            logits = self._compute_logits(output, embed_weight,
                                          logits_source, head_name)
            from .audio_utils import sample_token
            next_token = sample_token(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            if next_token in eos_ids:
                break
            context_embeds = torch.cat(
                [context_embeds, _embed([next_token])], dim=1)

        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens in "
              f"{(time.perf_counter() - start) * 1000:.0f}ms")
        resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

        from .audio_utils import postprocess_text_output
        postprocess_text_output(self.ctx)
        out_var = vlm.get("output", {}).get("variable",
                                            "global.generated_text")
        if out_var and resolved.get("global.transcription") is not None:
            resolved[out_var] = resolved["global.transcription"]
        return self.ctx.variable_resolver.resolve_all()

    def _build_masked_splice_ids(
            self, prompt: str, vlm: Dict[str, Any], pre: Dict[str, Any],
            apre: Dict[str, Any], n_img: int, n_aud: int, is_video: bool
    ) -> Tuple[List[int], Tuple[int, int], Tuple[int, int]]:
        """Assemble the prompt id sequence carrying both placeholder runs.

        Vendor renderer (processing_bailingmm): the modality marker inside
        the user turn is replaced by start + patch*N + end BEFORE
        encoding, the visual block adding a trailing newline; the turn is
        then wrapped by the role markers. Runtime mirror: the placeholder
        STRINGS are rebuilt from the topology-declared token ids via
        single-id decode (added tokens round-trip exactly), and the role
        wrapper comes from the build's declared prompt template — the
        engine holds no role/marker semantics of its own.

        Template source cascade (both data-driven, both named in the
        error): the request's `global.prompt_template` (a legitimate CLI
        input via --set) then `topology.flow.vlm.prompt_template`. The
        template must carry `{modality}` and `{text}`; substitution is by
        replace(), never format(), so braces inside the user prompt are
        literal.

        Returns (ids, image_span, audio_span) with each span a
        half-open [lo, hi) over the placeholder run — (0, 0) when the
        modality is absent."""
        tokenizer = self._get_tokenizer()
        resolved = self.ctx.variable_resolver.resolved

        def _tok_str(tid: Optional[int], what: str) -> str:
            if tid is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: {what} missing from the topology "
                    "preprocessing block — the registry must emit it.")
            # skip_special_tokens=False is MANDATORY: the placeholder
            # markers ARE special ids and would decode to "".
            return tokenizer.decode([int(tid)], skip_special_tokens=False)

        blocks = ""
        if n_img > 0:
            if is_video:
                _start = pre.get("video_start_token_id")
                _end = pre.get("video_end_token_id")
                _patch = vlm.get("video_token_id")
                _what = "video start/end/patch token id"
            else:
                _start = pre.get("image_start_token_id")
                _end = pre.get("image_end_token_id")
                _patch = vlm.get("image_token_id")
                _what = "image start/end/patch token id"
            blocks += (_tok_str(_start, _what)
                       + _tok_str(_patch, _what) * n_img
                       + _tok_str(_end, _what) + "\n")
        if n_aud > 0:
            _a_start = apre.get("audio_start_token_id")
            _a_end = apre.get("audio_end_token_id")
            _a_patch = vlm.get("audio_token_id")
            _what_a = "audio start/end/patch token id"
            blocks += (_tok_str(_a_start, _what_a)
                       + _tok_str(_a_patch, _what_a) * n_aud
                       + _tok_str(_a_end, _what_a))

        template = resolved.get("global.prompt_template")
        if not template:
            template = vlm.get("prompt_template")
        if not template:
            raise RuntimeError(
                "ZERO FALLBACK: this build declares no prompt template and "
                "its tokenizer embeds no chat template — the masked-splice "
                "contract cannot invent role markers. Provide "
                "topology.flow.vlm.prompt_template (registry-emitted) or "
                "the request input global.prompt_template "
                "(--set global.prompt_template='...'), carrying the "
                "{modality} and {text} markers.")
        template = str(template)
        if "{modality}" not in template or "{text}" not in template:
            raise RuntimeError(
                "ZERO FALLBACK: the prompt template must carry both the "
                "{modality} and the {text} marker; got: "
                f"{template[:120]!r}")
        text = template.replace("{modality}", blocks).replace("{text}", prompt)
        # padding=False is MANDATORY (the embedded tokenizer pads to its
        # declared model_max_length otherwise); the build's tokenizer
        # config drives BOS/EOS, which this contract never adds.
        ids = tokenizer.encode(text, padding=False, add_special_tokens=False)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = [int(i) for i in (ids[0] if ids and isinstance(ids[0], (list, tuple))
                                else ids)]

        def _span(patch_id: Optional[int], n: int, kind: str
                  ) -> Tuple[int, int]:
            if n <= 0:
                return (0, 0)
            positions = [i for i, tid in enumerate(ids)
                         if tid == int(patch_id)]
            first = positions[0] if positions else 0
            last = positions[-1] if positions else -1
            if (len(positions) != n
                    or positions != list(range(first, last + 1))):
                raise RuntimeError(
                    f"ZERO FALLBACK: the {kind} placeholder run was not "
                    f"recovered after templating ({len(positions)} "
                    f"positions, expected a contiguous run of {n}) — the "
                    "embedded tokenizer did not round-trip the placeholder "
                    "token.")
            return (first, last + 1)

        img_span = _span(vlm.get("video_token_id") if is_video
                         else vlm.get("image_token_id"), n_img,
                         "video" if is_video else "image")
        aud_span = _span(vlm.get("audio_token_id"), n_aud, "audio")
        if (img_span[1] > img_span[0] and aud_span[1] > aud_span[0]
                and not (img_span[1] <= aud_span[0]
                         or aud_span[1] <= img_span[0])):
            raise RuntimeError(
                "ZERO FALLBACK: the image and audio placeholder runs "
                f"overlap ({img_span} vs {aud_span}) — the two masks must "
                "be disjoint (the vendor asserts the same).")
        return ids, img_span, aud_span

    @staticmethod
    def _mrope_segments(seq_len: int, img_span: Tuple[int, int],
                        aud_span: Tuple[int, int],
                        vis_grid_llm: Optional[Tuple[int, int, int]]):
        """Segment layout for the M-RoPE builder: text runs between the
        modality spans, the visual span as a t/h/w grid, the audio span
        as a text-like run (the vendor's audio-only branch is a plain
        arange expanded over the three planes)."""
        spans = []
        if img_span[1] > img_span[0]:
            if vis_grid_llm is None:
                raise RuntimeError(
                    "ZERO FALLBACK: a visual span needs its llm grid "
                    "(t, h/merge, w/merge) for the M-RoPE builder.")
            spans.append((img_span[0], img_span[1], "visual", vis_grid_llm))
        if aud_span[1] > aud_span[0]:
            spans.append((aud_span[0], aud_span[1], "text", None))
        spans.sort(key=lambda s: s[0])
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int, float]]]] = []
        cursor = 0
        for lo, hi, kind, grid in spans:
            if lo > cursor:
                segments.append(("text", lo - cursor, None))
            if kind == "visual":
                gt, gh, gw = grid
                segments.append(("visual", hi - lo, (gt, gh, gw, 1.0)))
            else:
                segments.append(("text", hi - lo, None))
            cursor = hi
        if cursor < seq_len:
            segments.append(("text", seq_len - cursor, None))
        return segments

    # ─── mrope positions (internal get_rope_index port) ─────────────────

    @staticmethod
    def _build_mrope_positions(segments, device) -> Tuple[torch.Tensor, int]:
        """[3, 1, S] positions per the vendor get_rope_index contract.

        Text segment of length L: all three planes = offset + arange(L).
        Visual segment of llm grid (t, h, w, t_scale): temporal/height/
        width meshgrid indices + offset, the temporal index scaled by
        t_scale (position_id_per_seconds × second-per-grid semantics;
        1.0 for legacy mrope). Every segment starts at max(previous) + 1.
        Returns (positions, next_offset_for_decode)."""
        planes: List[torch.Tensor] = []
        offset = 0
        for kind, length, grid in segments:
            if kind == "text":
                pos = torch.arange(length, dtype=torch.long, device=device)
                seg = pos.view(1, -1).expand(3, -1) + offset
            else:
                gt, gh, gw, t_scale = grid
                t_idx = (torch.arange(gt, dtype=torch.float32, device=device)
                         * float(t_scale)).long()\
                    .view(-1, 1).expand(-1, gh * gw).flatten()
                h_idx = torch.arange(gh, dtype=torch.long, device=device)\
                    .view(1, -1, 1).expand(gt, -1, gw).flatten()
                w_idx = torch.arange(gw, dtype=torch.long, device=device)\
                    .view(1, 1, -1).expand(gt, gh, -1).flatten()
                seg = torch.stack([t_idx, h_idx, w_idx]) + offset
            planes.append(seg)
            offset = int(seg.max().item()) + 1
        positions = torch.cat(planes, dim=1).unsqueeze(1)      # [3, 1, S]
        return positions, offset

    # ─── tokenization around the image span ─────────────────────────────

    def _tokenize_around_span(self, prompt: str, span_token_id: int,
                              content_type: str) -> Tuple[List[int], List[int]]:
        """Chat-template the prompt with a modality content part
        ("image" or "audio"), encode, and split at the (single,
        contiguous) placeholder span.

        The placeholder span's ids are REPLACED by the modality embeds
        (start/end markers stay in the text halves when the template
        emits them outside the span)."""
        tokenizer = self._get_tokenizer()
        messages = [{
            "role": "user",
            "content": [
                {"type": content_type},
                {"type": "text", "text": prompt},
            ],
        }]
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            # Templates that only accept string content: placing the
            # modality token literally via its special-token string is
            # model-specific knowledge — refuse instead of guessing.
            raise RuntimeError(
                "ZERO FALLBACK: the embedded tokenizer could not apply its "
                f"chat template to a {content_type}+text message; the vlm "
                "flow requires a multimodal chat template.") from e
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = list(ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)
        positions = [i for i, tid in enumerate(ids) if tid == span_token_id]
        if not positions:
            raise RuntimeError(
                f"ZERO FALLBACK: chat template produced no {content_type} "
                f"placeholder (token id {span_token_id}) — cannot merge "
                "modality embeddings.")
        first, last = positions[0], positions[-1]
        if positions != list(range(first, last + 1)):
            raise RuntimeError(
                f"ZERO FALLBACK: {content_type} placeholder span is not "
                "contiguous — concat-merge equivalence does not hold.")
        return ids[:first], ids[last + 1:]

    def _get_tokenizer(self):
        tok = self.ctx.modules.get("tokenizer")
        if tok is None:
            raise RuntimeError("ZERO FALLBACK: vlm flow requires the embedded tokenizer.")
        return tok

    # ─── helpers (mirrors of audio_llm) ──────────────────────────────────

    def _get_compute_dtype(self) -> torch.dtype:
        from .audio_utils import get_compute_dtype
        return get_compute_dtype(self.ctx)

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state",
                    f"{comp_name}.output"]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return None
        for key in executor._weights:
            if "embed_tokens" in key or "token_embed" in key or key == "embed.weight":
                return executor._weights[key]
        # Second pass — the NeuroTax substring rule ("embed" in key, see
        # src/neurobrix/CLAUDE.md §0) with the vocab table as the
        # discriminator: a rank-2 weight whose rows equal the declared
        # vocab_size IS the token embedding (e.g. `word_embeddings.weight`).
        # Additive only: every build resolved by the exact names above
        # never reaches this pass.
        vocab = (self.ctx.pkg.defaults.get("lm_config", {}) or {}).get(
            "vocab_size")
        if vocab is None:
            return None
        for key, tensor in executor._weights.items():
            if ("embed" in key and tensor is not None
                    and getattr(tensor, "ndim", 0) == 2
                    and int(tensor.shape[0]) == int(vocab)):
                return tensor
        return None

    def _compute_logits(self, hidden_states: torch.Tensor,
                        embed_weight: Optional[torch.Tensor],
                        logits_source: str, head_name: Optional[str]) -> torch.Tensor:
        last_hidden = hidden_states[:, -1:, :]
        if logits_source == "lm_head" and head_name and head_name in self.ctx.executors:
            self._ensure_weights_loaded(head_name)
            executor = self.ctx.executors[head_name]
            for key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    w = tensor.to(device=last_hidden.device, dtype=last_hidden.dtype)
                    return torch.matmul(last_hidden, w.T)
        if embed_weight is not None:
            w = embed_weight.to(device=last_hidden.device, dtype=last_hidden.dtype)
            return torch.matmul(last_hidden, w.T)
        raise RuntimeError(
            "ZERO FALLBACK: no lm_head weight and no embed weight — "
            "cannot project hidden states to logits.")
