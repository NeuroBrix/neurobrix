"""Triton VLMEngine — zero-torch vision-conditioned LLM flow.

R33-pure mirror of core/flow/vlm.py (VLMEngine). Pattern:
vision_encoder(pixel_values, grid_thw) → merge with text embeddings at
the image-placeholder span → LLM(inputs_embeds, mrope position_ids) →
text.

All compute is NBXTensor + kernel wrappers; ZERO torch on the compute
path. Boundary notes (same doctrine as triton/flow/audio_llm.py):
- The CLI feeds pixel_values / image_grid_thw as host tensors; the
  executor's component-input boundary converts them for the Triton
  arena exactly as it does for any CLI input. This flow only aliases
  resolver slots (no compute).
- Reading the 3 grid integers and the sampled token id are
  decode-control boundary reads (the `.item()` class); the compute
  that produced every tensor stays on the Triton path.
- The tokenizer (ctx.modules) is pure text/ids — no tensors.

M-RoPE position_ids [3, 1, S] are built in numpy (allowed CPU glue)
per the vendor get_rope_index contract: text segments advance all
three planes sequentially from the running offset; an image segment of
llm-grid t×h×w gets t/h/w meshgrid indices + offset; each segment
starts at max(previous) + 1.

ZERO SEMANTIC / ZERO HARDCODE: everything reads topology.flow.vlm and
defaults.json.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.triton.memory_pool import release_flow_memory
from neurobrix.triton.flow.audio_llm import (
    _DTYPE_STR_TO_NBX,
    _sample_token_nbx,
)
from neurobrix.kernels import wrappers as w


def _read_small_ints(t: Any, count: int) -> List[int]:
    """Boundary metadata read: first `count` ints of a tiny tensor —
    NBXTensor (.numpy()), numpy array, or any host object exposing
    __array__ / flatten. Decode-control class, not compute."""
    if isinstance(t, NBXTensor):
        flat = t.numpy().reshape(-1)
    else:
        flat = np.asarray(t).reshape(-1)
    return [int(x) for x in flat[:count]]


def _read_small_float(t: Any) -> float:
    """Boundary metadata read: first element of a tiny float tensor
    (same class as _read_small_ints — decode-control, not compute)."""
    if isinstance(t, NBXTensor):
        return float(t.numpy().reshape(-1)[0])
    return float(np.asarray(t).reshape(-1)[0])


class TritonVLMEngine:
    """Vision-conditioned LLM (R33): encode image → merge embeds →
    mrope decode. Mirror of core/flow/vlm.py VLMEngine."""

    def __init__(
        self,
        ctx,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        self.ctx = ctx
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    def _compute_dtype(self) -> NBXDtype:
        from neurobrix.triton.dtype import resolve_compute_dtype
        s = resolve_compute_dtype(self.ctx)
        return _DTYPE_STR_TO_NBX.get(s, NBXDtype.float16)

    # ─── main ────────────────────────────────────────────────────────

    def execute(self) -> Dict[str, Any]:
        flow = self.ctx.pkg.topology.get("flow", {})
        vlm = flow.get("vlm", {})
        if not vlm:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires topology.flow.vlm.")
        defaults = self.ctx.pkg.defaults
        resolved = self.ctx.variable_resolver.resolved

        vision_name = vlm.get("vision_component")
        lm_name = vlm.get("lm_component")
        head_name = vlm.get("head_component")
        image_token_id = vlm.get("image_token_id")
        merge = vlm.get("spatial_merge_size")
        if not vision_name or not lm_name or image_token_id is None or not merge:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.vlm must declare "
                "vision_component, lm_component, image_token_id and "
                "spatial_merge_size.")

        dtype = self._compute_dtype()

        # ── Step 1: request modality (image XOR audio, plus text) ──
        # R30 mirror of the compiled flow's modality dispatch.
        in_cfg = vlm.get("input", {})
        pixel_values = resolved.get(
            in_cfg.get("image_variable", "global.pixel_values"))
        grid_thw = resolved.get(
            in_cfg.get("grid_variable", "global.image_grid_thw"))
        pixel_values_videos = resolved.get("global.pixel_values_videos")
        video_grid_thw = resolved.get("global.video_grid_thw")
        video_spg = resolved.get("global.video_second_per_grid")
        audio_path = resolved.get("global.audio_path")
        audio_name = vlm.get("audio_component")
        prompt = resolved.get(
            in_cfg.get("prompt_variable", "global.prompt"))
        if not prompt:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires a text prompt.")
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
        # R30 mirror of core/flow/vlm.py: a Qwen3-VL/omni-lineage LM graph
        # declares visual_pos_masks + deepstack_visual_embeds.N inputs —
        # the vision tower's multi-scale features are injected into its
        # first N decoder layers. GLM-class graphs keep the legacy path.
        _lm_graph_path = (Path(self.ctx.pkg.cache_path) / "components"
                          / lm_name / "graph.json")
        with open(_lm_graph_path, "r") as _f:
            _lm_dag_tensors = json.load(_f).get("tensors", {})
        has_deepstack = "input::visual_pos_masks" in _lm_dag_tensors
        ds_input_names = sorted(
            (tid[len("input::"):] for tid in _lm_dag_tensors
             if tid.startswith("input::deepstack_visual_embeds.")),
            key=lambda n: int(n.rsplit(".", 1)[1]))
        if has_deepstack and not ds_input_names:
            raise RuntimeError(
                "ZERO FALLBACK: LM graph declares visual_pos_masks but no "
                "deepstack_visual_embeds.N inputs — inconsistent trace.")

        deepstack_embeds: List[NBXTensor] = []
        if has_visual:
            # ── Step 2 (visual): vision encoder forward — the video path
            # rides the SAME tower with its own patch stream and grid
            # (temporal dim > 1); only the placeholder token and the
            # M-RoPE temporal scale differ downstream. R30 mirror. ──
            vis_pixels = pixel_values if has_image else pixel_values_videos
            vis_grid = grid_thw if has_image else video_grid_thw
            print(f"   [{vision_name}] Running vision forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(vision_name)
            resolved[f"{vision_name}.hidden_states"] = vis_pixels
            resolved["global.hidden_states"] = vis_pixels
            resolved[f"{vision_name}.grid_thw"] = vis_grid
            resolved["global.grid_thw"] = vis_grid
            self._execute_component(vision_name, "forward", None)
            modal_embeds = self._get_component_output(vision_name)
            if modal_embeds is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision component '{vision_name}' "
                    f"produced no output.")
            if modal_embeds.ndim == 2:
                n, h = modal_embeds.shape
                modal_embeds = modal_embeds.reshape(1, n, h)
            modal_embeds = modal_embeds.to(dtype)
            n_modal = modal_embeds.shape[1]
            if has_deepstack:
                # Tracer names the vision tuple's flattened extras
                # output_1..N ((hidden, [deepstack x3]) return).
                for k in range(1, len(ds_input_names) + 1):
                    t = resolved.get(f"{vision_name}.output_{k}")
                    if not isinstance(t, NBXTensor):
                        raise RuntimeError(
                            f"ZERO FALLBACK: LM graph declares "
                            f"{len(ds_input_names)} deepstack inputs but vision "
                            f"component '{vision_name}' produced no output_{k}.")
                    deepstack_embeds.append(t.to(dtype))
            print(f"   [{vision_name}] {n_modal} vision tokens in "
                  f"{(time.perf_counter() - start) * 1000:.0f}ms"
                  + (f" (+{len(deepstack_embeds)} deepstack levels)"
                     if deepstack_embeds else ""))
            if not self.ctx.persistent_mode:
                self._unload_component_weights(vision_name)
                release_flow_memory(self.ctx.primary_device)
        else:
            # ── Step 2 (audio): mel features → audio tower forward ──
            # R30 mirror of the compiled audio branch: the mel DSP is the
            # SHARED numpy core (R34 boundary I/O), features cross into
            # the arena as NBXTensor.
            print(f"   [{audio_name}] Running audio forward...")
            start = time.perf_counter()
            self._ensure_weights_loaded(audio_name)
            _aud_graph_path = (Path(self.ctx.pkg.cache_path) / "components"
                               / audio_name / "graph.json")
            _in_shape = None
            try:
                with open(_aud_graph_path, "r") as _f:
                    _in_shape = (json.load(_f).get("tensors", {})
                                 .get("input::input_features", {})
                                 .get("shape"))
            except Exception:
                pass
            _ap = vlm.get("audio_preprocessing") or {}
            _ptype = {"whisper_mel": "mel_spectrogram"}.get(
                _ap.get("type", "whisper_mel"), _ap.get("type"))
            from neurobrix.core.module.audio import mel_dsp
            feats_np = mel_dsp.extract_features_np(
                _ptype, str(audio_path), Path(self.ctx.pkg.cache_path),
                tuple(_in_shape) if _in_shape else None, params=_ap)
            while feats_np.ndim > 2:
                feats_np = feats_np[0]
            L = int(feats_np.shape[1])
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
            feats = NBXTensor.from_numpy(
                np.ascontiguousarray(feats_np)).to(dtype)
            resolved[f"{audio_name}.input_features"] = feats
            resolved["global.input_features"] = feats
            _flens = NBXTensor.from_numpy(np.array([L], dtype=np.int64))
            resolved[f"{audio_name}.feature_lens"] = _flens
            resolved["global.feature_lens"] = _flens
            _alens = NBXTensor.from_numpy(np.array([n_modal], dtype=np.int64))
            resolved[f"{audio_name}.aftercnn_lens"] = _alens
            resolved["global.aftercnn_lens"] = _alens
            self._execute_component(audio_name, "forward", None)
            modal_embeds = self._get_component_output(audio_name)
            if modal_embeds is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio component '{audio_name}' "
                    f"produced no output.")
            if modal_embeds.ndim == 2:
                n, h = modal_embeds.shape
                modal_embeds = modal_embeds.reshape(1, n, h)
            modal_embeds = modal_embeds.to(dtype)
            if modal_embeds.shape[1] != n_modal:
                raise RuntimeError(
                    f"ZERO FALLBACK: audio output has "
                    f"{modal_embeds.shape[1]} tokens but aftercnn({L}) "
                    f"expects {n_modal}.")
            print(f"   [{audio_name}] {n_modal} audio tokens "
                  f"(L={L} mel frames) in "
                  f"{(time.perf_counter() - start) * 1000:.0f}ms")
            if not self.ctx.persistent_mode:
                self._unload_component_weights(audio_name)
                release_flow_memory(self.ctx.primary_device)

        if has_visual:
            t, h, wgrid = _read_small_ints(vis_grid, 3)
            expected = t * (h // merge) * (wgrid // merge)
            if n_modal != expected:
                raise RuntimeError(
                    f"ZERO FALLBACK: vision output has {n_modal} tokens but "
                    f"grid {t}x{h}x{wgrid} / merge {merge} expects {expected}.")
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
        # CLI overrides first — R30 mirror of the compiled vlm flow (the
        # request's global.* variables beat the build defaults).
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
        # Declared-MoE fusion — R30 mirror of the compiled vlm flow (the
        # Stage-1 wall-5 class: without it only trace-fired experts run
        # and logits go near-uniform ⇒ degenerate repetition).
        lm_cfg = self.ctx.pkg.defaults.get("lm_config", {})
        _n_exp = lm_cfg.get("num_experts")
        _lm_exec = self.ctx.executors.get(lm_name)
        if _n_exp is not None and _n_exp > 1 and _lm_exec is not None:
            _norm_topk = lm_cfg.get("norm_topk_prob")
            if _norm_topk is None:
                raise RuntimeError(
                    "ZERO FALLBACK: norm_topk_prob missing from lm_config "
                    "for MoE model — add moe.norm_topk_prob to the registry.")
            _lm_exec.set_moe_config(norm_topk_prob=_norm_topk)
        embed_weight = self._get_embed_weight(lm_name)
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vlm stage '{lm_name}' requires "
                f"embed_tokens weight.")

        parts: List[NBXTensor] = []
        if prefix_ids:
            parts.append(self._embed_ids(prefix_ids, embed_weight, dtype))
        parts.append(modal_embeds)
        if suffix_ids:
            parts.append(self._embed_ids(suffix_ids, embed_weight, dtype))
        context_embeds = (NBXTensor.cat(parts, dim=1)
                          if len(parts) > 1 else modal_embeds)

        # Visual spans get the 3-D grid planes with a temporal scale;
        # audio spans use 1-D text-style positions (vendor audio-only
        # branch). Temporal scale — R30 mirror of the compiled flow:
        # position_id_per_seconds present → image ×pids, video
        # ×(second_per_grid × pids); absent (legacy mrope) → plain +1.
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
                    _spg = _read_small_float(video_spg)
                _t_scale = _spg * float(_pids)
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int, float]]]] = []
        if prefix_ids:
            segments.append(("text", len(prefix_ids), None))
        if has_visual:
            segments.append(("visual", n_modal,
                             (t, h // merge, wgrid // merge, _t_scale)))
        else:
            segments.append(("text", n_modal, None))
        if suffix_ids:
            segments.append(("text", len(suffix_ids), None))
        base_positions_np, next_pos = self._build_mrope_positions_np(segments)

        logits_source = ("lm_head"
                         if (head_name and head_name in self.ctx.executors)
                         else "embed_weight_tied")
        print(f"   [{lm_name}] Generating (max={max_tokens}, "
              f"context={context_embeds.shape[1]}, logits={logits_source})...")
        start = time.perf_counter()
        generated_ids: List[int] = []

        img_span_start = len(prefix_ids)
        if has_deepstack and not has_visual:
            # Audio requests on a DeepStack graph feed the empty-stub
            # form (all-False mask + zero-length embeds) — R30 mirror.
            _h_ds = context_embeds.shape[2]
            deepstack_embeds = [
                NBXTensor.from_numpy(np.zeros((0, _h_ds), dtype=np.float16))
                for _ in ds_input_names]
        for _step in range(max_tokens):
            n_gen = len(generated_ids)
            if n_gen:
                gen = np.arange(next_pos, next_pos + n_gen, dtype=np.int64)
                gen = np.broadcast_to(gen.reshape(1, 1, -1), (3, 1, n_gen))
                pos_np = np.concatenate([base_positions_np, gen], axis=2)
            else:
                pos_np = base_positions_np
            position_ids = NBXTensor.from_numpy(
                np.ascontiguousarray(pos_np))

            resolved["global.inputs_embeds"] = context_embeds
            resolved["inputs_embeds"] = context_embeds
            resolved["global.position_ids"] = position_ids
            resolved["position_ids"] = position_ids
            if has_deepstack:
                # Expanded [B, S, H] masked_scatter form (the graph
                # reduces it via [..., 0]); True exactly over the IMAGE
                # span (all-False for audio), generated tokens extend
                # with False — the per-step full re-forward matches
                # vendor semantics. numpy build = allowed CPU glue.
                _s = context_embeds.shape[1]
                _mask_np = np.zeros(
                    (1, _s, context_embeds.shape[2]), dtype=bool)
                if has_visual:
                    _mask_np[0, img_span_start:img_span_start + n_modal, :] = True
                _mask = NBXTensor.from_numpy(_mask_np)
                resolved["visual_pos_masks"] = _mask
                resolved["global.visual_pos_masks"] = _mask
                for _i, _t in enumerate(deepstack_embeds):
                    resolved[f"deepstack_visual_embeds.{_i}"] = _t
                    resolved[f"global.deepstack_visual_embeds.{_i}"] = _t

            self._execute_component(lm_name, "forward", None)
            output = self._get_component_output(lm_name)
            if output is None:
                break

            logits = self._compute_logits(
                output, embed_weight, logits_source, head_name)
            next_token = _sample_token_nbx(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            if next_token in eos_ids:
                break
            token_embed = self._embed_ids([next_token], embed_weight, dtype)
            context_embeds = NBXTensor.cat([context_embeds, token_embed],
                                           dim=1)
            if os.environ.get("NBX_VLM_STEP_SYNC"):
                # Binary race probe: force the flow's async tail (logits
                # projection on the head device, sampler, next-token
                # embedding, cat) to complete on EVERY device before the
                # next executor.run's rebind/transfers (current-device
                # sync alone is blind to the scattered placement).
                from neurobrix.kernels.nbx_tensor import DeviceAllocator
                _prev_d = DeviceAllocator.get_device()
                try:
                    for _d in range(8):
                        try:
                            DeviceAllocator.set_device(_d)
                            DeviceAllocator.sync_device()
                        except Exception:
                            break
                finally:
                    DeviceAllocator.set_device(_prev_d)

        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens in "
              f"{(time.perf_counter() - start) * 1000:.0f}ms")
        resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

        # ── Step 5: decode tokens → text (BOUNDARY: tokenizer) ──
        from neurobrix.triton.audio_frontend import (
            postprocess_text_output_np as postprocess_text_output,
        )
        postprocess_text_output(self.ctx)
        out_var = vlm.get("output", {}).get("variable",
                                            "global.generated_text")
        if out_var and resolved.get("global.transcription") is not None:
            resolved[out_var] = resolved["global.transcription"]
        return self.ctx.variable_resolver.resolve_all()

    # ─── mrope positions (numpy; mirror of core VLMEngine) ────────────

    @staticmethod
    def _build_mrope_positions_np(segments) -> Tuple[np.ndarray, int]:
        planes: List[np.ndarray] = []
        offset = 0
        for kind, length, grid in segments:
            if kind == "text":
                pos = np.arange(length, dtype=np.int64)
                seg = np.broadcast_to(pos.reshape(1, -1), (3, length)).copy()
                seg += offset
            else:
                gt, gh, gw, t_scale = grid
                # float32 scale then truncate — R30 mirror of the
                # compiled builder's (arange.float() * t_scale).long().
                t_vals = (np.arange(gt, dtype=np.float32)
                          * np.float32(t_scale)).astype(np.int64)
                t_idx = np.repeat(t_vals, gh * gw)
                h_idx = np.tile(np.repeat(np.arange(gh, dtype=np.int64), gw),
                                gt)
                w_idx = np.tile(np.arange(gw, dtype=np.int64), gt * gh)
                seg = np.stack([t_idx, h_idx, w_idx]) + offset
            planes.append(seg)
            offset = int(seg.max()) + 1
        positions = np.concatenate(planes, axis=1).reshape(3, 1, -1)
        return positions, offset

    # ─── tokenization around the image span (text-only boundary) ──────

    def _tokenize_around_span(self, prompt: str, span_token_id: int,
                              content_type: str
                              ) -> Tuple[List[int], List[int]]:
        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires the embedded tokenizer.")
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
            raise RuntimeError(
                "ZERO FALLBACK: the embedded tokenizer could not apply its "
                f"chat template to a {content_type}+text message; the vlm "
                "flow requires a multimodal chat template.") from e
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = list(ids[0] if ids and isinstance(ids[0], (list, tuple))
                   else ids)
        positions = [i for i, tid in enumerate(ids) if tid == span_token_id]
        if not positions:
            raise RuntimeError(
                f"ZERO FALLBACK: chat template produced no {content_type} "
                f"placeholder (token id {span_token_id}) — cannot merge "
                f"modality embeddings.")
        first, last = positions[0], positions[-1]
        if positions != list(range(first, last + 1)):
            raise RuntimeError(
                f"ZERO FALLBACK: {content_type} placeholder span is not "
                "contiguous — concat-merge equivalence does not hold.")
        return ids[:first], ids[last + 1:]

    # ─── helpers (NBXTensor mirrors) ───────────────────────────────────

    def _embed_ids(self, ids: List[int], embed_weight: NBXTensor,
                   dtype: NBXDtype) -> NBXTensor:
        idx = NBXTensor.from_numpy(np.array([ids], dtype=np.int64))
        emb = w.embedding(embed_weight, idx)
        return emb.to(dtype)

    def _get_component_output(self, comp_name: str) -> Optional[NBXTensor]:
        resolved = self.ctx.variable_resolver.resolved
        for key in (f"{comp_name}.output_0",
                    f"{comp_name}.last_hidden_state",
                    f"{comp_name}.output"):
            v = resolved.get(key)
            if isinstance(v, NBXTensor):
                return v
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[NBXTensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is not None:
            for key in executor._weights:
                if ("embed_tokens" in key or "token_embed" in key
                        or key == "embed.weight"):
                    return executor._weights[key]
        return None

    def _compute_logits(self, hidden_states: NBXTensor,
                        embed_weight: Optional[NBXTensor],
                        logits_source: str,
                        head_name: Optional[str]) -> NBXTensor:
        last_hidden = hidden_states[:, -1:, :]

        def _proj(weight: NBXTensor) -> NBXTensor:
            wt = weight.to(last_hidden.dtype)
            B, S, H = last_hidden.shape
            a2 = last_hidden.reshape(B * S, H)
            wtT = wt.transpose(0, 1)
            out = w.matmul_wrapper(a2, wtT)
            return out.reshape(B, S, out.shape[-1])

        if (logits_source == "lm_head" and head_name
                and head_name in self.ctx.executors):
            self._ensure_weights_loaded(head_name)
            executor = self.ctx.executors[head_name]
            for _key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    return _proj(tensor)
        if embed_weight is not None:
            return _proj(embed_weight)
        raise RuntimeError(
            "ZERO FALLBACK: no lm_head weight and no embed weight — "
            "cannot project hidden states to logits.")
