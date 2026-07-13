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

import time
import numpy as np
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

        # ── Step 1: vision inputs (CLI boundary; alias only) ──
        in_cfg = vlm.get("input", {})
        pixel_values = resolved.get(
            in_cfg.get("image_variable", "global.pixel_values"))
        grid_thw = resolved.get(
            in_cfg.get("grid_variable", "global.image_grid_thw"))
        prompt = resolved.get(
            in_cfg.get("prompt_variable", "global.prompt"))
        if pixel_values is None or grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires preprocessed image "
                f"inputs ({in_cfg.get('image_variable')}, "
                f"{in_cfg.get('grid_variable')}) — provide --input-image.")
        if not prompt:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires a text prompt.")

        # ── Step 2: vision encoder forward ──
        print(f"   [{vision_name}] Running vision forward...")
        start = time.perf_counter()
        self._ensure_weights_loaded(vision_name)
        resolved[f"{vision_name}.hidden_states"] = pixel_values
        resolved["global.hidden_states"] = pixel_values
        resolved[f"{vision_name}.grid_thw"] = grid_thw
        resolved["global.grid_thw"] = grid_thw
        self._execute_component(vision_name, "forward", None)
        vision_embeds = self._get_component_output(vision_name)
        if vision_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vision component '{vision_name}' "
                f"produced no output.")
        if vision_embeds.ndim == 2:
            n, h = vision_embeds.shape
            vision_embeds = vision_embeds.reshape(1, n, h)
        vision_embeds = vision_embeds.to(dtype)
        n_merged = vision_embeds.shape[1]
        print(f"   [{vision_name}] {n_merged} vision tokens in "
              f"{(time.perf_counter() - start) * 1000:.0f}ms")
        if not self.ctx.persistent_mode:
            self._unload_component_weights(vision_name)
            release_flow_memory(self.ctx.primary_device)

        t, h, wgrid = _read_small_ints(grid_thw, 3)
        expected = t * (h // merge) * (wgrid // merge)
        if n_merged != expected:
            raise RuntimeError(
                f"ZERO FALLBACK: vision output has {n_merged} tokens but "
                f"grid {t}x{h}x{wgrid} / merge {merge} expects {expected}.")

        # ── Step 3: tokenize prompt around the image span ──
        prefix_ids, suffix_ids = self._tokenize_around_image(
            str(prompt), image_token_id)

        # ── Step 4: LM decode with merged embeds + mrope positions ──
        from neurobrix.core.runtime.decode_bound import decode_bound
        max_tokens = decode_bound(defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError(
                "ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError(
                "ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError(
                "ZERO FALLBACK: eos_token_id missing from defaults.json.")
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple))
                      else [eos_token_id])
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        self._ensure_weights_loaded(lm_name)
        embed_weight = self._get_embed_weight(lm_name)
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vlm stage '{lm_name}' requires "
                f"embed_tokens weight.")

        parts: List[NBXTensor] = []
        if prefix_ids:
            parts.append(self._embed_ids(prefix_ids, embed_weight, dtype))
        parts.append(vision_embeds)
        if suffix_ids:
            parts.append(self._embed_ids(suffix_ids, embed_weight, dtype))
        context_embeds = (NBXTensor.cat(parts, dim=1)
                          if len(parts) > 1 else vision_embeds)

        segments: List[Tuple[str, int, Optional[Tuple[int, int, int]]]] = []
        if prefix_ids:
            segments.append(("text", len(prefix_ids), None))
        segments.append(("image", n_merged,
                         (t, h // merge, wgrid // merge)))
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
                gt, gh, gw = grid
                t_idx = np.repeat(np.arange(gt, dtype=np.int64), gh * gw)
                h_idx = np.tile(np.repeat(np.arange(gh, dtype=np.int64), gw),
                                gt)
                w_idx = np.tile(np.arange(gw, dtype=np.int64), gt * gh)
                seg = np.stack([t_idx, h_idx, w_idx]) + offset
            planes.append(seg)
            offset = int(seg.max()) + 1
        positions = np.concatenate(planes, axis=1).reshape(3, 1, -1)
        return positions, offset

    # ─── tokenization around the image span (text-only boundary) ──────

    def _tokenize_around_image(self, prompt: str, image_token_id: int
                               ) -> Tuple[List[int], List[int]]:
        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires the embedded tokenizer.")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            raise RuntimeError(
                "ZERO FALLBACK: the embedded tokenizer could not apply its "
                "chat template to an image+text message; the vlm flow "
                "requires a multimodal chat template.") from e
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = list(ids[0] if ids and isinstance(ids[0], (list, tuple))
                   else ids)
        positions = [i for i, tid in enumerate(ids) if tid == image_token_id]
        if not positions:
            raise RuntimeError(
                "ZERO FALLBACK: chat template produced no image placeholder "
                f"(token id {image_token_id}) — cannot merge vision "
                f"embeddings.")
        first, last = positions[0], positions[-1]
        if positions != list(range(first, last + 1)):
            raise RuntimeError(
                "ZERO FALLBACK: image placeholder span is not contiguous — "
                "concat-merge equivalence does not hold.")
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
