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
        image_token_id = vlm.get("image_token_id")
        merge = vlm.get("spatial_merge_size")
        if not vision_name or not lm_name or image_token_id is None or not merge:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.vlm must declare vision_component, "
                "lm_component, image_token_id and spatial_merge_size.")

        dtype = self._get_compute_dtype()
        device = self.ctx.primary_device

        # ── Step 1: vision inputs (preprocessed at the CLI boundary) ──
        in_cfg = vlm.get("input", {})
        pixel_values = resolved.get(in_cfg.get("image_variable", "global.pixel_values"))
        grid_thw = resolved.get(in_cfg.get("grid_variable", "global.image_grid_thw"))
        prompt = resolved.get(in_cfg.get("prompt_variable", "global.prompt"))
        if pixel_values is None or grid_thw is None:
            raise RuntimeError(
                "ZERO FALLBACK: vlm flow requires preprocessed image inputs "
                f"({in_cfg.get('image_variable')}, {in_cfg.get('grid_variable')}) — "
                "provide --input-image.")
        if not prompt:
            raise RuntimeError("ZERO FALLBACK: vlm flow requires a text prompt.")

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

        # ── Step 2: vision encoder forward ──
        print(f"   [{vision_name}] Running vision forward...")
        start = time.perf_counter()
        self._ensure_weights_loaded(vision_name)
        resolved[f"{vision_name}.hidden_states"] = pixel_values.to(device=device, dtype=dtype)
        resolved["global.hidden_states"] = resolved[f"{vision_name}.hidden_states"]
        resolved[f"{vision_name}.grid_thw"] = grid_thw.to(device=device)
        resolved["global.grid_thw"] = resolved[f"{vision_name}.grid_thw"]
        self._execute_component(vision_name, "forward", None)
        vision_embeds = self._get_component_output(vision_name)
        if vision_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: vision component '{vision_name}' produced no output.")
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)          # [1, n_merged, H]
        vision_embeds = vision_embeds.to(dtype=dtype)
        n_merged = vision_embeds.shape[1]
        deepstack_embeds: List[torch.Tensor] = []
        if has_deepstack:
            # The tracer names the vision tuple's flattened extras
            # output_1..N ((hidden, [deepstack x3]) return); each is
            # [n_merged, H_text], row-aligned with the merged hidden.
            for k in range(1, len(ds_input_names) + 1):
                t = resolved.get(f"{vision_name}.output_{k}")
                if t is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: LM graph declares "
                        f"{len(ds_input_names)} deepstack inputs but vision "
                        f"component '{vision_name}' produced no output_{k}.")
                deepstack_embeds.append(t.to(device=device, dtype=dtype))
        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{vision_name}] {n_merged} vision tokens in {elapsed:.0f}ms"
              + (f" (+{len(deepstack_embeds)} deepstack levels)"
                 if deepstack_embeds else ""))
        if not self.ctx.persistent_mode:
            self._unload_component_weights(vision_name)
            release_flow_memory(self.ctx.primary_device)

        # merged-token sanity: placeholder count must equal t*h*w/merge²
        t, h, w = (int(x) for x in grid_thw.reshape(-1)[:3])
        expected = t * (h // merge) * (w // merge)
        if n_merged != expected:
            raise RuntimeError(
                f"ZERO FALLBACK: vision output has {n_merged} tokens but "
                f"grid {t}x{h}x{w} / merge {merge} expects {expected}.")

        # ── Step 3: tokenize prompt around the image span ──
        prefix_ids, suffix_ids = self._tokenize_around_image(
            str(prompt), image_token_id, vlm)

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
        parts.append(vision_embeds.to(device=device))
        if suffix_ids:
            parts.append(_embed(suffix_ids))
        context_embeds = torch.cat(parts, dim=1)

        # segment layout for mrope positions: (kind, length, (t,h,w) or None)
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int]]]] = []
        if prefix_ids:
            segments.append(("text", len(prefix_ids), None))
        segments.append(("image", n_merged, (t, h // merge, w // merge)))
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
                # [..., 0]). True exactly over the image span; generated
                # tokens extend the mask with False, so the per-step full
                # re-forward matches vendor semantics (deepstack adds only
                # at visual positions).
                _s = context_embeds.shape[1]
                _mask2d = torch.zeros(1, _s, dtype=torch.bool, device=device)
                _mask2d[0, img_span_start:img_span_start + n_merged] = True
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

    # ─── mrope positions (internal get_rope_index port) ─────────────────

    @staticmethod
    def _build_mrope_positions(segments, device) -> Tuple[torch.Tensor, int]:
        """[3, 1, S] positions per the vendor get_rope_index contract.

        Text segment of length L: all three planes = offset + arange(L).
        Image segment of llm grid (t, h, w): temporal/height/width meshgrid
        indices + offset. Every segment starts at max(previous) + 1.
        Returns (positions, next_offset_for_decode)."""
        planes: List[torch.Tensor] = []
        offset = 0
        for kind, length, grid in segments:
            if kind == "text":
                pos = torch.arange(length, dtype=torch.long, device=device)
                seg = pos.view(1, -1).expand(3, -1) + offset
            else:
                gt, gh, gw = grid
                t_idx = torch.arange(gt, dtype=torch.long, device=device)\
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

    def _tokenize_around_image(self, prompt: str, image_token_id: int,
                               vlm: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """Chat-template the prompt with an image content part, encode, and
        split at the (single, contiguous) image placeholder span.

        The placeholder span = [image_start?] image_token(s) [image_end?] —
        the placeholder ids themselves are REPLACED by the vision embeds
        (start/end markers stay in the text halves when the template emits
        them outside the span; GLM emits begin/end markers around ONE
        image token, both markers kept)."""
        tokenizer = self._get_tokenizer()
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
            # Templates that only accept string content: place the image
            # token literally via its special-token string is model-specific
            # knowledge — refuse instead of guessing.
            raise RuntimeError(
                "ZERO FALLBACK: the embedded tokenizer could not apply its "
                "chat template to an image+text message; the vlm flow "
                "requires a multimodal chat template.") from e
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = list(ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)
        positions = [i for i, tid in enumerate(ids) if tid == image_token_id]
        if not positions:
            raise RuntimeError(
                "ZERO FALLBACK: chat template produced no image placeholder "
                f"(token id {image_token_id}) — cannot merge vision embeddings.")
        first, last = positions[0], positions[-1]
        if positions != list(range(first, last + 1)):
            raise RuntimeError(
                "ZERO FALLBACK: image placeholder span is not contiguous — "
                "concat-merge equivalence does not hold.")
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
        if executor is not None:
            for key in executor._weights:
                if "embed_tokens" in key or "token_embed" in key or key == "embed.weight":
                    return executor._weights[key]
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
