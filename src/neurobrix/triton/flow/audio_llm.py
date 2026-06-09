"""Triton AudioLLMEngine — zero-torch audio-conditioned LLM flow.

R33-pure port of core/flow/audio_llm.py (AudioLLMEngine). Pattern:
encoder(audio) → projector → LLM(inputs_embeds) → text.

All compute is NBXTensor + kernel wrappers; ZERO torch on the compute
path. Two sanctioned BOUNDARY helpers are imported from
core.flow.audio_utils — exactly as the other triton audio handlers do
(triton/flow/encoder_decoder.py, dual_ar.py): `preprocess_audio_input`
(mel/FFT, torch/torchaudio internally, output converted to NBXTensor
at the boundary) and `postprocess_text_output` (tokenizer decode of
the generated ids — pure text, no compute tensor). These are the same
torch-boundary subroutines the R33 doctrine explicitly allows for
audio I/O; the model forward, embedding, concat, logits and sampling
are all NBXTensor/Triton.

ZERO SEMANTIC: no model-specific knowledge. ZERO HARDCODE: all
parameters from the NBX container / defaults.json.
"""

import gc
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.kernels import wrappers as w


_DTYPE_STR_TO_NBX = {
    "float16": NBXDtype.float16, "fp16": NBXDtype.float16,
    "bfloat16": NBXDtype.bfloat16, "bf16": NBXDtype.bfloat16,
    "float32": NBXDtype.float32, "fp32": NBXDtype.float32,
}


def _nbx_f32_to_numpy(t: NBXTensor) -> np.ndarray:
    """R33-pure NBXTensor → host float32 numpy (D2H copy, no torch).
    Mirrors triton/flow/autoregressive.py::_read_ids_to_list."""
    import ctypes
    f = t.to(NBXDtype.float32).contiguous()
    n = f.numel()
    buf = (ctypes.c_char * (n * 4))()
    ctypes.cdll.LoadLibrary("libcudart.so").cudaMemcpy(
        ctypes.byref(buf), ctypes.c_void_p(f.data_ptr()), n * 4, 2)  # D2H
    return np.frombuffer(bytes(buf), dtype=np.float32).copy()


def _sample_token_nbx(
    logits: NBXTensor, temperature: float,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
) -> int:
    """NBXTensor mirror of core.flow.audio_utils.sample_token.

    Selecting one integer token id from a [1, S, V] logits tensor is a
    decode-control decision (the same `.item()` the core handler does);
    the compute that produced the logits stayed on the Triton path.
    """
    last = logits[:, -1, :] if logits.ndim == 3 else logits  # [1, V]

    if repetition_penalty != 1.0 and generated_ids:
        # Faithful mirror of the core per-id scaling, on a tiny [1,V]
        # vector at the decode-control boundary.
        row = _nbx_f32_to_numpy(last).reshape(-1)
        for tid in set(generated_ids):
            if row[tid] > 0:
                row[tid] /= repetition_penalty
            else:
                row[tid] *= repetition_penalty
        if temperature == 0.0:
            return int(np.argmax(row))
        last = NBXTensor.from_numpy(row.reshape(1, -1))

    if temperature == 0.0:
        idx = w.argmax_wrapper(last, dim=-1)
        return int(idx.item())

    # Mirror core: softmax(last / temperature) then multinomial.
    probs = w.softmax(last / float(temperature), dim=-1)
    nxt = w.multinomial_wrapper(probs, num_samples=1)
    return int(nxt.item())


class TritonAudioLLMEngine:
    """Audio-conditioned LLM (R33): encode audio → project → LLM
    autoregressive decode with audio embeddings. Mirror of
    core/flow/audio_llm.py AudioLLMEngine."""

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
        s = self.ctx.pkg.manifest.get("dtype", "float16")
        return _DTYPE_STR_TO_NBX.get(s, NBXDtype.float16)

    def execute(self) -> Dict[str, Any]:
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        forward_stages: List[Dict] = []
        ar_stage = None
        for s in stages:
            if s.get("execution", "forward") == "autoregressive":
                ar_stage = s
            else:
                forward_stages.append(s)

        if ar_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: audio_llm flow requires at least one "
                "'autoregressive' stage."
            )

        # ── Step 1: Preprocess audio (zero-torch numpy front-end) ──
        from neurobrix.triton.audio_frontend import (
            preprocess_audio_input_np as preprocess_audio_input,
            postprocess_text_output_np as postprocess_text_output,
        )
        preprocess_audio_input(self.ctx, audio_config, stages)

        # ── Step 2: Forward stages (encoder, projector) ──
        for stage in forward_stages:
            comp_name = stage["component"]
            if comp_name not in self.ctx.executors:
                print(f"   [{comp_name}] Skipped (not in executors)")
                continue
            print(f"   [{comp_name}] Running forward pass...")
            start = time.perf_counter()
            self._ensure_weights_loaded(comp_name)
            self._execute_component(comp_name, "forward", None)
            print(f"   [{comp_name}] Done in "
                  f"{(time.perf_counter() - start) * 1000:.0f}ms")
            self._store_output(comp_name)
            self._reshape_output_for_connections(comp_name)
            if not self.ctx.persistent_mode:
                self._unload_component_weights(comp_name)
                gc.collect()

        # ── Step 3: Autoregressive LLM decode with audio embeddings ──
        lm_name = ar_stage["component"]
        logits_source = ar_stage.get("logits_source", "lm_head")
        dtype = self._compute_dtype()

        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
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
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        self._ensure_weights_loaded(lm_name)
        embed_weight = self._get_embed_weight(lm_name)

        # Inject embed weight for weight-tied models (mirror core).
        if embed_weight is not None:
            executor = self.ctx.executors.get(lm_name)
            if executor is not None and hasattr(executor, "_weights"):
                dag = getattr(executor, "_dag", None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied in ("head.weight",
                                 "model.token_embed.weight"):
                        if (tied not in executor._weights
                                and f"param::{tied}" in tensors):
                            executor._weights[tied] = embed_weight

        audio_embeds = self._get_last_forward_output(forward_stages)
        if audio_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires "
                f"projected audio embeddings from forward stages, but "
                f"none found.")
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires "
                f"embed_tokens weight.")

        audio_embeds = audio_embeds.to(dtype)
        print(f"   [{lm_name}] Audio-LLM mode: audio_embeds "
              f"{tuple(audio_embeds.shape)}")

        # Build prompt: prefix_embeds + audio_embeds + suffix_embeds
        prefix_ids = defaults.get("stt_prefix_ids", [1])
        suffix_ids = defaults.get("stt_suffix_ids", [])

        parts: List[NBXTensor] = []
        if prefix_ids:
            parts.append(self._embed_ids(prefix_ids, embed_weight, dtype))
        parts.append(audio_embeds)
        if suffix_ids:
            parts.append(self._embed_ids(suffix_ids, embed_weight, dtype))
        context_embeds = (NBXTensor.cat(parts, dim=1)
                          if len(parts) > 1 else audio_embeds)

        print(f"   [{lm_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        generated_ids: List[int] = []

        for _step in range(max_tokens):
            seq_len = context_embeds.shape[1]
            position_ids = NBXTensor.from_numpy(
                np.arange(seq_len, dtype=np.int64).reshape(1, -1))

            res = self.ctx.variable_resolver.resolved
            res["global.inputs_embeds"] = context_embeds
            res["inputs_embeds"] = context_embeds
            res["global.position_ids"] = position_ids
            res["position_ids"] = position_ids

            self._execute_component(lm_name, "forward", None)

            output = self._get_component_output(lm_name)
            if output is None:
                break

            logits = self._compute_logits(
                output, embed_weight, logits_source)
            next_token = _sample_token_nbx(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            if next_token == eos_token_id:
                break

            token_embed = self._embed_ids(
                [next_token], embed_weight, dtype)
            context_embeds = NBXTensor.cat(
                [context_embeds, token_embed], dim=1)

        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens "
              f"in {(time.perf_counter() - start) * 1000:.0f}ms")

        self.ctx.variable_resolver.resolved[
            "global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            gc.collect()

        # ── Step 4: Decode tokens → text (BOUNDARY: tokenizer) ──
        postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers (NBXTensor mirrors of the core helpers) ──────────────

    def _embed_ids(self, ids: List[int], embed_weight: NBXTensor,
                   dtype: NBXDtype) -> NBXTensor:
        """F.embedding(ids, embed_weight) → NBXTensor [1, len(ids), H]."""
        idx = NBXTensor.from_numpy(
            np.array([ids], dtype=np.int64))
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

    def _get_last_forward_output(
            self, forward_stages: List[Dict]) -> Optional[NBXTensor]:
        for s in reversed(forward_stages):
            out = self._get_component_output(s["component"])
            if out is not None:
                return out
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[NBXTensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is not None:
            for key in executor._weights:
                if "embed_tokens" in key or "token_embed" in key:
                    return executor._weights[key]
        embed_executor = self.ctx.executors.get("embed_tokens")
        if embed_executor is not None:
            self._ensure_weights_loaded("embed_tokens")
            for key in embed_executor._weights:
                if key == "weight" or "embed" in key:
                    wt = embed_executor._weights[key]
                    if wt.ndim == 2 and wt.shape[0] > wt.shape[1]:
                        return wt
        return None

    def _compute_logits(
            self, hidden_states: NBXTensor,
            embed_weight: Optional[NBXTensor],
            logits_source: str) -> NBXTensor:
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, H]

        def _proj(weight: NBXTensor) -> NBXTensor:
            wt = weight.to(last_hidden.dtype)
            B, S, H = last_hidden.shape
            a2 = last_hidden.reshape(B * S, H)         # [1, H]
            wtT = wt.transpose(0, 1)                   # [H, V]
            out = w.matmul_wrapper(a2, wtT)            # [1, V]
            return out.reshape(B, S, out.shape[-1])    # [1, 1, V]

        if (logits_source == "lm_head"
                and "lm_head" in self.ctx.executors):
            self._ensure_weights_loaded("lm_head")
            executor = self.ctx.executors["lm_head"]
            for _key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    return _proj(tensor)
            return last_hidden

        if logits_source == "embed_weight_tied" and embed_weight is not None:
            return _proj(embed_weight)

        return last_hidden

    def _store_output(self, comp_name: str) -> None:
        resolved = self.ctx.variable_resolver.resolved
        for key in (f"{comp_name}.output_0",
                    f"{comp_name}.last_hidden_state"):
            if key in resolved:
                return

    def _reshape_output_for_connections(self, comp_name: str) -> None:
        """Multimodal frame pooling (DATA-DRIVEN), NBXTensor mirror of
        the core helper: audio_tower [B,T,D] → MMP-expected [B,T/p,D*p]."""
        connections = self.ctx.pkg.topology.get("connections", [])
        resolved = self.ctx.variable_resolver.resolved
        for conn in connections:
            src_key = conn.get("from", "")
            if not src_key.startswith(f"{comp_name}."):
                continue
            src = resolved.get(src_key)
            if not isinstance(src, NBXTensor) or src.ndim != 3:
                continue
            target_str = conn.get("to", "")
            parts = target_str.split(".")
            if len(parts) < 2:
                continue
            target_comp = parts[0]
            target_input = ".".join(parts[1:])
            target_executor = self.ctx.executors.get(target_comp)
            if target_executor is None:
                continue
            dag = getattr(target_executor, "_dag", None)
            if dag is None:
                continue
            target_feat = None
            for _tid, spec in dag.get("tensors", {}).items():
                if spec.get("input_name") == target_input:
                    shape = spec.get("shape", [])
                    if len(shape) >= 3:
                        feat = shape[-1]
                        if isinstance(feat, dict):
                            feat = feat.get("trace_value", feat)
                        if isinstance(feat, int):
                            target_feat = feat
                    break
            if target_feat is None:
                continue
            src_feat = src.shape[-1]
            if target_feat == src_feat:
                continue
            if target_feat > src_feat and target_feat % src_feat == 0:
                pool = target_feat // src_feat
                B, T, D = src.shape
                new_T = T // pool
                if new_T * pool <= T:
                    reshaped = src[:, :new_T * pool, :].reshape(
                        B, new_T, target_feat)
                    resolved[src_key] = reshaped
                    print(f"   [Reshape] {comp_name}: [{B}, {T}, {D}] "
                          f"→ [{B}, {new_T}, {target_feat}]")
