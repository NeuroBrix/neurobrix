"""Triton DualAR Engine — zero torch DualAR flow.

Ported from core/flow/dual_ar.py. Handles DualAR architecture:
backbone generates semantic tokens from 3D input [B, N+1, T],
then embeds tokens for codec decoder -> waveform.

No torch imports in hot path.
"""

import gc
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class TritonDualAREngine:
    """
    Triton-mode DualAR semantic token generation + codec decoding.

    topology.flow.audio:
        direction: tts
        stages:
          - component: model
            execution: dual_ar
          - component: codec.decoder
            execution: forward
    """

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

    def execute(self) -> Dict[str, Any]:
        """Execute DualAR pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        if not stages:
            raise RuntimeError("ZERO FALLBACK: dual_ar flow requires at least one stage.")

        # -- Step 1: Tokenize text input --
        from neurobrix.core.flow.audio_utils import preprocess_text_input
        preprocess_text_input(self.ctx)

        # -- Step 2: DualAR generation --
        backbone_stage = stages[0]
        comp_name = backbone_stage["component"]
        device_idx = _parse_device_idx(self.ctx.primary_device)
        DeviceAllocator.set_device(device_idx)

        max_tokens = defaults.get("max_tokens", 2048)
        temperature = defaults.get("temperature", 0.7)
        top_p = defaults.get("top_p", 0.8)
        eos_token_id = defaults.get("eos_token_id")

        print(f"   [{comp_name}] DualAR generation (max_tokens={max_tokens})...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)
        executor = self.ctx.executors[comp_name]

        # Get graph input shape: [B, N+1, T]
        dag = getattr(executor, '_dag', None)
        n_codebooks = 11
        trace_seq_len = 64
        if dag:
            for _tid, spec in dag.get("tensors", {}).items():
                if spec.get("input_name") == "inp":
                    shape = spec.get("shape", [])
                    if len(shape) >= 3:
                        n_val = shape[1]
                        n_codebooks = n_val if isinstance(n_val, int) else 11
                        t_val = shape[2]
                        trace_seq_len = t_val if isinstance(t_val, int) else 64
                    break

        # Get tokenized input
        input_ids = self.ctx.variable_resolver.resolved.get("global.input_ids")
        if input_ids is None:
            raise RuntimeError("ZERO FALLBACK: DualAR requires tokenized input_ids.")

        input_ids_np = _to_numpy(input_ids).astype(np.int64)
        if input_ids_np.ndim > 1:
            input_ids_np = input_ids_np.squeeze(0)
        prompt_len = len(input_ids_np)

        # Generate semantic tokens autoregressively
        generated_semantic: List[int] = []

        for step in range(max_tokens):
            cur_len = prompt_len + len(generated_semantic)
            if cur_len <= trace_seq_len:
                padded = np.zeros((1, n_codebooks, trace_seq_len), dtype=np.int64)
                padded[0, 0, :prompt_len] = input_ids_np[:prompt_len]
                for i, tok in enumerate(generated_semantic):
                    padded[0, 0, prompt_len + i] = tok
            else:
                # Sliding window
                all_semantic = list(input_ids_np[:prompt_len]) + generated_semantic
                window = all_semantic[-trace_seq_len:]
                padded = np.zeros((1, n_codebooks, trace_seq_len), dtype=np.int64)
                for i, tok in enumerate(window):
                    padded[0, 0, i] = tok

            padded_nbx = NBXTensor.from_numpy(padded)
            output = executor.run({"inp": padded_nbx})

            if isinstance(output, dict):
                logits = next(iter(output.values()))
            elif _is_tensor(output):
                logits = output
            else:
                logits = None

            if logits is None:
                break

            # Get logits at last real token position
            logits_np = _to_numpy(logits)
            pos = min(cur_len - 1, trace_seq_len - 1)
            if logits_np.ndim == 3:
                last_logits = logits_np[0, pos, :]
            else:
                last_logits = logits_np.flatten()

            next_token = _sample_token_np(last_logits, temperature, top_p=top_p)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            generated_semantic.append(next_token)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Generated {len(generated_semantic)} semantic tokens in {elapsed:.0f}ms")

        # -- Step 3: Embed semantic tokens -> codec input --
        embed_weight = None
        if hasattr(executor, '_weights'):
            best_vocab = 0
            for wname, wtensor in executor._weights.items():
                if 'embed' in wname and wname.endswith('.weight'):
                    if 'codebook' in wname or 'fast' in wname:
                        continue
                    w_shape = wtensor.shape if hasattr(wtensor, 'shape') else (0,)
                    if w_shape[0] > best_vocab:
                        best_vocab = w_shape[0]
                        embed_weight = wtensor

        if embed_weight is None:
            raise RuntimeError("ZERO FALLBACK: DualAR model must have embed.weight.")

        embed_np = _to_numpy(embed_weight).astype(np.float32)
        token_ids_np = np.array(generated_semantic, dtype=np.int64)
        max_id = int(token_ids_np.max()) if len(token_ids_np) > 0 else 0
        print(f"   [{comp_name}] Token range: 0..{max_id}, embed vocab: {embed_np.shape[0]}")
        if max_id >= embed_np.shape[0]:
            print(f"   [{comp_name}] WARNING: token {max_id} >= vocab {embed_np.shape[0]}, clamping")
            token_ids_np = np.clip(token_ids_np, 0, embed_np.shape[0] - 1)

        token_embeds = embed_np[token_ids_np]  # [T_gen, dim]
        codec_input_np = token_embeds[np.newaxis, :, :].transpose(0, 2, 1)  # [1, dim, T_gen]
        codec_input = NBXTensor.from_numpy(codec_input_np.astype(np.float32))
        print(f"   [{comp_name}] Embedded -> {list(codec_input.shape)} for codec.decoder")

        # Store for downstream
        self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = codec_input
        self.ctx.variable_resolver.resolved["global.generated_codes"] = NBXTensor.from_numpy(token_ids_np)
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_semantic

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()

        # -- Step 4: Codec decoder (forward stages) --
        for stage in stages[1:]:
            codec_name = stage["component"]
            if codec_name not in self.ctx.executors:
                print(f"   [{codec_name}] Skipped (not in executors)")
                continue

            print(f"   [{codec_name}] Running forward pass...")
            codec_start = time.perf_counter()
            self._ensure_weights_loaded(codec_name)

            if not self._try_chunked_forward(codec_name):
                self._execute_component(codec_name, "forward", None)

            codec_elapsed = (time.perf_counter() - codec_start) * 1000
            print(f"   [{codec_name}] Done in {codec_elapsed:.0f}ms")

            if not self.ctx.persistent_mode:
                self._unload_component_weights(codec_name)
                gc.collect()

        # -- Step 5: Output waveform --
        from neurobrix.core.flow.audio_utils import postprocess_audio_output
        postprocess_audio_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    def _try_chunked_forward(self, comp_name: str) -> bool:
        """Run chunked forward if input seq_len exceeds graph's trace-time seq_len."""
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return False

        dag = getattr(executor, '_dag', None)
        if dag is None:
            return False

        graph_input_name = None
        graph_seq_len = None
        for _tid, spec in dag.get("tensors", {}).items():
            iname = spec.get("input_name")
            if iname:
                shape = spec.get("shape", [])
                if len(shape) == 3:
                    graph_seq_len = shape[2]
                    if isinstance(graph_seq_len, dict):
                        graph_seq_len = graph_seq_len.get("trace_value", graph_seq_len)
                    graph_input_name = iname
                break

        if graph_input_name is None or not isinstance(graph_seq_len, int):
            return False

        # Find actual input tensor
        resolved = self.ctx.variable_resolver.resolved
        actual_input = None
        for key in [f"global.{graph_input_name}", graph_input_name]:
            val = resolved.get(key)
            if _is_tensor(val) and len(val.shape) == 3:
                actual_input = val
                break

        if actual_input is None:
            connections = self.ctx.pkg.topology.get("connections", [])
            for conn in connections:
                if conn.get("to", "") == f"{comp_name}.{graph_input_name}":
                    val = resolved.get(conn.get("from", ""))
                    if _is_tensor(val) and len(val.shape) == 3:
                        actual_input = val
                        break

        if actual_input is None:
            return False

        actual_seq = actual_input.shape[2]
        if actual_seq <= graph_seq_len:
            return False

        print(f"   [{comp_name}] Chunked: {actual_seq} frames -> {graph_seq_len}-frame blocks")
        waveform_chunks = []

        actual_np = _to_numpy(actual_input)

        for chunk_start in range(0, actual_seq, graph_seq_len):
            chunk_end = min(chunk_start + graph_seq_len, actual_seq)
            chunk = actual_np[:, :, chunk_start:chunk_end]

            if chunk.shape[2] < graph_seq_len:
                pad_width = ((0, 0), (0, 0), (0, graph_seq_len - chunk.shape[2]))
                chunk = np.pad(chunk, pad_width, mode='constant')

            chunk_nbx = NBXTensor.from_numpy(chunk.astype(np.float32))
            output = executor.run({graph_input_name: chunk_nbx})

            if isinstance(output, dict):
                out_tensor = next(iter(output.values()))
            elif _is_tensor(output):
                out_tensor = output
            else:
                out_tensor = None

            if out_tensor is not None:
                out_np = _to_numpy(out_tensor)
                if chunk_end - chunk_start < graph_seq_len and out_np.ndim >= 2:
                    ratio = (chunk_end - chunk_start) / graph_seq_len
                    trim_len = int(out_np.shape[-1] * ratio)
                    out_np = out_np[..., :trim_len]
                waveform_chunks.append(out_np)

        if waveform_chunks:
            full_output_np = np.concatenate(waveform_chunks, axis=-1)
            full_output = NBXTensor.from_numpy(full_output_np.astype(np.float32))
            self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = full_output
            self.ctx.variable_resolver.resolved["global.output_audio"] = full_output
            print(f"   [{comp_name}] Waveform: {list(full_output.shape)}")

        return True


# -----------------------------------------------------------------
# Module-level helpers (zero torch)
# -----------------------------------------------------------------

def _sample_token_np(
    logits_1d: np.ndarray, temperature: float,
    top_p: float = 1.0,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
) -> int:
    """Sample next token from logits (NumPy)."""
    logits = logits_1d.copy().astype(np.float64)

    if repetition_penalty != 1.0 and generated_ids:
        for tid in set(generated_ids):
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty

    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = logits / temperature
    logits -= logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        mask = np.ones_like(probs, dtype=bool)
        mask[sorted_idx[:cutoff]] = False
        probs[mask] = 0.0
        probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def _to_numpy(tensor) -> np.ndarray:
    """Convert any tensor to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _is_tensor(val) -> bool:
    """Check if val is any tensor type."""
    return isinstance(val, NBXTensor) or (hasattr(val, 'shape') and hasattr(val, 'dtype'))


def _parse_device_idx(device_str: str) -> int:
    """Parse device index from device string."""
    if device_str.startswith("cuda:"):
        try:
            return int(device_str.split(":")[-1].split(",")[0])
        except ValueError:
            return 0
    try:
        return int(device_str)
    except ValueError:
        return 0
