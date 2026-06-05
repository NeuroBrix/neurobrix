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

        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        # CLI sampling overrides (global.*) take precedence over the embedded
        # defaults, mirroring the compiled dual_ar flow (R30) — --temperature 0 ⇒
        # greedy. Without this the triton flow silently sampled at the default 0.7.
        _ov = self.ctx.variable_resolver.resolved
        max_tokens = decode_bound(_ov.get("global.max_tokens", defaults.get("max_tokens", 2048)))
        temperature = _ov.get("global.temperature", defaults.get("temperature", 0.7))
        top_p = _ov.get("global.top_p", defaults.get("top_p", 0.8))
        eos_token_id = defaults.get("eos_token_id")

        print(f"   [{comp_name}] DualAR generation (max_tokens={max_tokens})...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)
        executor = self.ctx.executors[comp_name]

        # Slow model input shape [B, N+1, T] — read the codebook-row count (N+1).
        # The full-grid generation feeds variable length, so the trace seq dim is
        # not needed (symbolic seq); only the row count is structural.
        dag = getattr(executor, '_dag', None)
        n_codebooks = 11
        if dag:
            for spec in dag.get("tensors", {}).values():
                if spec.get("input_name") == "inp":
                    shape = spec.get("shape", [])
                    if len(shape) >= 2 and isinstance(shape[1], int):
                        n_codebooks = shape[1]
                    break

        # Get tokenized input
        input_ids = self.ctx.variable_resolver.resolved.get("global.input_ids")
        if input_ids is None:
            raise RuntimeError("ZERO FALLBACK: DualAR requires tokenized input_ids.")

        input_ids_np = _to_numpy(input_ids).astype(np.int64)
        if input_ids_np.ndim > 1:
            input_ids_np = input_ids_np.squeeze(0)

        # Full DualAR generation (slow backbone + fast/depth transformer) — R30
        # mirror of core/flow/dual_ar.py (replaces the slow-AR-only + numpy
        # embedding shortcut). Orchestration in numpy, COMPUTE via executor.run on
        # NBXTensors (R33 — the flow handler may use numpy, never torch).
        import json as _json
        from pathlib import Path as _Path
        n_rows = n_codebooks                 # grid rows: 1 semantic + N acoustic
        num_codebooks = n_rows - 1           # acoustic codebooks (codec quantizer)
        acoustic_cb_size = 1024

        st_path = _Path(self.ctx.nbx_path_str) / "modules" / "tokenizer" / "special_tokens.json"
        special = _json.load(open(st_path)) if st_path.exists() else {}
        semantic_begin_id = special.get("<|semantic:0|>")
        if semantic_begin_id is None:
            raise RuntimeError("ZERO FALLBACK: '<|semantic:0|>' missing from tokenizer special_tokens.")
        im_end_id = special.get("<|im_end|>", eos_token_id)

        fast_exec = self.ctx.executors.get("model.fast")
        if fast_exec is None:
            raise RuntimeError("ZERO FALLBACK: DualAR requires 'model.fast' (re-trace with the fast split).")
        self._ensure_weights_loaded("model.fast")

        prompt_ids = [int(x) for x in np.asarray(input_ids_np).reshape(-1)]
        bos_id = defaults.get("bos_token_id")
        if bos_id is not None and prompt_ids and prompt_ids[0] == bos_id:
            prompt_ids = prompt_ids[1:]
        interleave_id = special.get("<|interleave|>")
        tok = self.ctx.modules.get("tokenizer")
        speaker_ids = []
        if tok is not None:
            try:
                _s = tok.encode("<|speaker:0|>", add_special_tokens=False)
            except TypeError:
                _s = tok.encode("<|speaker:0|>")
            speaker_ids = [int(x) for x in (_s.tolist() if hasattr(_s, "tolist") else _s)]
        prompt_ids = ([interleave_id] if interleave_id is not None else []) + speaker_ids + prompt_ids
        print(f"   [{comp_name}] prompt tokens ({len(prompt_ids)}): {prompt_ids[:16]}")

        rep_pen = _ov.get("global.repetition_penalty", defaults.get("repetition_penalty", 1.2))
        rep_window = 32
        sem_lo, sem_hi = semantic_begin_id, semantic_begin_id + 4096

        import os as _os_dbg
        _fixed_codes_env = _os_dbg.environ.get("NBX_DUALAR_FIXED_CODES", "")

        grid_cols = [[pid] + [0] * num_codebooks for pid in prompt_ids]
        acoustic_codes: List[List[int]] = []
        slow_history: List[int] = []

        if _fixed_codes_env:
            # Codec-isolation diagnostic: skip slow+fast AR; codes injected in Step 3.
            # The codec path is then validated independently of the generation (the
            # 4-mode kernel-oracle method — verify each half against its own oracle).
            print(f"   [{comp_name}] FIXED codes mode — generation skipped (codec isolation).")
        else:
            for _ in range(max_tokens):
                clen = len(grid_cols)
                inp = np.zeros((1, n_rows, clen), dtype=np.int64)
                for ci, col in enumerate(grid_cols):
                    for ri, val in enumerate(col):
                        inp[0, ri, ci] = val
                pos = clen - 1
                out = executor.run({"inp": NBXTensor.from_numpy(inp)})
                logits = out.get("logits") if isinstance(out, dict) else None
                hidden = out.get("hidden_states") if isinstance(out, dict) else None
                if logits is None or hidden is None:
                    raise RuntimeError("ZERO FALLBACK: slow model must output logits + hidden_states.")
                logits_np = _to_numpy(logits)
                hidden_np = _to_numpy(hidden)            # [1, clen, dim]
                slow_logits = logits_np[0, pos, :].astype(np.float64)
                _dsl = _os_dbg.environ.get("NBX_DUALAR_DUMP_SLOGITS", "")
                if _dsl and len(slow_history) == 0:
                    np.save(_dsl, slow_logits)
                masked = np.full_like(slow_logits, -np.inf)
                masked[sem_lo:sem_hi] = slow_logits[sem_lo:sem_hi]
                if im_end_id is not None:
                    masked[im_end_id] = slow_logits[im_end_id]
                slow_token = _sample_token_np(masked, temperature, top_p=top_p,
                                              repetition_penalty=rep_pen,
                                              generated_ids=slow_history[-rep_window:])
                if _os_dbg.environ.get("NBX_DEBUG_DECODE") == "1" and len(slow_history) < 16:
                    print(f"  [DBG-DUALAR] step={len(slow_history)} sem_token={int(slow_token)}", flush=True)
                if im_end_id is not None and slow_token == im_end_id:
                    break
                slow_history.append(slow_token)

                hidden_pos = hidden_np[:, pos, :]        # [1, dim], keep its dtype
                ctx = [max(0, slow_token - semantic_begin_id)]
                for _ in range(1, num_codebooks):
                    ctx_arr = np.zeros((1, num_codebooks - 1), dtype=np.int64)
                    for j, v in enumerate(ctx):
                        ctx_arr[0, j] = v
                    fout = fast_exec.run({"hidden": NBXTensor.from_numpy(hidden_pos),
                                          "codebook_context": NBXTensor.from_numpy(ctx_arr)})
                    flogits = fout.get("output") if isinstance(fout, dict) else fout
                    flogits_np = _to_numpy(flogits)
                    read_pos = len(ctx)
                    cb_logits = flogits_np[0, read_pos, :acoustic_cb_size]
                    sampled = _sample_token_np(cb_logits, temperature, top_p=top_p)
                    ctx.append(sampled)

                acoustic_codes.append(ctx)
                grid_cols.append([slow_token] + ctx)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Generated {len(acoustic_codes)} positions in {elapsed:.0f}ms")
        if not acoustic_codes and not _fixed_codes_env:
            raise RuntimeError("ZERO FALLBACK: DualAR produced no audio frames.")

        # -- Step 3: acoustic codes -> codec.quantizer.decode (RVQ) -> features --
        # Build the code grid [1, num_codebooks, T] and dequantize via the codec's
        # residual-VQ decoder (R30 mirror of core/flow/dual_ar.py). Replaces the old
        # numpy embedding shortcut that fed *semantic embeddings* to the decoder
        # instead of true RVQ features — the root of the near-silent triton audio.
        import os as _os_diag
        T = len(acoustic_codes)
        codes_np = np.zeros((1, num_codebooks, T), dtype=np.int64)
        for t, ac in enumerate(acoustic_codes):
            for r, v in enumerate(ac):
                codes_np[0, r, t] = int(v)

        _dump_codes = _os_diag.environ.get("NBX_DUALAR_DUMP_CODES", "")
        if _dump_codes:
            np.save(_dump_codes, codes_np)
            print(f"   [diag] dumped codes {list(codes_np.shape)} -> {_dump_codes}")
        _fixed_codes = _os_diag.environ.get("NBX_DUALAR_FIXED_CODES", "")
        if _fixed_codes:
            codes_np = np.load(_fixed_codes).astype(np.int64)
            T = codes_np.shape[2]
            print(f"   [diag] FIXED codes injected {list(codes_np.shape)} <- {_fixed_codes}")

        codec_q = "codec.quantizer"
        if codec_q not in self.ctx.executors:
            raise RuntimeError(f"ZERO FALLBACK: DualAR requires '{codec_q}' for RVQ decode.")
        self._ensure_weights_loaded(codec_q)
        q_out = self.ctx.executors[codec_q].run({"indices": NBXTensor.from_numpy(codes_np)})
        z = q_out.get("output") if isinstance(q_out, dict) else q_out
        if z is None:
            raise RuntimeError("ZERO FALLBACK: codec.quantizer.decode produced no features.")
        print(f"   [{codec_q}] decode codes {list(codes_np.shape)} -> features {list(z.shape)}")

        _dump_z = _os_diag.environ.get("NBX_DUALAR_DUMP_Z", "")
        if _dump_z:
            np.save(_dump_z, _to_numpy(z).astype(np.float32))
            print(f"   [diag] dumped z {list(z.shape)} -> {_dump_z}")

        # Bind RVQ features to the decoder input (topology connects
        # model.output_0 -> codec.decoder.x; mirror the compiled binding set).
        for key in ("global.x", "x", "codec.decoder.x", "model.output_0", f"{comp_name}.output_0"):
            self.ctx.variable_resolver.resolved[key] = z
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = [int(c[0]) for c in acoustic_codes]

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            self._unload_component_weights("model.fast")
            self._unload_component_weights(codec_q)
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

            # DAC decoder is fully symbolic (born-at-source seq) — single pass, no
            # trace-seq chunking (R30 mirror of core/flow/dual_ar.py).
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
    """Convert any tensor to numpy.

    For NBXTensor (Triton mode): upcasts bf16 to fp32 first (numpy has
    no native bf16), then D2H memcpy into a host buffer. Zero torch.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        t = tensor.contiguous()
        if t.dtype == NBXDtype.bfloat16:
            t = t.to(NBXDtype.float32)
        nb_to_np = {
            NBXDtype.float32: np.float32,
            NBXDtype.float16: np.float16,
            NBXDtype.int32:   np.int32,
            NBXDtype.int64:   np.int64,
        }
        np_dtype = nb_to_np.get(t.dtype)
        if np_dtype is None:
            t = t.to(NBXDtype.float32)
            np_dtype = np.float32
        arr = np.empty(t.shape, dtype=np_dtype)
        DeviceAllocator.memcpy(arr.ctypes.data, t.data_ptr(),
                               arr.nbytes, kind=2)
        return arr
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
