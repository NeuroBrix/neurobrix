"""
DualAR Engine — Fish-Speech/OpenAudio DualAR Flow

Handles DualAR architecture: backbone generates semantic tokens from
3D input [B, N+1, T], then embeds tokens for codec decoder → waveform.

ZERO SEMANTIC: No knowledge of "Fish-Speech" or "OpenAudio".
ZERO HARDCODE: All parameters from NBX container.
"""

import gc
from neurobrix.core.device_utils import device_empty_cache
import time
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow

# Shared default sampler seed (R27/R28) — MUST equal the triton dual_ar literal
# (triton/flow/dual_ar.py:_DUALAR_SEED) so the two separate code paths draw
# identical randoms when --seed is not passed.
_DUALAR_SEED = 1234


def _sample_token_np_core(logits_1d, temperature, top_p=1.0,
                          generated_ids=None, repetition_penalty=1.0, rng=None):
    """Numpy seeded sampler — byte-for-byte the same algorithm as the triton
    dual_ar `_sample_token_np`, so that at equal seed + equal (fp16-close) logits
    the pytorch and triton paths sample identical tokens. The pytorch path feeds
    it logits converted to fp64 numpy. R30 parity is the whole point; the two
    implementations are kept deliberately identical, not shared (R33 keeps the
    triton file torch-free, so it cannot import this torch-importing module)."""
    _draw = rng if rng is not None else np.random
    logits = np.asarray(logits_1d, dtype=np.float64).copy()
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
    return int(_draw.choice(len(probs), p=probs))


@register_flow("dual_ar")
class DualAREngine(FlowHandler):
    """
    DualAR semantic token generation + codec decoding.

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

    def execute(self) -> Dict[str, Any]:
        """Execute DualAR pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        if not stages:
            raise RuntimeError("ZERO FALLBACK: dual_ar flow requires at least one stage.")

        # ── Step 1: Tokenize text input ──
        from .audio_utils import preprocess_text_input
        preprocess_text_input(self.ctx)

        # ── Step 2: DualAR generation (slow backbone + fast/depth transformer) ──
        # Per position: slow backbone -> semantic token + hidden; then the fast
        # transformer (separate `model.fast` component) generates the N acoustic
        # codebooks by re-forwarding the growing codebook context under a causal
        # mask (mirror of vendor decode_one_token_ar, no KV cache). The N codes
        # per position are dequantized (codec.quantizer.decode) into the features
        # the codec decoder consumes. Replaces the old text-embedding shortcut
        # that fed the codec garbage (the 689 Hz carrier-tone bug).
        import json as _json
        from pathlib import Path as _Path
        comp_name = stages[0]["component"]  # "model"
        device = self.ctx.primary_device
        # CLI sampling overrides (global.*) take precedence over the embedded
        # defaults, mirroring the autoregressive flow — --temperature 0 ⇒ greedy.
        _ov = self.ctx.variable_resolver.resolved
        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        max_tokens = decode_bound(_ov.get("global.max_tokens", defaults.get("max_tokens", 2048)))
        temperature = _ov.get("global.temperature", defaults.get("temperature", 0.7))
        top_p = _ov.get("global.top_p", defaults.get("top_p", 0.8))

        # Special token ids from the embedded tokenizer (no hardcode): the
        # semantic-codebook offset and the stop token.
        st_path = _Path(self.ctx.nbx_path_str) / "modules" / "tokenizer" / "special_tokens.json"
        special = _json.load(open(st_path)) if st_path.exists() else {}
        semantic_begin_id = special.get("<|semantic:0|>")
        if semantic_begin_id is None:
            raise RuntimeError("ZERO FALLBACK: '<|semantic:0|>' missing from tokenizer special_tokens.")
        im_end_id = special.get("<|im_end|>", defaults.get("eos_token_id"))

        self._ensure_weights_loaded(comp_name)
        slow_exec = self.ctx.executors[comp_name]
        fast_exec = self.ctx.executors.get("model.fast")
        if fast_exec is None:
            raise RuntimeError("ZERO FALLBACK: DualAR requires the 'model.fast' component "
                               "(re-trace the model with the fast-transformer split).")
        self._ensure_weights_loaded("model.fast")

        # Slow model input shape [B, N+1, T]
        dag = getattr(slow_exec, '_dag', None)
        n_rows, trace_seq_len = 11, 64
        if dag:
            for _tid, spec in dag.get("tensors", {}).items():
                if spec.get("input_name") == "inp":
                    shape = spec.get("shape", [])
                    if len(shape) >= 3:
                        n_rows = shape[1] if isinstance(shape[1], int) else 11
                        trace_seq_len = shape[2] if isinstance(shape[2], int) else 64
                    break
        num_codebooks = n_rows - 1  # acoustic codebooks (semantic + residual)
        acoustic_cb_size = 1024     # residual codebook size (codec quantizer)

        input_ids = self.ctx.variable_resolver.resolved.get("global.input_ids")
        if input_ids is None:
            raise RuntimeError("ZERO FALLBACK: DualAR requires tokenized input_ids.")
        prompt_ids = [int(x) for x in input_ids.squeeze(0).tolist()]
        # Vendor builds ContentSequence(modality="interleave").append([TextPart(text)],
        # speaker=0): the exact prompt is <|interleave|> + "<|speaker:0|>" (encoded
        # as TEXT subwords — it is NOT a special token) + the text tokens, with no
        # leading BOS and no trailing im_end (generation emits the audio then
        # im_end). Verified against fish_speech ContentSequence.encode_for_inference.
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

        from .audio_utils import sample_token
        print(f"   [{comp_name}] DualAR generation (max_tokens={max_tokens}, N={num_codebooks})...")
        start = time.perf_counter()

        # Grid columns: each is [semantic, cb_1..cb_N]. Prompt columns carry the
        # text token in row 0 and zero acoustic codes.
        grid_cols = [[pid] + [0] * num_codebooks for pid in prompt_ids]
        acoustic_codes = []  # [num_codebooks] per generated position
        # Repetition penalty on the slow (semantic) head — vendor uses 1.2 with a
        # sliding window. Without it the AR loop never emits im_end (infinite loop).
        rep_pen = _ov.get("global.repetition_penalty", defaults.get("repetition_penalty", 1.2))
        slow_history: List[int] = []
        rep_window = 32
        # Deterministic shared-seed sampler RNG (R27/R28) — same seed + same numpy
        # algorithm as the triton dual_ar path, so the 4 modes draw identical
        # randoms and (at fp16-close logits) sample identical tokens. Greedy
        # (--temperature 0) remains the deterministic floor.
        _seed = _ov.get("global.seed")
        sampler_rng = np.random.RandomState(int(_seed) if _seed is not None else _DUALAR_SEED)

        # Constrained decoding: the slow head may only emit semantic tokens or the
        # stop token (vendor masks non-semantic logits). 4096 semantic ids.
        sem_lo, sem_hi = semantic_begin_id, semantic_begin_id + 4096

        for _ in range(max_tokens):
            # Feed the FULL grid (variable length). The slow model's seq dim is
            # symbolic (audio_token_grid convention), so it accepts any length and
            # sees the complete prompt+generation context (no fixed 23-window) —
            # this is what lets it emit im_end at the right time. O(n) re-forward
            # per step (the accepted decode_step "re-run full context" tier).
            clen = len(grid_cols)
            inp_t = torch.zeros(1, n_rows, clen, dtype=torch.long, device=device)
            for ci, col in enumerate(grid_cols):
                for ri, val in enumerate(col):
                    inp_t[0, ri, ci] = val
            pos = clen - 1
            out = slow_exec.run({"inp": inp_t})
            logits = out.get("logits") if isinstance(out, dict) else out
            hidden = out.get("hidden_states") if isinstance(out, dict) else None
            if logits is None or hidden is None:
                raise RuntimeError("ZERO FALLBACK: slow model must output logits + hidden_states.")

            slow_logits = logits[:, pos, :].clone()
            import os as _osdsl
            _dsl = _osdsl.environ.get("NBX_DUALAR_DUMP_SLOGITS", "")
            if _dsl and len(slow_history) == 0:
                import numpy as _npdsl
                _npdsl.save(_dsl, slow_logits.detach().float().cpu().numpy())
            # Mask to semantic ids + stop token.
            mask = torch.full_like(slow_logits, float("-inf"))
            mask[:, sem_lo:sem_hi] = slow_logits[:, sem_lo:sem_hi]
            if im_end_id is not None:
                mask[:, im_end_id] = slow_logits[:, im_end_id]
            slow_token = _sample_token_np_core(
                mask[0].detach().float().cpu().numpy(), temperature, top_p=top_p,
                repetition_penalty=rep_pen, generated_ids=slow_history[-rep_window:],
                rng=sampler_rng)
            import os as _os_dbg
            if _os_dbg.environ.get("NBX_DEBUG_DECODE") == "1" and len(slow_history) < 16:
                print(f"  [DBG-DUALAR] step={len(slow_history)} sem_token={int(slow_token)}",
                      flush=True)
            if im_end_id is not None and slow_token == im_end_id:
                break
            slow_history.append(slow_token)

            hidden_pos = hidden[:, pos, :]  # [1, dim]
            # codebook 0 (semantic VQ code) is the slow token mapped into [0, 4096);
            # the fast/depth transformer then generates the N-1 residual codebooks
            # autoregressively, conditioned on the growing codebook context.
            ctx = [max(0, slow_token - semantic_begin_id)]
            for _ in range(1, num_codebooks):
                ctx_t = torch.zeros(1, num_codebooks - 1, dtype=torch.long, device=device)
                for j, v in enumerate(ctx):
                    ctx_t[0, j] = v
                fout = fast_exec.run({"hidden": hidden_pos, "codebook_context": ctx_t})
                flogits = fout.get("output") if isinstance(fout, dict) else fout
                read_pos = len(ctx)  # depth position predicting the next codebook
                cb_logits = flogits[:, read_pos, :acoustic_cb_size]
                sampled = _sample_token_np_core(
                    cb_logits[0].detach().float().cpu().numpy(), temperature,
                    top_p=top_p, rng=sampler_rng)
                ctx.append(sampled)

            acoustic_codes.append(ctx)
            grid_cols.append([slow_token] + ctx)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Generated {len(acoustic_codes)} positions in {elapsed:.0f}ms")
        if not acoustic_codes:
            raise RuntimeError("ZERO FALLBACK: DualAR produced no audio frames.")

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            self._unload_component_weights("model.fast")
            gc.collect()
            device_empty_cache(device)

        # ── Step 3: codes -> codec.quantizer.decode (RVQ dequantize) -> features ──
        T = len(acoustic_codes)
        codes = torch.zeros(1, num_codebooks, T, dtype=torch.long, device=device)
        for t, ac in enumerate(acoustic_codes):
            for r, v in enumerate(ac):
                codes[0, r, t] = v

        # ── Diagnostic harness (default-off, §8 retained-diag class) — fixed-codes
        # injection for op-level cross-engine (compiled vs sequential) codec diffs.
        # The backbone emits different codes per engine even at fixed seed, so a
        # plain fingerprint diff of the codec is confounded; dump once, then replay
        # IDENTICAL codes into both engines to isolate the codec compute divergence.
        import os as _os_diag
        _dump_codes = _os_diag.environ.get("NBX_DUALAR_DUMP_CODES", "")
        _fixed_codes = _os_diag.environ.get("NBX_DUALAR_FIXED_CODES", "")
        if _dump_codes:
            import numpy as _np_d
            _np_d.save(_dump_codes, codes.detach().cpu().numpy())
            print(f"   [diag] dumped codes {list(codes.shape)} -> {_dump_codes}")
        if _fixed_codes:
            import numpy as _np_f
            _arr = _np_f.load(_fixed_codes)
            codes = torch.as_tensor(_arr, dtype=torch.long, device=device)
            T = codes.shape[2]
            acoustic_codes = [[int(codes[0, r, t]) for r in range(codes.shape[1])]
                              for t in range(T)]
            print(f"   [diag] FIXED codes injected {list(codes.shape)} <- {_fixed_codes}")

        self._ensure_weights_loaded("codec.quantizer")
        q_exec = self.ctx.executors["codec.quantizer"]
        # codec.quantizer.decode is fully symbolic in the .nbx (born-at-source seq:
        # the WindowLimitedTransformer mask/positions and the RVQ codebook gather are
        # symbol-tracked, the SDPA mask alignment slice recovers the seq symbol). The
        # variable-length code sequence runs in a single pass — no fixed q_seq window
        # chunking, which would have broken the window_size=128 cross-frame attention.
        z = q_exec.run({"indices": codes})
        z = z.get("output") if isinstance(z, dict) else z
        print(f"   [codec.quantizer] decode codes {list(codes.shape)} -> features {list(z.shape)}")
        _dump_z = _os_diag.environ.get("NBX_DUALAR_DUMP_Z", "")
        if _dump_z:
            import numpy as _np_z
            _zc = z.detach().float().cpu().numpy()
            _np_z.save(_dump_z, _zc)
            print(f"   [diag] z stats mean={_zc.mean():.5f} std={_zc.std():.5f} "
                  f"absmax={abs(_zc).max():.5f} -> {_dump_z}")
        # Bind features as the decoder input. The topology reconcile wires the
        # decoder input via the connection `model.output_0 -> codec.decoder.x`
        # (the quantizer is inserted by this handler, so the decoder's InputResolver
        # follows that connection back to `model.output_0`); bind there too so a
        # single full-sequence `_execute_component` forward resolves its input
        # (the old chunked path injected `executor.run({x: chunk})` directly and
        # bypassed this resolution).
        for key in ("global.x", "x", "codec.decoder.x", "model.output_0"):
            self.ctx.variable_resolver.resolved[key] = z
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = [c[0] for c in acoustic_codes]
        if not self.ctx.persistent_mode:
            self._unload_component_weights("codec.quantizer")
            gc.collect()
            device_empty_cache(device)

        # ── Step 4: codec.decoder (features -> waveform), single symbolic-seq pass ──
        # The DAC decoder is fully symbolic (unpad1d negative-end slice keeps the
        # upsampled length symbolic), so it runs the whole feature sequence at once —
        # no trace-seq chunking, no inter-block seams.
        for stage in stages[1:]:
            codec_name = stage["component"]
            if codec_name not in self.ctx.executors:
                continue
            self._ensure_weights_loaded(codec_name)
            print(f"   [{codec_name}] Running forward pass...")
            self._execute_component(codec_name, "forward", None)
            if not self.ctx.persistent_mode:
                self._unload_component_weights(codec_name)
                gc.collect()
                device_empty_cache(device)

        # ── Step 5: Output waveform ──
        from .audio_utils import postprocess_audio_output
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
            if isinstance(val, torch.Tensor) and val.dim() == 3:
                actual_input = val
                break

        if actual_input is None:
            connections = self.ctx.pkg.topology.get("connections", [])
            for conn in connections:
                if conn.get("to", "") == f"{comp_name}.{graph_input_name}":
                    val = resolved.get(conn.get("from", ""))
                    if isinstance(val, torch.Tensor) and val.dim() == 3:
                        actual_input = val
                        break

        if actual_input is None:
            return False

        actual_seq = actual_input.shape[2]
        if actual_seq <= graph_seq_len:
            return False

        print(f"   [{comp_name}] Chunked: {actual_seq} frames → {graph_seq_len}-frame blocks"
              f" (actual_input shape={list(actual_input.shape)})")
        waveform_chunks = []

        for chunk_start in range(0, actual_seq, graph_seq_len):
            chunk_end = min(chunk_start + graph_seq_len, actual_seq)
            chunk = actual_input[:, :, chunk_start:chunk_end]

            if chunk.shape[2] < graph_seq_len:
                pad_size = graph_seq_len - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            print(f"     chunk [{chunk_start}:{chunk_end}] shape={list(chunk.shape)} dtype={chunk.dtype}")
            output = executor.run({graph_input_name: chunk})

            if isinstance(output, dict):
                out_tensor = next(iter(output.values()))
            elif isinstance(output, torch.Tensor):
                out_tensor = output
            else:
                out_tensor = None

            if out_tensor is not None:
                if chunk_end - chunk_start < graph_seq_len and out_tensor.dim() >= 2:
                    ratio = (chunk_end - chunk_start) / graph_seq_len
                    trim_len = int(out_tensor.shape[-1] * ratio)
                    out_tensor = out_tensor[..., :trim_len]
                waveform_chunks.append(out_tensor)

        if waveform_chunks:
            full_output = torch.cat(waveform_chunks, dim=-1)
            self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = full_output
            self.ctx.variable_resolver.resolved["global.output_audio"] = full_output
            print(f"   [{comp_name}] Waveform: {list(full_output.shape)}")

        return True
