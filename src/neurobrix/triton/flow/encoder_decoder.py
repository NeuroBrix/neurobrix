"""Triton EncoderDecoderEngine — zero torch encoder-decoder flow.

Ported from core/flow/encoder_decoder.py. Handles models like Whisper:
encoder processes input features, decoder generates tokens autoregressively
with cross-attention from encoder output.

No torch imports in this file (except at audio preprocessing boundary).
"""

import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional

import os
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype
from neurobrix.triton.memory_pool import release_flow_memory
from neurobrix.triton.device_transfer import parse_device_idx


class TritonEncoderDecoderEngine:
    """
    Triton-mode encoder-decoder cross-attention flow.

    topology.flow.audio:
        direction: stt
        stages:
          - component: model.encoder
            execution: forward
          - component: model.decoder
            execution: autoregressive
            cross_attention_from: model.encoder
            logits_source: embed_weight_tied | self | lm_head
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
        """Execute encoder-decoder pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        # -- Step 1: Preprocess audio input (zero-torch numpy front-end) --
        from neurobrix.triton.audio_frontend import preprocess_audio_input_np
        preprocess_audio_input_np(self.ctx, audio_config, stages)

        # -- Step 2: Forward encoder --
        encoder_stage = None
        decoder_stage = None
        for s in stages:
            exec_type = s.get("execution", "forward")
            if exec_type == "forward":
                encoder_stage = s
            elif exec_type == "autoregressive":
                decoder_stage = s

        if encoder_stage is None or decoder_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: encoder_decoder flow requires one 'forward' stage "
                "(encoder) and one 'autoregressive' stage (decoder)."
            )

        enc_name = encoder_stage["component"]
        dec_name = decoder_stage["component"]

        # Long-form (D-STT-LONGFORM-CHUNKING, R30 mirror of the compiled
        # flow): audio longer than one window runs the vendor's
        # timestamp-seek algorithm (rules in
        # core/module/audio/stt_longform.py, numpy, shared). One window
        # == the exact pre-change path.
        _seek_ctx = getattr(self.ctx, "_stt_seek", None)
        _all_ids: list = []
        _all_texts: list = []
        _feat_var = audio_config.get("input", {}).get(
            "variable", "global.input_features")
        _feat_short = _feat_var.split(".")[-1]
        _seek = 0
        _seek_total = len(_seek_ctx["audio"]) if _seek_ctx else 0
        _ts_ids = None
        if _seek_ctx:
            from neurobrix.core.module.audio.stt_longform import whisper_timestamp_ids
            _ts_ids = whisper_timestamp_ids(defaults)
            if _ts_ids is None:
                raise RuntimeError(
                    "Long-form audio needs the model's timestamp token data "
                    "(no_timestamps_token_id) in defaults.json; this container "
                    "was built before the toolchain copied it. Rebuild it with "
                    "the current toolchain, or pass audio no longer than one "
                    f"window ({_seek_ctx['chunk_s']:.0f} s).")
            print(f"   [Audio·np] Long-form: timestamp seek over "
                  f"{_seek_total / _seek_ctx['sr']:.1f} s")

        _n_win = 0
        while True:
            if _seek_ctx and _n_win > 0:
                from neurobrix.core.module.audio.mel_dsp import whisper_window_mel
                _wf = _seek_ctx["build"](whisper_window_mel(_seek_ctx, _seek))
                self.ctx.variable_resolver.resolved[_feat_var] = _wf
                self.ctx.variable_resolver.resolved[_feat_short] = _wf
            _n_win += 1
            _mel_frames = int(self.ctx.variable_resolver.resolved[_feat_var].shape[-1])

            print(f"   [{enc_name}] Running encoder..." +
                  (f" (window {_n_win} at {_seek / _seek_ctx['sr']:.2f} s)"
                   if _seek_ctx else ""))
            start = time.perf_counter()
            self._ensure_weights_loaded(enc_name)
            self._execute_component(enc_name, "forward", None)
            enc_elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{enc_name}] Done in {enc_elapsed:.0f}ms")

            # Store encoder output for cross-attention
            encoder_output = _get_component_output(self.ctx, enc_name)
            if encoder_output is not None:
                self.ctx.variable_resolver.resolved[f"{enc_name}.output_0"] = encoder_output
            _prog0 = __import__("os").environ.get("NBX_DECODE_PROGRESS")
            if _prog0 and encoder_output is not None:  # gated diagnostic — encoder sanity
                try:
                    import numpy as _np
                    _eo = encoder_output.numpy()
                    with open(_prog0, "w") as _pf:
                        _pf.write(f"ENCODER shape={_eo.shape} l2={float(_np.linalg.norm(_eo)):.3f} "
                                  f"mean={float(_eo.mean()):.5f} std={float(_eo.std()):.5f} "
                                  f"nan={bool(_np.isnan(_eo).any())} "
                                  f"head={_np.round(_eo.flatten()[:6],4).tolist()}\n")
                        _pf.flush()
                except Exception:
                    pass

            _adv = self._decode_one_window(
                decoder_stage, dec_name, defaults, _all_ids, _all_texts,
                seek_ctx=_seek_ctx, ts_ids=_ts_ids, mel_frames=_mel_frames,
                encoder_frames=(int(encoder_output.shape[1])
                                if encoder_output is not None else None))
            if not _seek_ctx:
                break
            _seek += _adv if _adv else _seek_ctx["nsamp"]
            if _seek >= _seek_total:
                break

        if not self.ctx.persistent_mode:
            self._unload_component_weights(enc_name)
            self._unload_component_weights(dec_name)
            release_flow_memory(self.ctx.primary_device)

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = _all_ids
        if _n_win > 1:
            text = " ".join(t.strip() for t in _all_texts if t and t.strip())
            self.ctx.variable_resolver.resolved["global.transcription"] = text
            print(f"   [Output] Transcription ({_n_win} windows): "
                  f"{text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            from neurobrix.triton.audio_frontend import postprocess_text_output_np
            postprocess_text_output_np(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    def _decoder_kv_interceptor(self, dec_name: str, max_tokens: int):
        """Build and register the decoder's NBXTensor KV cache for one window,
        or None (recompute oracle requested, or no self-attention)."""
        if os.environ.get("NBX_KV_RECOMPUTE") == "1":
            return None
        executor = self.ctx.executors.get(dec_name)
        dag = getattr(executor, "_dag", None) if executor is not None else None
        if not dag:
            return None
        from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
        plan = decoder_self_attention_plan(dag)
        if plan is None:
            return None
        if not plan["arange_uids"] and not plan.get("position_slice_uids"):
            # One token per step needs a positional mechanism the cache can
            # offset; a graph with neither would decode every token at
            # position 0. Loud, and the recompute path (correct) instead.
            import sys as _sys
            print(f"[{dec_name}] KV cache REFUSED: the decoder graph carries no positional "
                  f"arange and no positional-table slice the cache could offset — "
                  f"recompute path (D-STT-KV-WHISPER-LARGE)", file=_sys.stderr, flush=True)
            return None
        from neurobrix.triton.kv_cache import TritonKVCache, TritonAttentionInterceptor
        interceptor = getattr(executor, "_decoder_kv_interceptor", None)
        if interceptor is None:
            dtype = parse_dtype(str(getattr(executor, "dtype", None) or "float16"))
            cache = TritonKVCache(num_layers=plan["num_layers"], num_kv_heads=plan["num_heads"],
                                  k_head_dim=plan["head_dim"], v_head_dim=plan["head_dim"],
                                  max_cache_len=int(max_tokens), dtype=dtype)
            interceptor = TritonAttentionInterceptor(cache=cache, num_heads=plan["num_heads"])
            variant = {
                "aten::_scaled_dot_product_efficient_attention": interceptor.intercept_efficient,
                "aten::_scaled_dot_product_cudnn_attention": interceptor.intercept_efficient,
                "aten::_scaled_dot_product_flash_attention": interceptor.intercept_flash,
            }
            per_uid = {uid: variant.get(dag["ops"][uid]["op_type"], interceptor.intercept)
                       for uid in plan["self_attn_uids"]}
            for uid in plan["arange_uids"]:
                per_uid[uid] = interceptor.intercept_arange
            for uid in plan.get("position_slice_uids") or []:
                per_uid[uid] = interceptor.intercept_position_slice
            executor.register_op_uid_interceptors(per_uid)
            executor._decoder_kv_interceptor = interceptor
            print(f"   [{dec_name}] KV cache (triton): {plan['num_layers']} self-attention layers cached, "
                  f"{len(plan['cross_attn_uids'])} cross-attentions native")
        interceptor.reset()
        return interceptor

    def _decode_one_window(self, decoder_stage, dec_name, defaults,
                           _all_ids, _all_texts, seek_ctx=None, ts_ids=None,
                           mel_frames=None, encoder_frames=None):
        """Decode one window; in long-form returns the seek advance in
        samples (None = consume the whole window). Mirror of the
        compiled flow."""
        # -- Step 3: Autoregressive decode with cross-attention --
        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        max_tokens = decode_bound(defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        decoder_start_token_id = defaults.get("decoder_start_token_id", eos_token_id)
        logits_source = decoder_stage.get("logits_source", "embed_weight_tied")
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        # Sampling-parameter contract: this path implements temperature
        # and the repetition penalty only. Anything else configured here
        # would be silently dropped, so refuse instead. See
        # core/module/sampling_contract.py.
        from neurobrix.core.module.sampling_contract import (
            enforce_sampling_support)
        _samp_cfg, _samp_explicit = _effective_sampling(
            self.ctx, defaults)
        enforce_sampling_support(
            "encoder_decoder (triton)", ("temperature", "repetition_penalty"),
            _samp_cfg, explicit=_samp_explicit)

        # Forced decoder positions and the prompt length (mirror of the
        # compiled flow; long-form drops a forced <|notimestamps|>).
        from neurobrix.core.module.audio.stt_longform import (
            whisper_begin_index, whisper_forced_map)
        forced_map = whisper_forced_map(defaults, ts_ids, timestamps=seek_ctx is not None)
        begin = whisper_begin_index(forced_map)

        print(f"   [{dec_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        self._ensure_weights_loaded(dec_name)

        # Get embed weight for weight-tied logits
        embed_weight = _get_embed_weight(self.ctx, dec_name)

        # Inject embed weight for weight-tied models
        if embed_weight is not None:
            executor = self.ctx.executors.get(dec_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

        device_idx = parse_device_idx(self.ctx.primary_device)
        DeviceAllocator.set_device(device_idx)
        generated_ids = [decoder_start_token_id]

        # R30 mirror of the compiled flow's decoder KV cache (NBXTensor brick):
        # self-attentions cached, cross-attentions native, positional arange
        # offset, one token per step. NBX_KV_RECOMPUTE=1 = the recompute oracle.
        kv = self._decoder_kv_interceptor(dec_name, max_tokens)
        for step in range(1, max_tokens):
            if kv is not None:
                if step > 1:
                    kv.update_position_offset()
                ids_np = np.array([generated_ids[-1:]], dtype=np.int64)
            else:
                ids_np = np.array([generated_ids], dtype=np.int64)
            input_ids = NBXTensor.from_numpy(ids_np)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            self._execute_component(dec_name, "forward", None)
            if kv is not None and step == 1:
                kv.set_decode_mode()

            decoder_output = _get_component_output(self.ctx, dec_name)
            if decoder_output is None:
                break

            logits = _compute_logits(
                self.ctx, decoder_output, embed_weight, logits_source
            )

            current_pos = len(generated_ids)
            if current_pos in forced_map and forced_map[current_pos] is not None:
                next_token = forced_map[current_pos]
            else:
                next_token = _sample_token_nbx(
                    logits, temperature,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                    defaults=defaults, ts_ids=ts_ids, begin=begin,
                    timestamps=seek_ctx is not None,
                )

            generated_ids.append(next_token)
            _prog = __import__("os").environ.get("NBX_DECODE_PROGRESS")
            if _prog:  # gated, off by default — file write is immune to stdout buffering
                _stat = ""
                try:
                    import numpy as _np
                    _do = decoder_output.numpy()
                    _lt = _do.reshape(-1, _do.shape[-1])[-1]  # last-token hidden (drives logits)
                    _stat = (f" dec_l2={float(_np.linalg.norm(_lt)):.4f}"
                             f" dec_mean={float(_lt.mean()):.5f}"
                             f" dec_head={_np.round(_lt[:4],4).tolist()}")
                except Exception as _e:
                    _stat = f" (dec-stat err: {_e})"
                with open(_prog, "a") as _pf:
                    _pf.write(f"step={step} last={next_token}{_stat}\n")
                    _pf.flush()
            if next_token == eos_token_id:
                break

        dec_elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{dec_name}] Generated {len(generated_ids)} tokens in {dec_elapsed:.0f}ms")

        # Timestamp seek — the vendor's rule (stt_longform.whisper_seek),
        # mirror of the compiled flow: drop the unfinished trailing
        # segment, advance to the last complete segment's end; the
        # advance is measured from the run's own mel/encoder frame ratio.
        advance_samples = None
        if seek_ctx is not None:
            from neurobrix.core.module.audio.stt_longform import (
                whisper_advance_samples, whisper_seek)
            _adv_idx, _keep = whisper_seek(generated_ids, begin, ts_ids[1], eos_token_id)
            if _adv_idx is not None and not encoder_frames:
                raise RuntimeError("ZERO FALLBACK: long-form seek needs the encoder output frame count.")
            if _adv_idx is not None and _adv_idx > 0:
                generated_ids = generated_ids[:_keep]
                advance_samples = whisper_advance_samples(
                    _adv_idx, mel_frames, encoder_frames, seek_ctx["hop"])
                print(f"   [{dec_name}] seek +{advance_samples / seek_ctx['sr']:.2f} s "
                      f"(end of the last complete segment)")
            else:
                print(f"   [{dec_name}] seek: whole window")
        _all_ids.extend(generated_ids)
        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            _all_texts.append(
                AudioOutputProcessor.decode_tokens(generated_ids, tokenizer))
        else:
            _all_texts.append(str(generated_ids))
        return advance_samples


# -----------------------------------------------------------------
# Module-level helpers (zero torch)
# -----------------------------------------------------------------

def _get_component_output(ctx, comp_name) -> Optional[NBXTensor]:
    """Get a component's primary output tensor."""
    resolved = ctx.variable_resolver.resolved
    for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
        if key in resolved:
            val = resolved[key]
            if isinstance(val, NBXTensor) or _is_tensor(val):
                return val
    return None


def _get_embed_weight(ctx, comp_name):
    """Get TOKEN embedding weight for weight-tied logits.

    NeuroTax standard: token_embed.weight (token embeddings).
    """
    executor = ctx.executors.get(comp_name)
    if executor is not None and hasattr(executor, '_weights'):
        for key in executor._weights:
            if "token_embed" in key:
                return executor._weights[key]
        best = None
        for key in executor._weights:
            if "embed" in key and executor._weights[key].ndim == 2:
                w = executor._weights[key]
                if best is None or w.shape[0] > best.shape[0]:
                    best = w
        return best
    return None


def _compute_logits(ctx, hidden_states, embed_weight, logits_source):
    """Compute logits from hidden states.

    For triton mode, we use the executor.run() path where possible.
    For weight-tied logits, we do a manual matmul via the kernel dispatcher.
    """
    # Extract last hidden state
    if hasattr(hidden_states, 'shape') and len(hidden_states.shape) >= 3:
        # hidden_states[:, -1:, :]
        last_dim = hidden_states.shape[1]
        # Use select to get last position
        last_hidden = hidden_states.select(1, last_dim - 1).unsqueeze(1)
    else:
        last_hidden = hidden_states

    if logits_source == "lm_head" and "lm_head" in ctx.executors:
        executor = ctx.executors["lm_head"]
        output = executor.run({"input": last_hidden})
        if isinstance(output, dict):
            return next(iter(output.values()))
        return output

    if logits_source == "embed_weight_tied" and embed_weight is not None:
        # matmul: last_hidden @ embed_weight.T
        # Use the graph executor's run for the matmul if available,
        # otherwise fall back to kernel dispatch
        from neurobrix.kernels.dispatch import dispatch
        w_t = embed_weight.transpose(0, 1) if embed_weight.ndim == 2 else embed_weight
        mm = dispatch("mm")
        if last_hidden.ndim > 2:
            # lm_head over [..., H]: the 2-D `mm` kernel needs flat [M, H].
            lead, hdim = last_hidden.shape[:-1], last_hidden.shape[-1]
            m = 1
            for d in lead:
                m *= d
            out = mm(last_hidden.reshape(m, hdim), w_t)
            return out.reshape(*lead, out.shape[-1])
        return mm(last_hidden, w_t)

    return last_hidden


def _sample_token_nbx(logits, temperature, generated_ids=None, repetition_penalty=1.0,
                      defaults=None, ts_ids=None, begin=1, timestamps=False) -> int:
    """Sample next token from logits.

    For sampling we need argmax or multinomial. In triton mode,
    we read logits to CPU via numpy for sampling (small tensor).
    `defaults`/`ts_ids`/`timestamps`: the vendor logit rules (suppression
    lists; the timestamp grammar in long-form) applied on the numpy row —
    shared with the compiled flow through stt_longform.
    """
    # Read last position logits to CPU
    if hasattr(logits, 'shape') and len(logits.shape) >= 3:
        last_logits_tensor = logits.select(1, logits.shape[1] - 1)
    else:
        last_logits_tensor = logits

    # Transfer to CPU via numpy for sampling
    last_logits_np = _to_numpy(last_logits_tensor)

    if last_logits_np.ndim > 1:
        last_logits_np = last_logits_np[0]  # First batch

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        for tid in set(generated_ids):
            if 0 <= tid < len(last_logits_np):
                if last_logits_np[tid] > 0:
                    last_logits_np[tid] /= repetition_penalty
                else:
                    last_logits_np[tid] *= repetition_penalty

    _maybe_log_topk(last_logits_np, len(generated_ids or []), "triton", "raw")
    if defaults is not None and (timestamps or defaults.get("suppress_tokens")
                                 or defaults.get("begin_suppress_tokens")):
        from neurobrix.core.module.audio.stt_longform import apply_whisper_logit_rules
        last_logits_np = np.array(last_logits_np, dtype=np.float32, copy=True)
        apply_whisper_logit_rules(last_logits_np, generated_ids or [], defaults, ts_ids,
                                  begin, timestamps=timestamps)
        _maybe_log_topk(last_logits_np, len(generated_ids or []), "triton", "rules")

    if temperature == 0.0:
        return int(np.argmax(last_logits_np))

    # Softmax in numpy
    logits_scaled = last_logits_np / temperature
    logits_scaled -= logits_scaled.max()  # numerical stability
    exp_logits = np.exp(logits_scaled)
    probs = exp_logits / exp_logits.sum()

    return int(np.random.choice(len(probs), p=probs))


def _maybe_log_topk(row, step: int, engine: str, stage: str) -> None:
    """NBX_DECODE_TOPK=<jsonl>: per-step top-4 ids/values + top-2 margin
    of one logits row (numpy), `stage` = "raw" (model logits) or
    "rules" (after the vendor logit rules). Same record shape as the
    vlm flows so engine records pair by (step, stage). Default-off."""
    import os as _os
    path = _os.environ.get("NBX_DECODE_TOPK")
    if not path:
        return
    import json as _json
    import numpy as _np
    r = _np.asarray(row, dtype=_np.float64).reshape(-1)
    top = _np.argsort(-r)[:4]
    vals = [float(r[i]) for i in top]
    rec = {"engine": engine, "stage": stage, "step": int(step),
           "ids": [int(i) for i in top], "vals": [round(v, 6) for v in vals],
           "margin12": round(vals[0] - vals[1], 6)}
    with open(path, "a") as f:
        _json.dump(rec, f)
        f.write("\n")


def _to_numpy(tensor) -> np.ndarray:
    """Convert any tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _is_tensor(val) -> bool:
    """Check if val is any tensor type."""
    return hasattr(val, 'shape') and hasattr(val, 'dtype')


_SAMPLING_PARAMS = ("temperature", "top_k", "top_p",
                    "repetition_penalty", "min_p")


def _effective_sampling(ctx, defaults):
    """Effective sampling config, and which parts the USER asked for.

    The registry's `defaults` is only half the story: a CLI flag or a
    serving request lands as a `global.*` variable in the resolver, and
    the autoregressive flow's own override table reads it from there.
    A guard that inspected `defaults` alone would miss exactly the case
    it exists to catch — whisper carries no `top_k` at all, so
    `--top-k 20` is invisible in `defaults` and visible only here.

    Returns (config, explicit) where config is defaults overlaid with
    the overrides, and explicit names the overridden parameters.
    """
    config = {k: defaults.get(k) for k in _SAMPLING_PARAMS}
    explicit = set()
    resolver = getattr(ctx, "variable_resolver", None)
    if resolver is None:
        return config, explicit
    for name in _SAMPLING_PARAMS:
        try:
            val = resolver.get(f"global.{name}", None)
        except Exception:
            continue
        if val is None:
            continue
        config[name] = val
        # EXPLICIT means the value DIFFERS from the registry default,
        # not merely that the resolver holds one. The configuration
        # pipeline populates `global.*` from defaults too, so presence
        # alone classifies every inherited field as a user request — the
        # first version did exactly that and refused MiniCPM-o for a
        # top_k=20 nobody had asked for. Re-requesting the default value
        # lands in the inherited branch, which is harmless: it is the
        # same value that was going to be ignored either way.
        try:
            same = float(val) == float(defaults.get(name))
        except (TypeError, ValueError):
            same = val == defaults.get(name)
        if not same:
            explicit.add(name)
    return config, explicit
