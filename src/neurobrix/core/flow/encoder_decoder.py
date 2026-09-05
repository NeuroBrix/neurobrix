"""
EncoderDecoderEngine — Encoder-Decoder with Cross-Attention Flow

Handles models like Whisper: encoder processes input features,
decoder generates tokens autoregressively with cross-attention
from encoder output.

ZERO SEMANTIC: No knowledge of "Whisper" or "speech".
ZERO HARDCODE: All parameters from NBX container.
"""

import time
import torch
from neurobrix.core.device_utils import device_multinomial
from neurobrix.core.memory.manager import release_flow_memory
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("encoder_decoder")
class EncoderDecoderEngine(FlowHandler):
    """
    Encoder-decoder cross-attention flow.

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
        """Execute encoder-decoder pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        # ── Step 1: Preprocess audio input ──
        from .audio_utils import preprocess_audio_input, postprocess_text_output
        preprocess_audio_input(self.ctx, audio_config, stages)

        # ── Step 2: Forward encoder ──
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

        # Long-form (D-STT-LONGFORM-CHUNKING): audio longer than one
        # window runs the vendor's timestamp-seek algorithm — decode a
        # window with timestamps, seek to the end of its last complete
        # segment, decode again from there (rules in
        # core/module/audio/stt_longform.py, shared with the triton
        # mirror). One window == the exact pre-change path. Weights stay
        # loaded across windows and unload once after the loop.
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
            print(f"   [Audio] Long-form: timestamp seek over "
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
            encoder_output = self._get_component_output(enc_name)
            if encoder_output is not None:
                self.ctx.variable_resolver.resolved[f"{enc_name}.output_0"] = encoder_output
            _prog0 = __import__("os").environ.get("NBX_DECODE_PROGRESS")
            if _prog0 and encoder_output is not None:  # gated diagnostic — compiled encoder ref
                try:
                    import numpy as _np
                    _eo = encoder_output.detach().float().cpu().numpy()
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
            from .audio_utils import postprocess_text_output
            postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    def _decoder_kv_wrapper(self, dec_name: str, max_tokens: int):
        """Build and register the decoder's KV cache for one window, or None
        (recompute oracle requested, or a graph without self-attention)."""
        import os as _os
        if _os.environ.get("NBX_KV_RECOMPUTE") == "1":
            return None
        executor = self.ctx.executors.get(dec_name)
        dag = getattr(executor, "_dag", None) if executor is not None else None
        if not dag:
            return None
        from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
        plan = decoder_self_attention_plan(dag)
        if plan is None:
            return None
        from neurobrix.core.runtime.graph.kv_cache_wrapper import (
            KVCacheAttentionWrapper, KVCacheConfig)
        wrapper = getattr(executor, "_decoder_kv_wrapper", None)
        if wrapper is None:
            dtype = getattr(executor, "dtype", None) or "float16"
            config = KVCacheConfig(
                num_layers=plan["num_layers"], num_kv_heads=plan["num_heads"],
                k_head_dim=plan["head_dim"], v_head_dim=plan["head_dim"],
                max_cache_len=int(max_tokens), dtype=str(dtype))
            wrapper = KVCacheAttentionWrapper(config, num_heads=plan["num_heads"])
            by_type = wrapper.get_interceptors()
            per_uid = {}
            for uid in plan["self_attn_uids"]:
                fn = by_type.get(dag["ops"][uid]["op_type"])
                if fn is None:
                    raise RuntimeError(f"ZERO FALLBACK: no KV interceptor for {dag['ops'][uid]['op_type']}")
                per_uid[uid] = fn
            for uid in plan["arange_uids"]:
                per_uid[uid] = wrapper.intercept_arange
            executor.register_op_uid_interceptors(per_uid)
            executor._decoder_kv_wrapper = wrapper
            print(f"   [{dec_name}] KV cache: {plan['num_layers']} self-attention layers cached, "
                  f"{len(plan['cross_attn_uids'])} cross-attentions native, "
                  f"heads={plan['num_heads']} d={plan['head_dim']} max={max_tokens}")
        wrapper.reset_for_new_sequence()
        return wrapper

    def _decode_one_window(self, decoder_stage, dec_name, defaults,
                           _all_ids, _all_texts, seek_ctx=None, ts_ids=None,
                           mel_frames=None, encoder_frames=None):
        """Decode one window; in long-form returns the seek advance in
        samples (None = consume the whole window)."""
        # ── Step 3: Autoregressive decode with cross-attention ──
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
            "encoder_decoder (compiled)", ("temperature", "repetition_penalty"),
            _samp_cfg, explicit=_samp_explicit)

        # Forced decoder positions (language/task tokens) and the prompt
        # length — in long-form the vendor drops a forced
        # <|notimestamps|> (stt_longform.whisper_forced_map).
        from neurobrix.core.module.audio.stt_longform import (
            whisper_begin_index, whisper_forced_map)
        forced_map = whisper_forced_map(defaults, ts_ids, timestamps=seek_ctx is not None)
        begin = whisper_begin_index(forced_map)

        print(f"   [{dec_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        self._ensure_weights_loaded(dec_name)

        # Get embed weight for weight-tied logits
        embed_weight = self._get_embed_weight(dec_name)

        # Inject embed weight for weight-tied models (LoRA breaks data_ptr tying)
        if embed_weight is not None:
            executor = self.ctx.executors.get(dec_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

        device = self.ctx.primary_device
        generated_ids = [decoder_start_token_id]

        # Decoder KV cache (the text flow's brick, adopted here): the
        # self-attentions cache their keys/values, the cross-attentions keep
        # the native path (their K/V are the encoder's, constant per window),
        # the positional arange is offset by the cache length, and every step
        # feeds ONE token — the prompt is forced token by token, so the first
        # step is a one-token prefill. Without it the decoder re-ran the whole
        # growing sequence every token (31 passes for 31 tokens, 2026-09-05).
        # NBX_KV_RECOMPUTE=1 keeps the recompute path as the oracle.
        kv = self._decoder_kv_wrapper(dec_name, max_tokens)
        for step in range(1, max_tokens):
            if kv is not None:
                if step > 1:
                    kv.update_position_offset()
                token_ids = [generated_ids[-1:]]
            else:
                token_ids = [generated_ids]
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            self._execute_component(dec_name, "forward", None)
            if kv is not None and step == 1:
                kv.set_decode_mode(actual_seq_len=1)

            decoder_output = self._get_component_output(dec_name)
            if decoder_output is None:
                break

            logits = self._compute_logits(decoder_output, embed_weight, logits_source)

            current_pos = len(generated_ids)
            if current_pos in forced_map and forced_map[current_pos] is not None:
                next_token = forced_map[current_pos]
            else:
                next_token = self._sample_token(
                    logits, temperature,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                    defaults=defaults, ts_ids=ts_ids, begin=begin,
                    timestamps=seek_ctx is not None,
                )

            generated_ids.append(next_token)
            _prog = __import__("os").environ.get("NBX_DECODE_PROGRESS")
            if _prog:  # gated diagnostic — compiled decoder ref trajectory
                _stat = ""
                try:
                    import numpy as _np
                    _do = decoder_output.detach().float().cpu().numpy()
                    _lt = _do.reshape(-1, _do.shape[-1])[-1]
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

        # Timestamp seek — the vendor's rule (stt_longform.whisper_seek):
        # an unfinished trailing segment is dropped and decoded again
        # from the last complete segment's end in the next window; a
        # single stamp at the very end, or no stamp pairs, consume the
        # whole window. The advance is measured in samples from the
        # run's own mel/encoder frame ratio, never assumed.
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
        # Per-window text (timestamp tokens stripped by form); joined by
        # the caller. Single-window callers still run the classic
        # postprocess for byte-identical output.
        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            _all_texts.append(
                AudioOutputProcessor.decode_tokens(generated_ids, tokenizer))
        else:
            _all_texts.append(str(generated_ids))
        return advance_samples

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get a component's primary output tensor."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get TOKEN embedding weight for weight-tied logits.

        NeuroTax standard: token_embed.weight (token embeddings).
        Must NOT match embed_positions.weight (positional embeddings)
        which has a different shape and purpose.
        """
        executor = self.ctx.executors.get(comp_name)
        if executor is not None:
            # Priority: exact token_embed match first
            for key in executor._weights:
                if "token_embed" in key:
                    return executor._weights[key]
            # Fallback: largest 2D embed weight (vocab_size > pos_table)
            best = None
            for key in executor._weights:
                if "embed" in key and executor._weights[key].ndim == 2:
                    w = executor._weights[key]
                    if best is None or w.shape[0] > best.shape[0]:
                        best = w
            return best
        return None

    def _compute_logits(
        self, hidden_states: torch.Tensor, embed_weight: Optional[torch.Tensor],
        logits_source: str,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
        last_hidden = hidden_states[:, -1:, :]

        if logits_source == "lm_head" and "lm_head" in self.ctx.executors:
            self._ensure_weights_loaded("lm_head")
            executor = self.ctx.executors["lm_head"]
            for key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    w = tensor.to(dtype=last_hidden.dtype)
                    return torch.matmul(last_hidden, w.T)
            return last_hidden

        if logits_source == "embed_weight_tied" and embed_weight is not None:
            w = embed_weight.to(dtype=last_hidden.dtype)
            return torch.matmul(last_hidden, w.T)

        return last_hidden

    def _sample_token(
        self, logits: torch.Tensor, temperature: float,
        generated_ids: Optional[List[int]] = None,
        repetition_penalty: float = 1.0,
        defaults: Optional[dict] = None, ts_ids=None, begin: int = 1,
        timestamps: bool = False,
    ) -> int:
        """Sample next token from logits."""
        last_logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                if last_logits[0, tid] > 0:
                    last_logits[0, tid] /= repetition_penalty
                else:
                    last_logits[0, tid] *= repetition_penalty

        # Vendor logit rules (suppression lists; the timestamp grammar
        # in long-form) — data the build carries; a container without
        # them keeps the exact device argmax path.
        if defaults is not None and (timestamps or defaults.get("suppress_tokens")
                                     or defaults.get("begin_suppress_tokens")):
            from neurobrix.core.module.audio.stt_longform import apply_whisper_logit_rules
            import numpy as _np
            _row = last_logits[0].detach().float().cpu().numpy()
            _maybe_log_topk(_row, len(generated_ids or []), "torch", "raw")
            apply_whisper_logit_rules(_row, generated_ids or [], defaults, ts_ids,
                                      begin, timestamps=timestamps)
            _maybe_log_topk(_row, len(generated_ids or []), "torch", "rules")
            if temperature == 0.0:
                return int(_np.argmax(_row))
            last_logits = torch.from_numpy(_row).to(
                device=last_logits.device, dtype=last_logits.dtype)[None]

        if temperature == 0.0:
            if __import__("os").environ.get("NBX_DECODE_TOPK"):
                _maybe_log_topk(last_logits[0].detach().float().cpu().numpy(),
                                len(generated_ids or []), "torch", "raw")
            return last_logits.argmax(dim=-1).item()
        probs = torch.softmax(last_logits / temperature, dim=-1)
        return device_multinomial(probs, 1).item()



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
