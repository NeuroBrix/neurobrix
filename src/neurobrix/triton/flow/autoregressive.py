"""Triton Flow Handler — zero torch autoregressive generation.

Ported from core/flow/autoregressive.py. Complete separation:
native mode uses torch tensors, triton mode uses NBXTensor throughout.

No torch imports in this file.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype

from neurobrix.triton.generator import TritonGenerator
from neurobrix.triton.session import TritonLMSession


def _flatten_tokenizer_output(token_ids: Any) -> List[int]:
    """Normalize tokenizer return into a flat List[int]."""
    if hasattr(token_ids, 'input_ids'):
        token_ids = token_ids['input_ids']
    elif (isinstance(token_ids, dict) or
          (hasattr(token_ids, '__getitem__') and not isinstance(token_ids, list)
           and 'input_ids' in token_ids)):
        token_ids = token_ids['input_ids']
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    return token_ids


def _build_generator_config(defaults: Dict, resolver: Any) -> Dict[str, Any]:
    """Build generator config from defaults.json — pure Python."""
    config = {
        "max_tokens": defaults.get("max_tokens", 512),
        "temperature": defaults.get("temperature", 1.0),
        "top_p": defaults.get("top_p", 1.0),
        "top_k": defaults.get("top_k", 0),
        "repetition_penalty": defaults.get("repetition_penalty", 1.0),
        "vocab_size": defaults.get("vocab_size", 32000),
        "eos_token_id": defaults.get("eos_token_id"),
        "pad_token_id": defaults.get("pad_token_id"),
        "_class_name": "TritonGenerator",
    }
    # Override from variable resolver if present.
    # Use explicit `is not None` — bare `or` treats valid falsy values
    # (temperature=0 for greedy, top_k=0 to disable) as missing.
    if resolver is not None:
        resolved = getattr(resolver, 'resolved', {})
        for key in ("max_tokens", "temperature", "top_p", "top_k"):
            gkey = f"global.{key}"
            if gkey in resolved and resolved[gkey] is not None:
                config[key] = resolved[gkey]
            elif key in resolved and resolved[key] is not None:
                config[key] = resolved[key]
    return config


class TritonAutoregressiveHandler:
    """Zero-torch autoregressive generation handler.

    Orchestrates: tokenize → prefill → decode loop → output.
    All tensors are NBXTensor. Sampling uses Triton kernels.
    """

    def __init__(self, ctx, execute_component_fn: Callable,
                 ensure_weights_fn: Callable, unload_weights_fn: Callable,
                 input_resolver=None, input_synthesizer=None,
                 output_extractor=None):
        self.ctx = ctx
        self._execute_component = execute_component_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn
        self._input_resolver = input_resolver
        self._input_synthesizer = input_synthesizer
        self._output_extractor = output_extractor
        self._active_session = None

    def execute(self) -> Dict[str, Any]:
        """Execute autoregressive generation — zero torch hot path."""
        gen_info = self.ctx.pkg.topology.get("flow", {}).get("generation", {})
        if not gen_info:
            raise RuntimeError("autoregressive_generation requires flow.generation info.")

        session = self._create_session(gen_info)
        self._active_session = session
        strategy = self._create_strategy(gen_info, session)
        generator = strategy.create_generator(
            self.ctx.pkg.defaults, self.ctx.variable_resolver)

        device_idx = self._parse_device_idx()
        generator.set_generation_params(device_idx=device_idx)

        input_ids = self._tokenize(gen_info, device_idx)
        batch_size = input_ids.shape[0]

        generator.set_prompt_ids(self._read_ids_to_list(input_ids))

        # Prefill
        hidden = session.prefill(input_ids, batch_size)

        # Decode loop
        import os as _os_dl
        _dump_path = _os_dl.environ.get("NBX_DUMP_LOGITS")
        for step_idx in generator:
            logits = strategy.get_logits(hidden, step_idx)
            if _dump_path and step_idx == 0:
                import json as _json_dl
                # logits shape may be (B, V) or (B, S, V); flatten to (V,)
                arr = logits
                if hasattr(arr, '_shape'):
                    flat = arr
                    while flat.ndim > 1:
                        flat = flat[0]
                    # NBXTensor → numpy via to(fp32)
                    from neurobrix.kernels.nbx_tensor import NBXDtype as _NBX
                    flat = flat.to(_NBX.float32)
                    n = flat.numel()
                    import ctypes as _ct
                    buf = (_ct.c_float * n)()
                    from neurobrix.kernels.nbx_tensor import (
                        DeviceAllocator as _DA)
                    _DA.memcpy(_ct.addressof(buf), flat.data_ptr(),
                               n * 4, kind=2)
                    vals = list(buf)
                else:
                    import torch as _torch_dl
                    vals = arr.detach().float().reshape(-1).cpu().tolist()
                # top-10
                idx_sorted = sorted(range(len(vals)), key=lambda i: -vals[i])[:10]
                top10 = [(i, vals[i]) for i in idx_sorted]
                with open(_dump_path, 'w') as _f:
                    _json_dl.dump({"engine": "triton", "vocab_size": len(vals),
                                   "top10": top10,
                                   "argmax": idx_sorted[0]}, _f)
                print(f"[NBX_DUMP_LOGITS] triton dumped top10 to {_dump_path}",
                      flush=True)
            next_token, is_done = generator.step(logits, step_idx)

            # Per-step token diagnostic (mirrors the compiled flow's NBX_DEBUG_DECODE)
            # for triton-seq-vs-sequential decode parity. R33-pure scalar read
            # (ctypes + DeviceAllocator, no torch).
            if _os_dl.environ.get("NBX_DEBUG_DECODE") == "1" and step_idx < 15:
                from neurobrix.kernels.nbx_tensor import (
                    NBXDtype as _NBXdt, DeviceAllocator as _DAdt)
                import ctypes as _ctdt
                _tk = next_token
                while hasattr(_tk, "ndim") and _tk.ndim > 1:
                    _tk = _tk[0]
                _tkf = _tk.to(_NBXdt.float32)
                _b = (_ctdt.c_float * 1)()
                _DAdt.memcpy(_ctdt.addressof(_b), _tkf.data_ptr(), 4, kind=2)
                print(f"  [DBG-TRITON] step={step_idx} token={int(round(_b[0]))}",
                      flush=True)

            if is_done:
                break

            token_embed = strategy.embed_token(next_token, step_idx)
            decode_ids, decode_embeds = strategy.prepare_decode_input(
                next_token, token_embed, batch_size)
            hidden = session.decode_step(decode_ids,
                                         inputs_embeds=decode_embeds)

        # Free LM + KV cache VRAM before the VQ decoder runs.  Janus's
        # language_model alone is ~13 GB bf16; adding the 249-op conv
        # decoder on top OOMs a 32 GB V100 unless the LM's arena is
        # released first. Text generation keeps the LM loaded (there is
        # no post-loop component), so this cleanup only triggers when
        # the strategy actually owns one.
        is_image = isinstance(strategy, TritonImageStrategy)
        if is_image:
            gen_info = self.ctx.pkg.topology.get("flow", {}).get(
                "generation", {})
            lm_name = gen_info.get("lm_component", "language_model")
            session.cleanup()
            self._unload_component_weights(lm_name)

        # Output
        generated_tokens = generator.get_generated_tokens()
        strategy.process_output(generated_tokens, self.ctx)

        if not is_image:
            session.cleanup()
        return self.ctx.variable_resolver.resolve_all()

    def _graph_lm_prefill(self, input_ids: NBXTensor) -> NBXTensor:
        """Delegate to active session. Required by runtime-guard hook."""
        if self._active_session is None:
            raise RuntimeError("No active session for prefill.")
        return self._active_session.prefill(input_ids, input_ids.shape[0])

    def _graph_lm_decode_step(self, input_ids: NBXTensor,
                              inputs_embeds=None) -> NBXTensor:
        """Delegate to active session. Required by runtime-guard hook."""
        if self._active_session is None:
            raise RuntimeError("No active session for decode.")
        return self._active_session.decode_step(input_ids, inputs_embeds)

    # ─── SETUP ────────────────────────────────────────────────────────

    def _parse_device_idx(self) -> int:
        """Parse device index from primary_device string.

        Handles: "cuda:2", "fgp:cuda:0,cuda:1", "0", "cuda".
        """
        dev = self.ctx.primary_device
        # "cuda:2" → 2
        if dev.startswith("cuda:"):
            try:
                return int(dev.split(":")[-1].split(",")[0])
            except ValueError:
                return 0
        # Compound: "fgp:cuda:0,cuda:1" → parse first cuda device
        if "cuda:" in dev:
            idx = dev.index("cuda:")
            num_str = dev[idx + 5:].split(",")[0].split(":")[0]
            try:
                return int(num_str)
            except ValueError:
                return 0
        # Plain int
        try:
            return int(dev)
        except ValueError:
            return 0

    def _tokenize(self, gen_info: Dict, device_idx: int) -> NBXTensor:
        """Tokenize prompt → NBXTensor of token IDs.

        Priority cascade mirrors native TextProcessor.tokenize:
          1. SFT format (Janus-style image AR)   — defaults.sft_format set
          2. HuggingFace chat_template            — LLM chat path
          3. Basic encode                         — fallback

        Image-AR models (gen_type == "autoregressive_image") NEVER go
        through chat_template even when the tokenizer exposes
        apply_chat_template — Janus's underlying DeepSeek tokenizer has the
        method but no template configured. Detection is data-driven via
        topology gen_type, independent of which family the model is
        packaged under (legacy "image" or current "multimodal").
        """
        if "tokenizer" not in self.ctx.modules:
            raise RuntimeError("autoregressive_generation requires 'tokenizer' module.")
        tokenizer = self.ctx.modules["tokenizer"]
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt", "")
        defaults = self.ctx.pkg.defaults
        gen_type = gen_info.get("type")
        is_image_ar = (gen_type == "autoregressive_image")

        sft_format = defaults.get("sft_format")
        special_token_ids = defaults.get("special_token_ids")

        # Resolve chat_mode (CLI override > defaults > False) — mirrors the
        # compiled TextProcessor.tokenize. Models with chat_mode=False (e.g. the
        # orpheus / openaudio TTS LMs, whose prompt is the bare templated text)
        # MUST NOT go through apply_chat_template: that wraps the text in the
        # full HF chat template (system prompt + role markers), producing a
        # ~10x longer prompt (orpheus: 39 vs 4 tokens) and a completely
        # different prefill → garbage decode. Was the orpheus triton bug.
        _SENT = object()
        _cli_chat = self.ctx.variable_resolver.resolved.get("global.chat_mode", _SENT)
        if _cli_chat is not _SENT and _cli_chat is not None:
            chat_mode = bool(_cli_chat)
        else:
            chat_mode = bool(defaults.get("chat_mode", False))

        if (is_image_ar and sft_format and special_token_ids
                and hasattr(tokenizer, "format_generation_prompt")):
            # Priority 1: SFT format (Janus-style image AR). Only taken for
            # image-AR models to avoid disturbing the LLM path.
            token_ids = tokenizer.format_generation_prompt(
                prompt=prompt,
                sft_format=sft_format,
                special_token_ids=special_token_ids,
                is_unconditional=False,
            )
        elif (chat_mode and not is_image_ar
              and hasattr(tokenizer, "apply_chat_template")
              and (not hasattr(tokenizer, "has_chat_template")
                   or tokenizer.has_chat_template())):
            # Priority 2: HF chat_template — ONLY when chat_mode is enabled
            # (TextProcessor parity). Preserves the Triton chat-LLM path
            # (TinyLlama, Qwen3, DeepSeek-MoE all set chat_mode=True).
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        elif hasattr(tokenizer, "encode_with_mask"):
            # Priority 3a: basic tokenization WITH special tokens (TextProcessor
            # parity for chat_mode=False models — orpheus, openaudio).
            _r = tokenizer.encode_with_mask(
                prompt, padding=False, add_special_tokens=True)
            token_ids = _r["input_ids"] if isinstance(_r, dict) else _r
        elif hasattr(tokenizer, "encode"):
            # Priority 3b: basic encode (with special tokens when supported).
            try:
                token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            except TypeError:
                token_ids = tokenizer.encode(prompt)
        else:
            raise RuntimeError("Tokenizer has no encode method.")

        token_ids = _flatten_tokenizer_output(token_ids)
        ids_np_cond = np.array([token_ids], dtype=np.int64)

        # CFG branch (image-AR only): tokenize unconditional variant and
        # concat → batch=2. Mirrors core/flow/autoregressive.py _tokenize:
        # graphs for Janus's gen_head/gen_embed/gen_aligner were traced at
        # batch=2 and the LM's RoPE bmm was traced at batch=2 as well, so
        # prefill MUST run at batch=2 for shapes to bind correctly.
        use_cfg = False
        if is_image_ar:
            cli_cfg = self.ctx.variable_resolver.resolved.get(
                "global.guidance_scale")
            cfg_weight = (float(cli_cfg) if cli_cfg is not None
                          else defaults.get("guidance_scale"))
            if cfg_weight is None:
                raise RuntimeError(
                    "guidance_scale missing from defaults.json for "
                    "autoregressive_image. Set via forge model_registry.yml.")
            use_cfg = float(cfg_weight) > 1.0

        if use_cfg:
            token_ids_un = tokenizer.format_generation_prompt(
                prompt=prompt,
                sft_format=sft_format,
                special_token_ids=special_token_ids,
                is_unconditional=True,
            )
            token_ids_un = _flatten_tokenizer_output(token_ids_un)
            ids_np_uncond = np.array([token_ids_un], dtype=np.int64)
            if ids_np_uncond.shape[1] != ids_np_cond.shape[1]:
                raise RuntimeError(
                    f"CFG cond/uncond token length mismatch: "
                    f"{ids_np_cond.shape[1]} vs {ids_np_uncond.shape[1]}")
            ids_np = np.concatenate([ids_np_cond, ids_np_uncond], axis=0)
        else:
            ids_np = ids_np_cond

        DeviceAllocator.set_device(device_idx)
        return NBXTensor.from_numpy(ids_np)

    def _read_ids_to_list(self, input_ids: NBXTensor) -> List[int]:
        """Read token IDs from GPU to Python list (small transfer)."""
        import ctypes
        n = input_ids.numel()
        buf = (ctypes.c_char * (n * 8))()
        ctypes.cdll.LoadLibrary('libcudart.so').cudaMemcpy(
            ctypes.byref(buf), ctypes.c_void_p(input_ids.data_ptr()),
            n * 8, 2)  # D2H
        return list(np.frombuffer(bytes(buf), dtype=np.int64))

    def _create_session(self, gen_info: Dict) -> TritonLMSession:
        """Create TritonLMSession with executor and KV cache."""
        from neurobrix.triton.sequence import TritonSequence

        lm_name = gen_info.get("lm_component", "language_model")
        # For models with 'model' as the main component (e.g., TinyLlama)
        if lm_name not in self.ctx.pkg.components:
            for name in self.ctx.pkg.components:
                if name not in ("lm_head",):
                    lm_name = name
                    break

        device_idx = self._parse_device_idx()

        # Load weights via lifecycle (same as native mode).
        # In triton mode, load_weights() routes to _load_weights_triton
        # which loads as NBXTensor directly. Same lifecycle, different format.
        self._ensure_weights_loaded(lm_name)

        # Get executor from context
        executor = self.ctx.executors.get(lm_name)
        if executor is None:
            raise RuntimeError(f"No executor for '{lm_name}'. "
                               f"Available: {list(self.ctx.executors.keys())}")

        # Get LM config for KV cache setup
        lm_config = self.ctx.pkg.defaults.get("lm_config", {})

        # MoE config — propagate norm_topk_prob to the executor BEFORE the
        # TritonSequence compiles. Mirror of core/flow/autoregressive.py:667-676.
        # Without this, detect_and_fuse_moe (run during executor.__init__ /
        # load_graph_from_dict) used the default True and the fused op's attrs
        # were never patched, so DeepSeek (norm_topk_prob=False) silently ran
        # with True → expert contributions normalized incorrectly → gibberish
        # output. top_k / num_experts are extracted from the trace by the
        # fusion pass itself, so only norm_topk_prob needs propagating.
        if lm_config:
            num_experts = lm_config.get("num_experts")
            if num_experts is not None and num_experts > 1:
                norm_topk = lm_config.get("norm_topk_prob")
                if norm_topk is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: norm_topk_prob missing from lm_config "
                        "for MoE model.\n"
                        "Add to forge model_registry.yml: moe.norm_topk_prob")
                executor.set_moe_config(norm_topk_prob=norm_topk)

        if not lm_config:
            extracted = self.ctx.pkg.topology.get("extracted_values", {}).get(lm_name, {})
            lm_config = {
                "num_layers": extracted.get("num_hidden_layers") or extracted.get("num_layers"),
                "num_heads": extracted.get("num_attention_heads") or extracted.get("num_heads"),
                "hidden_size": extracted.get("hidden_size"),
                "num_kv_heads": extracted.get("num_key_value_heads") or extracted.get("num_kv_heads"),
                "head_dim": extracted.get("head_dim"),
            }

        hidden_dim = lm_config.get("hidden_size", 2048)
        num_heads = lm_config.get("num_heads") or 32

        # Graph inputs
        cache_path = Path(self.ctx.pkg.cache_path)
        graph_path = cache_path / "components" / lm_name / "graph.json"
        with open(graph_path, 'r') as f:
            dag = json.load(f)

        graph_inputs = []
        for tid in dag.get("input_tensor_ids", []):
            name = tid.replace("input::", "")
            graph_inputs.append(name)

        uses_embeds = "inputs_embeds" in graph_inputs
        # All LLMs with position_ids input use absolute positions (RoPE, learnable).
        # Default True when graph has position_ids — the graph field is optional.
        uses_abs_pos = dag.get("uses_absolute_position",
                               "position_ids" in graph_inputs)

        # Detect SDPA ops in graph
        sdpa_types = {
            "aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_flash_attention",
            "aten::_scaled_dot_product_cudnn_attention",
        }
        graph_ops = dag.get("ops", {})
        has_sdpa = any(op.get("op_type") in sdpa_types for op in graph_ops.values())

        # Create KV cache from Prism plan (data-driven, zero hardcode)
        # KV cache only for compiled mode — sequential uses O(n) fallback
        kv_interceptor = None
        if has_sdpa and self.ctx.mode == "triton":
            from neurobrix.triton.kv_cache import TritonKVCache, TritonAttentionInterceptor

            kv_plan = getattr(self.ctx.plan, 'kv_cache_plan', None)
            if kv_plan is not None:
                # Prism path — uses precomputed budget
                cache_dtype = parse_dtype(kv_plan.dtype)
                kv_cache = TritonKVCache(
                    num_layers=kv_plan.num_layers,
                    num_kv_heads=kv_plan.num_kv_heads,
                    k_head_dim=kv_plan.k_head_dim,
                    v_head_dim=kv_plan.v_head_dim,
                    max_cache_len=kv_plan.max_cache_len,
                    dtype=cache_dtype,
                )
            else:
                # Legacy fallback from lm_config
                num_kv_heads = lm_config.get("num_kv_heads") or num_heads
                head_dim = lm_config.get("head_dim") or (hidden_dim // num_heads)
                num_layers = lm_config.get("num_layers") or 22
                from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
                max_tokens = decode_bound(self.ctx.pkg.defaults.get("max_tokens", 512))
                kv_cache = TritonKVCache(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    k_head_dim=head_dim,
                    v_head_dim=head_dim,
                    max_cache_len=max_tokens + 128,
                    dtype=NBXDtype.float16,
                )

            kv_interceptor = TritonAttentionInterceptor(
                cache=kv_cache, num_heads=num_heads)

            # Register interceptor on executor (applied when triton sequence compiles).
            # Per-variant: the efficient/cudnn/flash ATen ops have shifted positional
            # signatures (extra compute_log_sumexp / return_debug_mask arg) vs plain
            # SDPA; route each to its matching remap so is_causal / scale bind
            # correctly (else intercept() raises "multiple values for 'scale'").
            _variant_intercept = {
                "aten::_scaled_dot_product_efficient_attention": kv_interceptor.intercept_efficient,
                "aten::_scaled_dot_product_cudnn_attention": kv_interceptor.intercept_efficient,
                "aten::_scaled_dot_product_flash_attention": kv_interceptor.intercept_flash,
            }
            interceptors = {st: _variant_intercept.get(st, kv_interceptor.intercept)
                            for st in sdpa_types}
            executor.register_triton_interceptors(interceptors)

        return TritonLMSession(
            executor=executor,
            kv_wrapper=kv_interceptor,
            hidden_dim=hidden_dim,
            graph_inputs=graph_inputs,
            uses_embeds=uses_embeds,
            uses_absolute_position=uses_abs_pos,
            device_idx=device_idx,
        )

    def _create_strategy(self, gen_info: Dict,
                         session: TritonLMSession):
        """Create TextStrategy or ImageStrategy based on gen_type.

        Mirrors core/flow/autoregressive.py::_create_strategy.
        """
        gen_type = gen_info.get("type")
        defaults = self.ctx.pkg.defaults

        if gen_type == "autoregressive_image":
            head_name = gen_info.get("head_component", "gen_head")
            embed_name = gen_info.get("embed_component", "gen_embed")
            aligner_name = gen_info.get("aligner_component", "gen_aligner")
            decoder_name = gen_info.get("decoder_component", "gen_vision_model")

            for cn in (head_name, embed_name, aligner_name):
                if cn in self.ctx.executors:
                    self._ensure_weights_loaded(cn)

            head_exec = self.ctx.executors.get(head_name)
            embed_exec = self.ctx.executors.get(embed_name)
            aligner_exec = self.ctx.executors.get(aligner_name)
            if not all((head_exec, embed_exec, aligner_exec)):
                missing = [n for n, e in (
                    (head_name, head_exec),
                    (embed_name, embed_exec),
                    (aligner_name, aligner_exec)) if e is None]
                raise RuntimeError(
                    f"autoregressive_image requires executors: missing {missing}")

            cli_cfg = self.ctx.variable_resolver.resolved.get(
                "global.guidance_scale")
            cfg_weight = (float(cli_cfg) if cli_cfg is not None
                          else defaults.get("guidance_scale"))
            if cfg_weight is None:
                raise RuntimeError(
                    "guidance_scale missing from defaults.json for "
                    "autoregressive_image.")
            cfg_weight = float(cfg_weight)

            lm_vocab_size = defaults.get("lm_vocab_size")
            codebook_size = defaults.get("codebook_size")
            if lm_vocab_size is None or codebook_size is None:
                raise RuntimeError(
                    "lm_vocab_size or codebook_size missing from defaults.json "
                    "for autoregressive_image.")
            vq_token_offset = lm_vocab_size - codebook_size

            return TritonImageStrategy(
                head_executor=head_exec,
                embed_executor=embed_exec,
                aligner_executor=aligner_exec,
                decoder_name=decoder_name,
                ctx=self.ctx,
                ensure_weights_fn=self._ensure_weights_loaded,
                cfg_weight=cfg_weight,
                vq_token_offset=vq_token_offset,
                defaults=defaults,
                device_idx=self._parse_device_idx(),
            )

        # Default: text strategy (TinyLlama, Qwen3, DeepSeek-MoE, ...)
        head_name = gen_info.get("head_component", "lm_head")
        self._ensure_weights_loaded(head_name)
        head_executor = self.ctx.executors.get(head_name)
        return TritonTextStrategy(head_executor, session)

    def _unload_non_lm_weights(self, gen_info: Dict):
        """Unload non-persistent component weights after generation."""
        lm_name = gen_info.get("lm_component", "language_model")
        head_name = gen_info.get("head_component", "lm_head")
        for comp_name in list(self.ctx.executors.keys()):
            if comp_name not in (lm_name, head_name):
                self._unload_component_weights(comp_name)


class TritonTextStrategy:
    """Text LLM generation strategy — zero torch.

    Runs lm_head executor to get logits, prepares decode inputs.
    """

    def __init__(self, lm_head_executor, session: TritonLMSession):
        self._head = lm_head_executor
        self._session = session
        # Cache lm_head device index for D2D transfer in get_logits
        dev_str = getattr(lm_head_executor, 'device', 'cuda:0')
        self._head_device_idx = int(dev_str.split(':')[-1]) if ':' in str(dev_str) else 0

    def create_generator(self, defaults: Dict, resolver) -> TritonGenerator:
        config = _build_generator_config(defaults, resolver)
        return TritonGenerator(config)

    def get_logits(self, hidden: NBXTensor, step_idx: int) -> NBXTensor:
        """Run lm_head to get logits from last hidden state."""
        last_hidden = hidden.select(1, hidden.shape[1] - 1).unsqueeze(1)

        # Transfer hidden to lm_head's device if different (pipeline parallel)
        hidden_dev = getattr(last_hidden, '_device_idx', 0)
        head_dev = self._head_device_idx
        if hidden_dev != head_dev:
            dst = NBXTensor.empty(last_hidden._shape, last_hidden._dtype,
                                  f"cuda:{head_dev}")
            DeviceAllocator.memcpy(dst.data_ptr(), last_hidden.data_ptr(),
                                   last_hidden._nbytes)
            last_hidden = dst

        outputs = self._head.run({"input": last_hidden})
        # Find the output tensor
        for key, val in outputs.items():
            return val
        raise RuntimeError("lm_head produced no output.")

    def embed_token(self, next_token: NBXTensor, step_idx: int) -> None:
        """Text path: graph embeds token_ids internally → no separate embed step."""
        return None

    def prepare_decode_input(self, next_token: NBXTensor,
                             token_embed,
                             batch_size: int) -> Tuple[NBXTensor, None]:
        """Prepare input_ids for decode step (text: no embeds).

        Returns (decode_ids, None) — the None signals the session to
        embed inside the graph via its embed_tokens weight.
        """
        if next_token.ndim == 1:
            decode_ids = next_token.unsqueeze(1)
        else:
            decode_ids = next_token
        return decode_ids, None

    def process_output(self, generated_tokens: List[int], ctx) -> None:
        """Store generated tokens in variable resolver."""
        ctx.variable_resolver.resolved["global.output_tokens"] = generated_tokens
        ctx.variable_resolver.resolved["output_tokens"] = generated_tokens

        # SNAC audio post-processing for TTS models
        audio_output_type = ctx.pkg.defaults.get("audio_output_type")
        if audio_output_type == "snac_tokens":
            ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_tokens
            from neurobrix.core.flow.audio_utils import postprocess_audio_output
            postprocess_audio_output(ctx)


def _graph_input_name(executor, fallback: str) -> str:
    """Resolve the single graph input name from executor._dag."""
    dag = getattr(executor, '_dag', None)
    if not dag:
        return fallback
    for tid in dag.get("input_tensor_ids", []):
        if isinstance(tid, str) and "::" in tid:
            return tid.split("::", 1)[-1]
    return fallback


def _int64_shift(nbx: NBXTensor, offset: int) -> NBXTensor:
    """Add a scalar offset to an int64 NBXTensor via D2H → add → H2D.

    Used to shift VQ codebook indices (0..codebook_size-1) into the LM vocab
    range (lm_vocab_size - codebook_size .. lm_vocab_size-1) before feeding
    them back as language_model decode tokens. The transfer is tiny (one
    int64 per decode step × 576 steps) — not a hot-path concern.
    """
    import ctypes
    n = nbx.numel()
    host_buf = (ctypes.c_int64 * n)()
    DeviceAllocator.memcpy(ctypes.addressof(host_buf), nbx.data_ptr(),
                           n * 8, kind=2)  # D2H
    host_np = np.ctypeslib.as_array(host_buf).reshape(nbx.shape)
    shifted = host_np + offset
    return NBXTensor.from_numpy(np.ascontiguousarray(shifted))


class TritonImageStrategy:
    """VQ autoregressive image strategy — zero torch.

    Mirrors core/flow/autoregressive.py::ImageStrategy for the triton path.
    Runs gen_head with CFG-split inputs, uses gen_embed→gen_aligner to
    build decode-time inputs_embeds, and feeds the VQ sequence through
    gen_vision_model.decode_code at the end.
    """

    def __init__(self, head_executor, embed_executor, aligner_executor,
                 decoder_name: str, ctx, ensure_weights_fn: Callable,
                 cfg_weight: float, vq_token_offset: int,
                 defaults: Dict, device_idx: int):
        self._head = head_executor
        self._embed = embed_executor
        self._aligner = aligner_executor
        self._decoder_name = decoder_name
        self._ctx = ctx
        self._ensure_weights = ensure_weights_fn
        self._cfg_weight = cfg_weight
        self._use_cfg = cfg_weight > 1.0
        self._vq_token_offset = vq_token_offset
        self._defaults = defaults
        self._device_idx = device_idx
        self._head_in = _graph_input_name(head_executor, "x")
        self._embed_in = _graph_input_name(embed_executor, "input")
        self._aligner_in = _graph_input_name(aligner_executor, "x_or_tuple")

    def create_generator(self, defaults: Dict, resolver) -> TritonGenerator:
        """VQ image generator: fixed max_tokens = num_patches², no EOS."""
        image_size = defaults["image_size"]
        patch_size = defaults["patch_size"]
        num_patches = image_size // patch_size
        max_tokens = num_patches * num_patches
        config = {
            "max_tokens": max_tokens,
            "temperature": defaults["temperature"],
            "top_p": defaults["top_p"],
            "top_k": defaults["top_k"],
            "repetition_penalty": defaults["repetition_penalty"],
            "vocab_size": defaults["codebook_size"],
            "eos_token_id": None,
            "pad_token_id": None,
            "_class_name": "TritonVQImageGenerator",
        }
        if resolver is not None:
            resolved = getattr(resolver, 'resolved', {})
            for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
                gkey = f"global.{key}"
                if gkey in resolved and resolved[gkey] is not None:
                    config[key] = resolved[gkey]
        return TritonGenerator(config)

    def get_logits(self, hidden: NBXTensor, step_idx: int) -> NBXTensor:
        """Run gen_head on last hidden position, CFG-combine if enabled.

        Head was traced with batch=2 (CFG) but runs correctly at batch=1
        via symbolic dim binding — so we split cond/uncond and run twice,
        matching native ImageStrategy.
        """
        last = hidden.select(1, hidden.shape[1] - 1).unsqueeze(1)  # (B, 1, D)
        if self._use_cfg and last.shape[0] >= 2:
            l_cond = self._run_head(last.narrow(0, 0, 1).contiguous())
            l_uncond = self._run_head(last.narrow(0, 1, 1).contiguous())
            return l_uncond + self._cfg_weight * (l_cond - l_uncond)
        return self._run_head(last)

    def _run_head(self, hidden: NBXTensor) -> NBXTensor:
        outputs = self._head.run({self._head_in: hidden})
        for val in outputs.values():
            logits = val
            if logits.ndim == 3 and logits.shape[1] > 1:
                logits = logits.narrow(1, 0, 1)
            return logits
        raise RuntimeError("gen_head produced no output.")

    def embed_token(self, next_token: NBXTensor, step_idx: int) -> NBXTensor:
        """Look up VQ codebook embedding via gen_embed.

        gen_embed graph is one op (aten::embedding) traced with input shape
        (2, 1). The codebook is (16384, 8). We pass the raw sampled token
        without the LM-vocab offset — gen_embed indexes the VQ codebook.
        """
        tok = next_token
        if tok.ndim == 0:
            # Shouldn't happen with current sampler, but guard anyway.
            raise RuntimeError("Unexpected scalar next_token from sampler.")
        if tok.ndim == 1:
            tok = tok.unsqueeze(0) if tok.shape[0] == 1 else tok.unsqueeze(1)
        # Now tok is at least 2D (B, 1) — embed graph handles it.
        outputs = self._embed.run({self._embed_in: tok})
        for val in outputs.values():
            emb = val
            if emb.ndim == 3 and emb.shape[1] > 1:
                emb = emb.narrow(1, 0, 1)
            return emb
        raise RuntimeError("gen_embed produced no output.")

    def prepare_decode_input(self, next_token: NBXTensor,
                             token_embed: NBXTensor,
                             batch_size: int) -> Tuple[NBXTensor, NBXTensor]:
        """Align VQ embedding to LM hidden dim, shift token to LM vocab range.

        Returns (decode_token_ids, aligned_embeds).  The session feeds
        aligned_embeds as inputs_embeds; decode_token_ids is kept for
        bookkeeping / repetition_penalty context on the generator side.
        """
        outputs = self._aligner.run({self._aligner_in: token_embed})
        aligned = None
        for val in outputs.values():
            aligned = val
            break
        if aligned is None:
            raise RuntimeError("gen_aligner produced no output.")
        if aligned.ndim == 3 and aligned.shape[1] > 1:
            aligned = aligned.narrow(1, 0, 1)

        # Shift token into LM vocab range (VQ index → LM token id).
        decode_token = _int64_shift(next_token, self._vq_token_offset)
        if decode_token.ndim == 1:
            decode_token = decode_token.unsqueeze(0)  # → (1, S)
        if decode_token.ndim == 2 and decode_token.shape[1] != 1:
            decode_token = decode_token.narrow(1, 0, 1)

        if self._use_cfg:
            if aligned.shape[0] == 1:
                aligned = aligned.expand(2, -1, -1).contiguous()
            if decode_token.shape[0] == 1:
                decode_token = decode_token.expand(2, -1).contiguous()

        return decode_token, aligned

    def process_output(self, generated_tokens: List[int], ctx) -> None:
        """Decode the VQ token sequence into pixels via gen_vision_model."""
        ctx.variable_resolver.resolved["global.output_tokens"] = generated_tokens
        ctx.variable_resolver.resolved["output_tokens"] = generated_tokens

        self._ensure_weights(self._decoder_name)
        decoder = ctx.executors.get(self._decoder_name)
        if decoder is None:
            raise RuntimeError(
                f"Decoder '{self._decoder_name}' not found in executors.")

        num_tokens = len(generated_tokens)
        ids_np = np.array(generated_tokens, dtype=np.int64)
        DeviceAllocator.set_device(self._device_idx)
        code_b = NBXTensor.from_numpy(ids_np)

        decoder_in = _graph_input_name(decoder, "code_b")
        decoder_output = decoder.run({decoder_in: code_b})

        image = None
        for val in decoder_output.values():
            if hasattr(val, 'shape') and val.ndim == 4:
                image = val
                break
        if image is None:
            for val in decoder_output.values():
                image = val
                break
        if image is None:
            raise RuntimeError(
                f"gen_vision_model produced no output (ran on {num_tokens} tokens).")

        ctx.variable_resolver.resolved["global.output_image"] = image
        ctx.variable_resolver.resolved["output_image"] = image
