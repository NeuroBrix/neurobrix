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
    # Override from variable resolver if present
    if resolver is not None:
        resolved = getattr(resolver, 'resolved', {})
        for key in ("max_tokens", "temperature", "top_p", "top_k"):
            val = resolved.get(f"global.{key}") or resolved.get(key)
            if val is not None:
                config[key] = val
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
        for step_idx in generator:
            logits = strategy.get_logits(hidden, step_idx)
            next_token, is_done = generator.step(logits, step_idx)

            if is_done:
                break

            decode_ids = strategy.prepare_decode_input(next_token, batch_size)
            hidden = session.decode_step(decode_ids)

        # Output
        generated_tokens = generator.get_generated_tokens()
        strategy.process_output(generated_tokens, self.ctx)

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
        """Tokenize prompt → NBXTensor of token IDs."""
        if "tokenizer" not in self.ctx.modules:
            raise RuntimeError("autoregressive_generation requires 'tokenizer' module.")
        tokenizer = self.ctx.modules["tokenizer"]
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt", "")

        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        elif hasattr(tokenizer, 'encode'):
            token_ids = tokenizer.encode(prompt)
        else:
            raise RuntimeError("Tokenizer has no encode method.")

        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        # Convert to NBXTensor via numpy
        ids_np = np.array([token_ids], dtype=np.int64)
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
                max_tokens = self.ctx.pkg.defaults.get("max_tokens", 512)
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

            # Register interceptor on executor (applied when triton sequence compiles)
            interceptors = {st: kv_interceptor.intercept for st in sdpa_types}
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
                         session: TritonLMSession) -> "TritonTextStrategy":
        """Create text generation strategy."""
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

    def create_generator(self, defaults: Dict, resolver) -> TritonGenerator:
        config = _build_generator_config(defaults, resolver)
        return TritonGenerator(config)

    def get_logits(self, hidden: NBXTensor, step_idx: int) -> NBXTensor:
        """Run lm_head to get logits from last hidden state."""
        last_hidden = hidden.select(1, hidden.shape[1] - 1).unsqueeze(1)
        outputs = self._head.run({"input": last_hidden})
        # Find the output tensor
        for key, val in outputs.items():
            return val
        raise RuntimeError("lm_head produced no output.")

    def embed_token(self, next_token: NBXTensor, step_idx: int) -> NBXTensor:
        return next_token

    def prepare_decode_input(self, next_token: NBXTensor,
                             batch_size: int) -> NBXTensor:
        """Prepare input_ids for decode step."""
        if next_token.ndim == 1:
            return next_token.unsqueeze(1)
        return next_token

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
