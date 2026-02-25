"""
Autoregressive Flow Handler — Universal Loop

ZERO SEMANTIC: Executes token-by-token generation from topology.
ZERO HARDCODE: All configuration from NBX container.

Architecture (Strategy Pattern):
  AutoregressiveHandler            — universal loop
    ├── GraphLMSession             — executor + KV cache lifecycle
    └── GenerationStrategy (ABC)   — text/image differences
          ├── TextStrategy         — F.linear logits, F.embedding, no CFG
          └── ImageStrategy        — gen_head logits, gen_embed+aligner, CFG
"""

import gc
import os
import json
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.module.autoregressive.generator import AutoregressiveGenerator

from .base import FlowHandler, FlowContext, register_flow

_SENTINEL = object()


# ═══════════════════════════════════════════════════════════════════════════════
# GraphLMSession — Executor + KV Cache Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class GraphLMSession:
    """
    Encapsulates GraphExecutor + KVCacheWrapper lifecycle.

    The main loop never touches the executor or KV cache directly.
    Handles both O(1) decode (with KV cache) and O(n) fallback (without).
    """

    def __init__(
        self,
        executor,
        kv_wrapper,
        hidden_dim: int,
        graph_inputs: List[str],
        uses_embeds: bool,
        uses_absolute_position: bool = False,
    ):
        self.executor = executor
        self.kv_wrapper = kv_wrapper
        self.hidden_dim = hidden_dim
        self.graph_inputs = graph_inputs
        self.uses_embeds = uses_embeds
        self.uses_absolute_position = uses_absolute_position
        # O(n) fallback state (models without SDPA / no KV cache)
        self._accumulated_ids: Optional[torch.Tensor] = None
        self._accumulated_embeds: Optional[torch.Tensor] = None

    def prefill(self, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reset cache → run prefill → switch to decode mode → return hidden_states."""
        # Reset KV cache for new sequence
        if self.kv_wrapper is not None:
            self.kv_wrapper.reset_for_new_sequence()

        # Find correct device from executor weights (FGP may have distributed weights)
        device = self.executor.device
        if hasattr(self.executor, '_weights') and self.executor._weights:
            for weight_key, weight_tensor in self.executor._weights.items():
                if 'embed' in weight_key.lower():
                    device = weight_tensor.device
                    break
            else:
                first_weight = next(iter(self.executor._weights.values()))
                device = first_weight.device

        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        # Position IDs: absolute [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Enable hidden states capture BEFORE execution
        self.executor.enable_hidden_states_capture()

        # Build run inputs
        if self.uses_embeds:
            embed_weight = self.executor.get_embed_tokens()
            if embed_weight is None:
                raise RuntimeError(
                    "ZERO FALLBACK: Graph expects inputs_embeds but no embedding weight found."
                )
            with torch.no_grad():
                inputs_embeds = F.embedding(input_ids, embed_weight)
            run_inputs = {"inputs_embeds": inputs_embeds}
            if 'position_ids' in self.graph_inputs:
                run_inputs["position_ids"] = position_ids
        elif 'position_ids' in self.graph_inputs:
            run_inputs = {"input_ids": input_ids, "position_ids": position_ids}
        else:
            run_inputs = {"input_ids": input_ids}

        self.executor.run(run_inputs)

        # Switch to decode mode after prefill
        if self.kv_wrapper is not None:
            self.kv_wrapper.set_decode_mode(actual_seq_len=1)

        # Extract hidden states
        hidden_states = self.executor.get_hidden_states(
            expected_hidden_dim=self.hidden_dim,
            expected_batch_size=batch_size,
        )
        if hidden_states is None:
            raise RuntimeError(
                "ZERO FALLBACK: Prefill could not extract hidden_states.\n"
                "get_hidden_states() returned None."
            )
        # Ensure 3D [batch, seq, hidden_dim]
        while hidden_states.dim() > 3:
            hidden_states = hidden_states.squeeze(0)

        if self.kv_wrapper is None:
            # Init O(n) fallback accumulation
            if self.uses_embeds:
                embed_weight = self.executor.get_embed_tokens()
                if embed_weight is not None:
                    self._accumulated_embeds = F.embedding(input_ids, embed_weight)
            else:
                self._accumulated_ids = input_ids.clone()

        return hidden_states

    def decode_step(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single decode step. O(1) with KV cache, O(n) fallback."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if self.kv_wrapper is not None:
            # ── O(1) path: KV cache ──
            self.kv_wrapper.update_position_offset()
            seq_len = input_ids.shape[1]
            cache_len = self.kv_wrapper.get_cache_len()

            # Position IDs policy:
            # - Models with FLOAT arange (RoPE via interceptor): relative [[0]]
            #   The arange interceptor shifts float aranges by cache_len,
            #   so position_ids=[[0]] indexes row 0 of a shifted table → correct.
            # - Models with INT arange or image family: absolute [[cache_len]]
            #   INT aranges are NOT shifted by interceptor → needs explicit position.
            if self.uses_absolute_position:
                position_ids = torch.full((batch_size, seq_len), cache_len, dtype=torch.long, device=device)
            else:
                position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

            if self.uses_embeds:
                embeds_to_use = inputs_embeds if inputs_embeds is not None else self._embed_from_ids(input_ids)
                run_inputs = {"inputs_embeds": embeds_to_use}
                if 'position_ids' in self.graph_inputs:
                    embeds_batch = embeds_to_use.shape[0]
                    if embeds_batch != batch_size:
                        position_ids = torch.zeros((embeds_batch, seq_len), dtype=torch.long, device=device)
                    run_inputs["position_ids"] = position_ids
            elif 'position_ids' in self.graph_inputs:
                run_inputs = {"input_ids": input_ids, "position_ids": position_ids}
            else:
                run_inputs = {"input_ids": input_ids}

            self.executor.run(run_inputs)

        else:
            # ── O(n) fallback: re-run full context ──
            if self._accumulated_embeds is not None:
                # Accumulate embeds (VQ image path)
                embed_to_add = inputs_embeds if inputs_embeds is not None else self._embed_from_ids(input_ids)
                if embed_to_add.shape[0] != self._accumulated_embeds.shape[0]:
                    embed_to_add = embed_to_add.expand(self._accumulated_embeds.shape[0], -1, -1).contiguous()
                self._accumulated_embeds = torch.cat([self._accumulated_embeds, embed_to_add], dim=1)
                actual_len = self._accumulated_embeds.shape[1]
                run_inputs = {"inputs_embeds": self._accumulated_embeds}
                if 'position_ids' in self.graph_inputs:
                    pos_ids = torch.arange(actual_len, dtype=torch.long, device=device)
                    pos_ids = pos_ids.unsqueeze(0).expand(self._accumulated_embeds.shape[0], -1)
                    run_inputs["position_ids"] = pos_ids
                self.executor.run(run_inputs)
                batch_size = self._accumulated_embeds.shape[0]
            else:
                # Accumulate token IDs (text LLM path)
                new_tok = input_ids.to(self._accumulated_ids.device)
                if new_tok.dim() == 1:
                    new_tok = new_tok.unsqueeze(0)
                self._accumulated_ids = torch.cat([self._accumulated_ids, new_tok], dim=1)
                actual_len = self._accumulated_ids.shape[1]
                run_inputs = {"input_ids": self._accumulated_ids}
                if 'position_ids' in self.graph_inputs:
                    pos_ids = torch.arange(actual_len, dtype=torch.long, device=device)
                    pos_ids = pos_ids.unsqueeze(0).expand(self._accumulated_ids.shape[0], -1)
                    run_inputs["position_ids"] = pos_ids
                self.executor.run(run_inputs)
                batch_size = self._accumulated_ids.shape[0]

        # Extract hidden states
        hidden_states = self.executor.get_hidden_states(
            expected_hidden_dim=self.hidden_dim,
            expected_batch_size=batch_size,
        )
        if hidden_states is None:
            raise RuntimeError(
                "ZERO FALLBACK: Decode step could not extract hidden_states."
            )
        while hidden_states.dim() > 3:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states

    def cleanup(self):
        """Release per-request resources (KV cache, intermediates).

        Respects executor._persistent: when True, executor stays alive
        with weights in VRAM and compiled sequence cached. Only KV cache
        and per-request intermediates are cleared.
        """
        is_persistent = getattr(self.executor, '_persistent', False) if self.executor else False

        if self.executor is not None:
            self.executor.cleanup()  # Respects _persistent flag internally
            if not is_persistent:
                self.executor = None  # Drop reference (executor lives in ctx.executors)

        if self.kv_wrapper is not None:
            self.kv_wrapper.reset_for_new_sequence()
            self.kv_wrapper = None

        if not is_persistent:
            gc.collect()
            torch.cuda.empty_cache()

    def _embed_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs via executor's embed_tokens weight."""
        embed_weight = self.executor.get_embed_tokens()
        if embed_weight is None:
            raise RuntimeError("ZERO FALLBACK: Graph expects inputs_embeds but no embedding weight found.")
        with torch.no_grad():
            return F.embedding(input_ids, embed_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# GenerationStrategy — Text vs Image Differences
# ═══════════════════════════════════════════════════════════════════════════════

class GenerationStrategy(ABC):
    """Abstract strategy encapsulating text/image generation differences."""

    @abstractmethod
    def create_generator(self, defaults: Dict, resolver: Any) -> 'AutoregressiveGenerator':
        """Create the appropriate AutoregressiveGenerator."""

    @abstractmethod
    def get_logits(self, last_hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Compute logits from last hidden state."""

    @abstractmethod
    def embed_token(self, next_token: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Embed sampled token for next decode step."""

    @abstractmethod
    def prepare_decode_input(
        self, next_token: torch.Tensor, token_embed: torch.Tensor, batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare (input_ids, inputs_embeds) for the next decode step."""

    @abstractmethod
    def process_output(self, generated_tokens: torch.Tensor, ctx: FlowContext) -> None:
        """Process final generated tokens into output (text or image)."""


class TextStrategy(GenerationStrategy):
    """Strategy for pure text LLM generation (DeepSeek, Llama, etc.)."""

    def __init__(self, lm_head_weight: torch.Tensor, session: GraphLMSession):
        self._lm_head_weight = lm_head_weight
        self._session = session

    def create_generator(self, defaults: Dict, resolver: Any):
        from neurobrix.core.module.autoregressive.generator import AutoregressiveGenerator
        config = _build_generator_config(defaults, resolver, gen_type="autoregressive_text")
        return AutoregressiveGenerator(config)

    def get_logits(self, last_hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        # logits = F.linear(hidden_states, lm_head_weight)
        h = last_hidden
        if h.device != self._lm_head_weight.device:
            h = h.to(self._lm_head_weight.device)
        if h.dtype != self._lm_head_weight.dtype:
            h = h.to(self._lm_head_weight.dtype)
        return F.linear(h, self._lm_head_weight)

    def embed_token(self, next_token: torch.Tensor, step_idx: int) -> torch.Tensor:
        # No-op for text: the graph handles embedding internally via aten::embedding.
        # prepare_decode_input returns (input_ids, None) — token_embed is unused.
        return next_token

    def prepare_decode_input(
        self, next_token: torch.Tensor, token_embed: torch.Tensor, batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        decode_ids = next_token.unsqueeze(1) if next_token.dim() == 1 else next_token
        return decode_ids, None

    def process_output(self, generated_tokens: torch.Tensor, ctx: FlowContext) -> None:
        ctx.variable_resolver.resolved["global.output_tokens"] = generated_tokens
        ctx.variable_resolver.resolved["output_tokens"] = generated_tokens


class ImageStrategy(GenerationStrategy):
    """Strategy for VQ image generation (Janus, LlamaGen, etc.)."""

    def __init__(
        self,
        head_executor,
        embed_executor,
        aligner_executor,
        decoder_executor,
        output_extractor,
        ensure_weights_fn: Callable,
        cfg_weight: float,
        vq_token_offset: int,
        defaults: Dict,
    ):
        self._head = head_executor
        self._embed = embed_executor
        self._aligner = aligner_executor
        self._decoder = decoder_executor
        self._output_extractor = output_extractor
        self._ensure_weights = ensure_weights_fn
        self._cfg_weight = cfg_weight
        self._use_cfg = cfg_weight > 1.0
        self._vq_token_offset = vq_token_offset
        self._defaults = defaults

    def create_generator(self, defaults: Dict, resolver: Any):
        from neurobrix.core.module.autoregressive.generator import VQImageGenerator
        config = _build_generator_config(defaults, resolver, gen_type="autoregressive_image")
        return VQImageGenerator(config)

    def get_logits(self, last_hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        if self._use_cfg and last_hidden.shape[0] >= 2:
            # Run head separately for cond/uncond (head traced with batch=1)
            logits_cond = self._run_head(last_hidden[0:1, ...], step_idx)
            logits_uncond = self._run_head(last_hidden[1:2, ...], step_idx)
            return logits_uncond + self._cfg_weight * (logits_cond - logits_uncond)
        return self._run_head(last_hidden, step_idx)

    def embed_token(self, next_token: torch.Tensor, step_idx: int) -> torch.Tensor:
        # gen_embed component
        input_name = _get_graph_input_name(self._embed, "input_ids")
        embed_input = next_token
        output = self._embed.run({input_name: embed_input})
        token_embed = self._output_extractor.extract_embedding(output)
        if token_embed.dim() == 3 and token_embed.shape[1] > 1:
            token_embed = token_embed[:, 0:1, :]
        return token_embed

    def prepare_decode_input(
        self, next_token: torch.Tensor, token_embed: torch.Tensor, batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # gen_aligner: align VQ embedding to LM space
        input_name = _get_graph_input_name(self._aligner, "x")
        aligner_output = self._aligner.run({input_name: token_embed})
        aligned_embed = self._output_extractor.extract_embedding(aligner_output)
        if aligned_embed.dim() == 3 and aligned_embed.shape[1] > 1:
            aligned_embed = aligned_embed[:, 0:1, :]

        # Offset token to LM vocab range
        decode_token = next_token + self._vq_token_offset
        if decode_token.dim() == 1:
            decode_token = decode_token.unsqueeze(0).unsqueeze(1)
        elif decode_token.dim() == 2 and decode_token.shape[0] == 1:
            pass

        # Duplicate for CFG batch
        if self._use_cfg:
            if aligned_embed.shape[0] == 1:
                aligned_embed = aligned_embed.expand(2, -1, -1).contiguous()
            decode_token = decode_token.expand(2, -1).contiguous()

        return decode_token, aligned_embed

    def process_output(self, generated_tokens: torch.Tensor, ctx: FlowContext) -> None:
        # Load decoder weights
        decoder_name = ctx.pkg.topology["flow"]["generation"].get("decoder_component", "gen_vision_model")
        if decoder_name in ctx.executors:
            self._ensure_weights(decoder_name)

        decoder = ctx.executors.get(decoder_name)
        if decoder is None:
            raise RuntimeError(f"ZERO FALLBACK: Decoder executor '{decoder_name}' not found.")

        # Get decode shape from generator config
        image_size = self._defaults.get("image_size")
        patch_size = self._defaults.get("patch_size")
        codebook_dim = self._defaults.get("codebook_dim")
        if not all((image_size, patch_size, codebook_dim)):
            raise RuntimeError(
                "ZERO FALLBACK: VQ decode requires image_size, patch_size, codebook_dim in defaults."
            )
        num_patches = image_size // patch_size
        # Get decoder input name from graph
        decoder_dag = decoder._dag
        input_name = "input"
        if decoder_dag:
            input_tensor_ids = decoder_dag.get("input_tensor_ids", [])
            if input_tensor_ids:
                first_input = input_tensor_ids[0]
                if "::" in first_input:
                    input_name = first_input.split("::")[-1]

        decoder_output = decoder.run({input_name: generated_tokens})
        image = self._output_extractor.extract_image(decoder_output)

        ctx.variable_resolver.resolved["global.output_image"] = image
        ctx.variable_resolver.resolved["output_image"] = image

    def _run_head(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Run gen_head component on hidden states."""
        input_name = _get_graph_input_name(self._head, "x")
        output = self._head.run({input_name: hidden})
        logits = self._output_extractor.extract_logits(output)
        if logits.dim() == 3 and logits.shape[1] > 1:
            logits = logits[:, 0:1, :]
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# AutoregressiveHandler — Universal Loop
# ═══════════════════════════════════════════════════════════════════════════════

@register_flow("autoregressive_generation")
class AutoregressiveHandler(FlowHandler):
    """
    Universal autoregressive generation handler.

    Same ~40 line loop for ALL models (text LLM, VQ image).
    Strategy pattern encapsulates the 5 text/image differences.

    ZERO SEMANTIC: No domain knowledge beyond token generation.
    ZERO HARDCODE: Component names from topology.
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Optional[Any]],
        ensure_weights_fn: Callable[[str], None],
        unload_weights_fn: Callable[[str], None],
        input_resolver: Optional[Any] = None,
        input_synthesizer: Optional[Any] = None,
        output_extractor: Optional[Any] = None,
    ):
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn
        self._input_resolver = input_resolver
        self._input_synthesizer = input_synthesizer
        self._output_extractor = output_extractor
        self._active_session: Optional[GraphLMSession] = None

    def execute(self) -> Dict[str, Any]:
        """Execute autoregressive_generation flow — universal for all models."""
        # ── SETUP ──────────────────────────────────────────
        gen_info = self.ctx.pkg.topology.get("flow", {}).get("generation", {})
        if not gen_info:
            raise RuntimeError(
                "ZERO FALLBACK: autoregressive_generation requires flow.generation info."
            )
        gen_type = gen_info.get("type")

        session = self._create_session(gen_info)
        self._active_session = session
        strategy = self._create_strategy(gen_info, session)
        generator: Any = strategy.create_generator(self.ctx.pkg.defaults, self.ctx.variable_resolver)

        device = self._parse_device()
        generator.set_generation_params(device=torch.device(device))

        input_ids = self._tokenize(gen_info, device)
        batch_size = input_ids.shape[0]

        # Pass prompt token IDs to generator for repetition penalty context.
        # HuggingFace penalizes tokens from BOTH prompt and generated text.
        generator.set_prompt_ids(input_ids[0].tolist())

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.inference_mode():
            # ── PREFILL ────────────────────────────────────
            hidden = session.prefill(input_ids, batch_size)

            # ── DECODE LOOP ────────────────────────────────
            _debug_decode = os.environ.get("NBX_DEBUG_DECODE", "0") == "1"
            for step_idx in generator:

                logits = strategy.get_logits(hidden[:, -1:, :], step_idx)
                next_token, is_done = generator.step(logits, step_idx)

                if _debug_decode and step_idx < 15:
                    h_last = hidden[0, -1, :].float()
                    l_last = logits[0, -1, :].float() if logits.dim() == 3 else logits[0, :].float()
                    probs = torch.softmax(l_last, dim=-1)
                    top5_p, top5_i = torch.topk(probs, 5)
                    cache_info = session.kv_wrapper.get_cache_info() if session.kv_wrapper else {}
                    print(f"  [DBG] step={step_idx} token={next_token.item()} "
                          f"h_norm={h_last.norm():.1f} logit_max={l_last.max():.2f} "
                          f"top1_prob={top5_p[0]:.4f} top5={top5_i.tolist()} "
                          f"cache_seq={cache_info.get('seq_len', '?')} "
                          f"pos_off={cache_info.get('position_offset', '?')}")

                if is_done:
                    break

                token_embed = strategy.embed_token(next_token, step_idx)
                decode_ids, decode_embeds = strategy.prepare_decode_input(
                    next_token, token_embed, batch_size,
                )
                hidden = session.decode_step(decode_ids, inputs_embeds=decode_embeds)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ── OUTPUT + CLEANUP ───────────────────────────────
        generated_tokens = generator.get_generated_tokens()
        strategy.process_output(generated_tokens, self.ctx)

        session.cleanup()
        self._unload_non_lm_weights(gen_info)

        return self.ctx.variable_resolver.resolve_all()

    # ─── SETUP HELPERS ─────────────────────────────────────────────────────────

    def _parse_device(self) -> str:
        """Parse device from primary_device, handling compound strategy strings."""
        device_str = self.ctx.primary_device
        if ":" in device_str and device_str.split(":", 1)[0] in ("tp", "pp", "fgp", "zero3"):
            devices_part = device_str.split(":", 1)[1]
            return devices_part.split(",")[0]
        return device_str

    def _create_session(self, gen_info: Dict) -> GraphLMSession:
        """Create GraphLMSession with executor, KV cache, and interceptors."""
        from neurobrix.core.module.cache.factory import StateCacheFactory
        from neurobrix.core.runtime.factory import ExecutorFactory

        lm_name = gen_info.get("lm_component", "language_model")
        device = self._parse_device()

        # Get dtype from Prism allocation
        dtype = torch.float32
        if hasattr(self.ctx.plan, 'get_allocation'):
            comp_alloc = self.ctx.plan.get_allocation(lm_name)
            if comp_alloc and hasattr(comp_alloc, 'dtype'):
                dtype_val = comp_alloc.dtype
                if isinstance(dtype_val, torch.dtype):
                    dtype = dtype_val
                elif isinstance(dtype_val, str):
                    dtype = getattr(torch, dtype_val, torch.float32)

        # Get/create GraphExecutor
        cache_path = Path(self.ctx.pkg.cache_path)
        graph_path = cache_path / "components" / lm_name / "graph.json"
        if not graph_path.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: graph.json not found for '{lm_name}'.\n"
                f"Expected at: {graph_path}"
            )

        with open(graph_path, 'r') as f:
            dag = json.load(f)

        executor = self.ctx.executors.get(lm_name)
        if executor is None:
            allocation = None
            if hasattr(self.ctx.plan, 'components') and self.ctx.plan.components:
                allocation = self.ctx.plan.components.get(lm_name)
            elif hasattr(self.ctx.plan, 'allocations'):
                for alloc in self.ctx.plan.allocations:
                    if getattr(alloc, 'name', None) == lm_name:
                        allocation = alloc
                        break
            elif hasattr(self.ctx.plan, 'get_allocation'):
                allocation = self.ctx.plan.get_allocation(lm_name)

            if allocation is None:
                available = []
                if hasattr(self.ctx.plan, 'components'):
                    available = list(self.ctx.plan.components.keys())
                raise RuntimeError(
                    f"ZERO FALLBACK: No Prism allocation for '{lm_name}'.\n"
                    f"Available: {available}"
                )

            executor = ExecutorFactory.create(
                component=lm_name,
                allocation=allocation,
                nbx_path=str(self.ctx.pkg.root_path),
                dag=dag,
                mode=self.ctx.mode,
                skip_weights=True,
            )

            # FGP mode detection
            strategy = getattr(allocation, 'strategy', '')
            strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
            if 'fgp' in strategy_str.lower():
                executor._is_fgp_mode = True

            self.ctx.executors[lm_name] = executor

        # Propagate persistent mode from FlowContext to GraphExecutor
        # Must happen BEFORE first execute so cleanup() preserves weights
        if self.ctx.persistent_mode and hasattr(executor, '_persistent'):
            executor._persistent = True

        # Ensure weights loaded
        self._ensure_weights_loaded(lm_name)

        # Get lm_config
        lm_config = self.ctx.pkg.defaults.get("lm_config", {})
        if not lm_config:
            extracted = self.ctx.pkg.topology.get("extracted_values", {}).get(lm_name, {})
            lm_config = {
                "num_layers": extracted.get("num_hidden_layers") or extracted.get("num_layers"),
                "num_heads": extracted.get("num_attention_heads") or extracted.get("num_heads"),
                "hidden_size": extracted.get("hidden_size"),
                "num_kv_heads": extracted.get("num_key_value_heads"),
                "head_dim": extracted.get("head_dim"),
                "max_position_embeddings": extracted.get("max_position_embeddings"),
            }

        # MoE config — values MUST come from lm_config (built by forge from model_registry.yml)
        num_experts = lm_config.get("num_experts")
        if num_experts is not None and num_experts > 1:
            norm_topk = lm_config.get("norm_topk_prob")
            if norm_topk is None:
                raise RuntimeError(
                    "ZERO FALLBACK: norm_topk_prob missing from lm_config for MoE model.\n"
                    "Add to forge model_registry.yml: moe.norm_topk_prob"
                )
            executor.set_moe_config(norm_topk_prob=norm_topk)

        # Create KV cache
        kv_wrapper = StateCacheFactory.create(self.ctx, lm_config, device, dtype)

        # Detect SDPA ops
        sdpa_op_types = {
            "aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_flash_attention",
            "aten::_scaled_dot_product_cudnn_attention",
        }
        graph_ops = dag.get("ops", {})
        has_sdpa = any(op.get("op_type") in sdpa_op_types for op in graph_ops.values())

        if not has_sdpa:
            kv_wrapper = None
        else:
            # Register interceptors
            if hasattr(executor, 'register_op_interceptors'):
                interceptors = kv_wrapper.get_interceptors()
                executor.register_op_interceptors(interceptors)
            elif hasattr(executor, 'register_op_interceptor'):
                interceptors = kv_wrapper.get_interceptors()
                for sdpa_op, interceptor in interceptors.items():
                    executor.register_op_interceptor(sdpa_op, interceptor)

        # Detect graph inputs
        input_ids_list = dag.get('input_tensor_ids', [])
        graph_inputs = [tid.replace('input::', '') for tid in input_ids_list if tid.startswith('input::')]
        uses_embeds = 'inputs_embeds' in graph_inputs

        # Detect position_ids policy: absolute vs relative
        # Models with INT arange for RoPE (Janus, DeepSeek-V2 MLA) need absolute position_ids
        # Models with FLOAT arange (DeepSeek-V1, Llama) use relative (arange interceptor shifts)
        family = self.ctx.pkg.manifest.get("family")
        if family is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'family' missing from manifest.json. "
                "Add it via forge builder and rebuild."
            )
        uses_int_arange = _detect_int_arange_for_rope(dag)
        uses_absolute_position = family == "image" or uses_int_arange

        # Get hidden_dim
        lm_extracted = self.ctx.pkg.topology.get("extracted_values", {}).get(lm_name, {})
        hidden_dim = (
            lm_extracted.get("hidden_size")
            or self.ctx.pkg.defaults.get("lm_config", {}).get("hidden_size")
        )
        if hidden_dim is None:
            raise RuntimeError(
                f"ZERO FALLBACK: 'hidden_size' not found for '{lm_name}'."
            )

        return GraphLMSession(
            executor=executor,
            kv_wrapper=kv_wrapper,
            hidden_dim=hidden_dim,
            graph_inputs=graph_inputs,
            uses_embeds=uses_embeds,
            uses_absolute_position=uses_absolute_position,
        )

    def _create_strategy(self, gen_info: Dict, session: GraphLMSession) -> GenerationStrategy:
        """Create TextStrategy or ImageStrategy based on generation type."""
        gen_type = gen_info.get("type")
        device = self._parse_device()
        defaults = self.ctx.pkg.defaults

        # Get dtype from Prism allocation
        dtype = torch.float32
        lm_name = gen_info.get("lm_component", "language_model")
        if hasattr(self.ctx.plan, 'get_allocation'):
            comp_alloc = self.ctx.plan.get_allocation(lm_name)
            if comp_alloc and hasattr(comp_alloc, 'dtype'):
                dtype_val = comp_alloc.dtype
                if isinstance(dtype_val, torch.dtype):
                    dtype = dtype_val
                elif isinstance(dtype_val, str):
                    dtype = getattr(torch, dtype_val, torch.float32)

        if gen_type == "autoregressive_text":
            # Load lm_head weight
            if "lm_head" not in self.ctx.pkg.topology.get("components", {}):
                raise RuntimeError("ZERO FALLBACK: lm_head component not found for text generation.")

            lm_head_device = device
            if hasattr(self.ctx.plan, 'get_allocation'):
                lm_head_alloc = self.ctx.plan.get_allocation("lm_head")
                if lm_head_alloc and hasattr(lm_head_alloc, 'device'):
                    lm_head_device = str(lm_head_alloc.device)

            lm_head_weight = self._load_lm_head_weight(lm_head_device, dtype)
            return TextStrategy(lm_head_weight, session)

        elif gen_type == "autoregressive_image":
            # Get component executors
            head_name = gen_info.get("head_component", "gen_head")
            embed_name = gen_info.get("embed_component", "gen_embed")
            aligner_name = gen_info.get("aligner_component", "gen_aligner")

            for comp_name in [head_name, embed_name, aligner_name]:
                if comp_name in self.ctx.executors:
                    self._ensure_weights_loaded(comp_name)

            head_executor = self.ctx.executors.get(head_name)
            embed_executor = self.ctx.executors.get(embed_name)
            aligner_executor = self.ctx.executors.get(aligner_name)

            if not all((head_executor, embed_executor, aligner_executor)):
                missing = [n for n, e in [(head_name, head_executor), (embed_name, embed_executor), (aligner_name, aligner_executor)] if e is None]
                raise RuntimeError(f"ZERO FALLBACK: Missing VQ executors: {missing}")

            # CFG weight — MUST come from defaults (set by forge from model_registry.yml)
            cli_cfg = self.ctx.variable_resolver.get("global.guidance_scale", default=_SENTINEL)
            if cli_cfg is not _SENTINEL:
                cfg_weight = float(cli_cfg)
            else:
                cfg_weight = defaults.get("guidance_scale")
                if cfg_weight is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: guidance_scale missing from defaults for autoregressive_image.\n"
                        "Add to forge model_registry.yml: generation.guidance_scale"
                    )
                cfg_weight = float(cfg_weight)
            # VQ token offset
            lm_vocab_size = defaults.get("lm_vocab_size")
            codebook_size = defaults.get("codebook_size")
            if lm_vocab_size is None or codebook_size is None:
                raise RuntimeError(
                    "ZERO FALLBACK: lm_vocab_size or codebook_size not in defaults."
                )
            vq_token_offset = lm_vocab_size - codebook_size

            return ImageStrategy(
                head_executor=head_executor,
                embed_executor=embed_executor,
                aligner_executor=aligner_executor,
                decoder_executor=self.ctx.executors.get(gen_info.get("decoder_component", "gen_vision_model")),
                output_extractor=self._output_extractor,
                ensure_weights_fn=self._ensure_weights_loaded,
                cfg_weight=cfg_weight,
                vq_token_offset=vq_token_offset,
                defaults=defaults,
            )

        raise RuntimeError(f"ZERO FALLBACK: Unknown generation type '{gen_type}'.")

    def _tokenize(self, gen_info: Dict, device: str) -> torch.Tensor:
        """Tokenize prompt with optional CFG batching."""
        from neurobrix.core.module.text.processor import TextProcessor

        # Priority 0: Pre-tokenized input (from serving chat path).
        # Token IDs already include chat template + special tokens — no re-tokenization.
        _NO_PRETOKENIZED = object()
        pre_tokenized = self.ctx.variable_resolver.get("global.input_token_ids", default=_NO_PRETOKENIZED)
        if pre_tokenized is not _NO_PRETOKENIZED:
            if isinstance(pre_tokenized, list):
                input_ids = torch.tensor([pre_tokenized], dtype=torch.long, device=device)
            else:
                input_ids = pre_tokenized.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            return input_ids

        prompt = self.ctx.variable_resolver.get("global.prompt")
        if prompt is None:
            raise RuntimeError("ZERO FALLBACK: autoregressive_generation requires 'global.prompt' or 'global.input_token_ids'.")

        if "tokenizer" not in self.ctx.modules:
            raise RuntimeError("ZERO FALLBACK: autoregressive_generation requires 'tokenizer' module.")

        tp = TextProcessor(
            tokenizer=self.ctx.modules["tokenizer"],
            defaults=self.ctx.pkg.defaults,
            topology=self.ctx.pkg.topology,
            variable_resolver=self.ctx.variable_resolver,
        )

        gen_type = gen_info.get("type")
        defaults = self.ctx.pkg.defaults

        # CFG check (image only)
        use_cfg = False
        if gen_type == "autoregressive_image":
            cli_cfg = self.ctx.variable_resolver.get("global.guidance_scale", default=_SENTINEL)
            if cli_cfg is not _SENTINEL:
                cfg_weight = float(cli_cfg)
            else:
                cfg_weight = defaults.get("guidance_scale")
                if cfg_weight is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: guidance_scale missing from defaults.json for image model. "
                        "Add it to forge model_registry.yml and rebuild."
                    )
                cfg_weight = float(cfg_weight)
            use_cfg = cfg_weight > 1.0

        input_ids_cond = tp.tokenize(prompt, device, is_unconditional=False)

        if use_cfg:
            input_ids_uncond = tp.tokenize(prompt, device, is_unconditional=True)
            input_ids = torch.cat([input_ids_cond, input_ids_uncond], dim=0)
        else:
            input_ids = input_ids_cond

        return input_ids

    def _load_lm_head_weight(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Load lm_head weight for text LLM logits computation."""
        from neurobrix.core.io import WeightLoader

        loader = WeightLoader(str(self.ctx.pkg.root_path))
        weights = loader.load_component("lm_head", device, dtype)
        loader.close()

        lm_head_weight = weights.get("lm_head.weight") or weights.get("weight")
        if lm_head_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: lm_head.weight not found.\n"
                f"Available keys: {list(weights.keys())}"
            )

        return lm_head_weight

    def _unload_non_lm_weights(self, gen_info: Dict) -> None:
        """Unload component weights after generation."""
        comp_names = [
            gen_info.get("head_component", "gen_head"),
            gen_info.get("embed_component", "gen_embed"),
            gen_info.get("aligner_component", "gen_aligner"),
            gen_info.get("decoder_component", "gen_vision_model"),
        ]
        for comp_name in comp_names:
            if comp_name in self.ctx.executors:
                self._unload_component_weights(comp_name)

        gc.collect()
        torch.cuda.empty_cache()

    def _graph_lm_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate to active session. Required by runtime-guard hook."""
        if self._active_session is None:
            raise RuntimeError("ZERO FALLBACK: No active session for prefill.")
        return self._active_session.prefill(input_ids, input_ids.shape[0])

    def _graph_lm_decode_step(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegate to active session. Required by runtime-guard hook."""
        if self._active_session is None:
            raise RuntimeError("ZERO FALLBACK: No active session for decode.")
        return self._active_session.decode_step(input_ids, inputs_embeds=inputs_embeds)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-Level Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_generator_config(defaults: Dict, resolver: Any, gen_type: str) -> Dict:
    """Build generator config from defaults + CLI overrides."""
    image_size = defaults.get("image_size")
    patch_size = defaults.get("patch_size")
    codebook_size = defaults.get("codebook_size")
    codebook_dim = defaults.get("codebook_dim")
    max_tokens = defaults.get("max_tokens")

    if gen_type == "autoregressive_image":
        for required in ("image_size", "patch_size", "codebook_size", "codebook_dim"):
            if defaults.get(required) is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: {required} not found in defaults for autoregressive_image."
                )
        if max_tokens is None:
            assert image_size is not None and patch_size is not None  # guaranteed by check above
            max_tokens = (image_size // patch_size) ** 2

    config = {
        "_class_name": "VQImageGenerator" if gen_type == "autoregressive_image" else "AutoregressiveGenerator",
        "image_size": image_size,
        "patch_size": patch_size,
        "codebook_size": codebook_size,
        "codebook_dim": codebook_dim,
        "max_tokens": max_tokens,
        "temperature": defaults["temperature"],
        "top_p": defaults["top_p"],
        "top_k": defaults["top_k"],
        "repetition_penalty": defaults["repetition_penalty"],
        "eos_token_id": defaults.get("eos_token_id"),
        "pad_token_id": defaults.get("pad_token_id"),
    }

    # CLI overrides
    _CLI_OVERRIDES = [
        ("global.temperature", "temperature", float),
        ("global.repetition_penalty", "repetition_penalty", float),
        ("global.top_k", "top_k", int),
        ("global.top_p", "top_p", float),
        ("global.max_tokens", "max_tokens", int),
    ]
    for var_name, config_key, cast_fn in _CLI_OVERRIDES:
        cli_val = resolver.get(var_name, default=_SENTINEL)
        if cli_val is not _SENTINEL:
            config[config_key] = cast_fn(cli_val)

    return config


def _detect_int_arange_for_rope(dag: Dict) -> bool:
    """
    Detect if model needs absolute position_ids during KV cache decode.

    DATA-DRIVEN detection:
    - First arange op outputs float → model computes RoPE via float arange → relative pos OK
      (the arange interceptor shifts float aranges by cache_len)
    - First arange op outputs int64 → model indexes position table directly → absolute pos
    - NO arange ops → model uses pre-computed cos/sin tables indexed by position_ids
      → absolute pos (no aranges for the interceptor to shift)

    Handles: DeepSeek-V1 (float→relative), DeepSeek-V2 MLA (int→absolute),
             Janus (int→absolute), TinyLlama (no arange→absolute),
             Llama/Mistral (float→relative)
    """
    ops = dag.get("ops", {})
    tensors = dag.get("tensors", {})

    for _, op in ops.items():
        if op.get("op_type") == "aten::arange":
            out_tid = op.get("output_tensor_ids", [None])[0]
            if out_tid:
                t = tensors.get(out_tid, {})
                dtype = t.get("dtype", "")
                if "int" in dtype.lower():
                    return True
                return False
    # No arange ops found — model uses pre-computed RoPE tables indexed by
    # position_ids directly. Needs absolute position_ids since there are no
    # float aranges for the interceptor to shift.
    return True


def _get_graph_input_name(executor, fallback: str = "x") -> str:
    """Get expected input name from graph."""
    if not hasattr(executor, '_dag'):
        return fallback
    dag = executor._dag
    tensors = dag.get("tensors", {})
    for tdata in tensors.values():
        if tdata.get("is_input"):
            input_name = tdata.get("input_name")
            if input_name:
                return input_name
    return fallback
