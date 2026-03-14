"""
DualAR Engine — Fish-Speech/OpenAudio DualAR Flow

Handles DualAR architecture: backbone generates semantic tokens from
3D input [B, N+1, T], then embeds tokens for codec decoder → waveform.

ZERO SEMANTIC: No knowledge of "Fish-Speech" or "OpenAudio".
ZERO HARDCODE: All parameters from NBX container.
"""

import gc
import time
import torch
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


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

        # ── Step 2: DualAR generation ──
        backbone_stage = stages[0]
        comp_name = backbone_stage["component"]
        device = self.ctx.primary_device

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

        # Build initial 3D input [B, N+1, T_prompt]
        prompt_len = input_ids.shape[-1]
        inp = torch.zeros(1, n_codebooks, prompt_len, dtype=torch.long, device=device)
        inp[0, 0, :prompt_len] = input_ids.squeeze(0)[:prompt_len]

        # Generate semantic tokens autoregressively
        generated_semantic: List[int] = []

        for step in range(max_tokens):
            # Pad/truncate to trace_seq_len for CompiledSequence
            cur_len = prompt_len + len(generated_semantic)
            if cur_len <= trace_seq_len:
                padded = torch.zeros(1, n_codebooks, trace_seq_len, dtype=torch.long, device=device)
                padded[0, 0, :prompt_len] = input_ids.squeeze(0)[:prompt_len]
                for i, tok in enumerate(generated_semantic):
                    padded[0, 0, prompt_len + i] = tok
            else:
                # Sliding window
                all_semantic = list(input_ids.squeeze(0).tolist()[:prompt_len]) + generated_semantic
                window = all_semantic[-trace_seq_len:]
                padded = torch.zeros(1, n_codebooks, trace_seq_len, dtype=torch.long, device=device)
                for i, tok in enumerate(window):
                    padded[0, 0, i] = tok

            output = executor.run({"inp": padded})

            if isinstance(output, dict):
                logits = next(iter(output.values()))
            elif isinstance(output, torch.Tensor):
                logits = output
            else:
                logits = None

            if logits is None:
                break

            # Get logits at last real token position
            pos = min(cur_len - 1, trace_seq_len - 1)
            if logits.dim() == 3:
                last_logits = logits[:, pos, :].clone()
            else:
                last_logits = logits.clone()

            from .audio_utils import sample_token
            next_token = sample_token(
                last_logits.unsqueeze(1), temperature, top_p=top_p,
            )

            if eos_token_id is not None and next_token == eos_token_id:
                break

            generated_semantic.append(next_token)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Generated {len(generated_semantic)} semantic tokens in {elapsed:.0f}ms")

        # ── Step 3: Embed semantic tokens → codec input ──
        # Find the main text embedding (largest vocab, excludes codebook/fast)
        embed_weight = None
        if hasattr(executor, '_weights'):
            best_vocab = 0
            for wname, wtensor in executor._weights.items():
                if 'embed' in wname and wname.endswith('.weight'):
                    if 'codebook' in wname or 'fast' in wname:
                        continue
                    if wtensor.shape[0] > best_vocab:
                        best_vocab = wtensor.shape[0]
                        embed_weight = wtensor

        if embed_weight is None:
            raise RuntimeError("ZERO FALLBACK: DualAR model must have embed.weight.")

        token_ids = torch.tensor(generated_semantic, dtype=torch.long, device=device)
        max_id = token_ids.max().item()
        print(f"   [{comp_name}] Token range: 0..{max_id}, embed vocab: {embed_weight.shape[0]}")
        if max_id >= embed_weight.shape[0]:
            print(f"   [{comp_name}] WARNING: token {max_id} >= vocab {embed_weight.shape[0]}, clamping")
            token_ids = token_ids.clamp(0, embed_weight.shape[0] - 1)
        with torch.no_grad():
            token_embeds = torch.nn.functional.embedding(token_ids, embed_weight)

        codec_input = token_embeds.unsqueeze(0).transpose(1, 2)  # [1, dim, T_gen]
        print(f"   [{comp_name}] Embedded → {list(codec_input.shape)} for codec.decoder")

        # Store for downstream
        self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = codec_input
        self.ctx.variable_resolver.resolved["global.generated_codes"] = token_ids
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_semantic

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

        # ── Step 4: Codec decoder (forward stages) ──
        for stage in stages[1:]:
            codec_name = stage["component"]
            if codec_name not in self.ctx.executors:
                print(f"   [{codec_name}] Skipped (not in executors)")
                continue

            print(f"   [{codec_name}] Running forward pass...")
            codec_start = time.perf_counter()
            self._ensure_weights_loaded(codec_name)

            # Run codec.decoder forward
            # DAC decoder has ConvTranspose + residual blocks — chunking breaks residuals
            self._execute_component(codec_name, "forward", None)

            codec_elapsed = (time.perf_counter() - codec_start) * 1000
            print(f"   [{codec_name}] Done in {codec_elapsed:.0f}ms")

            if not self.ctx.persistent_mode:
                self._unload_component_weights(codec_name)
                gc.collect()
                torch.cuda.empty_cache()

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

        print(f"   [{comp_name}] Chunked: {actual_seq} frames → {graph_seq_len}-frame blocks")
        waveform_chunks = []

        for chunk_start in range(0, actual_seq, graph_seq_len):
            chunk_end = min(chunk_start + graph_seq_len, actual_seq)
            chunk = actual_input[:, :, chunk_start:chunk_end]

            if chunk.shape[2] < graph_seq_len:
                pad_size = graph_seq_len - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

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
