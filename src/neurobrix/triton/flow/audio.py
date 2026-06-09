"""Triton AudioEngine — zero torch orchestrator for audio flows.

Ported from core/flow/audio.py. Dispatches stage execution to
model-specific handlers. All tensor ops via NBXTensor + kernel wrappers.

Audio preprocessing (mel spectrogram, FFT) is a boundary operation:
it uses the AudioInputProcessor which internally uses torch/torchaudio.
Output is converted to NBXTensor at the boundary.

No torch imports in this file.
"""

import gc
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class TritonAudioEngine:
    """
    Triton-mode audio flow orchestrator.

    Reads flow.audio from topology.json and executes stages mechanically:
    1. Input preprocessing (audio->features or text->tokens)
    2. Stage execution (forward, native_kokoro, diffusion, native_acoustic_decoder)
    3. Output postprocessing (tokens->text or waveform->file)
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

    def _get_compute_dtype(self) -> str:
        """Return compute dtype as a STRING (e.g. "float16", "bfloat16").

        ZERO TORCH IN TRITON: this method MUST NOT import or return
        torch.dtype. Stage handlers that need a torch.dtype (accepted
        torch-boundary subroutines in core/flow/stages/) are responsible
        for converting the string to torch.dtype themselves.
        """
        return self.ctx.pkg.manifest.get("dtype", "float16")

    def execute(self) -> Dict[str, Any]:
        """Execute the audio pipeline from topology.json flow.audio."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio")
        if audio_config is None:
            raise RuntimeError(
                "ZERO FALLBACK: Audio flow requires topology.flow.audio section.\n"
                "Re-build the model with updated topology schema."
            )

        direction = audio_config.get("direction")
        if direction is None:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.direction is required ('stt' or 'tts')."
            )
        stages = audio_config.get("stages", [])
        if not stages:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.stages is empty.\n"
                "At least one stage is required."
            )

        # -- Step 1: Input preprocessing --
        input_config = audio_config.get("input", {})
        self._preprocess_input(input_config, direction, stages)

        # -- Step 2: Execute stages in order --
        for stage in stages:
            comp_name = stage["component"]
            execution = stage.get("execution", "forward")

            if comp_name not in self.ctx.executors:
                raise RuntimeError(
                    f"ZERO FALLBACK: Stage component '{comp_name}' not found in executors.\n"
                    f"Available: {list(self.ctx.executors.keys())}"
                )

            if execution == "forward":
                self._execute_forward_stage(stage)
            else:
                # The former native_kokoro / diffusion / native_acoustic_decoder
                # branches (which imported torch via core/flow/stages/) are gone:
                # Kokoro now runs its prosody-predictor + iSTFTNet decoder through
                # the graph in triton (all stages execution=forward), and
                # VibeVoice's diffusion + acoustic decoder run in the self-contained
                # zero-torch triton/flow/next_token_diffusion.py (flow type
                # next_token_diffusion, not audio). No flow=audio model reaches a
                # non-forward execution, so this is a hard ZERO FALLBACK — R33 is
                # restored for the triton audio stage loop (no core/flow/stages import).
                raise RuntimeError(
                    f"ZERO FALLBACK: triton audio flow supports only "
                    f"execution=forward (got '{execution}' for stage "
                    f"'{comp_name}'). Diffusion/acoustic-decoder TTS routes through "
                    f"the next_token_diffusion flow; Kokoro routes through the graph."
                )

        # -- Step 3: Output postprocessing --
        output_config = audio_config.get("output", {})
        self._postprocess_output(output_config, direction)

        return self.ctx.variable_resolver.resolve_all()

    # -----------------------------------------------------------------
    # Input preprocessing
    # -----------------------------------------------------------------

    def _preprocess_input(self, input_config: Dict, direction: str, stages: List) -> None:
        """Preprocess input based on modality and preprocessing type."""
        modality = input_config.get("modality", "audio" if direction == "stt" else "text")

        if modality == "audio":
            self._preprocess_audio_input(input_config, stages)
        elif modality == "text":
            self._preprocess_text_input(input_config)

    def _preprocess_audio_input(self, input_config: Dict, stages: List) -> None:
        """Load audio file and extract mel/raw features — ZERO TORCH (R33).

        Delegates to the numpy/NBX front-end in ``triton.audio_frontend``: it
        loads the waveform, runs the family-specific extractor (whisper/nemo/
        conformer/raw), pads/truncates to the trace dims and binds the result
        as ``NBXTensor`` on the encoder device. No torch / torchaudio anywhere
        on the triton compute path.
        """
        from neurobrix.triton.audio_frontend import preprocess_audio_input_np
        preprocess_audio_input_np(self.ctx, {"input": input_config}, stages)

    def _preprocess_text_input(self, input_config: Dict) -> None:
        """Tokenize text prompt for TTS/LLM-audio models."""
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None:
            raise RuntimeError(
                "ZERO FALLBACK: TTS model requires global.prompt.\n"
                "Use --prompt <text> to provide text input."
            )

        tts_template = self.ctx.pkg.defaults.get("tts_prompt_template")
        if tts_template and "{text}" in tts_template:
            prompt = tts_template.format(text=prompt)

        tokenizer = self.ctx.modules.get("tokenizer")

        # Phonemizer path (zero-torch: phonemizer/espeak g2p + numpy voicepack).
        phoneme_vocab = self.ctx.pkg.defaults.get("phoneme_vocab")
        if tokenizer is None and phoneme_vocab:
            from neurobrix.triton.audio_frontend import preprocess_phonemizer_input_np
            preprocess_phonemizer_input_np(self, prompt, phoneme_vocab)
            return

        if tokenizer is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

        # Detect tokenization style
        tokenization = input_config.get("tokenization", "auto")
        if tokenization == "auto":
            has_lm = any(
                k in self.ctx.executors or any(
                    ek.endswith(f".{k}") or ek == k for ek in self.ctx.executors
                )
                for k in ["language_model", "model", "lm_head"]
            )
            if not has_lm:
                flow = self.ctx.pkg.topology.get("flow", {})
                audio_stages = flow.get("audio", {}).get("stages", [])
                has_lm = any(s.get("execution") == "autoregressive" for s in audio_stages)
            tokenization = "llm" if has_lm else "diffusion"

        device = self.ctx.primary_device

        if tokenization == "llm":
            add_special = tts_template is None
            try:
                ids = tokenizer.encode(prompt, add_special_tokens=add_special)
            except TypeError:
                ids = tokenizer.encode(prompt)
            if not isinstance(ids, list):
                ids = list(ids)
            input_ids_np = np.array([ids], dtype=np.int64)
            input_ids = NBXTensor.from_numpy(input_ids_np)
            attention_mask_np = np.ones_like(input_ids_np)
            attention_mask = NBXTensor.from_numpy(attention_mask_np)
        else:
            from neurobrix.core.module.text.processor import TextProcessor
            tp = TextProcessor(
                tokenizer=tokenizer,
                defaults=self.ctx.pkg.defaults,
                topology=self.ctx.pkg.topology,
                variable_resolver=self.ctx.variable_resolver,
            )
            input_ids_torch, attention_mask_torch = tp.tokenize_for_diffusion(prompt, device)
            input_ids = _torch_to_nbx(input_ids_torch)
            attention_mask = _torch_to_nbx(attention_mask_torch) if attention_mask_torch is not None else None

        self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
        self.ctx.variable_resolver.resolved["input_ids"] = input_ids
        if attention_mask is not None:
            self.ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
            self.ctx.variable_resolver.resolved["attention_mask"] = attention_mask

    # -----------------------------------------------------------------
    # Stage execution
    # -----------------------------------------------------------------

    def _execute_forward_stage(self, stage: Dict) -> None:
        """Execute a single forward-pass stage."""
        comp_name = stage["component"]

        # Check if required inputs are available AND are tensors
        has_tensor_input = False
        comp_connections = self.ctx.connections_index.get(comp_name, {})
        if comp_connections:
            for _input_name, sources in comp_connections.items():
                for src in sources:
                    val = self.ctx.variable_resolver.resolved.get(src)
                    if isinstance(val, (NBXTensor,)) or _is_tensor(val):
                        has_tensor_input = True
                        break
                if has_tensor_input:
                    break
        else:
            executor = self.ctx.executors.get(comp_name)
            dag = getattr(executor, '_dag', None) if executor else None
            if dag:
                for _tid, spec in dag.get("tensors", {}).items():
                    iname = spec.get("input_name")
                    if iname:
                        for key in [iname, f"global.{iname}", f"{comp_name}.{iname}"]:
                            val = self.ctx.variable_resolver.resolved.get(key)
                            if isinstance(val, (NBXTensor,)) or _is_tensor(val):
                                has_tensor_input = True
                                break
                    if has_tensor_input:
                        break
        if not has_tensor_input:
            print(f"   [{comp_name}] Skipped (no tensor inputs available)")
            return

        print(f"   [{comp_name}] Running forward pass...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)
        # Fixed-length decoders (codec / iSTFT vocoder: the window-norm divisor and
        # the as_strided framing are baked at the trace frame count) must run at
        # EXACTLY the graph seq_len. Chunk a longer runtime input into trace-length
        # blocks — triton mirror of compiled AudioFlow._try_chunked_forward.
        # Data-driven: triggers only when the runtime 3D input seq_len differs from
        # the graph's trace seq_len; otherwise the normal single pass runs.
        if not self._try_chunked_forward(comp_name):
            self._execute_component(comp_name, "forward", None)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Done in {elapsed:.0f}ms")

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()

    def _try_chunked_forward(self, comp_name: str) -> bool:
        """Run a fixed-length decoder in trace-length chunks (triton-pure mirror of
        compiled AudioFlow._try_chunked_forward).

        The graph expects [1, C, graph_seq_len]; the runtime input may be longer
        (or shorter). The iSTFT window-norm divisor + as_strided framing are baked
        at graph_seq_len, so each block must run at EXACTLY that length. The primary
        frame input is chunked; frame-dependent aux inputs are chunked synchronously
        (detected by runtime dim != graph dim); static inputs (style) pass whole.
        Waveform chunks are concatenated. Returns True if chunking ran.
        R33-pure: NBXTensor narrow/zeros/cat only.
        """
        executor = self.ctx.executors.get(comp_name)
        dag = getattr(executor, '_dag', None) if executor else None
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

        resolved = self.ctx.variable_resolver.resolved
        connections_all = self.ctx.pkg.topology.get("connections", [])

        def _resolve(iname):
            # Accept NBXTensor (frame inputs from upstream triton components) AND
            # torch boundary tensors (e.g. the voicepack style vector) — convert
            # the latter to NBXTensor so all chunking stays NBX-pure.
            for c in connections_all:
                if c.get("to") == f"{comp_name}.{iname}":
                    v = resolved.get(c.get("from"))
                    if _is_tensor(v):
                        return _torch_to_nbx(v)
            for key in (f"global.{iname}", iname, f"{comp_name}.{iname}"):
                v = resolved.get(key)
                if _is_tensor(v):
                    return _torch_to_nbx(v)
            return None

        actual_input = _resolve(graph_input_name)
        if actual_input is None or actual_input.ndim != 3:
            return False
        actual_seq = actual_input.shape[2]
        if actual_seq == graph_seq_len:
            return False  # exact trace length → single pass

        # Frame-dependent aux inputs (runtime dim != graph dim) chunk synchronously;
        # static ones pass whole.
        aux_inputs = {}
        for _tid2, spec2 in dag.get("tensors", {}).items():
            iname2 = spec2.get("input_name")
            if not iname2 or iname2 == graph_input_name:
                continue
            val2 = _resolve(iname2)
            if val2 is None:
                continue
            cdim2 = glen2 = None
            gshape2 = spec2.get("shape", [])
            for d in range(min(len(gshape2), val2.ndim)):
                gd = gshape2[d]
                if isinstance(gd, dict):
                    gd = gd.get("trace_value", gd)
                if isinstance(gd, int) and val2.shape[d] != gd:
                    cdim2, glen2 = d, gd
                    break
            aux_inputs[iname2] = (val2, cdim2, glen2)

        def _pad_to(t, dim, length):
            if t.shape[dim] >= length:
                return t
            pad_shape = list(t.shape)
            pad_shape[dim] = length - t.shape[dim]
            z = NBXTensor.zeros(tuple(pad_shape), dtype=t.dtype, device=t.device)
            return NBXTensor.cat([t, z], dim=dim)

        print(f"   [{comp_name}] Chunked: {actual_seq} frames -> {graph_seq_len}-frame blocks")
        waveform_chunks = []
        for chunk_start in range(0, actual_seq, graph_seq_len):
            chunk_end = min(chunk_start + graph_seq_len, actual_seq)
            chunk = actual_input.narrow(2, chunk_start, chunk_end - chunk_start)
            chunk = _pad_to(chunk, 2, graph_seq_len)

            comp_inputs = {graph_input_name: chunk}
            block_idx = chunk_start // graph_seq_len
            for iname2, (val2, cdim2, glen2) in aux_inputs.items():
                if cdim2 is None:
                    comp_inputs[iname2] = val2
                else:
                    s2 = block_idx * glen2
                    take = max(0, min(glen2, val2.shape[cdim2] - s2))
                    c2 = val2.narrow(cdim2, s2, take) if take > 0 else val2.narrow(cdim2, 0, 0)
                    comp_inputs[iname2] = _pad_to(c2, cdim2, glen2)

            output = executor.run(comp_inputs)
            if isinstance(output, dict):
                out_tensor = next(iter(output.values()), None)
            elif isinstance(output, NBXTensor):
                out_tensor = output
            else:
                out_tensor = None
            if out_tensor is None:
                continue
            # Trim a padded last chunk proportionally.
            if chunk_end - chunk_start < graph_seq_len and out_tensor.ndim >= 2:
                ratio = (chunk_end - chunk_start) / graph_seq_len
                last = out_tensor.ndim - 1
                trim_len = int(out_tensor.shape[last] * ratio)
                out_tensor = out_tensor.narrow(last, 0, trim_len)
            waveform_chunks.append(out_tensor.contiguous())

        if not waveform_chunks:
            return False
        last = waveform_chunks[0].ndim - 1
        full_output = (waveform_chunks[0] if len(waveform_chunks) == 1
                       else NBXTensor.cat(waveform_chunks, dim=last).contiguous())
        resolved[f"{comp_name}.output_0"] = full_output
        resolved["global.output_audio"] = full_output
        print(f"   [{comp_name}] Waveform: {list(full_output.shape)}")
        return True

    # -----------------------------------------------------------------
    # Output postprocessing
    # -----------------------------------------------------------------

    def _postprocess_output(self, output_config: Dict, direction: str) -> None:
        """Postprocess output based on modality."""
        modality = output_config.get("modality", "text" if direction == "stt" else "audio")

        if modality == "text":
            self._postprocess_text_output(output_config)
        elif modality == "audio":
            self._postprocess_audio_output(output_config)

    def _postprocess_text_output(self, output_config: Dict) -> None:
        """Decode generated token IDs to text (STT)."""
        generated_ids = self.ctx.variable_resolver.resolved.get("global.generated_token_ids")
        if generated_ids is None:
            return

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
        else:
            text = str(generated_ids)

        variable = output_config.get("variable", "global.transcription")
        self.ctx.variable_resolver.resolved[variable] = text
        self.ctx.variable_resolver.resolved["global.transcription"] = text
        print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")

    def _postprocess_audio_output(self, output_config: Dict) -> None:
        """Process TTS output: decode audio tokens or store raw waveform."""
        # Delegate to shared audio_utils (these work with both torch and NBXTensor)
        from neurobrix.core.flow.audio_utils import postprocess_audio_output
        postprocess_audio_output(self.ctx)


# -----------------------------------------------------------------
# Module-level helpers (no torch imports)
# -----------------------------------------------------------------

def _is_tensor(val) -> bool:
    """Check if val is any tensor type (NBXTensor or torch.Tensor)."""
    if isinstance(val, NBXTensor):
        return True
    return hasattr(val, 'shape') and hasattr(val, 'dtype') and hasattr(val, 'device')


def _torch_to_nbx(tensor) -> NBXTensor:
    """Convert torch.Tensor to NBXTensor at the boundary.

    This is the ONLY place torch is used — at the preprocessing boundary.
    """
    if isinstance(tensor, NBXTensor):
        return tensor
    arr = tensor.detach().cpu().numpy()
    return NBXTensor.from_numpy(arr)


def _get_component_input_shape(ctx, comp_name) -> Optional[Tuple[int, ...]]:
    """Read first input tensor shape from component's graph (DATA-DRIVEN)."""
    if comp_name is None:
        return None
    executor = ctx.executors.get(comp_name)
    if executor is None:
        return None
    dag = getattr(executor, '_dag', None)
    if dag is None:
        return None
    for tid, spec in dag.get("tensors", {}).items():
        is_input = (
            spec.get("type") == "input"
            or spec.get("input_name") is not None
            or tid.startswith("input::")
        )
        if is_input:
            shape = spec.get("shape", [])
            resolved = []
            for dim in shape:
                if isinstance(dim, dict):
                    resolved.append(dim.get("trace_value", dim.get("trace", 0)))
                elif isinstance(dim, int):
                    resolved.append(dim)
                else:
                    resolved.append(0)
            return tuple(resolved)
    return None


def _find_model_config_path(ctx) -> Path:
    """Find model config path from NBX container."""
    nbx_path = Path(ctx.nbx_path_str)
    for subdir in ["modules/processor", "modules/tokenizer"]:
        candidate = nbx_path / subdir
        if candidate.exists():
            return candidate
    # Removed: legacy absolute-path fallback. The field held a
    # trace-host snapshot location and is no longer present in
    # correctly-built containers.
    raise RuntimeError(
        "Cannot find model config path. Expected `modules/processor/` "
        "or `modules/tokenizer/` inside the .nbx."
    )
