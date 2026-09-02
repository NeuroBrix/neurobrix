"""Triton RNNTEngine — zero torch RNNT transducer flow.

Ported from core/flow/rnnt.py. Handles RNNT/TDT models (Parakeet, etc.)
with greedy frame-by-frame decoding.

Architecture:
    encoder(audio_features) -> enc_out [B, D_enc, T]
    Greedy loop over T frames:
        decoder(prev_tokens) -> dec_out [B, D_dec, U]
        joint(enc_frame, dec_frame) -> logits [B, 1, 1, V]
        token = argmax(logits)

Audio preprocessing is a BOUNDARY operation (uses torch/torchaudio).
LSTM decoder + joint network run as native NumPy ops since their
tiny graphs have trace-time shapes incompatible with greedy decoding.

No torch imports in hot path.
"""

import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.triton.memory_pool import release_flow_memory
from neurobrix.triton.device_transfer import parse_device_idx


class TritonRNNTEngine:
    """
    Triton-mode RNNT transducer flow: encoder -> greedy decode (decoder + joint per frame).

    The encoder runs through the triton compiled sequence normally.
    The decoder (embedding + LSTM) and joint (linear + relu + linear) are
    executed with native NumPy ops using weights extracted from the executor,
    since their tiny graphs have trace-time shapes that don't match the
    variable runtime shapes of greedy decoding.
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
        """Execute RNNT pipeline: preprocess -> encoder -> greedy decode -> text."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})

        # Step 1: Audio preprocessing (stashes long-form windows —
        # D-STT-LONGFORM-CHUNKING, R30 mirror of the compiled flow)
        self._preprocess_audio(audio_config)
        _windows = getattr(self, "_stt_windows", None) or [None]
        _n_win = len(_windows)
        if _n_win > 1:
            print(f"   [Audio] Long-form: {_n_win} windows")

        tokens: list = []
        for _wi in range(_n_win):
            if _windows[_wi] is not None and _wi > 0:
                self._bind_window(*_windows[_wi])

            # Step 2: Run encoder
            start = time.perf_counter()
            self._ensure_weights_loaded("encoder")
            self._execute_component("encoder", "forward", None)
            enc_output = self._get_component_output("encoder")
            enc_ms = (time.perf_counter() - start) * 1000
            print(f"   [encoder] Done in {enc_ms:.0f}ms, output shape {enc_output.shape}"
                  + (f" (window {_wi + 1}/{_n_win})" if _n_win > 1 else ""))

            # Step 3: RNNT greedy decode (numpy-based for variable shapes)
            start = time.perf_counter()
            self._ensure_weights_loaded("decoder")
            self._ensure_weights_loaded("joint")

            _wtok = self._rnnt_greedy_decode(enc_output)
            if _n_win > 1:
                # Buffered-inference merge (stt_longform.rnnt_merge_window,
                # mirror of the compiled flow): tokens kept by emission
                # frame; the overlap in encoder frames is measured from
                # THIS window's encoder output against its mel span.
                from neurobrix.core.module.audio.stt_longform import rnnt_merge_window
                _w_enc = int(self._enc_frames)   # the FRAME axis the decode walked
                _ov_enc = int(round(self._stt_overlap_mel * _w_enc / self._stt_window_mel))
                _wtok = rnnt_merge_window(_wtok, self._token_frames, _wi, _n_win,
                                          _w_enc, _ov_enc)
            tokens.extend(_wtok)

            dec_ms = (time.perf_counter() - start) * 1000
            print(f"   [rnnt_decode] {len(tokens)} tokens in {dec_ms:.0f}ms")

        if not self.ctx.persistent_mode:
            self._unload_component_weights("encoder")
            self._unload_component_weights("decoder")
            self._unload_component_weights("joint")
            release_flow_memory(self.ctx.primary_device)

        # Step 4: Decode tokens to text
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = tokens
        text = self._decode_tokens(tokens)
        self.ctx.variable_resolver.resolved["global.transcription"] = text

        print(f"\n[Transcription]\n{text}")
        return self.ctx.variable_resolver.resolve_all()

    # -----------------------------------------------------------------
    # Audio preprocessing (BOUNDARY — torch used internally)
    # -----------------------------------------------------------------

    def _preprocess_audio(self, audio_config: Dict) -> None:
        """Load and preprocess audio to NeMo-compatible mel features.

        This is a BOUNDARY operation. Uses torch/torchaudio for mel spectrogram
        computation, then converts to NBXTensor.
        """
        audio_path = self.ctx.variable_resolver.resolved.get("global.audio_path")
        if audio_path is None:
            raise RuntimeError("ZERO FALLBACK: No audio_path provided.")

        device_idx = parse_device_idx(self.ctx.primary_device)
        DeviceAllocator.set_device(device_idx)

        print(f"   [Audio] Loading: {audio_path}")

        # NeMo mel in PURE NUMPY (zero torch) — bit-close mirror of the
        # torchaudio path (validated vs torch; STT-confirmed). The extractor
        # loads + resamples the audio itself.
        from neurobrix.triton.audio_frontend import _nemo_mel, _model_config_path
        _mp = _model_config_path(self.ctx)   # raises like the compiled mirror
        features_np = _nemo_mel(str(audio_path), _mp, None)   # [1, n_mels, frames]
        actual_frames = features_np.shape[2]
        print(f"   [Audio] Features: {features_np.shape} (nemo_mel·np, {actual_frames} frames)")

        # Pad/truncate to the encoder's traced frame count (numpy).
        expected_frames = 3000
        _dag_frames = None
        encoder_executor = self.ctx.executors.get("encoder")
        if encoder_executor and hasattr(encoder_executor, '_dag'):
            for tid in encoder_executor._dag.get("input_tensor_ids", []):
                tdata = encoder_executor._dag.get("tensors", {}).get(tid, {})
                shape = tdata.get("shape")
                if shape and len(shape) == 3:
                    expected_frames = _dag_frames = shape[2]
                    break
        # Long-form (mirror of the compiled rnnt flow): buffered
        # inference — overlapping windows (overlap from the family YAML
        # in seconds, converted with the extractor's own hop), tokens
        # merged by emission frame (stt_longform). Whole-utterance mel
        # stats, last window padded. Window length = the traced span
        # (D-PARAKEET-SYMBOLIC-T). NBX_DISABLE_STT_CHUNKING=1 = old path.
        import os as _os_sw
        self._stt_windows = None
        self._stt_window_mel = expected_frames
        self._stt_overlap_mel = 0
        if (actual_frames > expected_frames
                and _os_sw.environ.get("NBX_DISABLE_STT_CHUNKING") != "1"):
            from neurobrix.core.config.loader import get_family_config
            from neurobrix.core.module.audio.mel_dsp import nemo_mel_params
            from neurobrix.core.module.audio.stt_longform import rnnt_window_plan
            _lf = get_family_config("stt").get("long_form") or {}
            if "rnnt_overlap_seconds" not in _lf:
                raise RuntimeError(
                    "ZERO FALLBACK: stt.yml long_form.rnnt_overlap_seconds missing.")
            _mp_params = nemo_mel_params(_mp)
            if _dag_frames is None:
                raise RuntimeError(
                    "ZERO FALLBACK: long-form RNNT needs the encoder's traced frame "
                    "span from the DAG (no 3-D encoder input found).")
            if not _mp_params.get("source"):
                # Same hop/rate the mel was computed with (mirror of the
                # compiled flow): built-in NeMo values on a legacy build,
                # announced, never silent (D-PARAKEET-SYMBOLIC-T).
                print(f"   [Audio] NeMo preprocessor params: built-in values "
                      f"(no embedded config in this build) — "
                      f"sr={_mp_params['sr']} hop={_mp_params['hop']}")
            self._stt_overlap_mel = int(round(
                float(_lf["rnnt_overlap_seconds"]) * _mp_params["sr"] / _mp_params["hop"]))
            wins = []
            for _start, _valid in rnnt_window_plan(actual_frames, expected_frames,
                                                   self._stt_overlap_mel):
                f = features_np[:, :, _start:_start + _valid]
                if _valid < expected_frames:
                    f = np.concatenate(
                        [f, np.zeros((1, f.shape[1], expected_frames - _valid), np.float32)], axis=2)
                wins.append((NBXTensor.from_numpy(np.ascontiguousarray(f)),
                             NBXTensor.from_numpy(np.array([_valid], dtype=np.int64))))
            self._stt_windows = wins
            features_nbx, length_nbx = wins[0]
        else:
            if actual_frames < expected_frames:
                features_np = np.concatenate(
                    [features_np, np.zeros((1, features_np.shape[1],
                                            expected_frames - actual_frames), np.float32)], axis=2)
            elif actual_frames > expected_frames:
                features_np = features_np[:, :, :expected_frames]
            features_nbx = NBXTensor.from_numpy(np.ascontiguousarray(features_np))
            length_nbx = NBXTensor.from_numpy(
                np.array([min(actual_frames, expected_frames)], dtype=np.int64))

        self._bind_window(features_nbx, length_nbx)

    def _bind_window(self, features_nbx, length_nbx) -> None:
        # Bind under all possible keys
        for key in ["global.audio_signal", "global.input_features", "audio_signal",
                     "input::audio_signal", "encoder.audio_signal"]:
            self.ctx.variable_resolver.resolved[key] = features_nbx
        for key in ["global.length", "length", "input::length", "encoder.length"]:
            self.ctx.variable_resolver.resolved[key] = length_nbx

    # -----------------------------------------------------------------
    # RNNT greedy decode (numpy-based for variable shapes)
    # -----------------------------------------------------------------

    def _rnnt_greedy_decode(self, enc_output) -> List[int]:
        """
        RNNT greedy decoding with NumPy ops for decoder + joint.

        The decoder and joint are tiny networks whose traced graphs have
        hardcoded shapes. We extract their weights and run with NumPy
        to handle variable-length greedy decoding.
        """
        # Get encoder output as numpy
        enc_np = _to_numpy(enc_output)
        # enc_np is [B, D_enc, T_padded] -> transpose to [B, T_padded, D_enc]
        enc_out = np.transpose(enc_np, (0, 2, 1))
        T_padded = enc_out.shape[1]
        self._enc_frames = int(T_padded)   # encoder frame axis (long-form merge)

        # Compute actual encoder output length
        input_length = self.ctx.variable_resolver.resolved.get("global.length")
        if input_length is not None:
            actual_frames = int(_to_numpy(input_length).flat[0])
            sub_factor = self.ctx.pkg.defaults.get("subsampling_factor", 8)
            T = min((actual_frames + sub_factor - 1) // sub_factor, T_padded)
        else:
            T = T_padded

        # Extract weights to numpy
        dec_weights = self._extract_decoder_weights()
        joint_weights = self._extract_joint_weights()

        # TDT config
        defaults = self.ctx.pkg.defaults
        num_tdt_durations = defaults.get("num_tdt_durations", 1)
        blank_id = defaults.get("blank_id", 1024)
        vocab_size = defaults.get("vocab_size", 1024)

        # Initialize
        tokens: List[int] = []
        self._token_frames: List[int] = []
        blank_token = blank_id
        last_token = blank_token

        num_layers = dec_weights["num_layers"]
        hidden_size = dec_weights["hidden_size"]
        h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

        # NeMo's greedy loop, mirror of the compiled flow (see there):
        # g reused until a non-blank commits the state; the duration
        # head's skip advances time after every symbol.
        durations = defaults.get("tdt_durations")
        if num_tdt_durations > 1 and durations is None:
            durations = list(range(num_tdt_durations))
            print(f"   [rnnt_decode] TDT durations: built-in consecutive "
                  f"{durations} (not carried by this build)")
        max_symbols_per_step = int(defaults.get("max_symbols_per_step", 10))
        t = 0
        g = None
        h_next = c_next = None
        embed_weight = dec_weights["embedding"]
        while t < T:
            symbols_added = 0
            need_loop = True
            skip = 1
            while need_loop and symbols_added < max_symbols_per_step:
                if g is None:
                    dec_embed = embed_weight[last_token:last_token + 1]  # [1, D_dec]
                    dec_input = dec_embed[np.newaxis, :, :]  # [1, 1, D_dec]
                    dec_rnn_out, (h_next, c_next) = self._run_lstm_np(
                        dec_input, (h, c),
                        dec_weights["weight_ih"], dec_weights["weight_hh"],
                        dec_weights["bias_ih"], dec_weights["bias_hh"],
                        num_layers,
                    )
                    g = dec_rnn_out[0]  # [1, D_dec]
                enc_frame = enc_out[0:1, t, :]  # [1, D_enc]
                logits = self._run_joint_np(enc_frame, g, joint_weights)
                if num_tdt_durations > 1:
                    token_logits = logits[:vocab_size + 1]
                    skip = durations[int(np.argmax(logits[vocab_size + 1:]))]
                else:
                    token_logits = logits
                pred_token = int(np.argmax(token_logits))
                if pred_token != blank_id:
                    tokens.append(pred_token)
                    self._token_frames.append(t)   # encoder frame per token (long-form merge)
                    h, c = h_next, c_next
                    last_token = pred_token
                    g = None
                    if num_tdt_durations <= 1:
                        skip = 0
                elif num_tdt_durations <= 1:
                    skip = 1
                symbols_added += 1
                t += skip
                need_loop = skip == 0
            if symbols_added == max_symbols_per_step:
                t += 1
        return tokens

    def _run_lstm_np(self, input_seq, hx, weight_ih, weight_hh, bias_ih, bias_hh, num_layers):
        """Run the RNNT decoder LSTM through the triton-pure `lstm_wrapper`
        (NBXTensor + @triton.jit) instead of a hand-rolled NumPy cell — removing
        the NumPy compute debt (the prior `_lstm_cell_np`). The greedy loop stays
        in Python (data-dependent control flow); only the single-step input and
        h/c cross the NBXTensor boundary per call, and the weights are converted
        once and cached. Unidirectional, multi-layer, batch_first=False — the same
        kernel validated bit-exact for the Kokoro BiLSTMs.

        input_seq: [T, 1, D] numpy (T == 1 per RNNT step); hx = (h, c) numpy
        [num_layers, 1, H]. Returns (output [T,1,H] numpy, (h_n, c_n) numpy).
        """
        from neurobrix.kernels import wrappers as _w
        from neurobrix.kernels.nbx_tensor import NBXTensor

        params = getattr(self, "_dec_lstm_params_nbx", None)
        if params is None:
            params = []
            for layer in range(num_layers):
                params.append(NBXTensor.from_numpy(np.ascontiguousarray(weight_ih[layer], dtype=np.float32)))
                params.append(NBXTensor.from_numpy(np.ascontiguousarray(weight_hh[layer], dtype=np.float32)))
                params.append(NBXTensor.from_numpy(np.ascontiguousarray(bias_ih[layer], dtype=np.float32)))
                params.append(NBXTensor.from_numpy(np.ascontiguousarray(bias_hh[layer], dtype=np.float32)))
            self._dec_lstm_params_nbx = params

        x = NBXTensor.from_numpy(np.ascontiguousarray(input_seq, dtype=np.float32))
        h0 = NBXTensor.from_numpy(np.ascontiguousarray(hx[0], dtype=np.float32))
        c0 = NBXTensor.from_numpy(np.ascontiguousarray(hx[1], dtype=np.float32))
        out, h_n, c_n = _w.lstm_wrapper(
            x, [h0, c0], params, has_biases=True, num_layers=num_layers,
            bidirectional=False, batch_first=False,
        )
        return out.numpy(), (h_n.numpy(), c_n.numpy())

    def _run_joint_np(self, enc_frame, dec_frame, jw):
        """Run joint network with NumPy: project enc + dec, add, relu, output linear."""
        enc_proj = enc_frame @ jw["enc_weight"].T + jw["enc_bias"]
        dec_proj = dec_frame @ jw["dec_weight"].T + jw["dec_bias"]
        combined = np.maximum(enc_proj + dec_proj, 0)  # relu
        logits = combined @ jw["out_weight"].T + jw["out_bias"]
        return logits[0]

    # -----------------------------------------------------------------
    # Weight extraction
    # -----------------------------------------------------------------

    def _extract_decoder_weights(self) -> Dict[str, Any]:
        """Extract decoder weights (embedding + LSTM) and convert to numpy."""
        executor = self.ctx.executors["decoder"]
        w = executor._weights

        embedding = None
        weight_ih = []
        weight_hh = []
        bias_ih = []
        bias_hh = []

        for key, tensor in w.items():
            if tensor is None:
                continue
            k = key.lower()
            arr = _to_numpy_f32(tensor)
            if "embed" in k and arr.ndim == 2:
                embedding = arr
            elif "weight_ih_l" in k:
                layer = int(k.split("weight_ih_l")[-1])
                weight_ih.append((layer, arr))
            elif "weight_hh_l" in k:
                layer = int(k.split("weight_hh_l")[-1])
                weight_hh.append((layer, arr))
            elif "bias_ih_l" in k:
                layer = int(k.split("bias_ih_l")[-1])
                bias_ih.append((layer, arr))
            elif "bias_hh_l" in k:
                layer = int(k.split("bias_hh_l")[-1])
                bias_hh.append((layer, arr))

        if embedding is None:
            raise RuntimeError("ZERO FALLBACK: Decoder embedding weight not found")

        return {
            "embedding": embedding,
            "weight_ih": [t for _, t in sorted(weight_ih)],
            "weight_hh": [t for _, t in sorted(weight_hh)],
            "bias_ih": [t for _, t in sorted(bias_ih)],
            "bias_hh": [t for _, t in sorted(bias_hh)],
            "num_layers": len(weight_ih),
            "hidden_size": embedding.shape[1],
        }

    def _extract_joint_weights(self) -> Dict[str, Any]:
        """Extract joint network weights and convert to numpy."""
        executor = self.ctx.executors["joint"]
        w = executor._weights

        result = {}
        for key, tensor in w.items():
            if tensor is None:
                continue
            arr = _to_numpy_f32(tensor)
            if key.endswith("enc.weight"):
                result["enc_weight"] = arr
            elif key.endswith("enc.bias"):
                result["enc_bias"] = arr
            elif key.endswith("pred.weight"):
                result["dec_weight"] = arr
            elif key.endswith("pred.bias"):
                result["dec_bias"] = arr
            elif key.endswith("joint.2.weight"):
                result["out_weight"] = arr
            elif key.endswith("joint.2.bias"):
                result["out_bias"] = arr

        required = ["enc_weight", "enc_bias", "dec_weight", "dec_bias", "out_weight", "out_bias"]
        missing = [k for k in required if k not in result]
        if missing:
            raise RuntimeError(
                f"ZERO FALLBACK: Joint network missing weights: {missing}\n"
                f"Available: {list(w.keys())}"
            )
        return result

    # -----------------------------------------------------------------
    # Output utilities
    # -----------------------------------------------------------------

    def _get_component_output(self, comp_name: str):
        """Get the primary output tensor from a component."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.output"]:
            if key in resolved:
                return resolved[key]
        executor = self.ctx.executors.get(comp_name)
        if executor and hasattr(executor, '_last_outputs') and executor._last_outputs:
            return executor._last_outputs[0]
        raise RuntimeError(f"ZERO FALLBACK: No output found for {comp_name}")

    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode token IDs to text using sentencepiece tokenizer from NBX cache."""
        sp_path = self._find_tokenizer_model()
        if sp_path is not None:
            try:
                # R34 (Zero Outsider): NeuroBrix-internal SentencePiece, never
                # the `sentencepiece` vendor lib.
                from neurobrix.core.module.tokenizer.sp_proto import PySentencePiece
                with open(sp_path, "rb") as _spf:
                    sp = PySentencePiece.from_bytes(_spf.read())
                return sp.decode(tokens)
            except ImportError:
                pass

        cache_path = self.ctx.pkg.cache_path
        if cache_path is not None:
            vocab_paths = list(cache_path.rglob("*vocab.txt"))
            if vocab_paths:
                with open(vocab_paths[0]) as f:
                    vocab = [line.strip() for line in f]
                chars = [vocab[t] if t < len(vocab) else f"<{t}>" for t in tokens]
                return "".join(chars).replace("\u2581", " ").strip()

        raise RuntimeError(
            "ZERO FALLBACK: No sentencepiece tokenizer.model or vocab.txt in NBX cache.\n"
            "The model needs to be rebuilt with the tokenizer included."
        )

    def _find_tokenizer_model(self) -> Optional[Path]:
        """Find sentencepiece tokenizer.model in NBX cache."""
        cache_path = self.ctx.pkg.cache_path
        if cache_path is None:
            return None
        sp_paths = list(cache_path.rglob("*.model"))
        return sp_paths[0] if sp_paths else None


# -----------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------

def _to_numpy(tensor) -> np.ndarray:
    """Convert any tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _to_numpy_f32(tensor) -> np.ndarray:
    """Convert any tensor to float32 numpy array."""
    arr = _to_numpy(tensor)
    if arr.dtype in (np.float16, np.float64):
        arr = arr.astype(np.float32)
    return arr


