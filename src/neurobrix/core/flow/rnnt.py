"""
RNNTEngine — RNNT Transducer Flow Handler

Handles RNNT/TDT models (Parakeet, etc.) with greedy frame-by-frame decoding.

Architecture:
    encoder(audio_features) → enc_out [B, D_enc, T]
    Greedy loop over T frames:
        decoder(prev_tokens) → dec_out [B, D_dec, U]
        joint(enc_frame, dec_frame) → logits [B, 1, 1, V]
        token = argmax(logits)

ZERO SEMANTIC: No knowledge of "Parakeet" or "NeMo".
ZERO HARDCODE: All parameters from NBX container.
"""

from neurobrix.core.memory.manager import release_flow_memory
import time
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("rnnt")
class RNNTEngine(FlowHandler):
    """
    RNNT transducer flow: encoder → greedy decode (decoder + joint per frame).

    The encoder runs through CompiledSequence normally.
    The decoder (embedding + LSTM) and joint (linear + relu + linear) are
    executed with native PyTorch ops using weights extracted from the executor,
    since their tiny graphs (11 and 20 ops) have trace-time shapes that don't
    match the variable runtime shapes of greedy decoding.
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
        """Execute RNNT pipeline: preprocess → encoder → greedy decode → text."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})

        # Step 1: Audio preprocessing (stashes long-form windows —
        # D-STT-LONGFORM-CHUNKING: one mel window per traced frame span;
        # len==1 == the exact pre-change path)
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
            print(f"   [encoder] Done in {enc_ms:.0f}ms, output {enc_output.shape}"
                  + (f" (window {_wi + 1}/{_n_win})" if _n_win > 1 else ""))

            # Step 3: RNNT greedy decode
            start = time.perf_counter()
            self._ensure_weights_loaded("decoder")
            self._ensure_weights_loaded("joint")

            _wtok = self._rnnt_greedy_decode(enc_output)
            if _n_win > 1:
                # Buffered-inference merge (stt_longform.rnnt_merge_window):
                # each token carries the encoder frame it was emitted at;
                # the kept intervals tile the timeline exactly. The
                # overlap in encoder frames is measured from THIS
                # window's encoder output against its mel span.
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

    # ─────────────────────────────────────────────────────────
    # Audio preprocessing
    # ─────────────────────────────────────────────────────────

    def _preprocess_audio(self, audio_config: Dict) -> None:
        """Load and preprocess audio to NeMo-compatible mel features.

        NeMo preprocessing differs from Whisper:
        - n_fft=512, window=400 (25ms), hop=160 (10ms)
        - log mel with guard value
        - per_feature normalization (subtract mean, divide by std per mel bin)
        - optional dither (1e-5 Gaussian noise)
        """
        audio_path = self.ctx.variable_resolver.resolved.get("global.audio_path")
        if audio_path is None:
            raise RuntimeError("ZERO FALLBACK: No audio_path provided.")

        device = self.ctx.primary_device

        # Load audio
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # NeMo preprocessing params from the embedded model_config.yaml —
        # the single source the numpy mirror reads too
        # (mel_dsp.nemo_mel_params); the .nbx defaults keep precedence
        # for the values the build carries.
        from neurobrix.core.flow.audio_utils import find_model_config_path
        from neurobrix.core.module.audio.mel_dsp import nemo_mel_params
        _p = nemo_mel_params(find_model_config_path(self.ctx))
        defaults = self.ctx.pkg.defaults
        n_fft, win_length, hop_length, n_mels = _p["n_fft"], _p["win"], _p["hop"], _p["n_mels"]
        target_sr = _p["sr"]
        dither = defaults.get("dither", _p["dither"])

        # Resample to the model's rate if needed
        if sr != target_sr:
            import numpy as np
            target_len = int(len(audio) * target_sr / sr)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            sr = target_sr

        print(f"   [Audio] Loading: {audio_path}")

        waveform = torch.from_numpy(audio).to(device=device, dtype=torch.float32)

        # Pre-emphasis filter: y[n] = x[n] - 0.97*x[n-1]
        preemph = defaults.get("preemphasis", _p["preemph"])
        if preemph > 0:
            waveform = torch.cat([waveform[:1], waveform[1:] - preemph * waveform[:-1]])

        # Apply dither
        if dither > 0:
            waveform = waveform + dither * torch.randn_like(waveform)

        # STFT
        window = torch.hann_window(win_length, device=device)
        stft = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True, center=True, pad_mode="reflect",
        )
        power_spectrum = stft.abs().pow(2)  # [n_fft//2+1, frames]

        # Mel filterbank — R34 (Zero Outsider): NeuroBrix-internal numpy
        # filterbank (htk, no norm) instead of torchaudio.melscale_fbanks.
        # mel_dsp._mel_filters returns [n_mels, n_freqs]; transpose to
        # [n_freqs, n_mels] to match the torchaudio layout (~5e-6 identical).
        from neurobrix.core.module.audio.mel_dsp import _mel_filters as _nbx_mel_fb
        import numpy as _np_fb
        mel_fb = torch.from_numpy(
            _np_fb.ascontiguousarray(
                _nbx_mel_fb(sr, n_fft, n_mels, htk=True, norm=None).T)
        ).to(device=device, dtype=power_spectrum.dtype)  # [n_fft//2+1, n_mels]

        mel_spec = power_spectrum.T @ mel_fb  # [frames, n_mels]

        # Log with guard
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        # Per-feature normalization (NeMo normalize="per_feature")
        mean = mel_spec.mean(dim=0, keepdim=True)
        std = mel_spec.std(dim=0, keepdim=True).clamp(min=1e-5)
        mel_spec = (mel_spec - mean) / std

        # Transpose to [1, n_mels, frames] for encoder
        features = mel_spec.T.unsqueeze(0)  # [1, n_mels, frames]

        actual_frames = features.shape[2]
        print(f"   [Audio] Features: {features.shape} (nemo_mel, {actual_frames} frames)")

        # Pad to match encoder's traced frame count
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

        # Long-form (D-STT-LONGFORM-CHUNKING): buffered inference over
        # the encoder's window span — NeMo's own long-form recipe:
        # overlapping windows (overlap from the family YAML, in seconds,
        # converted with the extractor's own hop), one decode each,
        # tokens merged by emission frame (stt_longform). Whole-
        # utterance mel stats (the normalization above), last window
        # padded like a short clip. The window length is the traced
        # span: D-PARAKEET-SYMBOLIC-T (DETTE.md) names the symbolic-T
        # trace that removes that coupling. NBX_DISABLE_STT_CHUNKING=1
        # restores the truncating path for diagnosis.
        import os as _os_sw
        self._stt_windows = None
        self._stt_window_mel = expected_frames
        self._stt_overlap_mel = 0
        if (actual_frames > expected_frames
                and _os_sw.environ.get("NBX_DISABLE_STT_CHUNKING") != "1"):
            from neurobrix.core.config.loader import get_family_config
            from neurobrix.core.module.audio.stt_longform import rnnt_window_plan
            _lf = get_family_config("stt").get("long_form") or {}
            if "rnnt_overlap_seconds" not in _lf:
                raise RuntimeError(
                    "ZERO FALLBACK: stt.yml long_form.rnnt_overlap_seconds missing.")
            if _dag_frames is None:
                raise RuntimeError(
                    "ZERO FALLBACK: long-form RNNT needs the encoder's traced frame "
                    "span from the DAG (no 3-D encoder input found).")
            if not _p.get("source"):
                # The overlap is converted with the SAME hop/rate the mel
                # above was computed with; a legacy build without an
                # embedded NeMo config runs both on the built-in NeMo
                # values — said out loud, never silently (the rebuild
                # that embeds model_config.yaml is D-PARAKEET-SYMBOLIC-T).
                print(f"   [Audio] NeMo preprocessor params: built-in values "
                      f"(no embedded config in this build) — sr={sr} hop={hop_length}")
            self._stt_overlap_mel = int(round(float(_lf["rnnt_overlap_seconds"]) * sr / hop_length))
            wins = []
            for _start, _valid in rnnt_window_plan(actual_frames, expected_frames,
                                                   self._stt_overlap_mel):
                f = features[:, :, _start:_start + _valid]
                if _valid < expected_frames:
                    f = torch.nn.functional.pad(f, (0, expected_frames - _valid))
                wins.append((f, torch.tensor([_valid], dtype=torch.long, device=device)))
            self._stt_windows = wins
            features, length = wins[0]
        else:
            if actual_frames < expected_frames:
                features = torch.nn.functional.pad(features, (0, expected_frames - actual_frames))
            elif actual_frames > expected_frames:
                features = features[:, :, :expected_frames]
            length = torch.tensor([min(actual_frames, expected_frames)], dtype=torch.long, device=device)

        self._bind_window(features, length)

    def _bind_window(self, features, length) -> None:
        # Bind under all possible keys the executor might look for
        for key in ["global.audio_signal", "global.input_features", "audio_signal",
                     "input::audio_signal", "encoder.audio_signal"]:
            self.ctx.variable_resolver.resolved[key] = features
        for key in ["global.length", "length", "input::length", "encoder.length"]:
            self.ctx.variable_resolver.resolved[key] = length

    # ─────────────────────────────────────────────────────────
    # RNNT greedy decode
    # ─────────────────────────────────────────────────────────

    def _rnnt_greedy_decode(self, enc_output: torch.Tensor) -> List[int]:
        """
        RNNT greedy decoding with native PyTorch ops for decoder + joint.

        The decoder and joint are tiny networks (11 and 20 ops) whose traced
        graphs have hardcoded shapes. We extract their weights and run them
        natively to handle variable-length greedy decoding.

        Args:
            enc_output: Encoder output [B, D_enc, T] (transposed: time is last dim)

        Returns:
            List of decoded token IDs (without blank)
        """
        device = enc_output.device
        dtype = enc_output.dtype

        # enc_output is [B, D_enc, T_padded] — transpose to [B, T_padded, D_enc]
        enc_out = enc_output.transpose(1, 2)  # [1, T_padded, D_enc]
        T_padded = enc_out.shape[1]
        self._enc_frames = int(T_padded)   # encoder frame axis (long-form merge)

        # Compute actual encoder output length from input length
        # Conformer subsampling: 2 conv layers with stride 2 each = 4x reduction (typical)
        # NeMo ConformerEncoder: subsampling_factor from config, or compute from kernel/stride
        input_length = self.ctx.variable_resolver.resolved.get("global.length")
        if input_length is not None:
            actual_frames = input_length.item() if hasattr(input_length, 'item') else int(input_length)
            # Standard NeMo subsampling: (length - 1) // 2 applied twice = ~4x
            # More precisely: ceil(length / subsampling_factor)
            sub_factor = self.ctx.pkg.defaults.get("subsampling_factor", 8)
            T = min((actual_frames + sub_factor - 1) // sub_factor, T_padded)
        else:
            T = T_padded
        # Use encoder output dtype as compute dtype (set by DtypeEngine/Prism)
        dtype = enc_out.dtype

        # Extract decoder weights and cast to compute dtype
        dec_weights = self._extract_decoder_weights()
        joint_weights = self._extract_joint_weights()
        # Cast all weights to match encoder output dtype
        for k, v in dec_weights.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                dec_weights[k] = v.to(dtype)
            elif isinstance(v, list):
                dec_weights[k] = [t.to(dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in v]
        for k, v in joint_weights.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                joint_weights[k] = v.to(dtype)

        # TDT config
        defaults = self.ctx.pkg.defaults
        num_tdt_durations = defaults.get("num_tdt_durations", 1)
        blank_id = defaults.get("blank_id", 1024)  # Usually vocab_size
        vocab_size = defaults.get("vocab_size", 1024)

        # Initialize
        tokens: List[int] = []
        self._token_frames: List[int] = []
        blank_token = blank_id
        last_token = blank_token

        # LSTM hidden state: [num_layers, B, hidden_size]
        num_layers = dec_weights["num_layers"]
        hidden_size = dec_weights["hidden_size"]
        h = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=dtype)
        c = torch.zeros(num_layers, 1, hidden_size, device=device, dtype=dtype)

        # NeMo's greedy loop (GreedyTDTInfer._greedy_decode /
        # GreedyRNNTInfer, rnnt_greedy_decoding.py, read 2026-09-02):
        # the prediction network runs on (last_token, state) and its
        # output g is REUSED until a non-blank commits a new state; the
        # duration head's skip advances time after every symbol, blank or
        # not (a non-blank with skip 0 keeps the frame); a plain RNNT
        # stays on the frame after a non-blank and moves one frame on a
        # blank; max_symbols_per_step guards the frame.
        durations = defaults.get("tdt_durations")
        if num_tdt_durations > 1 and durations is None:
            durations = list(range(num_tdt_durations))
            print(f"   [rnnt_decode] TDT durations: built-in consecutive "
                  f"{durations} (not carried by this build)")
        max_symbols_per_step = int(defaults.get("max_symbols_per_step", 10))
        t = 0
        g = None                      # prediction-network output for (last_token, state)
        h_next = c_next = None
        with torch.inference_mode():
            while t < T:
                symbols_added = 0
                need_loop = True
                skip = 1
                while need_loop and symbols_added < max_symbols_per_step:
                    if g is None:
                        token_tensor = torch.tensor([[last_token]], dtype=torch.long, device=device)
                        dec_embed = torch.nn.functional.embedding(token_tensor, dec_weights["embedding"])
                        dec_input = dec_embed.transpose(0, 1)  # [seq=1, batch=1, D_dec]
                        dec_rnn_out, (h_next, c_next) = self._run_lstm(
                            dec_input, (h, c),
                            dec_weights["weight_ih"], dec_weights["weight_hh"],
                            dec_weights["bias_ih"], dec_weights["bias_hh"],
                            num_layers,
                        )
                        g = dec_rnn_out.squeeze(0)  # [1, D_dec]
                    enc_frame = enc_out[:, t, :]  # [1, D_enc]
                    logits = self._run_joint(enc_frame, g, joint_weights)
                    if num_tdt_durations > 1:
                        token_logits = logits[:vocab_size + 1]  # vocab + blank
                        skip = durations[int(logits[vocab_size + 1:].argmax().item())]
                    else:
                        token_logits = logits
                    pred_token = token_logits.argmax().item()
                    if pred_token != blank_id:
                        tokens.append(pred_token)
                        self._token_frames.append(t)   # encoder frame per token (long-form merge)
                        h, c = h_next, c_next          # commit the state, as the vendor does
                        last_token = pred_token
                        g = None
                        if num_tdt_durations <= 1:
                            skip = 0                   # plain RNNT: stay on the frame
                    elif num_tdt_durations <= 1:
                        skip = 1                       # plain RNNT: blank moves one frame
                    symbols_added += 1
                    t += skip
                    need_loop = skip == 0
                if symbols_added == max_symbols_per_step:
                    t += 1

        return tokens

    def _run_lstm(self, input_seq, hx, weight_ih, weight_hh, bias_ih, bias_hh, num_layers):
        """Run LSTM forward using extracted weights."""
        h, c = hx
        output = input_seq
        new_h_list = []
        new_c_list = []
        for layer in range(num_layers):
            h_l = h[layer:layer+1]
            c_l = c[layer:layer+1]
            output, (h_l, c_l) = self._lstm_cell_sequence(
                output, h_l, c_l,
                weight_ih[layer], weight_hh[layer],
                bias_ih[layer], bias_hh[layer],
            )
            new_h_list.append(h_l)
            new_c_list.append(c_l)
        new_h = torch.cat(new_h_list, dim=0)
        new_c = torch.cat(new_c_list, dim=0)
        return output, (new_h, new_c)

    @staticmethod
    def _lstm_cell_sequence(x, h, c, w_ih, w_hh, b_ih, b_hh):
        """Run LSTM over a sequence using manual cell computation."""
        seq_len = x.shape[0]
        outputs = []
        for i in range(seq_len):
            xi = x[i:i+1]  # [1, 1, D]
            xi = xi.squeeze(0)  # [1, D]
            h_sq = h.squeeze(0)  # [1, D]
            gates = xi @ w_ih.T + b_ih + h_sq @ w_hh.T + b_hh
            hidden_size = h.shape[2]
            i_gate = torch.sigmoid(gates[:, :hidden_size])
            f_gate = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
            g_gate = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
            o_gate = torch.sigmoid(gates[:, 3*hidden_size:])
            c_sq = c.squeeze(0)
            c_new = f_gate * c_sq + i_gate * g_gate
            h_new = o_gate * torch.tanh(c_new)
            h = h_new.unsqueeze(0)
            c = c_new.unsqueeze(0)
            outputs.append(h_new.unsqueeze(0))
        output = torch.cat(outputs, dim=0)  # [seq, 1, D]
        return output, (h, c)

    def _run_joint(self, enc_frame, dec_frame, jw):
        """Run joint network: project enc + dec, add, relu, output linear."""
        # enc_frame: [1, D_enc], dec_frame: [1, D_dec]
        enc_proj = enc_frame @ jw["enc_weight"].T + jw["enc_bias"]  # [1, D_joint]
        dec_proj = dec_frame @ jw["dec_weight"].T + jw["dec_bias"]  # [1, D_joint]
        combined = torch.relu(enc_proj + dec_proj)  # [1, D_joint]
        logits = combined @ jw["out_weight"].T + jw["out_bias"]  # [1, V]
        return logits.squeeze(0)  # [V]

    # ─────────────────────────────────────────────────────────
    # Weight extraction
    # ─────────────────────────────────────────────────────────

    def _extract_decoder_weights(self) -> Dict[str, Any]:
        """Extract decoder weights (embedding + LSTM) from GraphExecutor.

        NeMo RNNT decoder weight names:
          prediction.embed.weight: [vocab+1, D_dec]
          prediction.dec_rnn.lstm.weight_ih_l{N}: [4*D_dec, D_dec]
          prediction.dec_rnn.lstm.weight_hh_l{N}: [4*D_dec, D_dec]
          prediction.dec_rnn.lstm.bias_ih_l{N}: [4*D_dec]
          prediction.dec_rnn.lstm.bias_hh_l{N}: [4*D_dec]
        """
        executor = self.ctx.executors["decoder"]
        w = executor._weights

        # Find embedding by shape (2D with vocab-like first dim)
        embedding = None
        weight_ih = []
        weight_hh = []
        bias_ih = []
        bias_hh = []

        for key, tensor in w.items():
            if tensor is None:
                continue
            k = key.lower()
            if "embed" in k and tensor.ndim == 2:
                embedding = tensor
            elif "weight_ih_l" in k:
                layer = int(k.split("weight_ih_l")[-1])
                weight_ih.append((layer, tensor))
            elif "weight_hh_l" in k:
                layer = int(k.split("weight_hh_l")[-1])
                weight_hh.append((layer, tensor))
            elif "bias_ih_l" in k:
                layer = int(k.split("bias_ih_l")[-1])
                bias_ih.append((layer, tensor))
            elif "bias_hh_l" in k:
                layer = int(k.split("bias_hh_l")[-1])
                bias_hh.append((layer, tensor))

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
        """Extract joint network weights from GraphExecutor.

        NeuroTax standard names in NBX:
          enc.weight: [D_joint, D_enc]    enc.bias: [D_joint]
          pred.weight: [D_joint, D_dec]   pred.bias: [D_joint]
          joint.2.weight: [V, D_joint]    joint.2.bias: [V]
        """
        executor = self.ctx.executors["joint"]
        w = executor._weights

        result = {}
        for key, tensor in w.items():
            if tensor is None:
                continue
            if key.endswith("enc.weight"):
                result["enc_weight"] = tensor
            elif key.endswith("enc.bias"):
                result["enc_bias"] = tensor
            elif key.endswith("pred.weight"):
                result["dec_weight"] = tensor
            elif key.endswith("pred.bias"):
                result["dec_bias"] = tensor
            elif key.endswith("joint.2.weight"):
                result["out_weight"] = tensor
            elif key.endswith("joint.2.bias"):
                result["out_bias"] = tensor

        required = ["enc_weight", "enc_bias", "dec_weight", "dec_bias", "out_weight", "out_bias"]
        missing = [k for k in required if k not in result]
        if missing:
            raise RuntimeError(
                f"ZERO FALLBACK: Joint network missing weights: {missing}\n"
                f"Available: {list(w.keys())}"
            )
        return result

    # ─────────────────────────────────────────────────────────
    # Output utilities
    # ─────────────────────────────────────────────────────────

    def _get_component_output(self, comp_name: str) -> torch.Tensor:
        """Get the primary output tensor from a component."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.output"]:
            if key in resolved:
                return resolved[key]
        # Check executor outputs
        executor = self.ctx.executors.get(comp_name)
        if executor and hasattr(executor, '_last_outputs') and executor._last_outputs:
            return executor._last_outputs[0]
        raise RuntimeError(f"ZERO FALLBACK: No output found for {comp_name}")

    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode token IDs to text using sentencepiece tokenizer from NBX cache."""
        cache_path = self.ctx.pkg.cache_path

        # Look for sentencepiece model in NBX cache — R34 (Zero Outsider):
        # decode via the NeuroBrix-internal PySentencePiece (.model protobuf
        # parser), never the `sentencepiece` vendor lib.
        sp_path = self._find_tokenizer_model()
        if sp_path is not None:
            try:
                from neurobrix.core.module.tokenizer.sp_proto import PySentencePiece
                with open(sp_path, "rb") as _spf:
                    sp = PySentencePiece.from_bytes(_spf.read())
                return sp.decode(tokens)
            except ImportError:
                pass

        # Fallback: try vocab.txt in NBX cache
        if cache_path is not None:
            vocab_paths = list(cache_path.rglob("*vocab.txt"))
            if vocab_paths:
                with open(vocab_paths[0]) as f:
                    vocab = [line.strip() for line in f]
                chars = [vocab[t] if t < len(vocab) else f"<{t}>" for t in tokens]
                return "".join(chars).replace("▁", " ").strip()

        # ZERO FALLBACK: tokenizer missing from NBX container
        raise RuntimeError(
            "ZERO FALLBACK: No sentencepiece tokenizer.model or vocab.txt in NBX cache.\n"
            "The model needs to be rebuilt with the tokenizer included.\n"
            "Re-run the build toolchain: build --overwrite"
        )

    def _find_tokenizer_model(self) -> Optional[Path]:
        """Find sentencepiece tokenizer.model in NBX cache.

        ZERO FALLBACK: Runtime ONLY reads from NBX cache.
        If the tokenizer is missing, the toolchain builder needs to include it.
        """
        cache_path = self.ctx.pkg.cache_path
        if cache_path is None:
            return None
        # Search NBX cache for sentencepiece model files
        sp_paths = list(cache_path.rglob("*.model"))
        return sp_paths[0] if sp_paths else None
