"""Zero-torch audio front-end for the triton path.

NBXTensor / ctx glue around the shared vendor-free numpy DSP. The pure-numpy
mel / waveform extractors live in `core/module/audio/mel_dsp.py` (single source
of truth, R34 — no torch, no torchaudio, no librosa, no transformers) and are
re-exported here so the triton flow handlers keep importing them from
audio_frontend. This module keeps only the NBXTensor binding + ctx wiring so the
triton compute path imports NO torch. R33: numpy is CPU glue, never torch.

Validated bit-close to the original vendor extractors (whisper mel maxdiff
~1.7e-5, conformer ~1.3e-5; filterbank ~5e-6 — fp32-level), then proven by STT.
Preprocessing types: mel_spectrogram (Whisper/Voxtral), nemo_mel (Canary),
conformer (Granite), raw_waveform (Parakeet).

Two totally separate front-ends (R30/two-modes): the PyTorch flows call
core/flow/audio_utils.preprocess_audio_input (torch boundary wrap); the triton
flows call preprocess_audio_input_np here (numpy → NBXTensor). Both now share the
same DSP core (mel_dsp.py).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor

# Shared vendor-free DSP — single source of truth (compiled + triton).
from neurobrix.core.module.audio.mel_dsp import (  # noqa: F401
    _load_audio,
    _stft_power,
    _hz_to_mel,
    _mel_to_hz,
    _mel_filters,
    _whisper_mel,
    _conformer_mel,
    _nemo_mel,
    _raw_waveform,
    extract_features_np,
)


# ---------------------------------------------------------------------------
# binding — zero-torch mirror of audio_utils.preprocess_audio_input
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# torch-free helpers (duplicated from core/flow/audio_utils.py — deliberate
# two-modes duplication so the triton path imports nothing that pulls torch)
# ---------------------------------------------------------------------------
def _component_input_shape(ctx, comp_name):
    if comp_name is None:
        return None
    executor = ctx.executors.get(comp_name)
    if executor is None:
        return None
    dag = getattr(executor, "_dag", None)
    if dag is None:
        return None
    for tid, spec in dag.get("tensors", {}).items():
        if (spec.get("type") == "input" or spec.get("input_name") is not None
                or tid.startswith("input::")):
            resolved = []
            for dim in spec.get("shape", []):
                if isinstance(dim, dict):
                    resolved.append(dim.get("trace_value", dim.get("trace", 0)))
                elif isinstance(dim, int):
                    resolved.append(dim)
                else:
                    resolved.append(0)
            return tuple(resolved)
    return None


def _model_config_path(ctx) -> Path:
    nbx_path = Path(ctx.nbx_path_str)
    for subdir in ["modules/processor", "modules/tokenizer"]:
        cand = nbx_path / subdir
        if cand.exists():
            return cand
    raise RuntimeError(
        "Cannot find model config path. Expected modules/processor/ or "
        "modules/tokenizer/ inside the .nbx.")


# g2p is NeuroBrix-internal (ZO-3): text→IPA via the espeak-distilled lexicon
# embedded in the .nbx (modules/g2p/en_lexicon.txt.gz) + a stdlib LTS fallback,
# read by core.module.audio.g2p — NO `phonemizer`/`espeakng_loader`/`kokoro`
# import at runtime. The lexicon is byte-identical to espeak per word (incl.
# "Hello world" → həlˈoʊ wˈɜːld); embedded data retains espeak's license.


def _load_pt_numpy(path: str) -> np.ndarray:
    """Read a torch-saved .pt tensor (zip format) as numpy — zero torch. Tensor
    storage = largest data file; shape = the multi-dim pickle tuple whose product
    equals the element count."""
    import zipfile
    import io as _io
    import math
    import pickletools
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data_files = sorted([n for n in names if "/data/" in n],
                            key=lambda n: z.getinfo(n).file_size, reverse=True)
        arr = np.frombuffer(z.read(data_files[0]), dtype=np.float32).copy()
        pkl = z.read([n for n in names if n.endswith("data.pkl")][0])
    shape = None
    seq = []
    for op, a, _pos in pickletools.genops(_io.BytesIO(pkl)):
        if op.name in ("BININT", "BININT1", "BININT2"):
            seq.append(a)
        elif op.name.startswith("TUPLE"):
            k = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}.get(op.name, len(seq))
            t = tuple(seq[-k:]) if k else ()
            if len(t) >= 2 and math.prod(t) == arr.size and shape is None:
                shape = t
            seq = []
        elif op.name == "MARK":
            seq = []
    return arr.reshape(shape if shape is not None else (arr.size,))


def _set_device_for(ctx):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
    from neurobrix.triton.device_transfer import parse_device_idx
    _DA.set_device(parse_device_idx(getattr(ctx, "primary_device", "cuda:0")))


def _load_voicepack_np(engine, phoneme_count: int) -> None:
    """Torch-free Kokoro voicepack load + split (decoder/predictor styles)."""
    nbx_path = Path(engine.ctx.nbx_path_str)
    vdir = nbx_path / "modules" / "voices"
    if not vdir.exists():
        return
    vname = engine.ctx.pkg.defaults.get("voice", "af_heart")
    vp = vdir / f"{vname}.pt"
    if not vp.exists():
        files = sorted(vdir.glob("*.pt"))
        if not files:
            return
        vp = files[0]; vname = vp.stem
    voicepack = _load_pt_numpy(str(vp))
    if voicepack.ndim == 1:
        ref_s = voicepack[None]
    elif voicepack.ndim == 2:
        idx = min(phoneme_count, voicepack.shape[0] - 1)
        ref_s = voicepack[idx:idx + 1]
    elif voicepack.ndim == 3:
        idx = min(phoneme_count, voicepack.shape[0] - 1)
        ref_s = voicepack[idx]
    else:
        ref_s = voicepack.reshape(-1, voicepack.shape[-1])[0:1]
    split = ref_s.shape[-1] // 2
    _set_device_for(engine.ctx)
    sd = NBXTensor.from_numpy(np.ascontiguousarray(ref_s[:, :split]))
    sp = NBXTensor.from_numpy(np.ascontiguousarray(ref_s[:, split:]))
    for k in ["global.decoder_style", "decoder_style"]:
        engine.ctx.variable_resolver.resolved[k] = sd
    for k in ["global.predictor_style", "predictor_style"]:
        engine.ctx.variable_resolver.resolved[k] = sp
    print(f"   [Voicepack·np] Loaded '{vname}' (ref_s={tuple(ref_s.shape)})")


def preprocess_phonemizer_input_np(engine, prompt: str, phoneme_vocab: Dict) -> None:
    """Zero-torch g2p: text → IPA → phoneme IDs, bound as NBXTensor. Mirror of
    core/flow/stages/kokoro.preprocess_phonemizer_input (which uses torch)."""
    _lang_map = {"a": "en-us", "b": "en-gb"}
    klang = engine.ctx.pkg.defaults.get("phoneme_lang", "a")
    lang = _lang_map.get(klang, "en-us")
    from neurobrix.core.module.audio.g2p import g2p_phonemes
    phonemes = g2p_phonemes(prompt, engine.ctx.nbx_path_str, lang, klang)
    ids = [0]
    for ch in phonemes:
        if ch in phoneme_vocab:
            ids.append(phoneme_vocab[ch])
    ids.append(0)
    actual_len = len(ids)
    _set_device_for(engine.ctx)
    input_ids = NBXTensor.from_numpy(np.array([ids], dtype=np.int64))
    engine.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
    engine.ctx.variable_resolver.resolved["input_ids"] = input_ids
    print(f"   [Phonemizer·np] '{prompt[:60]}' -> {len(phonemes)} phonemes "
          f"-> {actual_len} IDs")
    text_lengths = NBXTensor.from_numpy(np.array([actual_len], dtype=np.int64))
    for k in ["global.text_lengths", "text_lengths", "input_lengths"]:
        engine.ctx.variable_resolver.resolved[k] = text_lengths
    # mask convention: True=PADDING; batch=1 no padding → all-False.
    text_mask = NBXTensor.from_numpy(np.zeros((1, actual_len), dtype=bool))
    for k in ["global.text_mask", "text_mask", "m"]:
        engine.ctx.variable_resolver.resolved[k] = text_mask
    _load_voicepack_np(engine, actual_len)


def postprocess_text_output_np(ctx) -> None:
    """Decode generated token IDs to text (torch-free mirror of audio_utils)."""
    generated_ids = ctx.variable_resolver.resolved.get("global.generated_token_ids")
    if generated_ids is None:
        return
    tokenizer = ctx.modules.get("tokenizer")
    if tokenizer is not None:
        from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
        text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
    else:
        text = str(generated_ids)
    ctx.variable_resolver.resolved["global.transcription"] = text
    print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")


def preprocess_audio_input_np(ctx, audio_config: Dict, stages: List[Dict]) -> None:
    """Load audio + extract features in numpy, bind as NBXTensor (no torch)."""
    get_component_input_shape = _component_input_shape
    find_model_config_path = _model_config_path

    input_config = audio_config.get("input", {})
    audio_path = ctx.variable_resolver.resolved.get("global.audio_path")
    if audio_path is None:
        raise RuntimeError("ZERO FALLBACK: Audio model requires global.audio_path "
                           "(use --audio <path>).")
    preprocessing = input_config.get("preprocessing")
    if preprocessing is None:
        raise RuntimeError("ZERO FALLBACK: topology.flow.audio.input.preprocessing required.")
    variable = input_config.get("variable", "global.input_features")

    first_comp = stages[0]["component"] if stages else None
    input_shape = get_component_input_shape(ctx, first_comp)
    # Auto-correct preprocessing from graph shape (mirror of the torch path).
    if input_shape and len(input_shape) >= 3:
        d1, d2 = input_shape[1], input_shape[2]
        if preprocessing == "raw_waveform":
            if d1 in (40, 64, 80, 128) and d2 > d1:
                preprocessing = "mel_spectrogram"
            elif d2 in (40, 64, 80, 128, 160, 256) and d1 > d2:
                preprocessing = "conformer"

    print(f"   [Audio·np] Loading: {audio_path}")
    feats = extract_features_np(preprocessing, str(audio_path),
                                Path(find_model_config_path(ctx)), input_shape)

    # Pad/truncate to trace-time dims (mirror of the torch path).
    if input_shape and len(input_shape) == feats.ndim and feats.ndim >= 3:
        for d in range(1, len(input_shape)):
            trace, actual = input_shape[d], feats.shape[d]
            if actual > trace:
                sl = [slice(None)] * feats.ndim
                sl[d] = slice(None, trace)
                feats = feats[tuple(sl)]
            elif actual < trace:
                ps = list(feats.shape); ps[d] = trace - actual
                feats = np.concatenate([feats, np.zeros(ps, np.float32)], axis=d)
    feats = np.ascontiguousarray(feats.astype(np.float32))
    print(f"   [Audio·np] Features: {tuple(feats.shape)} ({preprocessing})")

    # Place the feature tensor on the encoder's device (NBXTensor.from_numpy uses
    # the CURRENT DeviceAllocator device — without this it lands on cuda:0 while a
    # multi-GPU-placed encoder may be on another device → "cpu tensor" at conv).
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
    from neurobrix.triton.device_transfer import parse_device_idx
    _DA.set_device(parse_device_idx(getattr(ctx, "primary_device", "cuda:0")))
    nbx = NBXTensor.from_numpy(feats)
    ctx.variable_resolver.resolved[variable] = nbx
    short = variable.split(".")[-1] if "." in variable else variable
    ctx.variable_resolver.resolved[short] = nbx

    frames = feats.shape[-1] if preprocessing in ("mel_spectrogram", "nemo_mel") else feats.shape[1]
    length = NBXTensor.from_numpy(np.array([frames], dtype=np.int64))
    for k in ["global.audio_signal_length", "audio_signal_length", "global.length", "length"]:
        ctx.variable_resolver.resolved[k] = length
