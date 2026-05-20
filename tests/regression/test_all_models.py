"""Regression harness — every imported model × each runtime mode.

Each test launches `python3 -m neurobrix run ...` in a fresh subprocess,
captures stdout, and:

- for LLM models (`family == "llm"`):
  * runs with a 5-token greedy generation on a short prompt
  * parses the "[Output] Generated N tokens\\n\\n<text>" block
  * compares <text> to the golden file for (model, mode), if one exists
  * if no golden exists, fails with a clear message suggesting the
    capture command — NOT a silent pass (so CI never accidentally
    records a broken output as the new truth)

- for non-LLM models:
  * runs with default inputs, asserts exit code 0 only
  * golden artefacts (images, audio) are binary and not compared by
    default — the presence of a successful run is the v1 regression
    signal for these families

Usage
-----
    # Fast models only (LLM + audio — default):
    pytest tests/regression/ -v

    # Include slow families (image, video):
    pytest tests/regression/ -v --runslow

    # Capture or re-capture goldens (run when you've deliberately
    # changed a model's expected output):
    UPDATE_GOLDEN=1 pytest tests/regression/ -v

    # Single model × single mode:
    pytest tests/regression/ -v -k "TinyLlama and native"
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest


REPO = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["native", "triton"]

# Default LLM probe — short, stable, deterministic.
LLM_PROMPT = "Hello"
LLM_MAX_TOKENS = 5
LLM_TEMPERATURE = 0.0

# Shared audio asset used for STT inputs and TTS reference voices.
AUDIO_REF = REPO / "test_speech_ref.wav"

# Small (64x64 RGB, window_size-safe) image fed to `nbx upscale` for the
# upscaler family. Deliberately tiny: the project logo (1983x1536) OOMs
# at x4/x8 (52 GiB conv allocation), which would leave upscaler cells
# red for a sizing artefact unrelated to the dispatch under test.
IMAGE_REF = REPO / "test_upscale_input.png"

# Flow types that consume an audio file as the ONLY input (pure STT:
# Whisper encoder_decoder, Parakeet rnnt).  audio_llm is deliberately NOT
# here — Voxtral/canary-qwen/granite-speech take an audio file AND a text
# instruction prompt (audio-conditioned LLM), so they have their own
# dispatch branch below.
STT_FLOWS = {"encoder_decoder", "rnnt"}

# Flow types that generate audio and REQUIRE an audio reference voice
# (voice cloning / speaker conditioning).
TTS_WITH_REF_FLOWS = {"tts_llm", "dual_ar"}


# Import the discovery helper from conftest so the parametrize list
# matches exactly what the fixture reports.
sys.path.insert(0, str(Path(__file__).parent))
from conftest import (  # noqa: E402
    discover_models, discover_target_matrix, KNOWN_FAILURES, SLOW_FAMILIES,
)


def _marks_for(name: str, mode: str, family: str,
               status: str = "testable",
               skip_reason: str = "") -> List:
    """Build the pytest.mark list for a (model, mode) cell.

    `status='not_traced'` cells receive a `pytest.mark.skip` with the
    DETTE_TECHNIQUE_NON_OUVERTE reason — the harness lists them as
    coverage gaps without attempting to run them.
    """
    marks: List = []
    if status == "not_traced":
        marks.append(pytest.mark.skip(reason=skip_reason or
                                      "DETTE_TECHNIQUE_NON_OUVERTE: model not traced"))
        return marks
    if family in SLOW_FAMILIES:
        marks.append(pytest.mark.slow)
    for kn_name, kn_mode, reason in KNOWN_FAILURES:
        if kn_name == name and kn_mode in (None, mode):
            marks.append(pytest.mark.xfail(
                reason=reason, strict=False, run=True))
    return marks


def _build_parametrize() -> List:
    """Return a list of pytest.param entries for each (model, mode).

    Testable cells come from `discover_models()` (= cached .nbx
    inventory). Not-traced reference-matrix cells come from
    `discover_target_matrix()` and skip with reason — they make
    family-coverage gaps visible at every harness run instead of
    being silently absent.
    """
    params = []
    for m in discover_models():
        for mode in MODES:
            params.append(pytest.param(
                m, mode,
                marks=_marks_for(m["name"], mode, m["family"]),
                id=f"{m['name']}::{mode}",
            ))
    # Not-yet-traced reference matrix — single skipped cell per
    # model (no per-mode expansion: the model isn't built, so the
    # mode distinction is moot until it lands in the cache).
    for m in discover_target_matrix():
        params.append(pytest.param(
            m, "native",
            marks=_marks_for(m["name"], "native", m["family"],
                             status=str(m.get("status", "testable")),
                             skip_reason=str(m.get("skip_reason", ""))),
            id=f"{m['name']}::not_traced",
        ))
    return params


def _mode_for_gen_type(gen_type: str) -> str:
    """Map a build's `flow.generation.type` to the `--mode` value.

    Multimodal builds are traced for exactly one generation_type
    (`autoregressive_image` / `autoregressive_text`); `neurobrix run`
    rejects a `--mode` that doesn't match the trace.  The mode token is
    the last `_`-segment of the gen_type ("image" / "text").
    """
    tail = gen_type.rsplit("_", 1)[-1]
    return tail if tail in ("image", "text") else "image"


def _cli_inputs_for(family: str, flow: str, gen_type: str) -> List[str]:
    """Build the model-specific CLI inputs based on (family, flow, gen_type).

    NOTE: family=upscaler does NOT pass through here — it uses the
    dedicated `nbx upscale` subcommand (see `_run_neurobrix`), not
    `nbx run`. `nbx run` has no image-input wiring (P-IMAGE-INPUT-FLOW,
    deferred); `nbx upscale` is the validated production path.

    Dispatch matrix (family-first, then flow):
      family=llm                      → --prompt + --max-tokens
      family=multimodal               → --mode <gen_type> + --prompt
                                        (Janus: build is image-only, the
                                        CLI enforces --mode == trace type)
      flow in STT_FLOWS               → --audio only (Whisper, Parakeet)
      flow == audio_llm               → --audio + --prompt (Voxtral,
                                        canary-qwen, granite-speech:
                                        audio-conditioned LLM)
      flow in TTS_WITH_REF_FLOWS      → --prompt + --audio reference voice
                                        (chatterbox, openaudio-s1-mini)
      everything else                 → --prompt (Kokoro/VibeVoice TTS,
                                        diffusion image/video)
    """
    if family == "llm":
        return ["--prompt", LLM_PROMPT, "--max-tokens", str(LLM_MAX_TOKENS)]
    if family == "multimodal":
        return ["--mode", _mode_for_gen_type(gen_type),
                "--prompt", "Hello world"]
    if flow in STT_FLOWS:
        return ["--audio", str(AUDIO_REF)]
    if flow == "audio_llm":
        return ["--audio", str(AUDIO_REF), "--prompt", "Hello world"]
    if flow in TTS_WITH_REF_FLOWS:
        return ["--prompt", "Hello world", "--audio", str(AUDIO_REF)]
    return ["--prompt", "Hello world"]


def _runtime_python() -> str:
    """Return the Python interpreter to use for neurobrix subprocesses.

    pytest often runs under a system interpreter that doesn't have every
    optional dep (e.g. `transformers`/`tokenizers` pin mismatch), while a
    working venv elsewhere does. Override via env var when needed:

        NEUROBRIX_PYTHON=/path/to/venv/bin/python pytest tests/regression/

    Falls back to sys.executable when no override is set.
    """
    return os.environ.get("NEUROBRIX_PYTHON", sys.executable)


def _upscale_out_path(model: str, mode: str) -> Path:
    """Deterministic output path for an upscaler regression cell.

    Shared by `_run_neurobrix` (the `--output`) and `test_model_runs`
    (the existence/dimension assertion).
    """
    return Path("/tmp") / f"regression_upscale_{model}_{mode}.png"


def _run_out_path(model: str, mode: str, family: str, gen_type: str) -> Path:
    """Deterministic output path for a non-upscaler regression cell.

    Mirrors `_upscale_out_path`: writes under `/tmp` so the harness
    never leaves stray `output_<model>.<ext>` artefacts in the repo
    root (which used to happen when `nbx run` was invoked without
    `--output` and defaulted to cwd = `REPO`). The per-family
    extension matches the family YAML's `output.default_extension_per_mode`
    (kept as a small in-harness mapping to avoid importing
    `neurobrix.core.runtime.output_dispatch` from a subprocess
    harness).
    """
    if family == "multimodal":
        ext = "png" if (gen_type or "").startswith("autoregressive_image") else "txt"
    elif family in ("image",):
        ext = "png"
    elif family in ("video",):
        ext = "mp4"
    elif family in ("audio_llm", "llm", "vlm"):
        ext = "txt"
    elif family in ("audio", "tts", "stt"):
        # audio family covers both STT (transcription → txt by CLI dispatch)
        # and TTS (synthesis → wav); `neurobrix run` family-aware dispatch
        # handles the actual writer, we just need an extension the CLI
        # accepts without strict-mismatch error. .wav is the audio default.
        ext = "wav"
    else:
        ext = "out"
    return Path("/tmp") / f"regression_run_{model}_{mode}.{ext}"


def _upscale_scale(model: str) -> int | None:
    """Parse the integer scale factor from an upscaler model name.

    real-esrgan-x2 → 2, swin2SR-classical-sr-x4-64 → 4,
    swinir-classical-x2 → 2, hat-l-x4 → 4. None if unparseable.
    """
    m = re.search(r"x(\d+)", model)
    return int(m.group(1)) if m else None


def _png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG IHDR. Dependency-free (no PIL — the
    pytest interpreter may lack it). None if absent / not a PNG."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    import struct
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def _run_neurobrix(model: str, mode: str, family: str, flow: str,
                   gen_type: str,
                   timeout_s: int) -> subprocess.CompletedProcess:
    """Invoke the appropriate `neurobrix` subcommand in a subprocess.

    family=upscaler → `nbx upscale` (the validated image-SR path; `nbx
    run` has no image-input wiring — P-IMAGE-INPUT-FLOW, deferred).
    Everything else → `nbx run` with family-aware inputs.

    Harness mode → `nbx upscale --mode`: native↔compiled, triton↔triton.
    """
    if family == "upscaler":
        out_path = _upscale_out_path(model, mode)
        if out_path.exists():
            out_path.unlink()
        cmd = [
            _runtime_python(), "-m", "neurobrix", "upscale",
            "--model", model,
            "--input", str(IMAGE_REF),
            "--output", str(out_path),
            "--mode", "triton" if mode == "triton" else "compiled",
        ]
    else:
        cmd = [
            _runtime_python(), "-m", "neurobrix", "run",
            "--model", model,
            "--temperature", "0",
        ]
        # For families with a *file* output (image/video/audio binary),
        # pass --output under /tmp so the CLI does NOT default to
        # writing `output_<model>.<ext>` in the harness cwd (= REPO).
        # LLM (and text-mode multimodal) output goes to stdout and is
        # consumed by _parse_llm_text — those families must NOT get
        # --output (it would redirect the text away from stdout).
        text_only = (family == "llm" or
                     (family == "multimodal" and not
                      (gen_type or "").startswith("autoregressive_image")))
        if not text_only:
            out_path = _run_out_path(model, mode, family, gen_type)
            if out_path.exists():
                out_path.unlink()
            cmd.extend(["--output", str(out_path)])
        cmd.extend(_cli_inputs_for(family, flow, gen_type))
        if mode == "triton":
            cmd.append("--triton")

    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout_s, env=env, cwd=str(REPO),
    )


_OUTPUT_RE = re.compile(
    r"\[Output\] Generated \d+ tokens?\s*\n\n(.+?)(?:\Z|\n\n)",
    re.DOTALL,
)


def _parse_llm_text(stdout: str) -> str | None:
    m = _OUTPUT_RE.search(stdout)
    return m.group(1).rstrip() if m else None


def _golden_path(model: str, mode: str) -> Path:
    return GOLDEN_DIR / f"{model}__{mode}.txt"


def _update_golden() -> bool:
    return os.environ.get("UPDATE_GOLDEN", "").lower() in ("1", "true", "yes")


@pytest.mark.parametrize("model_meta,mode", _build_parametrize())
def test_model_runs(model_meta: Dict[str, str | int], mode: str) -> None:
    name = str(model_meta["name"])
    family = str(model_meta["family"])
    flow = str(model_meta["flow"])
    gen_type = str(model_meta.get("gen_type", "?"))
    timeout_s = int(model_meta["timeout_s"])

    try:
        r = _run_neurobrix(name, mode, family, flow, gen_type, timeout_s)
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"{name} / {mode}: timeout after {timeout_s}s. "
            f"Partial stdout: {(e.stdout or b'')[-400:]!r}"
        )

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        pytest.fail(
            f"{name} / {mode}: exit {r.returncode}\n"
            f"... tail:\n{tail}"
        )

    # Upscaler: beyond exit 0, assert a real super-resolved PNG was
    # written at the expected scale (R29 — inspect the artefact, not
    # just the exit code). The 64x64 input is window_size-safe, so no
    # processor padding → output dims are exactly input × scale.
    if family == "upscaler":
        out_path = _upscale_out_path(name, mode)
        in_size = _png_size(IMAGE_REF)
        out_size = _png_size(out_path)
        assert in_size is not None, f"test input PNG unreadable: {IMAGE_REF}"
        assert out_size is not None, (
            f"{name} / {mode}: no PNG written at {out_path}\n"
            f"stdout tail:\n{r.stdout[-600:]}"
        )
        scale = _upscale_scale(name)
        iw, ih = in_size
        ow, oh = out_size
        if scale is not None:
            assert (ow, oh) == (iw * scale, ih * scale), (
                f"{name} / {mode}: expected {iw*scale}x{ih*scale} "
                f"(input {iw}x{ih} × {scale}), got {ow}x{oh}"
            )
        else:
            assert ow > iw and oh > ih, (
                f"{name} / {mode}: output {ow}x{oh} not larger than "
                f"input {iw}x{ih} (scale unparseable from name)"
            )
        return

    # Non-LLM families: v1 regression = successful run (exit 0).
    if family != "llm":
        return

    text = _parse_llm_text(r.stdout)
    assert text is not None, (
        f"{name} / {mode}: could not parse '[Output] Generated N tokens' "
        f"block from stdout.\nTail:\n{r.stdout[-800:]}"
    )

    gp = _golden_path(name, mode)
    if _update_golden() or not gp.exists():
        gp.write_text(text + "\n", encoding="utf-8")
        if not _update_golden():
            pytest.skip(
                f"{name} / {mode}: no golden on record, captured "
                f"{gp.relative_to(REPO)} — re-run to compare")
        return

    expected = gp.read_text(encoding="utf-8").rstrip("\n")
    assert text == expected, (
        f"{name} / {mode}: output drift.\n"
        f"  expected: {expected!r}\n"
        f"  got:      {text!r}\n"
        f"If this change is intentional, re-capture with "
        f"UPDATE_GOLDEN=1 pytest tests/regression/ -k "
        f"'{name} and {mode}'"
    )
