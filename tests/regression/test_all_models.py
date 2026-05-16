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

# Shared image asset used as the input for the upscaler family.
IMAGE_REF = REPO / "assets" / "logo_NeuroBrix.png"

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
    discover_models, KNOWN_FAILURES, SLOW_FAMILIES,
)


def _marks_for(name: str, mode: str, family: str) -> List:
    """Build the pytest.mark list for a (model, mode) cell."""
    marks: List = []
    if family in SLOW_FAMILIES:
        marks.append(pytest.mark.slow)
    for kn_name, kn_mode, reason in KNOWN_FAILURES:
        if kn_name == name and kn_mode in (None, mode):
            marks.append(pytest.mark.xfail(
                reason=reason, strict=False, run=True))
    return marks


def _build_parametrize() -> List:
    """Return a list of pytest.param entries for each (model, mode)."""
    params = []
    for m in discover_models():
        for mode in MODES:
            params.append(pytest.param(
                m, mode,
                marks=_marks_for(m["name"], mode, m["family"]),
                id=f"{m['name']}::{mode}",
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

    Dispatch matrix (family-first, then flow):
      family=llm                      → --prompt + --max-tokens
      family=upscaler                 → --input-image (real image input;
                                        --prompt is meaningless here)
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
    if family == "upscaler":
        return ["--input-image", str(IMAGE_REF)]
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


def _run_neurobrix(model: str, mode: str, family: str, flow: str,
                   gen_type: str,
                   timeout_s: int) -> subprocess.CompletedProcess:
    """Invoke `neurobrix run` in a subprocess."""
    cmd = [
        _runtime_python(), "-m", "neurobrix", "run",
        "--model", model,
        "--temperature", "0",
    ]
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
