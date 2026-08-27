"""Serve WARM path gate — the class response to the warm-boundary lesson.

The serving warm path (InferenceEngine.generate — the same code the
serve daemon dispatches to) is a separate input/output surface from
the CLI cold path, and gaps there have repeatedly slept through cold
gates (4th recurrence 2026-08-26: the upscaler family fell through
to the video input contract and broke on serve warm, BOTH engines,
while every cold cell was green — S3_FINDING_SERVE_UPSCALER_WARM).

Contract of this gate: at least one row per family exercises the
warm path in every battery, with the OUTPUT SHAPE AND TYPE verified,
on both engines, under short budgets (a budget-killed cell reads as
"slow" when it is broken — the Flex lesson).

Currently covered: upscaler (the family whose warm break motivated
the gate). Extend one small row per family as warm coverage grows —
never delete a family from here once added.

Runnable: python3 -m pytest tests/regression/test_serve_warm.py -q

REQUIRES THE FULL UNMASKED RIG: cells plan against autodetect (which
enumerates via nvidia-smi and is BLIND to CUDA_VISIBLE_DEVICES —
D-AUTODETECT-VISIBLE-MASK), so running the suite under a restricted
mask produces false failures ("invalid device ordinal" class; 13 of
them in the 2026-08-27 supervisor spot-check). A module guard skips
the suite with a clear message instead. Rows that need pinning do it
themselves (_PINNED_ROWS: mask + MATCHING profile together, per
cell, in the subprocess).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Guard: refuse a restricted parent environment loudly-but-kindly.
# (The pinned rows set their own mask inside their subprocess env —
# the parent process must stay unmasked.)
if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
    pytest.skip(
        "test_serve_warm requires the FULL unmasked rig: autodetect is "
        "blind to CUDA_VISIBLE_DEVICES (D-AUTODETECT-VISIBLE-MASK) and "
        "a restricted mask yields false 'invalid device ordinal' "
        "failures (supervisor spot-check 2026-08-27). Unset "
        "CUDA_VISIBLE_DEVICES and re-run.",
        allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]

UPSCALER_MODEL = "real-esrgan-x4"
UPSCALE_FACTOR = 4
INPUT_SIZE = 64  # synthetic input — self-contained, no asset debt


def _synthetic_input(tmp_path: Path) -> Path:
    """A small deterministic RGB gradient — content is irrelevant to
    the gate (shape/type verified, not pixels)."""
    from PIL import Image

    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    img.putdata([(x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
                 for y in range(INPUT_SIZE) for x in range(INPUT_SIZE)])
    p = tmp_path / "warm_input.png"
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# One warm row per family — the zoo-wide sweep, permanent (2026-08-26
# supervisor decision after the 4th warm-boundary recurrence). Small
# models, short configs; every row verifies output PRESENCE + SHAPE/
# TYPE, never just exit codes. rnnt (parakeet) is absent:
# BLOCKED-ON-ARTIFACT (legacy family field — S2 prereg re-trace
# motion); add its row when the re-traced artifact lands.

AUDIO_REF = REPO / "benchmarks" / "assets" / "jfk_11s.wav"
IMAGE_REF = REPO / "benchmarks" / "assets" / "apple_448.png"

FAMILY_ROWS = [
    pytest.param(
        "TinyLlama-1.1B-Chat-v1.0",
        {"prompt": "Hello, how are you?", "max_tokens": 5},
        "text", id="llm"),
    pytest.param(
        "whisper-large-v3-turbo",
        {"audio_path": str(AUDIO_REF)},
        "text", id="stt"),
    pytest.param(
        "Kokoro-82M",
        {"prompt": "Hello world."},
        "media:wav", id="tts"),
    pytest.param(
        "Voxtral-Mini-3B-2507",
        {"audio_path": str(AUDIO_REF), "prompt": "Hello world",
         "temperature": 0.0},
        "text", id="audio_llm",
        marks=pytest.mark.xfail(
            reason="sampling capability gate refuses defaults-carried "
                   "top_k/top_p on the audio_llm paths even at "
                   "temperature 0 (landed a9aab29; chantier "
                   "P-SAMPLING-CONSOLIDATION) and serving has no "
                   "top-k/top-p plumb-through to neutralize them — "
                   "warm-sweep finding 2026-08-26",
            strict=False)),
    pytest.param(
        "GLM-4.1V-9B-Thinking",
        {"image_path": str(IMAGE_REF),
         "prompt": "What is in this image?", "max_tokens": 5},
        "text", id="vlm"),
    pytest.param(
        "Janus-Pro-7B",
        {"prompt": "a red apple on a wooden table", "mode": "image"},
        "media:png", id="multimodal"),
    pytest.param(
        "Sana-1600M-MultiLing",
        {"prompt": "a red apple on a wooden table", "steps": 4},
        "media:png", id="image"),
    pytest.param(
        "Wan2.1-T2V-1.3B-Diffusers",
        {"prompt": "a red fox running through snow", "steps": 2,
         "num_frames": 5, "height": 480, "width": 832},
        "media:mp4", id="video"),
]


_CELL_TIMEOUT_S = {"video": 900}
_DEFAULT_CELL_TIMEOUT_S = 600

# Rows that run PINNED (mask + MATCHING single-GPU profile together —
# the closure-config pattern; autodetect is mask-blind,
# D-AUTODETECT-VISIBLE-MASK). Wan-T2V: the unpinned weight_sharding
# plan places the patch-embed conv activations on a 16 GB card and
# launch-OOMs (warm-sweep finding 2026-08-26; pinned 32 GB PASS) —
# a Prism placement class, not a warm-path defect; unpin when that
# placement finding is fixed.
_PINNED_ROWS = {
    "Wan2.1-T2V-1.3B-Diffusers": {"visible": "2", "hardware": "v100-32g"},
}


@pytest.mark.parametrize("mode", ["compiled", "triton"])
@pytest.mark.parametrize("model,gen_kwargs,verify", FAMILY_ROWS)
def test_family_serve_warm(model: str, gen_kwargs: dict, verify: str,
                           mode: str, request) -> None:
    """Each cell runs in a FRESH subprocess (warm_cell_runner.py):
    one model per process is the production serve shape, and it
    isolates the in-process unload live-set accumulation
    (P-SERVE-UNLOAD-LIVE-SET — the sweep's own finding). The
    subprocess timeout enforces the short-budget rule for real: a
    hang fails the cell instead of the battery."""
    import subprocess
    import sys as _sys

    if model == "GLM-4.1V-9B-Thinking" and mode == "triton":
        pytest.skip("GLM-4.1V triton generation exceeds budgets — open "
                    "cold finding (baseline-reproduced 2026-08-26, "
                    "antireg_2026_08_26_rope_fix verdict); unskip with "
                    "its fix")

    try:
        from neurobrix.cli.utils import find_model
        find_model(model)
    except Exception:
        pytest.skip(f"{model} not in the local cache")

    family_id = request.node.callspec.id.split("-")[0]
    timeout_s = _CELL_TIMEOUT_S.get(family_id, _DEFAULT_CELL_TIMEOUT_S)
    spec = {"model": model, "mode": mode,
            "gen_kwargs": dict(gen_kwargs), "verify": verify}
    import json as _json
    import os as _os
    env = {**_os.environ, "PYTHONPATH": str(REPO / "src"),
           "PYTHONUNBUFFERED": "1"}
    pin = _PINNED_ROWS.get(model)
    if pin:
        env["CUDA_VISIBLE_DEVICES"] = pin["visible"]
        spec["hardware"] = pin["hardware"]
    try:
        r = subprocess.run(
            [_sys.executable, str(Path(__file__).parent
                                  / "warm_cell_runner.py"),
             _json.dumps(spec)],
            capture_output=True, text=True, timeout=timeout_s, env=env,
            cwd=str(REPO))
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"{model}/{mode}: warm cell timeout after "
                    f"{timeout_s}s. Partial stdout: "
                    f"{(e.stdout or '')[-300:]!r}")
    if r.returncode != 0 or "WARM-CELL-OK" not in r.stdout:
        tail = (r.stderr or r.stdout or "")[-800:]
        pytest.fail(f"{model}/{mode}: warm cell rc={r.returncode}\n"
                    f"... tail:\n{tail}")


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_upscaler_serve_warm_output_shape(mode: str, tmp_path: Path) -> None:
    from PIL import Image

    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.serving.engine import InferenceEngine

    try:
        from neurobrix.cli.utils import find_model
        find_model(UPSCALER_MODEL)
    except Exception:
        pytest.skip(f"{UPSCALER_MODEL} not in the local cache")

    engine = InferenceEngine(UPSCALER_MODEL,
                             get_or_create_default_profile(), mode=mode)
    engine.load()
    try:
        input_png = _synthetic_input(tmp_path)
        expected = INPUT_SIZE * UPSCALE_FACTOR

        for request_idx in (1, 2):  # request 2 = the warm-reuse property
            result = engine.generate(image_path=str(input_png))
            assert "outputs" in result, (
                f"warm request {request_idx} ({mode}): no outputs in "
                f"result — keys {sorted(result)}")
            out_path = tmp_path / f"warm_out_{mode}_{request_idx}.png"
            saved = engine.save_output(result["outputs"], str(out_path))
            im = Image.open(saved)
            assert im.size == (expected, expected) and im.mode == "RGB", (
                f"warm request {request_idx} ({mode}): expected "
                f"{expected}x{expected} RGB, got {im.size} {im.mode}")
    finally:
        engine.unload()
