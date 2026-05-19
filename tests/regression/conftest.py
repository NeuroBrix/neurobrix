"""Pytest config for the regression harness.

Discovers imported models from ~/.neurobrix/cache/, reads each one's
manifest.json + topology.json to get (family, flow_type), and exposes
the inventory as a fixture.

Adds a `--runslow` flag so heavy models (diffusion, video) can be
skipped by default and explicitly included when desired:

    pytest tests/regression/                 # fast models only
    pytest tests/regression/ --runslow       # include slow models
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pytest


CACHE_ROOT = Path(os.path.expanduser("~/.neurobrix/cache"))


def pytest_configure(config):
    """Auto-detect an active venv for neurobrix subprocesses.

    `test_all_models.py::_run_neurobrix` spawns `[_runtime_python(), "-m",
    "neurobrix", ...]` in a subprocess. `_runtime_python()` returns
    `NEUROBRIX_PYTHON` (if set) else `sys.executable`. When pytest itself
    runs under `/usr/bin/python3` (as happens when `pytest` is installed
    at `~/.local/bin/pytest` with a system-python shebang), `sys.executable`
    points at that system Python, whose user-site packages are often a
    different version from the developer's working venv — notably:

      - `transformers` at user-site may pin `tokenizers<0.20` while the
        real tokenizers install is 0.21+.
      - `mistral_common` may be installed only in the venv.
      - `neurobrix` may not be installed in the system site at all.

    The result is a pile of spurious `::native` audio failures that
    every session re-diagnoses from scratch. If the user has a
    VIRTUAL_ENV exported and that venv's python can import both
    `neurobrix` and `transformers`, prefer it. The existing
    `NEUROBRIX_PYTHON` env override always wins — this hook only sets
    a default when nothing was chosen explicitly.

    Single session-scope probe (~500 ms), not called per test.
    """
    if os.environ.get("NEUROBRIX_PYTHON"):
        return  # user explicitly chose an interpreter — respect it
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return
    candidate = Path(venv) / "bin" / "python"
    if not candidate.exists():
        return
    try:
        probe = subprocess.run(
            [str(candidate), "-c", "import neurobrix, transformers"],
            capture_output=True, timeout=10,
        )
    except Exception:
        return
    if probe.returncode == 0:
        os.environ["NEUROBRIX_PYTHON"] = str(candidate)


# Per-family heuristics for how long a single greedy run is expected to
# take at most. Smaller models override per-name below.
FAMILY_TIMEOUT_S: Dict[str, int] = {
    "llm":   180,
    "audio": 240,
    "image": 600,
    "video": 900,
    # upscaler runs `nbx upscale` on a 64x64 fixture — the tiny tensor
    # is instant; the budget covers model load + Triton kernel compile.
    "upscaler": 240,
}

# Per-model overrides (name → seconds). Use when a model needs more than
# the family default (big MoE, 4K diffusion, etc.).
MODEL_TIMEOUT_S: Dict[str, int] = {
    "TinyLlama-1.1B-Chat-v1.0":     60,
    "orpheus-3b-0.1-ft":            180,
    "Qwen3-30B-A3B-Thinking-2507":  420,
    "deepseek-moe-16b-chat":        420,
    "Sana_1600M_4Kpx_BF16":         900,
    "SANA-Video_2B_720p_diffusers": 1200,
}


# Families / flows marked "slow" by default — included only with --runslow.
SLOW_FAMILIES = {"image", "video"}


# Known-broken cells and the reason — recorded as xfail.  Keep this list
# short, current, and ALWAYS with a reason.  Every entry is a promise
# that we know about the breakage and we're not hiding a regression
# behind it.  Use (model_name, mode_or_None, reason).  mode_or_None=None
# means both modes.  When a fix lands, REMOVE the entry — leaving stale
# xfails masks regressions.
KNOWN_FAILURES: List[Tuple[str, str | None, str]] = [
    # ------------------------------------------------------------------
    # Triton audio — genuine runtime blockers on the --triton path.
    # Native paths for these all PASS (Batch 1 harness extension landed
    # flow-aware CLI dispatch).  What remains is per-family Triton work.
    # ------------------------------------------------------------------
    ("whisper-large",          "triton", "Triton encoder_decoder audio flow not validated end-to-end yet"),
    ("whisper-large-v3-turbo", "triton", "Triton encoder_decoder audio flow not validated end-to-end yet"),
    ("parakeet-tdt-1.1b",      "triton", "Triton rnnt flow not validated end-to-end yet"),
    # P-AUDIO-LLM-TRITON-FLOW: triton/flow/audio_llm.py now ported
    # (TritonAudioLLMEngine, R33-pure). Voxtral validated
    # compiled<->triton byte-identical → un-xfail'd. canary-qwen /
    # granite-speech reach the new handler but hit pre-existing
    # triton kernel-op gaps (NOT the flow port) — xfail kept,
    # residual blockers named for follow-up.
    ("canary-qwen-2.5b",       "triton", "P-TRITON-NBXTENSOR-REPEAT-MISSING — flow ported; encoder aten::repeat → NBXTensor has no .repeat (dispatch.py:161 _meta_repeat)"),
    ("granite-speech-3.3-8b",  "triton", "P-TRITON-SAFE-SOFTMAX-MISSING — flow ported; encoder triton compile aborts on missing op aten::_safe_softmax (sequence.py:1526) + prior CFormer projector note"),

    # ------------------------------------------------------------------
    # TTS models that work in --native but need reference-voice plumbing
    # in the Triton tts_llm / dual_ar flow handlers.
    # ------------------------------------------------------------------
    ("chatterbox",         "triton", "triton/flow/tts_llm.py doesn't wire audio_path reference voice"),
    ("openaudio-s1-mini",  "triton", "triton/flow/dual_ar.py doesn't wire audio_path reference voice"),

    # ------------------------------------------------------------------
    # Kokoro — triton only. _execute_native_text_encoder still passes
    # NBXTensor to torch.nn.functional.embedding — the stage handler
    # needs an NBXTensor→torch boundary conversion for the Triton path.
    # (::native was un-xfailed: the aten::cudnn_batch_norm undefined-
    # tensor regression introduced by ea90d66 was fixed indirectly in
    # the zero3 refactor era — see
    # docs/follow-ups/archive/kokoro_cudnn_batch_norm_regression.md and
    # docs/verdicts/p_kokoro_native_cudnn_batch_norm/verdict.md.)
    # ------------------------------------------------------------------
    ("Kokoro-82M", "triton", "_execute_native_text_encoder passes NBXTensor to torch.nn.functional.embedding — stage needs NBX→torch boundary"),

    # ------------------------------------------------------------------
    # VibeVoice — structural contract violation: DDPM loop + ConvNext1d
    # acoustic decoder run as native PyTorch outside TensorDAG.  Triton
    # mode (ABSOLUTE ZERO torch) rejects this.  Native mode works in
    # practice but the CLI invocation profile the harness uses doesn't
    # match what the model expects (needs speaker ref / script format).
    # Needs forge re-trace to integrate DDPM as a neural component.
    # ------------------------------------------------------------------
    ("VibeVoice-1.5B",     None, "TensorDAG contract violation — DDPM + ConvNext1d outside graph; needs forge re-trace"),
]


def _read_manifest(model_dir: Path) -> Dict[str, str]:
    """Return (family, flow_type, gen_type) for a cached model.  '?' if missing.

    gen_type is `topology.flow.generation.type` — multimodal builds are
    traced for exactly one generation_type (Janus image-only vs text-only),
    and `neurobrix run` enforces `--mode` matching that type, so the harness
    must derive the mode from the build rather than guess it.
    """
    family = "?"
    flow = "?"
    gen_type = "?"
    mf = model_dir / "manifest.json"
    if mf.exists():
        try:
            family = json.loads(mf.read_text()).get("family", "?")
        except Exception:
            pass
    tp = model_dir / "topology.json"
    if tp.exists():
        try:
            flow_obj = json.loads(tp.read_text()).get("flow", {})
            flow = flow_obj.get("type", "?")
            gen_type = flow_obj.get("generation", {}).get("type", "?")
        except Exception:
            pass
    return {"family": family, "flow": flow, "gen_type": gen_type}


def discover_models() -> List[Dict[str, str | int]]:
    """Scan the neurobrix cache for all importable models."""
    if not CACHE_ROOT.exists():
        return []
    out: List[Dict[str, str | int]] = []
    for sub in sorted(CACHE_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "manifest.json").exists():
            continue
        meta = _read_manifest(sub)
        name = sub.name
        timeout = MODEL_TIMEOUT_S.get(
            name, FAMILY_TIMEOUT_S.get(meta["family"], 300))
        out.append({
            "name": name,
            "family": meta["family"],
            "flow": meta["flow"],
            "gen_type": meta["gen_type"],
            "timeout_s": timeout,
        })
    return out


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run slow regression tests (diffusion, video).",
    )


@pytest.fixture(scope="session")
def model_inventory() -> List[Dict[str, str | int]]:
    return discover_models()


def pytest_collection_modifyitems(config, items):
    """Skip slow tests (image, video) unless --runslow is passed.

    Must live in conftest.py, not in a test module, or pytest will not
    discover the hook during collection.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="skipped; pass --runslow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
