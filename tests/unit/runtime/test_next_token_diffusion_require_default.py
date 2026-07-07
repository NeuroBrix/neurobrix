"""P-ZERO-FALLBACK-SWEEP — next_token_diffusion semantic keys required.

Pins the HIGH finding of engine audit #2 (2026-07-05): the
next_token_diffusion flow (both engines) silently defaulted six
semantic keys (cfg_scale, speech_scaling_factor,
ddpm_num_inference_steps, ddpm_beta_schedule, prediction_type,
ddpm_num_steps) plus the two same-class adjacent keys
(speech_bias_factor, acoustic_vae_dim) — inventing vendor constants
when the .nbx defaults.json omits them, where the sibling
encoder_decoder flow does exemplary ZERO FALLBACK.

Both engines carry their OWN `_require_default` (R30 symmetric fix,
total path separation — no shared helper across engines); this test
pins both.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_next_token_diffusion_require_default.py
"""
from __future__ import annotations

import pytest

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.flow.next_token_diffusion import (
    _require_default as _require_default_compiled,
)
from neurobrix.triton.flow.next_token_diffusion import (
    _require_default as _require_default_triton,
)

_ENGINES = pytest.mark.parametrize(
    "require_default",
    [_require_default_compiled, _require_default_triton],
    ids=["compiled", "triton"],
)

# The audit's six keys + the two same-class adjacent keys.
_SEMANTIC_KEYS = (
    "cfg_scale",
    "speech_scaling_factor",
    "speech_bias_factor",
    "ddpm_num_inference_steps",
    "ddpm_num_steps",
    "ddpm_beta_schedule",
    "prediction_type",
    "acoustic_vae_dim",
)


@_ENGINES
def test_present_key_returned(require_default):
    defaults = {"cfg_scale": 1.3, "ddpm_num_steps": 1000}
    assert require_default(defaults, "cfg_scale") == 1.3
    assert require_default(defaults, "ddpm_num_steps") == 1000


@_ENGINES
@pytest.mark.parametrize("key", _SEMANTIC_KEYS)
def test_missing_semantic_key_raises_naming_key(require_default, key):
    defaults = {k: 1 for k in _SEMANTIC_KEYS if k != key}
    with pytest.raises(RuntimeError) as excinfo:
        require_default(defaults, key)
    msg = str(excinfo.value)
    assert "ZERO FALLBACK" in msg
    assert key in msg
    assert "defaults.json" in msg


@_ENGINES
def test_engines_carry_separate_helpers(require_default):
    """Path separation: the two engines must not share the helper."""
    assert _require_default_compiled is not _require_default_triton
    assert (_require_default_compiled.__module__
            == "neurobrix.core.flow.next_token_diffusion")
    assert (_require_default_triton.__module__
            == "neurobrix.triton.flow.next_token_diffusion")
