"""A container whose keys were written under another NeuroTax version is refused at load, by name.

NeuroTax 5.0 renamed two canonical tokens (the SwiGLU gate `gate` -> `ffn_gate`, diffusers'
`attn1` `self_attn` -> `attn`) so the parser is a fixed point on its own output. The engine's
readers follow the parser — the MoE fusion finds experts by the parser's tokens — so a
container still named under 0.1 would load, find no expert, and run unfused without a word.
The loader refuses it before any weight I/O and names the version and the fix.

What these would do without the door: the two refusal tests fail (the load proceeds past the
manifest and dies later, or not at all).
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.nbx.neurotax import NEUROTAX_VERSION


def _container(tmp_path, manifest):
    # The directory carries the manifest's model_name: the loader refuses a misnamed container
    # before it reads the version, and that refusal would pass for the door (or hide it).
    d = tmp_path / manifest["model_name"]
    (d / "runtime").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    for rel in ("topology.json", "runtime/variables.json", "runtime/defaults.json"):
        (d / rel).write_text("{}")
    return d


# 5.0: the NeuroTax 5.1 vocabulary renamed 22 749 keys 5.0 had written (rule 6, 2026-10-04) — a
# 5.0 container holds keys the 5.1 readers (the RNNT joint, the TTS vocoder's token table) no
# longer find.
@pytest.mark.parametrize("manifest", [{"neurotax_version": "0.1"}, {"neurotax_version": "5.0"}, {}])
def test_another_version_is_refused_by_name(tmp_path, manifest):
    with pytest.raises(RuntimeError, match="NEUROTAX VERSION") as e:
        NBXRuntimeLoader().load(str(_container(tmp_path, {"model_name": "m", **manifest})))
    assert repr(manifest.get("neurotax_version")) in str(e.value)
    assert NEUROTAX_VERSION in str(e.value) and "neurotax_rename" in str(e.value)


def test_the_engines_version_passes_the_door(tmp_path):
    try:
        NBXRuntimeLoader().load(str(_container(
            tmp_path, {"model_name": "m", "neurotax_version": NEUROTAX_VERSION})))
    except RuntimeError as e:
        assert "NEUROTAX VERSION" not in str(e)
    except Exception:
        pass  # the empty fixture fails further on; only the door is under test
