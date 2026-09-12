"""The output resolution is read from the container, wherever the container puts it.

`_container_output_size` derives (height, width) from a traced LATENT input times
the VAE scale. It is what the resolution cascade asks when a request names none,
and what `latent_height` is derived from. Two of its own gates refused containers
that declare everything needed, and two video models died for it on 2026-09-11 on
`Key 'latent_height' not found in runtime/defaults.json`.

* **The flow type was a second gate in front of a narrower one.** The SHAPE test —
  rank 4 or 5 with two integer trailing extents — is the real discriminator, and
  no text or audio component satisfies it (an LLM's `hidden_states` is rank 3).
  The flow check sat in front of it and refused `Wan2.2-I2V-A14B`, whose flow is
  `static_graph` and whose backbone declares `hidden_states [1, 36, 5, 10, 12]`.
* **The VAE was never consulted.** `Open-Sora-v2`'s backbone takes a FLATTENED
  latent (`img [1, 60, 64]`, rank 3) and cannot answer — while its VAE declares
  `z [1, 16, 9, 14, 22]`, the same latent, in the same container.

Neither is an invention: a decoder's input extents ARE latent extents, and the
container states them. Inventing a resolution would be the engine papering over a
build-side limit, which is a separate and forbidden thing.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_container_output_size_reads_the_container.py
"""
from __future__ import annotations

import types

from neurobrix.core.runtime.executor import RuntimeExecutor


def _ex(topology, scale=8):
    ex = object.__new__(RuntimeExecutor)
    ex.pkg = types.SimpleNamespace(
        topology=topology, manifest={"vae_scale_factor": scale}, defaults={})
    return ex


def _topo(flow, **components):
    return {"version": "0.1", "flow": {"type": flow},
            "components": {n: {"shapes": s} for n, s in components.items()}}


# The live declarations, verbatim from the 2026-09-12 cached topologies.
ALLEGRO_TI2V = [3, 12, 7, 18, 26]
WAN22 = [1, 36, 5, 10, 12]
OPENSORA_BACKBONE = [1, 60, 64]          # FLATTENED — rank 3, cannot answer
OPENSORA_VAE_Z = [1, 16, 9, 14, 22]      # the same latent, rank 5


def test_a_static_graph_flow_with_a_spatial_latent_answers():
    """Wan2.2-I2V: the flow gate refused what the shape test accepts."""
    ex = _ex(_topo("static_graph", transformer={"hidden_states": WAN22}))
    assert ex._container_output_size({}) == (10 * 8, 12 * 8)


def test_a_flattened_backbone_falls_through_to_the_vae():
    """Open-Sora-v2: rank 3 at the backbone, rank 5 at the VAE, one container."""
    ex = _ex(_topo("iterative_process",
                   transformer={"img": OPENSORA_BACKBONE},
                   vae={"z": OPENSORA_VAE_Z}))
    assert ex._container_output_size({}) == (14 * 8, 22 * 8)


def test_the_backbone_wins_when_both_can_answer():
    """The backbone's latent is the one a request scales; the VAE is the fallback."""
    ex = _ex(_topo("iterative_process",
                   transformer={"hidden_states": ALLEGRO_TI2V},
                   vae={"z": OPENSORA_VAE_Z}))
    assert ex._container_output_size({}) == (18 * 8, 26 * 8)


def test_a_text_container_still_answers_nothing():
    """The control. An LLM's hidden_states is rank 3 and must not resolve.

    This is what the removed flow gate was believed to be protecting; the shape
    test does it, and does it without refusing a diffusion container whose flow
    happens to be named something else.
    """
    ex = _ex(_topo("autoregressive_generation",
                   transformer={"hidden_states": [1, 23, 4096]}))
    assert ex._container_output_size({}) is None


def test_an_audio_container_still_answers_nothing():
    ex = _ex(_topo("audio", transformer={"hidden_states": [1, 128, 3000]}))
    assert ex._container_output_size({}) is None


def test_no_scale_no_answer():
    """Without a VAE scale the latent extents cannot be turned into pixels."""
    ex = _ex(_topo("iterative_process", transformer={"hidden_states": WAN22}), scale=None)
    assert ex._container_output_size({}) is None


def test_a_container_declaring_no_latent_anywhere_answers_nothing():
    """Silence when the container is silent — never a guessed resolution."""
    ex = _ex(_topo("iterative_process", transformer={"timestep": [1]}))
    assert ex._container_output_size({}) is None
