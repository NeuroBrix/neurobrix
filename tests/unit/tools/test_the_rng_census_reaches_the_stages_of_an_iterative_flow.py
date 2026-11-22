"""A flow of `type: iterative_process` names its components in pre_loop, loop.components and post_loop, not in
`order`; every request runs them, so the RNG census must call them reached by the default request (before, a
drawing component of such a flow, e.g. an encoder in pre_loop, was reported UNREACHED)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from tests.unit.workshop_configuration import the_workshop_configuration  # noqa: E402
with the_workshop_configuration():
    import rng_census as RC  # noqa: E402

TOPOLOGY = {"flow": {"type": "iterative_process",
                     "pre_loop": ["text_encoder", "image_encoder"],
                     "loop": {"driver": "scheduler", "components": ["denoiser"]},
                     "post_loop": ["decoder"]}}


def test_every_stage_of_an_iterative_flow_is_reached_by_the_default_request():
    for comp in ("text_encoder", "image_encoder", "denoiser", "decoder"):
        assert RC.request_reaching(TOPOLOGY, comp) == "default", comp
    assert RC.request_reaching(TOPOLOGY, "nowhere") is None


def test_a_flow_of_another_type_does_not_have_its_loop_stages_read():
    topo = {"flow": {"type": "autoregressive", "pre_loop": ["x"], "order": []}}
    assert RC.request_reaching(topo, "x") is None
