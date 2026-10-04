"""`FlowBindings` reads the flags of the container it is built from, whoever builds it.

The flow's flag reads (`vace_control_conditioning`, `zero_pad_embeddings`) go through
`registry_flags.get_component_flag`, which answers from `nbx.component_flags` — a table filled when a
container is OPENED. The run opens it; the derived census does not, so every flag read there answered
its default: Wan2.1-VACE's control encoder was keyed at batch 1 while the run feeds it the
(inactive, reactive) pair, batch 2 (the Mac's zero-miss gate, 2026-10-04 19:19, `aten.convolution::0`
conv2d_forward (2, 3, 162, 162, ...)). `test_the_vace_control_encoder_runs_the_pair` could not see it:
it replaces the flag reader. Injection: the registration removed from `FlowBindings.__init__` -> RED."""
from neurobrix.core.prism.flow_bindings import FlowBindings
from neurobrix.nbx import component_flags


def test_a_never_opened_container_still_binds_the_vace_pair():
    component_flags.clear()
    topo = {"flow": {"type": "iterative_process", "pre_loop": ["text_encoder", "vae_encoder"],
                     "loop": {"components": ["transformer"]}, "post_loop": ["vae"]},
            "connections": [{"from": "global.image", "to": "vae_encoder.args"}],
            "extracted_values": {"transformer": {"vace_control_conditioning": True}}}
    fb = FlowBindings(topo, None, "a-container-nobody-opened")
    assert fb.vace_control_batch("vae_encoder") == 2
    assert fb.vace_control_batch("vae") is None


def test_a_container_without_the_flag_binds_no_pair():
    component_flags.clear()
    topo = {"flow": {"type": "iterative_process", "pre_loop": ["vae_encoder"],
                     "loop": {"components": ["transformer"]}, "post_loop": ["vae"]},
            "connections": [{"from": "global.image", "to": "vae_encoder.args"}],
            "extracted_values": {"transformer": {}}}
    assert FlowBindings(topo, None, "another-container").vace_control_batch("vae_encoder") is None
