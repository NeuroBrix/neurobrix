"""The flow's generating language model is priced at the longest pass it runs: its prompt, at most the prompt budget its
cache is sized with (`defaults.prompt_margin`, the state engine's own figure) — never at the whole cache in one pass.
Before: its sequence axis took the trace extent under the placement floor; Janus-Pro-7B (traced at 704 positions =
576 image tokens + 128) priced its text logits over 5 x 704 rows, a 1 375 MB activation figure (KV included) where the
run held ~300 MB beside its weights and cache (the Dell's split on a V100, 2026-10-28), and on the idle Mac its lazy
plan overshot the pool by 388 MB and fell to layer_streaming. RED on lot-2026-10-28. No card."""
from neurobrix.core.prism.flow_bindings import FlowBindings
from tests.unit.prism._pinned_machine import container_root, pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def test_the_generating_lm_binds_its_sequence_to_the_prompt_budget():
    import json
    root = container_root("Janus-Pro-7B")
    fb = FlowBindings(json.loads((root / "topology.json").read_text()), root)
    assert fb.prompt_prefill_length("language_model") == 128
    assert fb.prompt_prefill_length("vision_model") is None and fb.prompt_prefill_length("gen_head") is None


def test_janus_triton_on_the_idle_mac_is_planned_lazy_with_its_head_priced_at_the_prompt(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the idle Mac")
    # the planning process's own memory, pinned at what a fresh run holds (the judged run said resident 99 MB): in a
    # long test process it grows and enters the plan's host side (+463 MB in the full suite)
    import neurobrix.core.prism.host_footprint as HF
    monkeypatch.setattr(HF, "process_footprint_now", lambda: 100 << 20)
    p, _ = _plan("Janus-Pro-7B", "triton")
    assert p.strategy in ("lazy_sequential", "single_gpu", "single_gpu_lifecycle"), (p.strategy, p.unified_rungs_tried)
    act_mb = p.component_memory["language_model"].activation_bytes / 2 ** 20
    assert act_mb < 1000, act_mb               # 130 MB of activations + 660 MB of KV, not 715 + 660
