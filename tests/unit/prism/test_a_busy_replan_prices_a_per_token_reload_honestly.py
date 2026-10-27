"""A busy re-plan takes the cheapest plan that fits, priced with what each plan really does: a strategy that re-reads a
generating component's weights for every token pays that reload at the rate the loader really reloads (the profile's
measured `weight_reload_gbps`: read, decode, convert, upload), not at the page-locked copy rate (the supervisor
2026-10-27 20:10, rule 13). Before: Janus-Pro-7B triton on the Mac with 677 MB taken by other applications (16 732 MB
free) re-planned to layer_streaming "estimated 169 s" — its language model's two segments re-read every token, 716
loads in 38 min, a ~50x slower plan bought for a 465 MB shortfall (stable f8def0b9, 2026-10-27 19:3x). No card."""
from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def test_janus_triton_on_a_busy_mac_is_not_re_planned_into_a_per_token_reload(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 16732, "the Mac with 677 MB taken by other applications")
    p, _ = _plan("Janus-Pro-7B", "triton")
    # with 16 732 MB free no faster plan fits (lazy_sequential needs ~17 034 MB): the re-plan may stream, but it must
    # say what streaming costs — the measured run took 2 348 s for 716 reloads; the estimate must be of that order,
    # never the 169 s the page-locked price gave
    if p.strategy == "layer_streaming":
        assert p.estimated_s is not None and p.estimated_s > 0.5 * 2348 * 577 / 716, (p.estimated_s, p.selection_reason[:300])
        # the supervisor's decision 2026-10-27 20:45: no refusal, a clear warning — what the fast plan lacks, both
        # times, and what to do
        why = p.selection_reason
        assert "lacks" in why and "close an application" in why and " h)" in why, why[:400]


def test_a_per_token_reload_is_priced_at_the_measured_reload_rate():
    from neurobrix.core.prism.time_estimate import host_facts
    h = host_facts({"host_matmul_gflops": 3125, "host_memory_gbps": 132, "weight_reload_gbps": 2.1})
    assert abs(h.reload_bps - 2.1e9) < 1
