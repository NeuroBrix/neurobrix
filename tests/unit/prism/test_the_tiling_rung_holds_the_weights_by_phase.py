"""The op-level tiling rung holds the weights resident WHILE EACH COMPONENT RUNS — the flow's release schedule —
not every component's weights at once.

The Mac, 2026-10-25 17:56 / 18:06, Sana_1600M_4Kpx_BF16 native at the gate request on the M4 Pro: op_level_tiling
declined "the weights alone (11 411 MB) do not fit the room (11 021 MB): tiling cuts activations, not weights", and
the plan fell to lazy_sequential / layer_streaming — while the VAE, the one component whose activations overflow,
runs with 303 MB of weights beside nothing (the text encoder is released after its run, the transformer before the
post-loop stage: core/flow/residency, both engines' flows). Resident-and-cut is the rank above every streamed plan
(the owner, 15:16); a rung that demands the whole model resident at once cannot offer it to any model of several
components.

What this test would do if the code were wrong: the rung declines on the sum of the weights and the plan is a
streamed one — the first assertion fails. Seen red on 34171d6c, green with `held_of` (the weights resident while
the component runs). No card: the Mac's pinned profile, the plan derived."""
import pytest
from neurobrix.core.prism import PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, no_door, pin_host, profile

MODEL = "Sana_1600M_4Kpx_BF16"


def _request(c):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import trace_request as TR
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    args = create_parser().parse_args(["run", "--model", MODEL, *TR.gate_request(MODEL), "--compiled"])
    man = c.get_manifest() or {}
    return request_input_config(args, man, man.get("family"), c.cache_path)


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_sana_4k_on_the_pinned_mac_the_tiling_rung_holds_the_weights_by_phase_and_prices_the_tiled_vae(monkeypatch, mode):
    """The rung no longer refuses on the sum of the weights; it then prices the VAE under the plan the detector
    offers and either claims it (resident, the VAE tiled: the plan's memory is the tiled figure and its host side
    fits the pool) or says what the tiled VAE still holds and where — never a refusal on figures nobody priced.
    Measured 2026-10-25 on the pinned Mac: compiled, the tiled VAE still holds three whole 128-channel tensors at
    the pixel shuffle (the broadcast chain's clone materialises on torch — the Triton engine has the proxy, R30);
    triton, the bf16 VAE's chain is priced fp32 by the width model (the Triton branch's FP32-class stores under
    bf16, the 15:27 contract) — two engine-side debts this cell names by their figures while they stand."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    c = NBXContainer.load(str(container_root(MODEL)))              # absent: a named skip
    s = PrismSolver()
    p = s.solve_smart(c, profile(APPLE_M4_PRO), _request(c), mode=mode)
    refused = {r[0]: r for r in (p.rejected or [])}
    assert all("tiling cuts activations, not weights" not in str(r) for r in (p.rejected or [])), refused
    if p.strategy == "op_level_tiling":
        assert "vae" in (p.runtime_op_tiling or {}), sorted(p.runtime_op_tiling or {})
        fp = p.host_footprint or {}
        assert (int(fp.get("total_bytes", 0)) - int(fp.get("resident_bytes", 0))) / 2**20 <= s._unified_host_pool_mb(profile(APPLE_M4_PRO)), fp
    else:
        why = str(refused.get("op_level_tiling"))      # the floor before the detector, or the priced plan after it
        assert ("still hold" in why or "are live before" in why) and " MB" in why and "::" in why, (p.strategy, why)
