"""Janus-Pro-7B triton on a V100-16GB at the CUDA gate request (supervisor 2026-10-28 19:05): its language model was
priced 13 180 MB of weights + 920 MB of activations x the Triton arena factor 1.3 + a 5 % "fragmentation buffer"
(705 MB), where the rack measured 14 151 MB whole (weights 13 181 + KV 660 + ~310) — zero3 at 384-668 s instead of
lazy whole at 113 s.

Two terms that measure nothing:
- the KV cache (660 MB) sits inside the language model's activation figure, and the arena factor multiplied it: the KV
  is ONE allocation the state engine makes once (a cache, sized by the plan), never the arena's per-op activations;
- the 5 % buffer was added on top of an activation figure that is the walk of the live set op by op, and on weights
  counted exactly from the index: on the rack the whole run peaked 51 MB above weights + activations (14 151 vs 14 100),
  and the compiled engine's per-op V100 trace of Open-Sora-v2's VAE held the walk within 0.38 GB at every op
  (2026-10-23). A buffer is added only to a figure that was not walked."""
import sys
from pathlib import Path

import pytest

from neurobrix.nbx import NBXContainer
from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.config import get_prism_defaults
from tests.unit.prism._pinned_machine import pin_host, pin_dedicated_card, profile, V100_16GB, container_root
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import no_door

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import trace_request as TR  # noqa: E402

MODEL = "Janus-Pro-7B"
MB = 1024 * 1024

pytestmark = pytest.mark.skipif(not container_root(MODEL).exists(), reason=f"{MODEL} not in the cache")


def _price(monkeypatch):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "V100")
    c = NBXContainer.load(str(container_root(MODEL)))
    args = create_parser().parse_args(["run", "--model", MODEL, *TR.gate_request(MODEL), "--triton"])
    man = c.get_manifest() or {}
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    s.solve_smart(c, profile(V100_16GB), request_input_config(args, man, man.get("family"), c.cache_path),
                  mode="triton")
    return s, c, seen["language_model"]


def test_the_arena_factor_never_multiplies_the_kv_cache(monkeypatch):
    s, c, lm = _price(monkeypatch)
    kv = s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    assert kv > 0 and lm.kv_bytes == kv, (lm.kv_bytes, kv)
    factor = float(get_prism_defaults()["triton_arena_activation_factor"])
    want = ((lm.activation_bytes - kv) * factor + kv) / MB
    assert s._live_activation_mb(lm) == pytest.approx(want), (s._live_activation_mb(lm), want)


def test_a_walked_live_set_carries_no_added_buffer(monkeypatch):
    _, _, lm = _price(monkeypatch)
    assert lm.activation_profiled
    assert lm.overhead_bytes == 0, f"{lm.overhead_bytes / MB:.0f} MB added to a walked figure"
