"""A decoder's plan is priced at or above what the card held running it, with a named margin.

The oracle is the CARD, not the walk: each request below ran certified-only on a V100-SXM2-32GB of
the rack and `tools/run_with_peaks.py` read the component's rise above its own weights. The plan is
the one `neurobrix run` forms for that request (`run.request_input_config`, `solve_smart`), so the
figure under test is the one placement reads.

* mochi-1-preview, compiled, 7 frames 320x576 (card 3, 2026-10-09): the vae held 4 706 MB above its
  weights (campaigns/2026_10_09_transient_proof/run_A2_mochi_vae.log). The peak is
  `aten.convolution::33`, where cuDNN's tensor-core path holds NHWC copies of the fp16 input
  (636 MB), output (540 MB) and weight — priced by `op_transients.library_layout_transient_bytes`
  from the profile's `conv.library_layout_copies`. Before it the plan said 3 780 MB (and
  7 560 MB at the guidance batch, which the decode does not run).
* Sana_1600M_4Kpx_BF16, Triton, 3072x4096 (card 2, 2026-10-09): the vae held 18 434 MB above its
  weights with the allocator's pool off (run_B2np_sana4k_vae.log). With the pool on the reading is
  30 383 MB (run_B2_sana4k_vae.log): the extra ~12 GB is freed blocks parked in the pool, which
  `DeviceAllocator.malloc_cuda` flushes and retries on an out-of-memory, so it is not a
  requirement. The peak is `aten.relu::18`, which copies its permuted (strided) input contiguous
  — 6 144 MB the plan did not hold (`op_transients.contiguous_copy_bytes`). Before it, and before
  the residual chain's in-place merge stopped aliasing a zero-sized sentinel, the plan said
  15 368 MB.

The named margin is the solver's per-component overhead (`overhead_bytes`): the price compared is
activation + overhead. The upper bound refuses a gross overprice (more than 10 % over the card).

SEEN RED (2026-10-09): `library_layout_transient_bytes` returning 0 -> the mochi cell reads
3 780 + 206 MB at `aten.native_group_norm::33`; `contiguous_copy_bytes` returning 0 -> the Sana
cell reads 15 368 + 799 MB at `aten.convolution::62`.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/prism/test_a_decode_is_priced_at_or_above_what_the_card_held.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import PrismSolver
from neurobrix.nbx.container import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, impose_rung,
                                              pin_dedicated_card, profile)

V100_32GB = {**V100_16GB, "id": "scenario-v100-32gb",
             "devices": [{**V100_16GB["devices"][0], "model": "Tesla V100-SXM2-32GB", "memory_mb": 32768}]}
MB = 2 ** 20


def _priced(monkeypatch, model, mode_flag, request_args, component):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    pin_dedicated_card(monkeypatch, 32501, 267, "the rack's card 2, 2026-09-25")
    impose_rung(monkeypatch, 32768)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", model, mode_flag, *request_args,
                                       "--prompt", "a cat"])
    request = request_input_config(args, man, man.get("family"), c.cache_path)
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    plan = s.solve_smart(c, profile(V100_32GB), request, mode=mode_flag.lstrip("-"))
    m = seen[component]
    print(f"{model}/{component} {mode_flag}: activation {m.activation_bytes / MB:.1f} MB + overhead "
          f"{m.overhead_bytes / MB:.1f} MB at {m.peak_op_uid}; tiling {plan.component_tiling}")
    return m, plan


@pytest.mark.parametrize("model, mode_flag, request_args, measured_mb, peak_op", [
    ("mochi-1-preview", "--compiled", ["--num-frames", "7", "--height", "320", "--width", "576"],
     4_706, "aten.convolution::33"),
    ("Sana_1600M_4Kpx_BF16", "--triton", ["--height", "3072", "--width", "4096"],
     18_434, "aten.relu::18"),
])
def test_the_decode_is_priced_at_or_above_the_card(monkeypatch, model, mode_flag, request_args,
                                                   measured_mb, peak_op):
    m, plan = _priced(monkeypatch, model, mode_flag, request_args, "vae")
    assert "vae" not in (plan.component_tiling or {}), plan.component_tiling   # the run was whole
    price = (m.activation_bytes + m.overhead_bytes) / MB
    assert price >= measured_mb, (price, measured_mb, m.peak_op_uid)
    assert price <= 1.10 * measured_mb, (price, measured_mb, m.peak_op_uid)
    assert m.peak_op_uid == peak_op, m.peak_op_uid
