"""A compiled conv3d is priced at max(floor, fraction x input) — the floor being what the device's conv3d takes whatever
its size (`memory.compiled_conv3d_workspace_floor_bytes`, core/prism/op_workspace.py).

The Mac's law (RUN 37b, 2026-11-05, M4 Pro, MPS watermark bisection of one conv3d per process, 36 shapes): under the MPS
heap, ~1 GiB, no conv3d runs; above ~800 MB of input the workspace is 1.08-1.60 x the input. Before this the floor sat
in the vendor file and no code read it (rule 1: a field no code reads is a defect): a small conv3d was priced at
1.6 x a few MB on a device that cannot start it under 1 GiB.

What this test does if the code were wrong: the floor ignored — a small conv on the M4 Pro priced at its fraction (red at
the first test); the floor applied as an addition or on cuDNN — a large conv priced above 1.6 x its input, or a V100
conv priced above 0 (red at the second and third); a vendor file stating the fraction without the floor — red at the
fourth; a missing floor defaulted — red at the fifth. Card-less."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.attention_scratch import vendor_memory_of
from neurobrix.core.prism.op_workspace import conv3d_workspace_bytes

VENDORS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors"
GiB = 1 << 30


def _compiled_hook(brand, architecture):
    S = PrismSolver(); S._mode = "compiled"
    prof = SimpleNamespace(devices=[SimpleNamespace(memory_mb=24576, architecture=architecture, brand=brand, index=0)])
    return S._conv3d_workspace(prof)


def test_a_small_conv3d_on_the_m4_pro_is_priced_at_the_mps_heap():
    hook = _compiled_hook("apple", "apple_m4_pro")
    assert hook((1, 96, 3, 64, 64), (96, 96, 3, 3, 3), {}, 2, 2) == GiB


def test_a_large_conv3d_on_the_m4_pro_is_priced_at_its_fraction():
    hook = _compiled_hook("apple", "apple_m4_pro")
    x = (1, 128, 8, 482, 850)                                   # 838 MB at 2 bytes, above the floor at 1.6
    n = 1 * 128 * 8 * 482 * 850 * 2
    assert hook(x, (128, 128, 3, 3, 3), {}, 2, 2) == int(n * 1.6) > GiB


def test_cudnn_takes_no_floor():
    hook = _compiled_hook("nvidia", "volta")
    assert hook((1, 96, 3, 64, 64), (96, 96, 3, 3, 3), {}, 2, 2) == 0


def test_every_vendor_file_that_prices_a_conv3d_states_its_floor():
    missing = []
    for f in sorted(VENDORS.glob("*/*.yml")):
        mem = (yaml.safe_load(f.read_text()) or {}).get("memory") or {}
        if "compiled_conv3d_workspace_input_fraction" in mem:
            b = mem.get("compiled_conv3d_workspace_floor_bytes")
            if not isinstance(b, int) or b < 0:
                missing.append(f"{f.parent.name}/{f.name}: {b!r}")
    assert not missing, missing


def test_a_vendor_without_the_floor_is_refused_by_name():
    mem = dict(vendor_memory_of("apple", "apple_m4_pro")); mem.pop("compiled_conv3d_workspace_floor_bytes")
    with pytest.raises(KeyError, match="compiled_conv3d_workspace_floor_bytes"):
        conv3d_workspace_bytes(mem, 1000)
