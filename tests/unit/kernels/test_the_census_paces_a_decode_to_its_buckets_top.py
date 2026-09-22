"""A census shadow walks a request loop by the KEY CLASSES it produces, never iteration by
iteration (the owner, 2026-09-21 23:31): a decode of 2 048 tokens costs 2 048 shadow steps
where the bucketed key changes only at the ladder's tops. `census.pace(length)` answers how
many positions a loop may skip to land on the current bucket's top; outside the shadow it
answers 0 so no served path moves.

Measured on chatterbox (16 GB profile, 2 048 speech tokens): 541.7 s unpaced, 37.9 s paced,
the decode's keys identical; what differed was the vocoder's value-filtered length, a class
the shadow enumerates on its own.

The probe runs behind the census door in a subprocess, bound to the Volta profile the way the
census tool binds it (the ladder is the profile's: exact to 64, 16 to 256, 32 to 1 024, 128 to
8 192). Shapes: 64 is the last exact length (the next, 65, keys at 80: 15 to skip); 255 sits
one under the top 256 (0); 256's successor keys at 288 under the 32-step (31); 1 024's at
1 152 under the 128-step (127) — one length per ladder step, so a wrong step size at any rung
fails its own case.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

PROBE = textwrap.dedent("""
    import json
    from neurobrix.kernels import census
    census.install(hardware_profile={"devices": [{"brand": "nvidia", "model": "Tesla V100-SXM2-16GB",
                                                  "compute_capability": "7.0", "architecture": "volta"}]})
    paced = {n: census.pace(n) for n in (64, 65, 255, 256, 1024)}
    census._ACTIVE["census"] = False
    live = {n: census.pace(n) for n in (64, 1024)}
    from neurobrix.triton import kv_cache as kvc
    wrapper_cls = next(c for c in vars(kvc).values() if isinstance(c, type) and hasattr(c, "skip_positions"))
    w = wrapper_cls.__new__(wrapper_cls)
    layers = {i: type("L", (), {"current_len": 5})() for i in range(2)}
    w.cache = type("C", (), {"_layers": layers})()
    w.skip_positions(12)
    print(json.dumps({"paced": paced, "live": live, "kv": [l.current_len for l in layers.values()]}))
""")


def _probe() -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
           "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_pace_lands_the_next_position_on_its_buckets_top_and_nothing_moves_live():
    got = _probe()
    assert got["paced"] == {"64": 15, "65": 14, "255": 0, "256": 31, "1024": 127}, got
    assert got["live"] == {"64": 0, "1024": 0}, got
    assert got["kv"] == [17, 17], got
