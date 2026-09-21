"""A census runs behind the door `CUDA_VISIBLE_DEVICES=`: no driver answers which vendor profile
applies, `arch_smem_budget` resolves EMPTY and caches empty, and every key the shadow records is
composed as if no profile existed — the bucket ladder unread, the request dimension exact
(measured 2026-09-21: the 16 GB and 32 GB catalogue censuses recorded matmul M = 226 and 3 136
where the ladder says 240 and 3 200), the SMEM budget unread. The census names its hardware
profile (`--hardware`), and the profile names its device's compute capability: the shadow binds
the launcher's target from it, so the vendor profile — ladder, budgets, config spaces — is the
one the certified keys will be served under.

Shapes: 226 is Wan's traced conditioning length (a real key of the catalogue), whose bucket top
on the Volta ladder is 240 — a value the exact form cannot produce by accident.
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
    import json, sys
    from neurobrix.kernels import census
    census.install(hardware_profile={"devices": [{"brand": "nvidia", "model": "Tesla V100-SXM2-16GB",
                                                  "compute_capability": "7.0", "architecture": "volta"}]})
    from neurobrix.kernels.ops import _configs
    from neurobrix.kernels.autotune_bucket import bucket_of
    prof = _configs.active_vendor_profile()
    print(json.dumps({"profile": prof.get("_profile"), "bucket_M_226": bucket_of("M", 226),
                      "smem": _configs.arch_smem_budget()}))
""")


def _probe() -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
           "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_shadow_resolves_the_profile_its_hardware_names_and_buckets_on_its_ladder():
    got = _probe()
    assert got["profile"] and "volta" in str(got["profile"]).lower(), got
    assert got["bucket_M_226"] == 240, got
    assert got["smem"], got
