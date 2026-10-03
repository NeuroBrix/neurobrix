"""A streamed plan states the device bytes it holds at its worst moment — its window — and the
unified-memory reservation takes that, not the components' sum.

The Mac (2026-10-03 21:59): b859a41f reserved `planned_memory_mb` on a unified host, and for a
layer_streaming plan that is every component at full residency (granite-speech 18 085 MB), so
granite, Janus, orpheus and Flex were never admitted on a 24 GB Mac. The window is what stays
resident beside the segments plus the largest segment's weights and the activations alive with it.

What would this file do if the code were wrong? The window not set on a streamed plan -> the first
test RED; the reservation taking total_memory_mb on a streamed plan -> the second RED.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
MODEL = "TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.skipif(not (CACHE / MODEL).exists(), reason=f"{MODEL} is not in this catalogue")
def test_a_streamed_plan_states_a_window_inside_its_rung():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
           "NBX_FORCE_STRATEGY": "layer_streaming", "NBX_PRISM_BUDGET_MB": "1024",
           "PYTHONPATH": str(REPO / "src"), "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--model", MODEL, "--prompt", "hi",
                        "--max-tokens", "4", "--triton", "--hardware", "default-ff6008b7",
                        "--explain-plan", "--json"], env=env, capture_output=True, text=True,
                       timeout=600, cwd=str(REPO))
    t = r.stdout
    plan = json.loads(t[t.index("{"):])
    plan = plan.get("plan", plan)
    assert plan["strategy"] == "layer_streaming", plan["strategy"]
    w = plan["device_window_mb"]
    assert w is not None and 0 < w <= 1024 < plan["planned_memory_mb"], (w, plan["planned_memory_mb"])


def test_the_unified_reservation_takes_the_window_of_a_streamed_plan():
    from neurobrix.core.prism.solver import unified_device_bytes
    unified = NS(devices=[NS(has_unified_memory=True)])
    assert unified_device_bytes(NS(total_memory_mb=18085.0, device_window_mb=9000.0), unified) == 9000 << 20
    assert unified_device_bytes(NS(total_memory_mb=18085.0, device_window_mb=None), unified) == 18085 << 20
    assert unified_device_bytes(NS(total_memory_mb=18085.0, device_window_mb=9000.0),
                                NS(devices=[NS(has_unified_memory=False)])) == 0
