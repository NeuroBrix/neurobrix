"""When every component fits the rung alone but not together, the layer-streaming rung streams the largest
one instead of declining with an empty streamed set.

The Mac (2026-10-03 22:00): orpheus-3b-0.1-ft triton on a 24 GB unified host with ~10.4 GB usable —
"layer_streaming declined: no room for a single segment: ... beside the streamed []", every
whole-component strategy rejected on the KV budget: a refusal where the owner's rule says the engine
streams. Reproduced here with the Mac's profile (_pinned_machine.APPLE_M4_PRO, its default-9f169c79) and the
host reader set to 8 400 MB available; no card.

What would this file do if the code were wrong? The empty streamed set kept -> no plan, RED.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
MODEL = "orpheus-3b-0.1-ft"

DRIVER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from neurobrix.core import host_memory as HM
    from neurobrix.core.prism import solver as S, loader as LD
    st = HM.MemoryState(available_mb=8400, source="simulated")
    HM.memory_state = lambda: st
    S.memory_state = lambda: st
    S._census_shadow_active = lambda: False      # the census door supplies the device; the room is simulated
    LD.HARDWARE_DIR = Path(sys.argv[1])
    from neurobrix.cli import main
    sys.argv = ["neurobrix", "run", "--model", sys.argv[2], "--prompt", "hello", "--seed", "42", "--triton",
                "--hardware", sys.argv[3], "--explain-plan", "--json"]
    main()
""")


@pytest.mark.skipif(not (CACHE / MODEL).exists(), reason=f"{MODEL} is not in this catalogue")
def test_a_model_whose_components_fit_alone_but_not_together_is_streamed(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "PYTHONNOUSERSITE": "1"}
    import yaml
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO
    (tmp_path / f"{APPLE_M4_PRO['id']}.yml").write_text(yaml.safe_dump(APPLE_M4_PRO))
    r = subprocess.run([sys.executable, "-c", DRIVER, str(tmp_path), MODEL, APPLE_M4_PRO["id"]],
                       env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO))
    t = r.stdout
    assert "{" in t, (r.stderr or t)[-1500:]
    plan = json.loads(t[t.index("{"):])
    plan = plan.get("plan", plan)
    assert plan["strategy"] == "layer_streaming", plan["strategy"]
    assert 0 < plan["device_window_mb"] < 8400 < plan["planned_memory_mb"] + 8400
