"""When every component fits the rung alone but not together, the layer-streaming rung streams the largest
one instead of declining with an empty streamed set.

The Mac (2026-10-03 22:00): orpheus-3b-0.1-ft triton on a 24 GB unified host with ~10.4 GB usable —
"layer_streaming declined: no room for a single segment: ... beside the streamed []", every
whole-component strategy rejected on the KV budget: a refusal where the owner's rule says the engine
streams. Reproduced here with the Mac's profile (_pinned_machine.APPLE_M4_PRO, its default-9f169c79) and the
host reader set to 8 400 MB available; no card. Since 2026-10-25 (the owner, 15:16) the reading no longer sizes
the plan (it is the execution door), so the room is imposed through the census door at the 8 192 rung.

What would this file do if the code were wrong? The empty streamed set kept -> no plan, RED.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests.conftest import installed_container

REPO = Path(__file__).resolve().parents[3]
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


def test_a_model_whose_components_fit_alone_but_not_together_is_streamed(tmp_path):
    installed_container(MODEL)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1", "NBX_PRISM_BUDGET_MB": "8192",
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
    # The owner's rule: the engine never refuses — at the 8 192 rung orpheus's components fit alone and the plan
    # loads them one phase at a time (lazy_sequential, the rank above streaming since 2026-10-25) or streams the
    # largest; a refusal (the empty streamed set this file was written against) raises before the JSON and fails
    # the assertion above. Neither plan computes on the host.
    assert plan["strategy"] in ("lazy_sequential", "layer_streaming"), plan["strategy"]
    assert not str(plan["strategy"]).startswith("cpu"), plan["strategy"]
    if plan["strategy"] == "layer_streaming":
        assert 0 < plan["device_window_mb"] <= 8192, plan["device_window_mb"]
