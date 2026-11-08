"""A non-persistent executor of a lazy plan releases its language model at each request's end through the strategy's
own unload, so a second and third request hold no more than the first (2026-11-08, A's gate).

Before: the autoregressive flows ended a request with the session's bare `executor.cleanup()`, which released the
weight dict but never uninstalled the strategy (`RuntimeExecutor._unload_component_weights` does: a lazy plan's zero3
pins and ratchet first). Every later request pinned a second copy — deepseek-moe-16b-chat under `run --turns`:
30 434 MB of pinned host at turn 1, 60 868 MB at turn 2, killed in turn 3; DeepSeek-Coder-V2-Lite on a 32 GB card:
a device OOM in turn 2. Cold serving (a lazy plan behind `neurobrix serve`) ran the same path request after request.

What this test does if the code is wrong: a later request fails, the pinned host partition the loader reports grows,
the device holds, after the third request, a second copy of the model's weights (its plan's weight bytes), or the requests' texts differ — red (red on lot-a
f1a2c81c5 on a 32 GB V100: request 2's replay copied a weight into the freed memory of the first load, a CUDA error). Device test:
deepseek-moe-16b-chat (lazy_sequential on a V100), the machine's own profile, certified-only, Triton, a child process
on the card CUDA_VISIBLE_DEVICES names."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODEL = "deepseek-moe-16b-chat"

CHILD = r'''
import json, sys
from neurobrix.cli.utils import find_model
from neurobrix.nbx import NBXContainer
from neurobrix.core.prism import PrismSolver, load_profile
from neurobrix.core.prism.autodetect import get_or_create_default_profile
from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.core.runtime.executor import RuntimeExecutor
from neurobrix.core.runtime.output_dispatch import _extract_text
from neurobrix.cli import create_parser
from neurobrix.cli.commands.run import request_input_config
from neurobrix.triton.dtype import declare_hardware_profile
model = sys.argv[1]
hw = load_profile(get_or_create_default_profile())
declare_hardware_profile(hw)
path = find_model(model)
c = NBXContainer.load(str(path))
man = c.get_manifest() or {}
args = create_parser().parse_args(["run", "--model", model, "--prompt", "Why is the sky blue?", "--max-tokens", "8",
                                   "--triton"])
plan = PrismSolver().solve_smart(c, hw, request_input_config(args, man, man.get("family"), c.cache_path), mode="triton")
print("STRATEGY " + plan.strategy, flush=True)
from neurobrix.core.runtime.lm_facts import session_lm_name
_lm = session_lm_name(json.load(open(path / "topology.json"))["flow"]["generation"], plan.components)
print("LM_WEIGHT_BYTES %d" % plan.component_memory[_lm].weight_bytes, flush=True)
ex = RuntimeExecutor(NBXRuntimeLoader().load(str(path)), plan, mode="triton")
for i in range(3):
    o = ex.execute({"global.prompt": "Why is the sky blue?", "global.max_tokens": 8, "global.temperature": 0.0})
    print("TEXT " + json.dumps(_extract_text(o, ex)), flush=True)
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.sync_device()
    print("LIVE %d" % sum(DA._cuda_live_bytes.values()), flush=True)
'''


def test_three_requests_on_a_lazy_plan_hold_what_the_first_holds(tmp_path):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("device test: name the card with CUDA_VISIBLE_DEVICES")
    sys.path.insert(0, str(REPO / "tools"))
    import atelier
    if not (Path(atelier.root("cache")) / MODEL).exists():
        pytest.skip(f"{MODEL} is not in the engine cache")
    env = dict(os.environ, PYTHONNOUSERSITE="1", NBX_AUTOTUNE_CERTIFIED_ONLY="1",
               PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}")
    p = subprocess.run([sys.executable, "-c", CHILD, MODEL], env=env, capture_output=True, text=True, timeout=3000,
                       cwd=str(tmp_path))
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    strategy = re.search(r"STRATEGY (\S+)", out).group(1)
    if strategy not in ("lazy_sequential", "zero3", "component_placement_lazy", "layer_streaming"):
        pytest.skip(f"{MODEL} plans {strategy} on this card: not a lazy plan")
    pinned = [int(m) for m in re.findall(r"on pinned host \(~(\d+)MB\)", out)]
    live = [int(m) for m in re.findall(r"^LIVE (\d+)$", out, re.M)]
    texts = [json.loads(t) for t in re.findall(r"^TEXT (.*)$", out, re.M)]
    assert len(texts) == 3 and len(set(texts)) == 1, texts
    # the pinned host partition the loader reports (a plan that pins), and the device memory left after each request:
    # the same after the third request as after the first — nothing of an earlier request's load is still held
    assert len(set(pinned)) <= 1, f"the pinned host partition grew request after request: {pinned} MB"
    lm_bytes = int(re.search(r"LM_WEIGHT_BYTES (\d+)", out).group(1))
    assert len(live) == 3 and live[2] - live[0] < lm_bytes, (
        f"the device holds an earlier request's load: {live} bytes after each request, the model's weights "
        f"{lm_bytes} bytes")
