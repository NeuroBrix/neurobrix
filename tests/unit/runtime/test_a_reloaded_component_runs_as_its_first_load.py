"""A component released and loaded again under one plan computes as its first load did (2026-11-08).

The Triton engine binds the constants whose consumers compute in fp32 once, when its sequence compiles
(`GraphExecutor._bind_fp32_constants`). A non-persistent executor (a CLI run, a cold-serving plan) releases its weights
after each request and keeps the compiled sequence; the reload brought the weights back at the loader's width and
nothing bound them again — every request after the first ran those constants at fp16 (TinyLlama's
block.0.input_norm.weight), and a replay plan frozen during the first request's decode met the fp16 copy: "replay slot
mismatch at slot 8" on the second turn of `neurobrix run --turns`.

The rebind must reproduce the first load's binding exactly, not re-derive it: DeepSeek-Coder-V2-Lite-Instruct's first
compile bound a narrower set than the contract names once its keys are reconciled, so a rebind under the contract
made block.0.input_norm.weight fp32 where the first load ran it fp16, and the second `run --turns` turn stopped on a
certified-only miss (matmul (128, 3072, 2048) fp32 x fp16 at block.0's query, 2026-11-08 22:08). The proof is the
dtype of every weight the language model's executor runs with, request 1 against request 2.

What this test does if the code is wrong: the second identical request on the same non-persistent executor dies on the
replay's slot mismatch or a certified-only miss, runs a weight at another dtype than the first request, or its
first-step logits / tokens differ from the first request's — red (TinyLlama red on lot-a f1a2c81c5, DeepSeek-Coder
red on 8e3abefe0). Device test: the models from the engine cache, on the card CUDA_VISIBLE_DEVICES names, the
machine's own profile, certified-only, in a child process."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODELS = ["TinyLlama-1.1B-Chat-v1.0", "DeepSeek-Coder-V2-Lite-Instruct"]

CHILD = r'''
import hashlib, json, os, sys
from neurobrix.cli.utils import find_model
from neurobrix.nbx import NBXContainer
from neurobrix.core.prism import PrismSolver, load_profile
from neurobrix.core.prism.autodetect import get_or_create_default_profile
from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.core.runtime.executor import RuntimeExecutor
from neurobrix.cli import create_parser
from neurobrix.cli.commands.run import request_input_config
from neurobrix.triton.dtype import declare_hardware_profile
from neurobrix.core.runtime.graph_executor import GraphExecutor
model, dump = sys.argv[1], sys.argv[2]
seen = {}                      # request -> {component: {weight: dtype}} at the component's first run of the request
request = [0]
_real = GraphExecutor._run_triton_compiled
def _snap(self, *a, **k):
    by = seen.setdefault(request[0], {})
    comp = str(getattr(self, "_component_name", None) or id(self))
    if comp not in by:
        by[comp] = {str(n): str(getattr(w, "nbx_dtype", type(w).__name__)) for n, w in (self._weights or {}).items()}
    return _real(self, *a, **k)
GraphExecutor._run_triton_compiled = _snap
hw = load_profile(get_or_create_default_profile())
declare_hardware_profile(hw)
path = find_model(model)
c = NBXContainer.load(str(path))
man = c.get_manifest() or {}
args = create_parser().parse_args(["run", "--model", model, "--prompt", "Why is the sky blue?", "--max-tokens", "12",
                                   "--triton"])
plan = PrismSolver().solve_smart(c, hw, request_input_config(args, man, man.get("family"), c.cache_path), mode="triton")
ex = RuntimeExecutor(NBXRuntimeLoader().load(str(path)), plan, mode="triton")
out = []
for i in range(2):
    request[0] = i
    os.environ["NBX_DUMP_LOGITS"] = dump
    o = ex.execute({"global.prompt": "Why is the sky blue?", "global.max_tokens": 12, "global.temperature": 0.0})
    toks = [int(t) for t in (o.get("output_tokens") or o.get("global.output_tokens"))]
    logits = json.load(open(dump))["logits"]
    out.append({"tokens": toks, "logits": hashlib.sha256(json.dumps(logits).encode()).hexdigest(),
                "dtypes": seen.get(i, {})})
print("RESULT " + json.dumps(out))
'''


@pytest.mark.parametrize("model", MODELS)
def test_a_second_request_after_a_reload_gives_the_first_requests_bytes(tmp_path, model):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("device test: name the card with CUDA_VISIBLE_DEVICES")
    sys.path.insert(0, str(REPO / "tools"))
    import atelier
    if not (Path(atelier.root("cache")) / model).exists():
        pytest.skip(f"{model} is not in the engine cache")
    env = dict(os.environ, PYTHONNOUSERSITE="1", NBX_AUTOTUNE_CERTIFIED_ONLY="1",
               PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}")
    p = subprocess.run([sys.executable, "-c", CHILD, model, str(tmp_path / "logits.json")], env=env,
                       capture_output=True, text=True, timeout=1200, cwd=str(tmp_path))
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    first, second = json.loads(p.stdout.split("RESULT ", 1)[1].splitlines()[0])
    assert first["dtypes"], "no component ran on the Triton engine: the snapshot saw nothing"
    for comp, dt in first["dtypes"].items():
        moved = {n: (d, second["dtypes"].get(comp, {}).get(n)) for n, d in dt.items()
                 if second["dtypes"].get(comp, {}).get(n) != d}
        assert not moved, f"{comp}: the reload bound {len(moved)} weight(s) otherwise, e.g. {list(moved.items())[:4]}"
    assert len(first["tokens"]) > 2
    assert second == first, f"the reloaded request computed otherwise: {first} vs {second}"
