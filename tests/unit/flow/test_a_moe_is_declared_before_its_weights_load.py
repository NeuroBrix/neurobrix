"""A flow declares its language model's MoE before it loads the model's weights, in every flow of both engines.

The load reads the graph to choose what it loads; a plan that maps the routed experts from their files
(`ComponentAllocation.expert_files`, unified memory) reads them off the FUSED graph
(`core/runtime/expert_files.expert_loader_keys`). An llm-family graph is fused at construction, a vlm-family
graph only by the flow's declaration (`GraphExecutor.set_moe_config`) — and the vlm flows declared AFTER the load:
Qwen3-Omni-30B-A3B-Instruct on the Mac's 24 GB unified pool, both engines, 2026-10-14 ("the plan maps this
component's routed experts from its files, and its graph holds no fused routed layer"). Declaring first is the
executor's documented order (`_load_what_the_rewrite_added`: nothing to do before the first load, the load then
reads the rewritten graph).

What this test does if the code is wrong: it fails for every method of the flows that loads the language model
(`_ensure_weights_loaded(lm_name)`) before it declares the MoE (`set_moe_config`) — red on main at the six vlm
stages and the two autoregressive sessions.

Run: python -m pytest tests/unit/flow/test_a_moe_is_declared_before_its_weights_load.py
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
FLOWS = ["core/flow/vlm.py", "triton/flow/vlm.py", "core/flow/autoregressive.py", "triton/flow/autoregressive.py"]


def _sites(path):
    tree = ast.parse((SRC / path).read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loads, declares = [], []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_ensure_weights_loaded" and node.args and isinstance(node.args[0], ast.Name) \
                        and node.args[0].id == "lm_name":
                    loads.append(node.lineno)
                elif node.func.attr == "set_moe_config":
                    declares.append(node.lineno)
        if loads and declares:
            yield fn.name, min(loads), min(declares)


@pytest.mark.parametrize("path", FLOWS)
def test_every_language_model_load_follows_its_moe_declaration(path):
    sites = list(_sites(path))
    assert sites, f"{path}: no method both loads the language model and declares its MoE — the test would be empty"
    early = [f"{name} (load line {ld}, declaration line {dc})" for name, ld, dc in sites if ld < dc]
    assert not early, f"{path}: the weights load before the MoE is declared in {early}"
