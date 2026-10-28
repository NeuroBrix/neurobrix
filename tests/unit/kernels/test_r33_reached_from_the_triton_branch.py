"""R33 over the import graph the Triton branch REALLY reaches (supervisor 2026-10-28 20:10).

`test_r33_torch_import_guard.py` scans two directories, `kernels/` and `triton/`. The Triton entry points also reach
`core/` — Prism, the DtypeEngine, the shared flows, the modules (tokenizers, processors), the runtime's resolvers —
and a torch import there was outside every gate: `triton/flow/audio.py` tokenized a TTS prompt through
`core/module/text/processor.TextProcessor`, whose compiled branch imports torch, and converted whatever came back with
`.detach().cpu().numpy()`.

The graph is walked with `ast` from every module under `triton/`, through EVERY import — module level, inside a
function, under a condition (R33 counts conditional, transitive and dead-branch imports). Two tiers:

1. IMPORT time, no exception: the modules reached through module-level edges only (what `import` of a Triton module
   executes) import no torch at module level.
2. EXECUTION, a list that can only shrink: every reached module outside `kernels/` (that package has its own list)
   that imports torch anywhere is named in REACHED below with the KINDS of its imports — "type" (under
   `TYPE_CHECKING`, never executed), "deferred" (inside a function: a compiled-engine branch behind the mode test),
   "module" (at module level of a module the Triton branch reaches only through deferred edges). A new module, a new
   kind in a listed one, an entry no longer reached or no longer importing torch: each fails, and the message says
   which line of REACHED to change. The list shrinks as the shared bricks hand their compiled branch to one seam.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
ROOT = "neurobrix"

# module -> the kinds of its torch imports, as found on 2026-10-28 (lot-2026-10-28 ced5ee7b + this branch)
REACHED = {
    "neurobrix.core.components.base": {"type"},
    "neurobrix.core.components.handlers.text_encoder_handler": {"deferred", "type"},
    "neurobrix.core.components.handlers.vae_handler": {"deferred", "type"},
    "neurobrix.core.device_utils": {"deferred"},
    "neurobrix.core.dtype.calibration": {"deferred", "type"},
    "neurobrix.core.dtype.config": {"deferred", "type"},
    "neurobrix.core.dtype.converter": {"deferred", "type"},
    "neurobrix.core.dtype.engine": {"module"},
    "neurobrix.core.flow.audio_utils": {"deferred", "type"},
    "neurobrix.core.flow.base": {"type"},
    "neurobrix.core.io.loader": {"module"},
    "neurobrix.core.io.memory": {"module"},
    "neurobrix.core.io.weight_loader": {"module"},
    "neurobrix.core.memory.manager": {"type"},
    "neurobrix.core.module.audio.input_processor": {"deferred", "type"},
    "neurobrix.core.module.audio.output_processor": {"deferred"},
    "neurobrix.core.module.output_processor": {"type"},
    "neurobrix.core.module.temporal_stream_runner": {"deferred"},
    "neurobrix.core.module.tiling_engine": {"deferred", "type"},
    "neurobrix.core.module.tokenizer.sp_tokenizer": {"deferred"},
    "neurobrix.core.prism.cpu_config": {"deferred", "type"},
    "neurobrix.core.prism.facts_probe": {"deferred"},
    "neurobrix.core.prism.solver": {"type"},
    "neurobrix.core.runtime.debug": {"deferred"},
    "neurobrix.core.runtime.graph.attention_mask": {"module"},
    "neurobrix.core.runtime.graph.compiled_ops": {"module"},
    "neurobrix.core.runtime.graph.compiled_sequence": {"deferred", "module"},
    "neurobrix.core.runtime.graph.device_attention": {"module"},
    "neurobrix.core.runtime.graph.execution_context": {"type"},
    "neurobrix.core.runtime.graph.expert_store": {"module"},
    "neurobrix.core.runtime.graph.memory_pool": {"deferred", "type"},
    "neurobrix.core.runtime.graph.run_generator": {"module"},
    "neurobrix.core.runtime.graph.sequential_dispatcher": {"module"},
    "neurobrix.core.runtime.graph.tensor_resolver": {"deferred", "type"},
    "neurobrix.core.runtime.graph_executor": {"deferred", "type"},
    "neurobrix.core.runtime.output_dispatch": {"deferred"},
    "neurobrix.core.runtime.resolution.input_synthesizer": {"deferred", "type"},
    "neurobrix.core.runtime.resolution.output_extractor": {"type"},
    "neurobrix.core.runtime.resolution.variable_resolver": {"deferred", "type"},
    "neurobrix.core.runtime.shape_resolver": {"type"},
    "neurobrix.core.strategies.base": {"deferred"},
    "neurobrix.core.strategies.expert_store": {"deferred"},
    "neurobrix.core.strategies.zero3": {"deferred", "type"},
    "neurobrix.nbx.loader": {"type"},
}


def _modules(src: Path = SRC) -> dict:
    out = {}
    for p in src.rglob("*.py"):
        if "triton_kernels_ref" in str(p):
            continue                                    # reference tree, never executed
        parts = list(p.relative_to(src.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = p
    return out


def _is_torch(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


def _scan(name: str, mods: dict, top_only: bool):
    """(the package modules `name` imports, the kinds of its torch imports). `top_only`: the edges an import of
    `name` executes — module level, outside function bodies and outside `if TYPE_CHECKING:`."""
    path = mods[name]
    pkg = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    targets, kinds = set(), set()

    def target(node):
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if node.level:
            base = pkg.split(".")[:len(pkg.split(".")) - (node.level - 1)]
            mod = ".".join(base + ([node.module] if node.module else []))
        else:
            mod = node.module or ""
        return [mod] + [f"{mod}.{a.name}" for a in node.names]

    def visit(node, kind):
        for ch in ast.iter_child_nodes(node):
            k = kind
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                k = "deferred"
            elif isinstance(ch, ast.If) and "TYPE_CHECKING" in ast.unparse(ch.test) and kind == "module":
                visit(ast.Module(body=ch.body, type_ignores=[]), "type")
                visit(ast.Module(body=ch.orelse, type_ignores=[]), kind)
                continue
            if isinstance(ch, (ast.Import, ast.ImportFrom)):
                names = target(ch)
                if any(_is_torch(n) for n in names[:1] if not getattr(ch, "level", 0)) or (
                        isinstance(ch, ast.Import) and any(_is_torch(n) for n in names)):
                    kinds.add(k)
                if not top_only or k == "module":
                    targets.update(names)
            visit(ch, k)

    visit(ast.parse(path.read_text()), "module")
    found = set()
    for t in targets:
        while t and t not in mods and "." in t:
            t = t.rsplit(".", 1)[0]
        if t in mods:
            parts = t.split(".")
            found.update(".".join(parts[:i]) for i in range(2, len(parts) + 1) if ".".join(parts[:i]) in mods)
    return found, kinds


def _reached(mods: dict, top_only: bool) -> dict:
    """{module reached from triton/: the module that first reached it}."""
    start = sorted(m for m in mods if m.startswith(f"{ROOT}.triton"))
    via = {m: None for m in start}
    stack = list(start)
    while stack:
        m = stack.pop()
        for d in _scan(m, mods, top_only)[0]:
            if d not in via:
                via[d] = m
                stack.append(d)
    return via


def _torch_in_reach(mods: dict, top_only: bool) -> dict:
    via = _reached(mods, top_only)
    out = {}
    for m in via:
        kinds = _scan(m, mods, top_only)[1]
        if kinds and not m.startswith(f"{ROOT}.kernels"):
            out[m] = (kinds, via[m])
    return out


def test_importing_the_triton_branch_imports_no_torch():
    mods = _modules()
    bad = {m: k for m, (k, _) in _torch_in_reach(mods, top_only=True).items() if "module" in k}
    assert not bad, f"a module the Triton branch imports at import time imports torch at module level: {bad}"


def test_every_torch_import_the_triton_branch_reaches_is_listed_with_its_kinds():
    found = _torch_in_reach(_modules(), top_only=False)
    new = {m: (sorted(k), via) for m, (k, via) in found.items() if m not in REACHED}
    assert not new, ("torch imported by a module the Triton branch reaches, outside REACHED (module: kinds, reached "
                     "from):\n" + "\n".join(f"  {m}: {k} <- {v}" for m, (k, v) in sorted(new.items()))
                     + "\nThe Triton branch takes NBXTensor end to end: hand the compiled branch to the one seam "
                       "(`variable_resolver.to_engine_container`) instead of importing torch here.")
    grown = {m: (sorted(k), sorted(REACHED[m])) for m, (k, _) in found.items() if m in REACHED and k - REACHED[m]}
    assert not grown, f"a listed module gained a kind of torch import (found, listed): {grown}"


@pytest.mark.parametrize("module", sorted(REACHED))
def test_every_entry_is_still_reached_and_still_imports_torch(module):
    found = _torch_in_reach(_modules(), top_only=False)
    assert module in found, f"{module} no longer imports torch or is no longer reached: delete its line from REACHED"
    assert found[module][0] == REACHED[module], (
        f"{module}: kinds now {sorted(found[module][0])}, listed {sorted(REACHED[module])} — narrow its line")


def test_the_walk_is_seen_failing_on_an_injected_import(tmp_path):
    """A triton module importing, inside a function, a core module that imports torch under a condition: both edges
    are followed and the import is named, with its kind."""
    pkg = tmp_path / ROOT
    (pkg / "triton").mkdir(parents=True)
    (pkg / "core").mkdir()
    for d in (pkg, pkg / "triton", pkg / "core"):
        (d / "__init__.py").write_text("")
    (pkg / "triton" / "flow.py").write_text("def run():\n    from neurobrix.core import brick\n    return brick\n")
    (pkg / "core" / "brick.py").write_text("def place(x, mode):\n    if mode != 'triton':\n        import torch\n")
    (pkg / "core" / "typed.py").write_text("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n")
    (pkg / "triton" / "eager.py").write_text("from ..core import typed\n")
    mods = _modules(pkg)
    found = _torch_in_reach(mods, top_only=False)
    assert found["neurobrix.core.brick"][0] == {"deferred"}
    assert found["neurobrix.core.typed"][0] == {"type"}
    assert "neurobrix.core.brick" not in _torch_in_reach(mods, top_only=True)
