"""Every flow the catalogue declares is reached by the dispatcher, and no flow module is an orphan.

`core/flow/stages/kokoro.py` and `vibevoice.py` sat in the tree for weeks reached by nothing (the
native Kokoro handlers had no caller; the VibeVoice ones answered executions of the `audio` flow
that no `audio` container declares) and still read raw vendor names the complete vocabulary renames.
They were removed on 2026-10-04 (the supervisor's decision), the one function the audio flow
reaches moving to `core/audio_frontend.py`. These cells keep an orphan from coming back unseen:

* every flow type the catalogue declares (`data/catalogue_flows.json`, read from the 49 cached
  containers' topology.json) and every family declares (`config/families/*.yml` `execution.flow_type`) has
  its dispatch branch (`GraphExecutor._create_flow_handler`), its compiled module that registers it,
  and its Triton module;
* the `audio` flow, which dispatches on each stage's `execution`, runs every execution its
  containers declare, with ONE set in both engines;
* the other flows that read `execution` name every value their containers declare; those that do
  not read it carry their declared executions as labels — pinned here, so a new one is seen;
* every module under `core/flow/` and `triton/flow/` is imported by another engine module (an
  orphan stage file goes red).

What these would do with an orphan stage file re-added: the last cell fails naming it; with a
declared execution the dispatcher does not run: the audio cell fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
CATALOGUE = json.loads((Path(__file__).parent / "data" / "catalogue_flows.json").read_text())["flows"]

# Flows whose handlers drive their components by name and never read a stage's `execution`: the
# values their containers declare are labels (Forge writes them), measured 2026-10-04.
LABELS_ONLY = {
    "autoregressive_generation": {"autoregressive", "forward"},
    "dual_ar": {"dual_ar", "forward"},
    "rnnt": {"rnnt_greedy", "rnnt_joint", "forward"},
    "next_token_diffusion": {"diffusion", "native_acoustic_decoder", "forward"},
}


def _declared_flow_types():
    return {v["type"] for v in CATALOGUE.values() if v["type"]} | _family_flow_types()


def _family_flow_types():
    out = set()
    for y in (ROOT / "config" / "families").glob("*.yml"):
        ft = ((yaml.safe_load(y.read_text()) or {}).get("execution") or {}).get("flow_type")
        if ft:
            out.add(ft)
    return out


def _executions(flow_type):
    out = set()
    for v in CATALOGUE.values():
        if v["type"] == flow_type:
            out |= set(v["executions"])
    return out


def _dispatch_src():
    return (ROOT / "core" / "runtime" / "executor.py").read_text()


def _modules(flow_type):
    from neurobrix.core.flow.base import COMPILED_FLOW_MODULES
    core = COMPILED_FLOW_MODULES[flow_type]
    branch = re.search(rf'flow_type == "{re.escape(flow_type)}":(.*?)(?:\n        elif |\n        else:)',
                       _dispatch_src(), re.S)
    assert branch, f"no dispatch branch for flow '{flow_type}'"
    triton = re.search(r"from (neurobrix\.triton\.flow\.\w+) import", branch.group(1))
    assert triton, f"the dispatch branch of '{flow_type}' imports no Triton flow module"
    path = lambda dotted: ROOT.parent / (dotted.replace(".", "/") + ".py")  # noqa: E731
    return path(core), path(triton.group(1))


def test_the_catalogue_is_the_49_and_the_families_declare_flows():
    assert len(CATALOGUE) == 49
    assert {"autoregressive_generation", "audio_llm", "iterative_process"} <= _family_flow_types()


def test_every_declared_flow_type_is_dispatched_in_both_engines():
    for t in sorted(_declared_flow_types()):
        core, triton = _modules(t)
        assert core.exists() and triton.exists(), t
        assert f'@register_flow("{t}")' in core.read_text(), f"{core} does not register '{t}'"


def test_the_audio_flow_runs_every_execution_its_containers_declare():
    from neurobrix.core.flow import audio as core_audio
    from neurobrix.triton.flow import audio as triton_audio
    assert core_audio.STAGE_EXECUTIONS == triton_audio.STAGE_EXECUTIONS
    assert _executions("audio") <= set(core_audio.STAGE_EXECUTIONS)


def test_a_flow_that_reads_executions_names_every_declared_one_and_the_others_are_pinned():
    for t in sorted(_declared_flow_types() - {"audio"}):
        core, triton = _modules(t)
        srcs = core.read_text(), triton.read_text()
        declared = _executions(t)
        if all('"execution"' not in s for s in srcs):
            assert declared == LABELS_ONLY.get(t, set()), (t, declared)
            continue
        for e in sorted(declared):
            for s, p in zip(srcs, (core, triton)):
                assert f'"{e}"' in s, f"{p.name} reads executions and never names '{e}' ({t})"


def test_no_flow_module_is_an_orphan():
    texts = {p: p.read_text() for p in ROOT.rglob("*.py") if "triton_kernels_ref" not in p.parts}
    orphans = []
    for base in (ROOT / "core" / "flow", ROOT / "triton" / "flow"):
        for p in sorted(base.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            dotted = ".".join(p.relative_to(ROOT.parent).with_suffix("").parts)
            name, pkg = p.stem, ".".join(p.parent.relative_to(ROOT.parent).parts)
            pats = [re.compile(rf"\b{re.escape(dotted)}\b"),
                    re.compile(rf"\bfrom {re.escape(pkg)} import [^\n]*\b{name}\b"),
                    re.compile(rf"\bfrom \.{name} import\b"), re.compile(rf"\bfrom \. import [^\n]*\b{name}\b")]
            used = any(q != p and (any(r.search(t) for r in pats[:2]) or
                                   (q.parent == p.parent and any(r.search(t) for r in pats[2:])))
                       for q, t in texts.items())
            if not used:
                orphans.append(p.relative_to(ROOT).as_posix())
    assert orphans == [], f"flow modules no engine module imports: {orphans}"
