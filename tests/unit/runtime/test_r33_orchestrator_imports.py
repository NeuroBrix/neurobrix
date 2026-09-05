"""R33, import-time half, for the shared orchestrator.

The modules both engines load (the executor stack, the registries, the
dtype table, the container loader, the device utilities) must import
without torch: a `--triton` process never loads the ATen branch. Each
module below is imported in a subprocess where torch is BLOCKED
(`tools/r33_import_without_torch.py`); a module that pulls torch at
import, itself or transitively, fails with the importer named.

The list is the pin: a module leaves it only with the peel that makes it
clean, and never comes back. The gate is seen failing on an injected
import (`test_the_gate_is_seen_failing_on_an_injected_import`).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
TOOL = REPO / "tools" / "r33_import_without_torch.py"

# Shared orchestrator modules proven torch-free at import (batch A of the
# 2026-09-05 peel). Extended by every later batch.
ORCHESTRATOR_MODULES = [
    "neurobrix.core.runtime.tensor_compat",
    "neurobrix.core.device_utils",
    "neurobrix.core.flow",
    "neurobrix.core.flow.base",
    "neurobrix.core.strategies",
    "neurobrix.core.strategies.base",
    "neurobrix.core.strategies.triton",
    "neurobrix.core.module.scheduler.base",
    "neurobrix.core.dtype",
    "neurobrix.core.dtype.config",
    "neurobrix.core.dtype.converter",
    "neurobrix.core.runtime.graph",
    "neurobrix.core.prism.cpu_config",
    "neurobrix.core.prism",
    "neurobrix.core.prism.solver",
    "neurobrix.core.strategies.zero3",
    "neurobrix.core.runtime.graph.tensor_resolver",
    "neurobrix.core.runtime.graph.memory_pool",
    "neurobrix.core.runtime.graph.execution_context",
    "neurobrix.core.components",
    "neurobrix.core.components.base",
    "neurobrix.core.module.output_processor",
    "neurobrix.nbx",
    "neurobrix.nbx.loader",
    "neurobrix.core.memory",
    "neurobrix.core.memory.manager",
    # batch B (the shared executor stack)
    "neurobrix.core.dtype.calibration",
    "neurobrix.core.runtime.precision_contract",
    "neurobrix.core.runtime.shape_resolver",
    "neurobrix.core.runtime.resolution",
    "neurobrix.core.runtime.resolution.variable_resolver",
    "neurobrix.core.runtime.resolution.input_synthesizer",
    "neurobrix.core.runtime.resolution.output_extractor",
    "neurobrix.core.module.tiling_engine",
    "neurobrix.core.runtime.graph_executor",
    "neurobrix.core.runtime.factory",
    "neurobrix.core.runtime.executor",
    # batch D (the input boundary: tokens, images, audio enter the engine's container)
    "neurobrix.core.module.tokenizer",
    "neurobrix.core.module.tokenizer.sp_tokenizer",
    "neurobrix.core.module.tokenizer.factory",
    "neurobrix.core.module.vision.input_processor",
    "neurobrix.core.module.audio.input_processor",
    "neurobrix.cli.commands.run",
    "neurobrix.cli.commands.upscale",
    "neurobrix.core.module.text",
    "neurobrix.core.module.text.processor",
    "neurobrix.core.runtime.output_dispatch",
    "neurobrix.core.module.audio.output_processor",
    "neurobrix.serving.engine",
]


def _run(names, extra_path=None):
    env = {"PYTHONPATH": str(SRC) + (f":{extra_path}" if extra_path else ""), "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home())}
    return subprocess.run([sys.executable, str(TOOL), *names], capture_output=True, text=True, env=env)


def test_the_shared_orchestrator_imports_without_torch():
    proc = _run(ORCHESTRATOR_MODULES)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    assert f"{len(ORCHESTRATOR_MODULES)}/{len(ORCHESTRATOR_MODULES)} imported without torch" in proc.stdout


def test_the_gate_is_seen_failing_on_an_injected_import(tmp_path):
    (tmp_path / "r33_injected_orchestrator.py").write_text(textwrap.dedent("""
        def helper():
            return 1
        import torch  # injected
    """))
    proc = _run(["r33_injected_orchestrator"], extra_path=tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "FAIL r33_injected_orchestrator" in proc.stdout
    # A dead-branch import counts as well: the blocker refuses the import
    # statement itself, wherever it sits.
    (tmp_path / "r33_injected_conditional.py").write_text(textwrap.dedent("""
        import os
        if os.environ.get("NEVER_SET_XYZ") is None:
            import torch.nn  # conditional, still executed at import
    """))
    proc = _run(["r33_injected_conditional"], extra_path=tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "FAIL r33_injected_conditional" in proc.stdout
