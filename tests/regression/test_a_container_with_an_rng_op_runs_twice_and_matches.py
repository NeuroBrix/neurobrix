"""A container that draws inside its graph runs twice and matches — the
permanent guard D-RNG-DRAW-UNARMED-IN-A-FLOW owes (filed 2026-09-11).

Kokoro-82M once gave two wav files from two runs of one tree, one seed, one
request: the divergence began at `aten.rand::0` and propagated. The fix (the
RNG stream armed at the executor for every flow) is on the trunk; this guard
is what keeps it there. The list of containers is not typed here: it is
`tools/rng_census.py`, read from the graphs, so a flow added tomorrow with a
draw and no armed stream is caught by the suite and not by a byte gate on a
stale branch.

Each cell: the request that reaches the drawing component (read from the
topology — the default request, or the speech leg through `--mode audio`),
`--seed 42`, run twice in fresh processes, the output artefact hashed each
time. Two hashes, one value. A component no request the topology names can
reach is a FAILURE with that sentence, never a skip: a guard that skipped
would read as an answer.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src pytest tests/regression/test_a_container_with_an_rng_op_runs_twice_and_matches.py -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import CACHE_ROOT, FAMILY_TIMEOUT_S, MODEL_TIMEOUT_S, _read_manifest
from .test_all_models import REPO, _cli_inputs_for, _run_out_path, _runtime_python

sys.path.insert(0, str(REPO / "tools"))
import rng_census as RC  # noqa: E402

MODES = ["native", "triton"]
SPEECH_MAX_TOKENS = "32"        # the bounded decode of the voice bench row (benchmarks/config/rows.yml)


def _cells():
    census = RC.containers_with_rng_ops(CACHE_ROOT)
    params = []
    for name, comps in census.items():
        for mode in MODES:
            params.append(pytest.param(name, comps, mode, id=f"{name}::{mode}"))
    return params


def _request(name: str, comps: dict, mode: str):
    """(argv, output path, unreached components) — the request that reaches the drawing components."""
    meta = _read_manifest(CACHE_ROOT / name)
    topo = json.loads((CACHE_ROOT / name / "topology.json").read_text(encoding="utf-8"))
    reach = {c: RC.request_reaching(topo, c) for c in comps}
    unreached = [c for c, r in reach.items() if r is None]
    family, flow, gen_type = meta["family"], meta["flow"], meta["gen_type"]
    args = _cli_inputs_for(family, flow, gen_type, name)
    out = _run_out_path(name, mode, family, gen_type)
    if "speech" in reach.values():
        if "--mode" in args:
            args[args.index("--mode") + 1] = "audio"
        else:
            args += ["--mode", "audio"]
        args += ["--max-tokens", SPEECH_MAX_TOKENS]
        out = out.with_suffix(".wav")
    cmd = [_runtime_python(), "-u", "-m", "neurobrix", "run", "--model", name, "--seed", "42",
           "--output", str(out), *args]
    if mode == "triton":
        cmd.append("--triton")
    return cmd, out, unreached


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,comps,mode", _cells())
def test_a_container_with_an_rng_op_runs_twice_and_matches(name, comps, mode):
    cmd, out, unreached = _request(name, comps, mode)
    assert not unreached, (f"{name}: the drawing component(s) {unreached} are reached by no request the "
                           f"topology names — this guard cannot claim they ran; name the leg in the topology")
    timeout = max(MODEL_TIMEOUT_S.get(name, FAMILY_TIMEOUT_S.get(_read_manifest(CACHE_ROOT / name)["family"], 300)), 900)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"}
    env.pop("NBX_FORCE_RAND_SEED", None)           # the claim is the executor's stream, not a pinned draw
    shas = []
    for rep in (1, 2):
        if out.exists():
            out.unlink()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(REPO))
        assert proc.returncode == 0, (f"{name} {mode} run {rep} rc={proc.returncode}\n"
                                      f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}")
        assert out.exists(), f"{name} {mode} run {rep}: no artefact at {out}"
        shas.append(_sha(out))
    assert shas[0] == shas[1], (f"{name} {mode}: two runs of one request and one seed gave two artefacts "
                                f"({shas[0][:16]} vs {shas[1][:16]}) — an RNG draw in {list(comps)} is not on "
                                f"the executor's armed stream")
