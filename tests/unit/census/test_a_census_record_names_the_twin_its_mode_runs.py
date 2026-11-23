"""A census input record names the code of the branch its jobs ran (qa 1286-1): the derivation runs Triton modes only
(tools/derived_census.py MODE_FLAGS), so a record whose units name the PyTorch side's twin (pytorch/prism,
pytorch/neurotax) would not mark a certificate moved when the Triton twin it really executed changes.

What it would do if the claim were wrong: one record of a Triton job naming a `repo:src/neurobrix/pytorch/` unit or
file is listed by model, red."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import derived_census as DC  # noqa: E402

SIDE_OF_MODE = {mode: ("triton" if mode.startswith("triton") else "pytorch") for mode in DC.MODE_FLAGS}


def test_the_derivation_runs_only_triton_modes():
    assert set(SIDE_OF_MODE.values()) == {"triton"}, SIDE_OF_MODE


def test_every_record_names_the_twin_of_the_branch_its_jobs_ran():
    wrong = []
    for path in sorted((REPO / "src/neurobrix/config/census").glob("*/*/*.inputs.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            sides = {SIDE_OF_MODE[job.split("|", 1)[0]] for job in rec.get("jobs", {})}
            other = {"triton": "pytorch", "pytorch": "triton"}
            for side in sides:
                stem = f"repo:src/neurobrix/{other[side]}/"
                named = [k for k in (*rec.get("units", {}), *rec.get("files", {})) if k.startswith(stem)]
                if named:
                    wrong.append((path.parent.name + "/" + path.name, rec.get("model"), named[0], len(named)))
    assert not wrong, wrong[:5]
