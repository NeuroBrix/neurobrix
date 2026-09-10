"""Standing gate: no campaign harness reads a key its brick never emits.

The repository's other vacuity gates assert that a numeric BOUND is not too
loose, and they live beside the kernels they guard. This one guards the
harnesses, because the defect they could not see was neither a loose bound nor
in a test: `tools/vendor_correctness_cell.py::m_psnr_db` read `d.get("psnr", 0.0)`
against a brick emitting `psnr_db`, so every image comparison scored 0.0 dB and
NO image cell could ever report AGREES. A gate with no true branch is the
strongest vacuity there is, and it reported DIVERGES on two renders a human
eyeballed as correct.

The audit itself lives in `tools/harness_metric_key_audit.py`. This file makes
it standing, and — because a gate that has never fired has never been tested —
also proves it still fails on the defect it was written for.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AUDIT = REPO / "tools" / "harness_metric_key_audit.py"


def test_audit_exists() -> None:
    assert AUDIT.is_file(), f"the harness key audit is missing: {AUDIT}"


def test_no_harness_reads_a_key_its_brick_never_emits() -> None:
    """The audit over every campaign harness. A silent default fails the gate."""
    r = subprocess.run([sys.executable, str(AUDIT)], cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        "a harness metric reads a key its brick does not emit — it returns a "
        "verdict computed on a default:\n" + r.stdout[-3000:])


def test_the_audit_still_fails_on_the_defect_it_was_written_for(tmp_path) -> None:
    """A gate that has never fired has never been tested.

    Rebuilds the original defect — a metric reading `psnr` from a brick that
    emits `psnr_db` — and asserts the audit reports it AND exits non-zero.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "image_fidelity.py").write_text(
        "import json\n"
        "def compare(ref, test):\n"
        "    return {'ref': ref, 'test': test, 'psnr_db': 40.0, 'ssim': 0.99}\n"
        "def main():\n"
        "    print(json.dumps(compare('a', 'b')))\n")
    (tools / "consumer.py").write_text(
        "import json, subprocess, sys, os\n"
        "def m_psnr_db(ours, theirs, bound):\n"
        "    p = subprocess.run([sys.executable, 'tools/image_fidelity.py', theirs, ours,\n"
        "                        '--json'], capture_output=True, text=True)\n"
        "    d = json.loads(p.stdout)\n"
        "    return d, float(d.get('psnr', 0.0)) >= float(bound)\n")

    r = subprocess.run(
        [sys.executable, str(AUDIT),
         "--harnesses", "tools/consumer.py", "--bricks", "tools/image_fidelity.py"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert "SILENT-DEFAULT" in r.stdout, (
        "the audit did not report the defect it exists for:\n" + r.stdout)
    assert "'psnr'" in r.stdout or '"psnr"' in r.stdout, r.stdout
    assert r.returncode != 0, "the audit reported the defect but did not fail the gate"


def test_the_audit_reads_a_bricks_returned_payload() -> None:
    """A brick's keys are usually in a RETURNED dict, not an inline dumps().

    `tools/image_fidelity.py` is exactly that shape, and reading only the inline
    form once made this audit blind to the very brick it was written for.
    """
    sys.path.insert(0, str(REPO / "tools"))
    from harness_metric_key_audit import emitted_keys        # noqa: E402

    keys, complete = emitted_keys(REPO / "tools" / "image_fidelity.py")
    assert complete, "image_fidelity's emit set was not readable"
    assert "psnr_db" in keys, f"psnr_db not seen among {sorted(keys)}"
    assert "psnr" not in keys, "psnr is not emitted and must not be claimed as such"


def test_a_write_is_not_a_read() -> None:
    """`d["k"] = v` enriches a payload; counting it as a read cries wolf."""
    sys.path.insert(0, str(REPO / "tools"))
    from harness_metric_key_audit import read_keys           # noqa: E402

    fn = ast.parse("def f(d):\n    d['written'] = 1\n    return d['read']\n").body[0]
    keys = {r["key"] for r in read_keys(fn)}
    assert "read" in keys
    assert "written" not in keys, "a store-context subscript was counted as a read"


@pytest.mark.parametrize("payload,key,ok", [
    ({"psnr_db": 40.0}, "psnr_db", True),
    ({"psnr_db": 40.0}, "psnr", False),
])
def test_the_runtime_contract_is_loud(payload, key, ok) -> None:
    """`require_key` never returns a default: it raises, naming what was emitted."""
    sys.path.insert(0, str(REPO / "tools"))
    from harness_contract import HarnessContractError, require_key   # noqa: E402

    if ok:
        assert require_key(payload, key, produced_by="brick") == payload[key]
    else:
        with pytest.raises(HarnessContractError) as e:
            require_key(payload, key, produced_by="brick")
        assert "psnr_db" in str(e.value), "the failure must name what the brick DID emit"
