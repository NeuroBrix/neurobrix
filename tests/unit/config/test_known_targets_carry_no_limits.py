"""The known-targets list carries no hard limit, and cannot contradict the driver.

`config/vendors/KNOWN_TARGETS.md` exists for two things: good defaults where no
measurement exists, and naming a target in a certificate path. It is NOT a
source of hardware limits, and this is the gate that keeps it that way.

The reason is measured, not stylistic. One flash-attention tile
(BLOCK_M=128, BLOCK_N=64, BLOCK_HEADDIM=128) compiled by Triton 3.6 needs
98 304 bytes of shared memory on sm_70 and 164 352 on sm_86. The cost is a
function of the TARGET as much as of the tile, so no table of cards can hold the
answer — the quantity is not a property of the card alone. The limit comes from
the driver and the cost from the compiler; both are exact.

An entry in that file that contradicted the driver would be false by
construction. Since the file is not allowed to state limits at all, the way to
enforce that is to check it states none.

Run: PYTHONPATH=src python -m pytest tests/unit/config/test_known_targets_carry_no_limits.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config"
       / "vendors" / "KNOWN_TARGETS.md")

# Keys that name a hard limit. None of them may be ASSERTED here as this
# target's value — the driver owns every one of them.
LIMIT_KEYS = (
    "max_shared_memory_per_block",
    "max_threads_per_block",
    "max_registers_per_block",
    "warp_size",
    "max_grid",
    "regs_per_sm",
)


def test_the_file_exists_and_states_its_purpose():
    assert DOC.is_file(), f"{DOC} is missing"
    text = DOC.read_text()
    assert "carries no hard limit" in text, (
        "the file must state, in its first lines, that no hard limit is read "
        "from it — otherwise the next reader will add one")
    assert "false by construction" in text


def test_no_hard_limit_key_is_asserted():
    """A limit key appearing as a declared value is the failure this guards."""
    text = DOC.read_text()
    offenders = []
    for key in LIMIT_KEYS:
        for m in re.finditer(rf"{re.escape(key)}\s*[:=]\s*\S", text):
            offenders.append(f"{key} at offset {m.start()}")
    assert not offenders, (
        "KNOWN_TARGETS.md declares a hardware limit: " + ", ".join(offenders)
        + ". The driver owns these. Remove the value; name the target only.")


def test_no_byte_sized_limit_is_offered_as_this_target_s_budget():
    """`96 KB` in the prose that EXPLAINS the defect is fine; a table row that
    hands a per-target byte budget is not. The distinction is the table: a
    limit-shaped number inside a row keyed on a capability is a budget, however
    it is spelled."""
    text = DOC.read_text()
    bad = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        keyed_on_target = re.match(r"^(\d+\.\d+|gfx[0-9a-f]+)", cells[0])
        if keyed_on_target and re.search(r"\d+\s*(KB|MB|bytes)\b", " ".join(cells[1:])):
            bad.append(line.strip()[:90])
    assert not bad, (
        "a row keyed on a compute capability or gfx target hands a byte budget:\n  "
        + "\n  ".join(bad))


def test_the_sm120_trap_is_written_down():
    """sm_120 and sm_100 share the Blackwell name and no binary compatibility.
    A family name that silently means two targets costs a day; it is written."""
    text = DOC.read_text()
    assert re.search(r"sm_120.{0,80}NOT compatible.{0,40}sm_100", text, re.S | re.I), (
        "the sm_120 / sm_100 incompatibility must be stated explicitly")


def test_every_axis_names_its_source_and_when_it_was_read():
    """A target list without a dated source rots silently."""
    text = DOC.read_text()
    for vendor in ("NVIDIA", "AMD", "Intel"):
        i = text.index(f"## {vendor}")
        section = text[i:text.find("\n## ", i + 1)]
        assert "Source" in section or "source" in section, f"{vendor}: no source cited"
        assert re.search(r"20\d\d-\d\d-\d\d", section), (
            f"{vendor}: no date on the source — a table read once and never "
            f"dated cannot be re-checked")


def test_it_does_not_contradict_this_machine_s_driver():
    """The property the file claims for itself, checked where it can be: any
    capability this machine reports must appear, and the file must not offer a
    ceiling for it that differs from what the driver says."""
    try:
        from neurobrix.kernels.launcher import arch, max_shared_memory_per_block
        cap, limit = arch(), max_shared_memory_per_block()
    except Exception:
        pytest.skip("no CUDA driver here")
    if limit is None:
        pytest.skip("driver did not answer")
    spelled = f"{cap // 10}.{cap % 10}"
    text = DOC.read_text()
    assert re.search(rf"^\|\s*{re.escape(spelled)}\s*\|", text, re.M), (
        f"this machine reports capability {spelled} and the list does not name it")
    # And the file offers no number that could be mistaken for its budget.
    row = next(l for l in text.splitlines()
               if l.strip().startswith(f"| {spelled} "))
    assert not re.search(r"\d+\s*(KB|MB|bytes)\b", row), (
        f"the row for {spelled} carries a byte figure; the driver says {limit} "
        f"and the file must not offer a second opinion")
