"""A plan Prism accepts on unified memory is one the host ledger can admit: the resident memory the
descent takes off as "already out of the reading" is what the process holds NOW, never a high-water mark.

DeepSeek-Coder-V2-Lite-Instruct, `--explain-plan --json` on the Mac (tree 1e94a1dc, 2026-10-04 20:25,
15.5-15.7 GB free): layer_streaming at the 12 288 rung, a 11 304 MB window, a host side of 17 128 MB
(window + resident 5 025 + transient 800). The descent accepted 17 128 - 5 025 = 12 103 MB; the ledger
(tools/regression_matrix.py `plan_host_need`, `reserve_host`) reserves the whole 17 128 against the
machine's reading, so three replay rounds never admitted the cell. The resident term was ru_maxrss on
macOS — the process's HIGH-WATER mark: the same process read at least 12 288 MB free (it planned that
rung, base 0), so at the reading it held at most 15 500 - 12 288 = 3 212 MB, not 5 025.

The machine is pinned as the Mac's figures give it: the machine's reading 15 500 MB, the process holding
3 200 MB now (so it reads 12 300 MB free itself) after a 5 025 MB high-water mark. No card.
"""
import json
import resource
import sys
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism import host_footprint as HF
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, no_door, pin_host, profile

MODEL = "DeepSeek-Coder-V2-Lite-Instruct"
MACHINE_FREE_MB = 15500           # the machine's reading, the one the ledger reserves against
HELD_NOW_MB = 3200                # what the planning process holds when it reads
HIGH_WATER_MB = 5025              # its high-water mark: the Mac's plan record's resident


def _pin_the_mac(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, MACHINE_FREE_MB - HELD_NOW_MB, "the Mac, 2026-10-04 20:25, read in-process")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(HF, "process_footprint_now", lambda: HELD_NOW_MB << 20)
    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=HIGH_WATER_MB << 20))


def _plan(monkeypatch):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    try:
        root = container_root(MODEL)
        c = NBXContainer.load(str(root))
    except Exception as e:                                # a machine without the container
        pytest.skip(f"{MODEL} not in this machine's cache: {e}")
    man = json.loads((root / "manifest.json").read_text())
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "Write a quicksort in Python.",
                                       "--max-tokens", "32", "--triton"])
    ic = request_input_config(args, man, man.get("family"), root)
    _pin_the_mac(monkeypatch)
    return PrismSolver().solve_smart(c, profile(APPLE_M4_PRO), ic, mode="triton")


def test_the_resident_term_is_what_the_process_holds_now(monkeypatch):
    fp = _plan(monkeypatch).host_footprint
    assert fp["resident_bytes"] >> 20 == HELD_NOW_MB, fp["resident_bytes"] >> 20


def test_the_ledger_admits_what_prism_accepted(monkeypatch):
    """The ledger's need is the whole host side against the machine's reading."""
    p = _plan(monkeypatch)
    total = p.host_footprint["total_bytes"] >> 20
    assert total <= MACHINE_FREE_MB, (p.strategy, p.device_window_mb, total, p.unified_rungs_tried)
