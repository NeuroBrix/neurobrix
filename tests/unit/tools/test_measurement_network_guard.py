"""The guard refuses a machine that is not on the network it measures from.

2026-09-10: this machine joined 192.168.1.0/24 instead of the lab's
10.0.0.0/24. Traffic to the file server went to the internet gateway, a 12 GB
stage died at 2 GB, and it was diagnosed after the fact, from the corpse. A
machine that joins the wrong network once will join it again.

Two things here are pinned because each was got wrong once while writing it:

  * the `mount` PARSER. The first version looked for `" nfs "` while macOS
    ends the line with `(nfs)`, so it found no servers at all and the guard
    passed in silence. A detector that finds nothing is worse than none, and
    it cannot be caught by running it on a healthy machine — which is exactly
    what a test with captured output is for.

  * the CRITERION. The first version connected to each server's nfsd with a
    2 s timeout. Run while a 12 GB stage saturated the link, every connect
    timed out and the guard refused a machine that was demonstrably on the
    right network. Subnet membership answers the question actually asked and
    cannot be perturbed by load; reachability stayed as a warning.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_measurement_network_guard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import check_measurement_environment as guard  # noqa: E402


# Captured verbatim from `mount` on this machine, 2026-09-11.
MACOS = """/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
devfs on /dev (devfs, local, nobrowse)
10.0.0.40:/home/mlops/NeuroBrix_System on /Users/hocine/Mounts/Dell-NeuroBrix (nfs)
10.0.0.20:/nvme/neurobrix_cache on /Users/hocine/Mounts/Super-NeuroBrix-Cache (nfs)
10.0.0.20:/data/models on /Users/hocine/Mounts/Super-Models (nfs)
map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)
"""

# The Linux form, which the same parser has to read: the rig has two sides.
LINUX = """/dev/nvme0n1p2 on / type ext4 (rw,relatime)
192.168.100.1:/data/models on /home/mlops/NeuroBrix_System/models type nfs (rw,vers=4.2)
192.168.100.1:/nvme/ai_work on /mnt/nvme_ai_work type nfs4 (rw,relatime)
tmpfs on /run type tmpfs (rw,nosuid)
"""


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

def test_the_macos_form_is_read():
    assert guard.parse_nfs_servers(MACOS) == ["10.0.0.40", "10.0.0.20"]


def test_the_linux_form_is_read():
    assert guard.parse_nfs_servers(LINUX) == ["192.168.100.1"]


def test_a_server_named_twice_is_listed_once():
    assert guard.parse_nfs_servers(MACOS).count("10.0.0.20") == 1


def test_non_nfs_lines_are_not_servers():
    assert guard.parse_nfs_servers(
        "/dev/disk3s1s1 on / (apfs, sealed, local)\ndevfs on /dev (devfs)\n") == []


def test_a_hostname_is_not_mistaken_for_an_address():
    """The check compares /24s, so only a literal address can answer it."""
    assert guard.parse_nfs_servers(
        "fileserver:/data on /mnt/data (nfs)\n") == []


def test_the_parser_finds_something_on_this_machine_if_anything_is_mounted():
    """The silence that hid the first version: a parser that matches nothing
    looks identical to a machine with no mounts."""
    import subprocess
    out = subprocess.run(["mount"], capture_output=True, text=True).stdout
    if ":" not in out or "nfs" not in out.lower():
        pytest.skip("no NFS mounted here")
    assert guard.parse_nfs_servers(out), (
        "mount output names NFS mounts and the parser found none")


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------

def _with(monkeypatch, servers, local):
    monkeypatch.setattr(guard, "_nfs_servers_and_local_addresses",
                        lambda: (servers, local))


def test_the_wrong_subnet_is_refused_and_both_addresses_are_named(monkeypatch):
    _with(monkeypatch, ["10.0.0.20", "10.0.0.40"], ["192.168.1.163"])
    problems = guard.check_on_the_measurement_network()
    assert problems, "a machine on 192.168.1.0/24 measuring from 10.0.0.0/24 must refuse"
    text = problems[0]
    assert "192.168.1.163" in text, "the address it sees is not named"
    assert "10.0.0.20" in text and "expected 10.0.0.0/24" in text, (
        "the address it expected is not named")


def test_the_right_subnet_passes(monkeypatch):
    _with(monkeypatch, ["10.0.0.20"], ["10.0.0.217", "100.76.114.44"])
    assert guard.check_on_the_measurement_network() == []


def test_one_interface_on_the_right_network_is_enough(monkeypatch):
    """A machine with VM bridges and a VPN holds several addresses; only one
    of them has to reach the lab."""
    _with(monkeypatch, ["10.0.0.20"],
          ["192.168.1.163", "10.211.55.2", "10.0.0.217"])
    assert guard.check_on_the_measurement_network() == []


def test_no_mounts_is_not_a_refusal(monkeypatch):
    """A machine that measures nothing over the network is not off it."""
    _with(monkeypatch, [], ["192.168.1.163"])
    assert guard.check_on_the_measurement_network() == []


def test_a_machine_with_no_address_at_all_is_refused_and_says_so(monkeypatch):
    _with(monkeypatch, ["10.0.0.20"], [])
    problems = guard.check_on_the_measurement_network()
    assert problems and "no routable address" in problems[0]


def test_reachability_is_a_warning_and_never_a_refusal():
    """The transient that made the first version refuse a healthy machine.
    Kept structural: this must not depend on a server being up right now."""
    import inspect
    refusal = inspect.getsource(guard.main)
    assert "problems += check_on_the_measurement_network()" in refusal
    assert "warnings += warn_servers_answer()" in refusal
    assert "problems += warn_servers_answer()" not in refusal
