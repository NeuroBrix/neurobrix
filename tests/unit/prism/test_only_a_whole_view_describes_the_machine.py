"""The shared `default.yml` is the MACHINE's profile. A process that sees part
of the machine may not write it.

`default.yml` is what a process with no `CUDA_VISIBLE_DEVICES` reads — the
battery is such a process. On 2026-09-17 it read `2 x Tesla V100-SXM2-16GB`,
32 GB total, on a rack of two 16 GB and two 32 GB cards: 96 GB, two thirds of
it invisible, written by a run pinned to `CUDA_VISIBLE_DEVICES=0,1`.

The branch that writes it is only reached by a process that HAS a mask set, so
its only writers are by construction the ones most likely to be partial. The
guard was `tag != "cpu"` — the single partial view that had been caught, a
process seeing no card at all (2026-09-13). Naming the instance instead of the
class left every other partial view holding the pen.

Pairs with `tests/unit/kernels/test_the_device_classes_come_from_the_visible_mask.py`:
NVML is the authority for what the RACK has and the wrong one for what THIS
PROCESS may touch; here the wrong direction is the other one — a process-scoped
answer overwriting a rack-scoped file.
"""

import pytest

from neurobrix.core.prism import autodetect


def _profile(n_devices):
    return {"devices": [{"index": i, "memory_mb": 16384} for i in range(n_devices)],
            "summary": {"total_gpus": n_devices}}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A profile directory of our own, and a machine whose size we dictate."""
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    return tmp_path


def _arrange(monkeypatch, *, machine, seen, tag="abcd1234"):
    monkeypatch.setattr(autodetect, "_machine_device_count", lambda: machine)
    monkeypatch.setattr(autodetect, "_visible_set_tag", lambda: tag)
    monkeypatch.setattr(autodetect, "detect_hardware", lambda: _profile(seen))


def test_a_whole_view_writes_the_machines_shared_profile(rig, monkeypatch):
    _arrange(monkeypatch, machine=4, seen=4)
    assert autodetect.get_or_create_default_profile() == "default-abcd1234"
    assert (rig / "default.yml").exists(), "a process seeing every card describes the machine"
    assert (rig / "default-abcd1234.yml").exists()


def test_a_partial_view_writes_only_its_own_profile(rig, monkeypatch):
    # The exact 2026-09-17 shape: pinned to 2 of the rack's 4 cards.
    _arrange(monkeypatch, machine=4, seen=2)
    assert autodetect.get_or_create_default_profile() == "default-abcd1234"
    assert (rig / "default-abcd1234.yml").exists(), "it still gets its OWN profile"
    assert not (rig / "default.yml").exists(), \
        "two of four cards is not the machine and must not be written as it"


def test_a_process_that_sees_no_card_still_may_not(rig, monkeypatch):
    # The 2026-09-13 instance, which must stay fixed by the general rule.
    _arrange(monkeypatch, machine=4, seen=0, tag="cpu")
    autodetect.get_or_create_default_profile()
    assert not (rig / "default.yml").exists()


def test_an_unknowable_machine_size_is_not_a_permission(rig, monkeypatch):
    # No nvidia-smi, a non-NVIDIA box: the writer cannot prove it sees
    # everything, so it does not write the file others cannot check.
    _arrange(monkeypatch, machine=None, seen=2)
    autodetect.get_or_create_default_profile()
    assert not (rig / "default.yml").exists()


def test_an_existing_profile_is_reused_and_nothing_is_written(rig, monkeypatch):
    _arrange(monkeypatch, machine=4, seen=4)
    (rig / "default-abcd1234.yml").write_text("id: already-here\n")
    assert autodetect.get_or_create_default_profile() == "default-abcd1234"
    assert not (rig / "default.yml").exists(), "an early return writes nothing at all"
    assert (rig / "default-abcd1234.yml").read_text() == "id: already-here\n"


def test_the_machine_count_ignores_the_mask(monkeypatch):
    """It must ask NVML, which is blind to CUDA_VISIBLE_DEVICES — that
    blindness is the whole reason it is the right authority here."""
    seen = {}

    class R:
        returncode = 0
        stdout = "0\n1\n2\n3\n"

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    monkeypatch.setattr(autodetect.subprocess, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert autodetect._machine_device_count() == 4, \
        "the machine has four cards whatever this process may touch"
    assert "nvidia-smi" in seen["cmd"][0]


def test_a_missing_nvidia_smi_is_none_not_zero(monkeypatch):
    # None means "unknown" and zero would mean "a machine with no GPUs" — and
    # zero would compare equal to a cpu-only detection and grant the write.
    def boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(autodetect.subprocess, "run", boom)
    assert autodetect._machine_device_count() is None
    assert autodetect._describes_the_whole_machine(_profile(0)) is False
