"""`memory.pool.enabled`, `parked_cap_bytes`, `fit_slack`: the pool's rules for a device, answered by the library from
the device's file — the allocator's switch and cap, and what a planner prices the pool by. The NeuroBrix planner
(Prism) read the same keys and the same switch a second time from its own copy of the reading until 2026-11-12.

What this test would do if the code were wrong: the switch read off when it is on (or the reverse), a cap where the
file states none, a cap not the file's fraction of the device, a negative slack — RED."""
import pytest

from nbxtensor.memory import pool


def test_the_switch_is_on_unless_set_to_zero(monkeypatch):
    monkeypatch.delenv("NBX_ALLOC_POOL", raising=False)
    assert pool.enabled() is True
    monkeypatch.setenv("NBX_ALLOC_POOL", "0")
    assert pool.enabled() is False
    monkeypatch.setenv("NBX_ALLOC_POOL", "1")
    assert pool.enabled() is True


@pytest.fixture
def described(monkeypatch):
    files = {("v", "capped"): {"alloc_pool_parked_cap_fraction": 0.25, "alloc_pool_fit_slack_fraction": 1.0},
             ("v", "bare"): {}, ("v", "odd"): {"alloc_pool_fit_slack_fraction": -0.5,
                                               "alloc_pool_parked_cap_fraction": None}}
    from nbxtensor.backend import device_files
    monkeypatch.setattr(device_files, "param",
                        lambda vendor, name, section, key, default: files[(vendor, name)].get(key, default)
                        if section == "memory" else default)


def test_the_cap_is_the_file_s_fraction_of_the_device_and_none_is_unbounded(described):
    assert pool.parked_cap_bytes("v", "capped", 16 << 30) == 4 << 30
    assert pool.parked_cap_bytes("v", "bare", 16 << 30) == 0
    assert pool.parked_cap_bytes("v", "odd", 16 << 30) == 0


def test_the_slack_is_the_file_s_and_never_negative(described):
    assert pool.fit_slack("v", "capped") == 1.0
    assert pool.fit_slack("v", "bare") == 0.0 and pool.fit_slack("v", "odd") == 0.0


def test_the_shipped_files_answer_as_the_planner_read_them():
    # nvidia/volta caps at half the card (its file); a 32 GiB card parks at most 16 GiB
    assert pool.parked_cap_bytes("nvidia", "volta", 32 << 30) == 16 << 30


def test_the_application_s_door_answers_the_same(described, monkeypatch):
    import nbxtensor.memory as memory
    monkeypatch.setenv("NBX_ALLOC_POOL", "0")
    assert memory.pool_enabled() is False
    assert memory.pool_parked_cap_bytes("v", "capped", 8 << 30) == 2 << 30 and memory.pool_fit_slack("v", "capped") == 1.0
