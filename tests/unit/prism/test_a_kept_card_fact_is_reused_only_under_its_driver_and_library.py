"""qa on 1215 (NEEDS, the orchestrator's decision): the expandable pool's chunk and the reload rates were kept by card
model alone and copied by `from_siblings` into every generated profile of the machine, whatever driver or library
measured them. Each is now kept with what it was measured under (`kept_launch_facts.card_key`: the driver the OS states,
the library with its commit for a source tree); a fact under another key is dropped (`drop_stale`) and measured again,
in the profile itself and in a sibling it would be taken from.

What each test would do if the code were wrong: a chunk of another driver reused with no sweep (red), a sweep paid again
under the same key (red), a sibling's stale chunk taken (red), reload rates of another library kept (red)."""
from __future__ import annotations

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import kept_launch_facts as K
from tests.unit.prism.test_class_facts_are_measured_once_per_card_model import (  # noqa: F401 — the fixture
    C16, CALLS, _read, _write, hw)


def _sweeps():
    return [c for c in CALLS if c[0] == "vmm"]


def test_a_chunk_kept_under_another_driver_is_swept_again_and_never_under_the_same(hw, monkeypatch):
    CALLS.clear()
    _write(hw, "default-one", [dict(C16, index=0)])
    A.ensure_measured_facts("default-one")
    assert _sweeps() == [("vmm", 0)]
    assert _read(hw, "default-one")["devices"][0]["measured"]["vmm_chunk_under"] == K.card_key()
    A.ensure_measured_facts("default-one")
    assert _sweeps() == [("vmm", 0)]                                       # the same key: never paid twice
    monkeypatch.setattr(K, "driver_key", lambda: "nvidia 999.99")          # another driver
    added = A.ensure_measured_facts("default-one")
    assert _sweeps() == [("vmm", 0), ("vmm", 0)]                           # measured again
    assert any("vmm_chunk_bytes" in s for s in added["stale_dropped"])
    assert _read(hw, "default-one")["devices"][0]["measured"]["vmm_chunk_under"]["driver"] == "nvidia 999.99"


def test_a_sibling_s_chunk_of_another_library_is_not_taken(hw, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(K, "library_key", lambda: "0.1.11+old")
    _write(hw, "default-one", [dict(C16, index=0)])
    A.ensure_measured_facts("default-one")
    assert _sweeps() == [("vmm", 0)]
    monkeypatch.setattr(K, "library_key", lambda: "0.1.11+new")
    _write(hw, "default-two", [dict(C16, index=0), dict(C16, index=1)])
    A.ensure_measured_facts("default-two")
    assert _sweeps() == [("vmm", 0), ("vmm", 0)]                           # the sibling's chunk was of the old library
    monkeypatch.setattr(K, "library_key", lambda: "0.1.11+new")
    _write(hw, "default-three", [dict(C16, index=0)])
    A.ensure_measured_facts("default-three")
    assert _sweeps() == [("vmm", 0), ("vmm", 0)]                           # default-two's, of this library: taken


def test_reload_rates_kept_under_another_library_are_dropped_and_under_this_one_kept():
    rates = {"weight_reload_gbps": {"triton": {"bfloat16": 2.0}}, "weight_reload_source": ["triton bfloat16: 2.0 GB/s"]}
    stale = {"devices": [], "host_measured": dict(rates, weight_reload_under={"driver": K.driver_key(), "library": "0.1.9"})}
    assert K.drop_stale(stale) and stale["host_measured"] == {}
    unkeyed = {"devices": [], "host_measured": dict(rates)}
    assert K.drop_stale(unkeyed) and unkeyed["host_measured"] == {}        # a rate with no key is stale
    fresh = {"devices": [], "host_measured": dict(rates, weight_reload_under=K.card_key())}
    assert K.drop_stale(fresh) == [] and fresh["host_measured"]["weight_reload_gbps"]["triton"]["bfloat16"] == 2.0
    host = {}
    K.put_reload(host, "triton", "bfloat16", 2.5, None, "now")
    assert host["weight_reload_under"] == K.card_key()


def test_the_driver_key_is_read_from_the_os_and_names_something():
    assert K.driver_key() and isinstance(K.driver_key(), str)
    assert K.card_key() == {"driver": K.driver_key(), "library": K.library_key()}
