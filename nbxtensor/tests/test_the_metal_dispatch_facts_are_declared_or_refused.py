"""The Metal runtime's two dispatch facts (command buffers in flight, dispatches per buffer) come from the device file, or the
runtime refuses by name — they were 32 / 512 written in metal_device.py under a bare `except` (the Dell 2026-11-05 01:20).
What this does if the fallback came back: a silent 32 — RED."""
import pytest

from nbxtensor.backend import metal_device as M


@pytest.mark.parametrize("key", ["max_command_buffers_in_flight", "max_dispatches_per_command_buffer"])
def test_an_undeclared_dispatch_fact_is_refused_by_name(monkeypatch, key):
    import nbxtensor.ops.wrappers as W
    monkeypatch.setattr(W, "_arch_dispatch_param", lambda k, d: None)
    with pytest.raises(RuntimeError, match=f"dispatch.{key}"):
        M._declared_dispatch_fact(key)


def test_a_declared_dispatch_fact_is_read_as_stated(monkeypatch):
    import nbxtensor.ops.wrappers as W
    monkeypatch.setattr(W, "_arch_dispatch_param", lambda k, d: 48)
    assert M._declared_dispatch_fact("max_command_buffers_in_flight") == 48


def test_every_apple_device_file_states_both_facts():
    import yaml
    from pathlib import Path
    files = sorted((Path(M.__file__).parent / "devices" / "apple").glob("*.yml"))
    assert files
    for f in files:
        d = (yaml.safe_load(f.read_text()) or {}).get("dispatch") or {}
        for key in ("max_command_buffers_in_flight", "max_dispatches_per_command_buffer"):
            assert int(d.get(key) or 0) > 0, f"{f.name}: dispatch.{key}"
