"""The render checkpoint must survive exactly the events it exists for.

It exists because the rack has one breaker and no UPS. So the cases that
matter are not the happy path: they are a file truncated mid-write, a
checkpoint belonging to a different run, and a scheduler that says it
cannot be resumed. Each of those must lead to "start from step 0", never to
a wrong resume and never to a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.runtime.render_checkpoint import RenderCheckpoint


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    ck = RenderCheckpoint.from_env(model="unit-model", num_steps=8,
                                   extra={"seed": 42, "cfg": 7.5})
    assert ck is not None
    return ck


# --- enable / disable -------------------------------------------------------

def test_ENABLED_by_default(monkeypatch, tmp_path):
    """POLICY REVERSED 2026-09-04, on the supervisor's instruction.

    It was off unless asked for. The rack has one breaker and no UPS, and a cut
    at hour thirteen of a fourteen-hour render loses everything — so a resume
    point that must be remembered is one nobody has when they need it.

    The cost objection that justified opt-in is answered by the TIME GATE
    instead (below): a render finishing inside the interval writes nothing.
    """
    monkeypatch.delenv("NBX_RENDER_CHECKPOINT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert RenderCheckpoint.from_env(model="m", num_steps=8) is not None


def test_a_short_render_writes_nothing(monkeypatch, tmp_path):
    """Default-on must not mean every render pays. The save is gated on
    ELAPSED TIME, so a render that finishes inside the gate never writes —
    which is what makes 'always on' affordable."""
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    monkeypatch.delenv("NBX_RENDER_CHECKPOINT_EVERY", raising=False)
    ck = RenderCheckpoint.from_env(model="m", num_steps=20)
    assert ck is not None and ck.min_interval_s > 0
    assert [i for i in range(19) if ck.should_save(i, 20)] == [], (
        "a render finishing inside the time gate must not write at all")


def test_an_explicit_step_interval_overrides_the_time_gate(monkeypatch, tmp_path):
    """Asking for every N steps is asking for every N steps."""
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT_EVERY", "4")
    ck = RenderCheckpoint.from_env(model="m", num_steps=20)
    assert ck is not None and ck.min_interval_s == 0


def test_a_one_step_render_has_nothing_to_resume_into(monkeypatch, tmp_path):
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    assert RenderCheckpoint.from_env(model="m", num_steps=1) is None


def test_explicit_zero_is_disabled(monkeypatch):
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", "0")
    assert RenderCheckpoint.from_env(model="m", num_steps=8) is None


def test_interval_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT_EVERY", "4")
    ck = RenderCheckpoint.from_env(model="m", num_steps=10)
    assert ck is not None
    assert [i for i in range(10) if ck.should_save(i, 10)] == [3, 7, 9], (
        "saves on the interval, and always on the final step"
    )


# --- round trip -------------------------------------------------------------

def test_latent_and_scheduler_state_round_trip(checkpoint):
    latent = np.random.RandomState(0).randn(1, 4, 8, 8).astype(np.float32)
    state = {
        "_step_index": 3,
        "lower_order_nums": 2,
        "this_order": 2,
        "model_outputs": [np.ones((2, 2), dtype=np.float32), None],
        "timestep_list": [981, None],
        "last_sample": np.zeros((2, 2), dtype=np.float32),
    }
    checkpoint.save(3, latent, state)

    resumed = checkpoint.load()
    assert resumed is not None
    step, restored_latent, restored_state = resumed
    assert step == 4, "load returns the NEXT step to run"
    np.testing.assert_array_equal(restored_latent, latent)
    assert restored_state["_step_index"] == 3
    assert restored_state["lower_order_nums"] == 2
    assert restored_state["this_order"] == 2
    assert restored_state["timestep_list"] == [981, None]
    np.testing.assert_array_equal(restored_state["model_outputs"][0], np.ones((2, 2)))
    assert restored_state["model_outputs"][1] is None
    np.testing.assert_array_equal(restored_state["last_sample"], np.zeros((2, 2)))


def test_no_checkpoint_means_start_fresh(checkpoint):
    assert checkpoint.load() is None


# --- the failure modes it exists for ---------------------------------------

def test_a_different_run_is_refused(tmp_path, monkeypatch):
    """Resuming into a different render would produce an output matching
    neither run, and nothing would report it."""
    monkeypatch.setenv("NBX_RENDER_CHECKPOINT", str(tmp_path))
    first = RenderCheckpoint.from_env(model="m", num_steps=8, extra={"seed": 1})
    assert first is not None
    first.save(2, np.zeros((2, 2), dtype=np.float32), {"_step_index": 2})

    other = RenderCheckpoint.from_env(model="m", num_steps=8, extra={"seed": 2})
    assert other is not None
    assert other.path != first.path, "a different run gets a different file"

    # and even pointed at the same file, the fingerprint refuses it
    other.path = first.path
    assert other.load() is None


def test_a_truncated_checkpoint_is_ignored_not_fatal(checkpoint):
    """Exactly what a power cut produces. It must cost the render its steps,
    not its process."""
    checkpoint.save(2, np.zeros((4, 4), dtype=np.float32), {"_step_index": 2})
    raw = checkpoint.path.read_bytes()
    checkpoint.path.write_bytes(raw[: len(raw) // 2])
    assert checkpoint.load() is None


def test_write_is_atomic_so_a_cut_cannot_destroy_a_good_checkpoint(checkpoint):
    """The temp file must not be left behind, and the live file must always be
    a complete checkpoint."""
    checkpoint.save(1, np.ones((4, 4), dtype=np.float32), {"_step_index": 1})
    checkpoint.save(2, np.full((4, 4), 2.0, dtype=np.float32), {"_step_index": 2})
    assert not checkpoint.path.with_suffix(".npz.tmp").exists()
    resumed = checkpoint.load()
    assert resumed is not None and resumed[0] == 3


def test_unserialisable_state_is_rejected_loudly(checkpoint):
    """A scheduler handing back something that cannot travel must fail here,
    where it is a checkpoint bug, rather than silently at resume."""
    from neurobrix.core.runtime.render_checkpoint import _flatten_state

    with pytest.raises(TypeError, match="cannot travel in a checkpoint"):
        _flatten_state({"weird": object()})


def test_clear_removes_the_file(checkpoint):
    checkpoint.save(1, np.zeros((2, 2), dtype=np.float32), {"_step_index": 1})
    assert checkpoint.path.exists()
    checkpoint.clear()
    assert not checkpoint.path.exists()
    checkpoint.clear()   # idempotent


def test_module_is_torch_free():
    """R33: the triton engine needs this same checkpoint."""
    import inspect

    from neurobrix.core.runtime import render_checkpoint

    source = inspect.getsource(render_checkpoint)
    assert "import torch" not in source
