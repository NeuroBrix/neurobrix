"""The frozen-plan replay is the Triton engine's own decision, not an environment switch (the supervisor's order
2026-10-28 08:20, S3 1b): on by default with no variable set, every sequence it cannot vouch for refused BY NAME once,
and a recorded plan's slab released when an allocation on its card needs the room. Before: `NBX_TRITON_REPLAY=1`,
`NBX_REPLAY_KV_DECODE=1`, `NBX_REPLAY_EXPERT_TAIL=1` had to be set, an ineligible sequence fell back silently, and a slab
stayed held beside every other component. No card."""
import os
import subprocess
import sys
from types import SimpleNamespace


def test_the_replay_is_on_with_no_variable_set():
    env = {k: v for k, v in os.environ.items() if not k.startswith("NBX_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "import neurobrix.triton.replay as r; print(r.ENABLED)"],
                         env=env, capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1:] == ["True"], (out.stdout[-500:], out.stderr[-500:])


def test_an_ineligible_sequence_is_refused_by_name_once(capsys):
    from neurobrix.triton import replay
    seq = SimpleNamespace(dag={"component_name": "decoder"}, _is_multi_device=False, _replay_ineligible=False,
                          _op_uid_interceptors={}, _op_interceptors={},
                          _ops=[SimpleNamespace(op_type="aten.randn"), SimpleNamespace(op_type="aten.mm")])
    assert replay.signature(seq) is None
    assert replay.signature(seq) is None
    said = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[Replay] decoder: not replayed")]
    assert len(said) == 1 and "random" in said[0], said


def test_a_per_op_callback_is_refused_by_name(capsys):
    from neurobrix.triton import replay
    seq = SimpleNamespace(dag={"component_name": "model"})
    assert replay.maybe_run(seq, False, lambda *a: None) is False
    assert "[Replay] model: not replayed — a per-op callback" in capsys.readouterr().out


def test_an_idle_slab_is_released_when_its_card_needs_the_room(monkeypatch, capsys):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    from neurobrix.triton import replay
    assert replay.release_idle_slabs in DeviceAllocator._idle_reclaimers

    freed = []

    class Slab:
        def __init__(self, dev, size):
            self.dev_idx, self.size, self.allocs = dev, size, {}

        def retire(self):
            freed.append((self.dev_idx, self.size))

    plan = replay.FrozenPlan.__new__(replay.FrozenPlan)
    plan.slab, plan.other_slabs, plan.graph = Slab(1, 64 << 20), {}, None
    other = replay.FrozenPlan.__new__(replay.FrozenPlan)
    other.slab, other.other_slabs, other.graph = Slab(0, 32 << 20), {}, None

    class Seq:
        pass
    s = Seq()
    s.dag = {"component_name": "transformer"}
    s._replay_plans = {("a",): plan, ("b",): other}
    replay._PLAN_SEQS.add(s)
    try:
        assert replay.release_idle_slabs(1) == 64 << 20
    finally:
        replay._PLAN_SEQS.discard(s)
    assert freed == [(1, 64 << 20)] and list(s._replay_plans) == [("b",)], (freed, s._replay_plans)
    assert "[Replay] transformer: a recorded plan's slab released on device 1" in capsys.readouterr().out
