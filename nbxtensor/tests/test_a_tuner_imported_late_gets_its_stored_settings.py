"""A tuner whose kernel module is imported after the process's first autotuned launch gets the machine's stored
settings too (`autotune_cache.seed_late`, called from the launch door `_configs._seed`), once.

Before (2026-11-12, the real-use campaign's D15): the artifact was seeded once, at the first autotuned launch, into
the kernel modules already imported. A serving daemon imports its paged decode attention and its MoE decode kernels
at its first request: ten kernels never received their 357 stored entries, and every new daemon measured them again
("measuring kernel configurations once" on every daemon, the artifact growing by nothing).

What this test does if the code is wrong: the late tuner gets nothing — red; it is seeded twice, or a non-member of
its declared space is seeded — red; the launch door stops offering a late tuner its settings — red. Card-less."""
from types import SimpleNamespace

from nbxtensor.ops import autotune_cache as A


def _tuner():
    cfg = SimpleNamespace(kwargs={"BLOCK": 64}, num_warps=4, num_stages=2)
    return SimpleNamespace(configs=[cfg], cache={})


STORED = {
    "nbxtensor.ops.late.kernel::(128, 'fp16')": {"kwargs": {"BLOCK": 64}, "num_warps": 4, "num_stages": 2},
    "nbxtensor.ops.late.kernel::(256, 'fp16')": {"kwargs": {"BLOCK": 32}, "num_warps": 4, "num_stages": 2},  # not a member
    "nbxtensor.ops.other.kernel::(128, 'fp16')": {"kwargs": {"BLOCK": 64}, "num_warps": 4, "num_stages": 2},
}


def test_a_late_tuner_is_seeded_once_with_its_members_only(monkeypatch):
    t = _tuner()
    monkeypatch.setattr(A, "_autotuners", lambda: iter([("nbxtensor.ops.late.kernel", t)]))
    monkeypatch.setattr(A, "_stored", lambda: STORED)
    monkeypatch.setattr(A, "_config_from_dict", lambda d: ("cfg", d["kwargs"]["BLOCK"]))
    monkeypatch.setattr(A, "_SEEDED_TUNERS", set())
    assert A.seed_late(t) == 1
    assert t.cache == {(128, "fp16"): ("cfg", 64)}
    assert A.seed_late(t) == 0                       # once


def test_the_launch_door_offers_every_tuner_its_settings(monkeypatch):
    from nbxtensor.ops import _configs as C
    from nbxtensor.ops import autotune_certified as Cert
    offered = []
    monkeypatch.setattr(C, "_SEEDED", [True])        # the process-wide seed already ran
    monkeypatch.setattr(A, "seed_late", lambda at: offered.append(at) or 2)
    monkeypatch.setattr(A, "qual_of", lambda at: "nbxtensor.ops.late.kernel")
    overridden = []
    monkeypatch.setattr(Cert, "override_seeded", lambda tuners=None: overridden.append(tuners) or 0)
    monkeypatch.setattr(Cert, "note_local", lambda n: None)
    t = _tuner()
    C._seed(t)
    assert offered == [t] and overridden == [[("nbxtensor.ops.late.kernel", t)]]
