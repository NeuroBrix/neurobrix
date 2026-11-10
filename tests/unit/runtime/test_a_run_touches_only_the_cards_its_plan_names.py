"""A run synchronizes, and reads the runtime's growth on, the devices its plan and profile name — never "the current
device" (card 0 in a fresh process) nor every visible card: each of those created a context on a card the plan did not
name (the rack, 2026-11-10: 306 MB on card 0 for a compiled run on card 2; on all four cards under the Triton engine).

What these tests would do if the code were wrong: a device the plan does not name is synchronized, or the growth
baseline reads every visible card — RED."""
from types import SimpleNamespace as NS

from neurobrix.core import device_utils


def _plan():
    return NS(components={"model": NS(devices=["zero3:cuda:2"], expert_cards={"cuda:3": {}, "cuda:2": {}}),
                          "lm_head": NS(devices=["cuda:2"], expert_cards={}),
                          "tokenizer": NS(devices=["cpu"], expert_cards={})})


def test_the_plans_devices_are_its_components_and_its_expert_cards():
    assert device_utils.plan_devices(_plan()) == ["cuda:2", "cuda:3"]


def test_sync_plan_synchronizes_those_devices_and_no_other(monkeypatch):
    seen = []
    monkeypatch.setattr(device_utils, "device_sync", lambda d=None: seen.append(d))
    device_utils.sync_plan(_plan())
    assert seen == ["cuda:2", "cuda:3"]


def test_the_growth_baseline_reads_the_profiles_cards_only(monkeypatch):
    from neurobrix.cli.commands import run
    from neurobrix.core.runtime import runtime_growth
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "device_count", staticmethod(lambda: 4))
    asked = []
    monkeypatch.setattr(runtime_growth, "untracked_mb", lambda devs: asked.append(list(devs)) or {d: 0.0 for d in devs})
    profile = NS(devices=[NS(index=2, has_unified_memory=False)])
    _, devs, _ = run.growth_baseline("triton", profile)
    assert devs == [2] and asked == [[2]]
