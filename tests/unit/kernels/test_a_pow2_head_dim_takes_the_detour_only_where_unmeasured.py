"""The flash route's zero-pad detour at a power-of-two head dim >= 128 applies on an arch that has not
measured the kernel correct there, and not on one whose profile says it did (Volta on Triton 3.8,
2026-10-03: the 33-case correctness oracle passes with the detour off; D 128 runs 65 ms instead of
2 196). One function decides for the wrapper and the derived census alike.

What would this file do if the code were wrong? The flag ignored -> D 128 still padded on Volta,
the first test RED; the flag read as true when absent -> the second RED; Volta's profile without
the flag -> the third RED.
"""
from pathlib import Path

import yaml

from neurobrix.kernels import launch_keys as LK
from neurobrix.kernels.ops import _configs as K

REPO = Path(__file__).resolve().parents[3]


def test_a_measured_arch_runs_the_dim_as_it_is(monkeypatch):
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {"flash": {"pow2_head_dim_correct": True}})
    assert [LK.flash_headdim_detour(d) for d in (64, 96, 128, 256)] == [64, 96, 128, 256]


def test_an_unmeasured_arch_keeps_the_detour(monkeypatch):
    for prof in ({}, {"flash": {}}, {"flash": {"pow2_head_dim_correct": "yes"}}):
        monkeypatch.setattr(K, "active_vendor_profile", lambda prof=prof: prof)
        assert [LK.flash_headdim_detour(d) for d in (96, 128, 256)] == [96, 129, 257]
    assert LK.flash_headdim_detour(128, enabled=False) == 128


def test_volta_states_its_measurement_and_every_other_arch_declares_it_unmeasured():
    """Every arch profile DECLARES the flag (test_smem_budget's door: a key a sibling declares is declared
    or its omission reasoned) — true only where the correctness oracle ran with the detour off (Volta), false
    everywhere it was not measured. Until 2026-10-04 this cell asserted the flag ABSENT on the other NVIDIA
    archs, against that door: the two tests could not both pass, and the kernels suite had not been run on
    a card."""
    vendors = REPO / "src" / "neurobrix" / "config" / "vendors"
    declared = {f"{p.parent.name}/{p.stem}": (yaml.safe_load(p.read_text()).get("flash") or {}).get("pow2_head_dim_correct")
                for p in sorted(vendors.glob("*/*.yml"))}
    assert declared["nvidia/volta"] is True
    others = {k: v for k, v in declared.items() if k != "nvidia/volta"}
    assert others and all(v is False for v in others.values()), {k: v for k, v in others.items() if v is not False}
