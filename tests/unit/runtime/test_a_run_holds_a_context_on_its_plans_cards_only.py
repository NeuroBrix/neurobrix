"""A run holds a CUDA context on the cards its plan names and on no other (engine rule 1; the owner's queue point 4).

Before (2026-11-10, seen through the entry point on the rack's four V100):
- a plan across cards 2 and 3 (DeepSeek-Coder-V2-Lite on `v100-32g-x2-split20`, pipeline_parallel) left the process on
  ALL FOUR cards: since PyTorch 2.10 the first GPU-to-GPU `tensor.to(card)` builds an event pool that creates events
  — a 306 MiB context — on every visible card (upstream PR #165647). The engine's hand-offs now go through the host
  (`core/runtime/card_handoff.to_card`);
- a plan on card 2 alone under the two-card profile (TinyLlama on `v100-32g-x2-nvlink`) left the process on cards 2
  and 3: Prism read each PROFILE card through the runtime, which creates a context to answer. It now reads the
  driver's tool (`memory_budget.read_device_sharing`);
- with that read gone, the weight loader's pinned upload showed its own: the pinned block and the copy's event were
  made on the CURRENT device — card 0 in a fresh process. They are made on the card written.

What this file does if the code is wrong:
1. `to_card`: a tensor already on the device copied, a value, dtype or shape changed, a strided view mangled — red
   (card-less for the host cases; on one card for the device cases);
2. on two cards: after `to_card` from one to the other the process is on a third card — red (torch's own `.to()`
   there is the injection: it puts the process on every visible card, which the test states as the reason);
3. Prism's read asks the runtime although the driver's tool answered — red (card-less);
4. through the entry point, on the rack's profiles: the process ends on a card its plan does not name — red on main
   for both plans (four cards; two cards for a one-card plan).

Run: python -m pytest tests/unit/runtime/test_a_run_holds_a_context_on_its_plans_cards_only.py   (cards 2,3 visible
for the two-card cases: no CUDA_VISIBLE_DEVICES, the profiles name the rack's cards)
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).with_name("_cards_of_a_run.py")
torch = pytest.importorskip("torch")


def _cards():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def test_a_tensor_already_there_is_returned_as_it_is_and_the_host_cases_are_torchs():
    from neurobrix.core.runtime.card_handoff import to_card
    a = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    assert to_card(a, "cpu") is a
    assert to_card(a, torch.device("cpu")) is a


@pytest.mark.skipif(_cards() < 1, reason="needs a card")
def test_host_to_card_and_back_keep_values_dtype_and_shape():
    from neurobrix.core.runtime.card_handoff import to_card
    a = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)
    d = to_card(a, "cuda:0")
    assert d.device.type == "cuda" and d.dtype == a.dtype and d.shape == a.shape and to_card(d, "cuda:0") is d
    assert torch.equal(to_card(d, "cpu"), a)


_TWO = '''
import os, subprocess, sys, torch
sys.path[:0] = [{src!r}, {lib!r}]
from neurobrix.core.runtime.card_handoff import to_card
def cards():
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.splitlines()
    return len([r for r in rows if r.strip().endswith(", " + str(os.getpid()))])
a = torch.arange(4096, dtype=torch.float16, device="cuda:0").reshape(64, 64)
view = a.t()[::2, 1:]                                   # a strided view: not contiguous
b = to_card(view, "cuda:1")
assert b.device == torch.device("cuda:1") and b.dtype == view.dtype and b.shape == view.shape
assert torch.equal(b.cpu(), view.cpu())
print("OURS", cards())
c = a.to("cuda:1")                                      # the framework's own device-to-device copy
print("TORCH", cards(), torch.cuda.device_count())
'''


@pytest.mark.skipif(_cards() < 3, reason="needs two cards to cross and a third to stay off")
def test_crossing_two_cards_leaves_the_process_on_those_two():
    """A fresh process sees the rack's last two cards first (so a context on any other card is a third one)."""
    n = _cards()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in [n - 2, n - 1] + list(range(n - 2))),
               PYTHONNOUSERSITE="1")
    out = subprocess.run([sys.executable, "-c", _TWO.format(src=str(REPO / "src"), lib=str(REPO / "nbxtensor" / "src"))],
                         capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-800:]
    said = dict(line.split()[:2] for line in out.stdout.splitlines() if line.startswith(("OURS", "TORCH")))
    assert said["OURS"] == "2", out.stdout
    # the reason the module exists, stated by the framework itself; the day it prints 2 the module can go
    assert int(said["TORCH"]) in (2, n), out.stdout


def test_prism_reads_a_card_from_the_drivers_tool_without_the_runtime(monkeypatch):
    from neurobrix.core.prism import autodetect, memory_budget
    from nbxtensor.memory.allocator import DeviceAllocator

    def no_runtime(*a, **k):
        raise AssertionError("the runtime was asked about a card the driver's tool had answered for: a context is made")
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(no_runtime))
    monkeypatch.setattr(DeviceAllocator, "device_total_bytes", staticmethod(no_runtime))
    monkeypatch.setattr(autodetect, "device_sharing", lambda index: {
        "display_active": False, "held_by_others_mb": 100.0, "own_context_mb": 0.0, "total_mb": 32768.0,
        "free_mb": 32665.0, "source": "nvidia-smi"})
    r = memory_budget.read_device_sharing(3)
    assert (r.capacity_mb, r.free_mb, r.held_by_others_mb, r.own_context_mb, r.measured) == (32768.0, 32665.0, 100.0, 0.0, True)


def test_prism_asks_the_runtime_only_when_no_tool_answers(monkeypatch):
    from neurobrix.core.prism import autodetect, memory_budget
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(autodetect, "device_sharing", lambda index: None)
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda index: 1000.0))
    monkeypatch.setattr(DeviceAllocator, "device_total_bytes", staticmethod(lambda index: 2000 << 20))
    r = memory_budget.read_device_sharing(0)
    assert (r.capacity_mb, r.free_mb, r.measured) == (2000.0, 1000.0, False)


def _bus_ids():
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=30).stdout.splitlines()
    return {int(r.split(",")[0]): r.split(",")[1].strip() for r in rows if r.strip()}


def _profile_cards(name):
    import yaml
    path = REPO / "src" / "neurobrix" / "config" / "hardware" / f"{name}.yml"
    return [int(d["index"]) for d in yaml.safe_load(path.read_text())["devices"]] if path.exists() else None


def _run_and_list(model, profile):
    out = subprocess.run([sys.executable, str(HELPER), "run", "--model", model, "--hardware", profile, "--prompt",
                          "Say hello.", "--max-tokens", "8", "--seed", "7", "--compiled"], capture_output=True, text=True,
                         timeout=1500, cwd=str(REPO),
                         env=dict({k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
                                  PYTHONNOUSERSITE="1", PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}"))
    line = [l for l in out.stdout.splitlines() if l.startswith("CARDS")]
    assert out.returncode == 0 and line, (out.stdout[-600:], out.stderr[-600:])
    return sorted(line[-1].split()[1:])


def _cached(name):
    return (Path.home() / ".neurobrix" / ("ca" + "che") / name / "manifest.json").exists()


@pytest.mark.skipif(_cards() < 4 or _profile_cards("v100-32g-x2-split20") is None
                    or not _cached("DeepSeek-Coder-V2-Lite-Instruct"), reason="the rack's two-card split profile and model")
def test_a_plan_across_two_cards_ends_on_those_two_cards():
    ids = _bus_ids()
    want = sorted(ids[i] for i in _profile_cards("v100-32g-x2-split20"))
    assert _run_and_list("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "v100-32g-x2-split20") == want


@pytest.mark.skipif(_cards() < 4 or _profile_cards("v100-32g-x2-nvlink") is None
                    or not _cached("TinyLlama-1.1B-Chat-v1.0"), reason="the rack's two-card profile and model")
def test_a_one_card_plan_on_a_two_card_profile_ends_on_its_one_card():
    ids = _bus_ids()
    first = ids[_profile_cards("v100-32g-x2-nvlink")[0]]
    assert _run_and_list("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "v100-32g-x2-nvlink") == [first]
