"""A test that picks a device index from NVML and allocates on it with CUDA is
reading one namespace and writing another.

`CUDA_VISIBLE_DEVICES` is a CUDA-runtime mask: it renumbers ordinals for
everything that goes through libcuda. `nvidia-smi` answers from NVML, which is
outside that mask and always reports the whole board. On this rack the 32G
cards are physical 2 and 3, so `test_prefill_determinism.py::_gpu_classes`
returned big=2 under `CUDA_VISIBLE_DEVICES=0`, declined to skip, and asked for
`cuda:2` where only ordinal 0 exists. The suite went red at
`DeviceAllocator.set_device(2)` (2026-09-17, card 0, candidate stack) — a
failure about the pin, not about the prefill route the file tests.

The failure is the dangerous kind: not an error but a WRONG ANSWER, because 2
is a valid integer in both namespaces and simply names different cards.
"""

import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator

from .test_prefill_determinism import _gpu_classes

MIB = 1024 * 1024


def test_both_classes_are_found_when_both_are_visible(monkeypatch):
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: [(0, 16384 * MIB), (1, 32768 * MIB)]))
    assert _gpu_classes() == (0, 1)


def test_a_pinned_16g_card_reports_no_big_card_so_the_32g_cell_skips(monkeypatch):
    # The exact shape of `CUDA_VISIBLE_DEVICES=0` on this rack. Before the fix
    # this returned (0, 2) — an ordinal from a board the process cannot touch.
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: [(0, 16384 * MIB)]))
    small, big = _gpu_classes()
    assert small == 0
    assert big is None, "a masked-away 32G card must not be offered to an allocation"


def test_a_pinned_32g_card_reports_no_small_card(monkeypatch):
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: [(0, 32768 * MIB)]))
    small, big = _gpu_classes()
    assert big == 0
    assert small is None


def test_the_ordinal_returned_is_the_one_the_runtime_gave_not_a_position(monkeypatch):
    # If the helper ever enumerates positionally it reads right here and wrong
    # on `CUDA_VISIBLE_DEVICES=3,1`, where the runtime's ordinals are 0 and 1
    # but the classes are not in board order.
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: [(0, 32768 * MIB), (1, 16384 * MIB)]))
    assert _gpu_classes() == (1, 0)


def test_no_runtime_at_all_is_not_a_crash(monkeypatch):
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: []))
    assert _gpu_classes() == (None, None)

    def boom():
        raise RuntimeError("no GPU runtime")
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory", staticmethod(boom))
    assert _gpu_classes() == (None, None)


def test_the_helper_does_not_shell_out_to_nvidia_smi(monkeypatch):
    # The injection that turns the old implementation red. `nvidia-smi` is the
    # wrong authority for a question whose answer will be used as a CUDA ordinal.
    import subprocess

    def refuse(*a, **kw):
        raise AssertionError("_gpu_classes must not ask NVML for a CUDA ordinal")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(DeviceAllocator, "visible_device_memory",
                        staticmethod(lambda: [(0, 16384 * MIB)]))
    assert _gpu_classes() == (0, None)


@pytest.mark.skipif(DeviceAllocator.device_count() == 0, reason="no GPU on this host")
def test_on_a_real_card_the_ordinals_are_usable_for_an_allocation():
    """The method's activation proof: every ordinal it returns must actually
    take an allocation. A list that merely looks plausible is what the old
    helper produced."""
    from neurobrix.kernels.nbx_tensor import NBXTensor

    devs = DeviceAllocator.visible_device_memory()
    assert devs, "a host with a device must report at least one"
    assert len(devs) == DeviceAllocator.device_count()
    for idx, total in devs:
        assert total > 0
        t = NBXTensor.empty((8,), dtype="float16", device=f"cuda:{idx}")
        assert t.data_ptr() != 0
