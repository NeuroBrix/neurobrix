"""A host whose OS probes find no accelerator has none: the detection says so by
name and never imports a compute branch to look again.

2026-10-11 (Studio item 4, the Dell's decision on the residual): every OS
cascade ended in `_detect_torch_fallback`, which imported torch when the OS
probes found nothing. On a CPU-only host that made discovery pay a framework
import (several seconds) to learn that there is no device, and it made the
hardware profile depend on which torch build happened to be installed. R33:
no driver-probe torch import.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_no_device_found_is_no_device.py
"""
import sys
import types

import pytest

from neurobrix.core.prism import autodetect


class _TorchMustNotBeRead(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"device detection read torch.{name}")


_EMPTY_LIST = lambda: []                     # noqa: E731
_EMPTY_PAIR = lambda: ([], None)             # noqa: E731

_OS_PROBES = {
    "Linux": {"_visible_device_filter": lambda: None, "_parse_nvidia_smi": _EMPTY_LIST, "_parse_rocm": _EMPTY_LIST,
              "_parse_xpu": _EMPTY_LIST, "_parse_tenstorrent": _EMPTY_LIST, "_parse_moore_threads": _EMPTY_LIST,
              "_parse_biren": _EMPTY_LIST, "_parse_iluvatar": _EMPTY_LIST, "_parse_cambricon": _EMPTY_LIST,
              "_parse_lspci": _EMPTY_PAIR},
    "Darwin": {"_parse_system_profiler": _EMPTY_LIST},
    "Windows": {"_parse_nvidia_smi": _EMPTY_LIST, "_parse_rocm": _EMPTY_LIST, "_parse_xpu": _EMPTY_LIST,
                "_parse_wmi": _EMPTY_PAIR},
}


@pytest.mark.parametrize("os_type", sorted(_OS_PROBES))
def test_nothing_found_by_the_os_probes_is_no_device(monkeypatch, os_type):
    for name, probe in _OS_PROBES[os_type].items():
        monkeypatch.setattr(autodetect, name, probe)
    monkeypatch.setitem(sys.modules, "torch", _TorchMustNotBeRead("torch"))
    assert autodetect._detect_gpus(os_type) == ([], "none")
