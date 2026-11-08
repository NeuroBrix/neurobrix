"""Every Prism test plans on the container and the facts it pins, never on what this machine's runs recorded: the
engine's calibration store (`core/dtype/calibration.STORE_ROOT`, which `core/runtime/runtime_growth` reads too) is
read as empty. A test that wants a record sets its own store (`monkeypatch.setattr(calibration, "STORE_ROOT", ...)`);
a test about WHERE the engine home's directories are (not what they hold) is marked `engine_home_layout`."""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import pin_no_calibration


def pytest_configure(config):
    config.addinivalue_line("markers", "engine_home_layout: the test reads where the engine home's directories are, "
                                       "never a record in them; the calibration store is not pinned")


@pytest.fixture(autouse=True)
def _no_machine_calibration_store(request, monkeypatch):
    if request.node.get_closest_marker("engine_home_layout") is None:
        pin_no_calibration(monkeypatch)
