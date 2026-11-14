"""The library's suite on a host without a device: its card tests skip by name (`_no_device.py`, the one mechanism the
engine's suite registers too)."""
from _no_device import (pytest_configure, pytest_make_collect_report, pytest_runtest_call,  # noqa: F401
                        pytest_runtest_setup)
