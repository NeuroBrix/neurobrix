"""nbxtensor.host: what an application provides, and what the library owns or runs without (API.md section 11)."""
import pytest

from nbxtensor import host
from nbxtensor.backend.device_names import arch_string


@pytest.fixture
def no_application(monkeypatch):
    monkeypatch.setattr(host, "_PROVIDED", {})


def test_the_arch_string_is_the_librarys_own():
    assert arch_string("Apple M4 Pro") == "apple-m4-pro"
    assert arch_string("  Tesla V100-SXM2-16GB ") == "tesla-v100-sxm2-16gb"
    assert "device_arch_string" not in host.SERVICES


def test_optional_hooks_run_without_an_application(no_application):
    assert host.need("installation_refusals")("/tmp/anything") == []
    assert host.need("census")().active() is False


def test_a_needed_service_nobody_provided_is_refused_by_name(no_application):
    with pytest.raises(RuntimeError, match="vendor_config"):
        host.need("vendor_config")
