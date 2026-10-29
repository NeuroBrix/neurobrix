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



def test_with_no_observer_installed_nothing_runs_dry():
    """The launch observer is the library's own optional hook (`ops.launch_observer`), no longer a host service: with
    none installed every launch runs, and a dry run asked of no observer is refused by name."""
    import pytest as _pytest
    from nbxtensor.ops import launch_observer as LO
    saved = LO._OBSERVER
    try:
        LO.set_launch_observer(None)
        assert LO.launch_observer().active() is False
        assert LO.launch_observer().record(None, (1,)) is None      # a live launch offers its key; none observes it
        with _pytest.raises(RuntimeError, match="no launch observer installed"):
            LO.launch_observer().shadow_run()
        with _pytest.raises(TypeError, match="no callable record, shadow_run"):
            LO.set_launch_observer(type("Half", (), {"active": staticmethod(lambda: True)}))
        assert "census" not in host.SERVICES
    finally:
        LO._OBSERVER = saved


def test_a_needed_service_nobody_provided_is_refused_by_name(no_application):
    with pytest.raises(RuntimeError, match="vendor_config"):
        host.need("vendor_config")
