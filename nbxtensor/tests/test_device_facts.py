"""What the library answers by itself when no application provides it (API.md section 11)."""
import pytest

from nbxtensor import host


@pytest.fixture
def no_application(monkeypatch):
    monkeypatch.setattr(host, "_PROVIDED", {})


def test_the_host_reading_is_the_librarys_own(no_application):
    state = host.need("memory_state")()
    assert state.source and (state.available_mb is None or state.available_mb >= 0)


def test_the_tiling_policy_is_the_librarys_file(no_application):
    policy = host.need("tiling_policy")()
    assert "sdpa" in policy and isinstance(policy, dict)
    policy["sdpa"] = None                                       # a copy: the library's table is untouched
    assert host.need("tiling_policy")()["sdpa"] is not None


def test_the_certified_store_defaults_to_the_users_cache(no_application, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert host.need("certified_dir")() == tmp_path / "nbxtensor" / "certified"


def test_an_application_still_overrides(monkeypatch):
    monkeypatch.setattr(host, "_PROVIDED", {})
    host.provide("tiling_policy", lambda: {"sdpa": {"x": 1}})
    assert host.need("tiling_policy")() == {"sdpa": {"x": 1}}
