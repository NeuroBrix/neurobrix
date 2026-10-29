"""What the library answers by itself, with no application around it (API.md section 11): measured facts are read by
the library, settings have a default and a setter taking plain data, and none of them is a host service any more."""
import pytest

from nbxtensor import host


def test_the_host_reading_is_the_librarys_own():
    from nbxtensor.backend.host_memory import memory_state
    state = memory_state()
    assert state.source and (state.available_mb is None or state.available_mb >= 0)


def test_the_tiling_policy_is_the_librarys_file():
    from nbxtensor.ops.tiling_policy import tiling_table
    policy = tiling_table()
    assert "sdpa" in policy and isinstance(policy, dict)
    policy["sdpa"] = None                                       # a copy: the library's table is untouched
    assert tiling_table()["sdpa"] is not None


def test_a_caller_sets_its_own_tiling_and_restores_the_librarys():
    from nbxtensor.ops.tiling_policy import set_tiling, tiling_table
    from nbxtensor.ops.tiling_sizes import _value
    own = tiling_table()
    try:
        set_tiling({"sdpa": {"x": 1}})
        assert tiling_table() == {"sdpa": {"x": 1}}
        with pytest.raises(KeyError, match="states no conv2d"):  # the rules read the table in force
            _value("conv2d", "band_bytes")
        with pytest.raises(TypeError):
            set_tiling([("sdpa", 1)])
    finally:
        set_tiling(None)
    assert tiling_table() == own


def test_the_certified_store_defaults_to_the_users_cache(monkeypatch, tmp_path):
    from nbxtensor.ops import autotune_certified as C
    monkeypatch.delenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", raising=False)
    monkeypatch.setattr(C, "_STORE", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert C.directory() == tmp_path / "nbxtensor" / "certified"
    C.set_store(tmp_path / "committed")
    assert C.directory() == tmp_path / "committed"
    C.set_store(None)
    assert C.directory() == tmp_path / "nbxtensor" / "certified"


@pytest.mark.parametrize("gone", ["memory_state", "host_shares_memory_with_device", "tiling_policy", "certified_dir"])
def test_a_fact_or_setting_the_library_holds_is_no_service(gone):
    with pytest.raises(KeyError, match="not a service"):
        host.need(gone)
