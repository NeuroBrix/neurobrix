"""A flash launch whose query rows are fewer than its keys (a continuing turn's prefill) and whose own key the directory
does not hold is served the certified SQUARE key of the same key length, never a run-time sweep (task 203, decided
2026-11-15 11:45). On apple_m4_pro a served five-turn chat swept (20,128) (23,192) (18,256) (20,320) at run time, and
two fresh installs of one release gave different bytes on turns 3-5.

What this test does if the code is wrong:
1. the default of (20, 128, 64, ...) is not (128, 128, 64, ...), or a q >= Tk key or another kernel gets one (red);
2. a launch whose directory holds only the square key is not served it, or a certified q < Tk entry of its own loses
   to the default (red: the 300 shipped q < Tk entries would change bytes);
3. a replay cache's earlier sweep of the key is kept on a warm machine instead of the square key's config (red);
4. a seeded key the directory cannot read (a string that is no dtype: `UnreadableKey`) stops the seed, or loses its
   local setting (red); any OTHER error of the seeded lookup is swallowed and the key silently keeps its sweep (red:
   seen red 2026-11-15, the `except Exception: pass` this replaced returned 0 and said nothing).
No card: the directory and the tuner are stand-ins."""
import pytest

from nbxtensor.ops import autotune_certified as C
from nbxtensor.ops import launch_keys as L

T = ("fp32",) * 7
SQUARE, TURN, OWN = (128, 128, 64) + T, (20, 128, 64) + T, (32, 160, 64) + T


class _Tuner:
    arg_names, keys = [], ["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "headdim"]

    def __init__(self):
        self.cache = {}


def _entry(block):
    return {"config": {"kwargs": {"BLOCK_M": block, "BLOCK_N": block}, "num_warps": 4, "num_stages": 1}}


@pytest.fixture
def directory(monkeypatch):
    held = {SQUARE: _entry(32), OWN: _entry(16)}
    monkeypatch.setattr(C, "lookup", lambda qual, tuner, key, **kw: held.get(tuple(key)))
    monkeypatch.setattr(C, "enabled", lambda: True)
    monkeypatch.setattr(C, "executing_memory_gb", lambda *a: 18)
    return held


def test_the_default_of_a_turn_prefill_key_is_its_square_key():
    assert L.square_default_key(L.FLASH, TURN) == SQUARE
    assert L.square_default_key(L.UNIT_FLASH, (20, 128, 64, False)) == (128, 128, 64, False)
    assert L.square_default_key(L.FLASH, SQUARE) is None                       # already square
    assert L.square_default_key(L.FLASH, (288, 128, 64) + T) is None           # q > Tk: no default
    assert L.square_default_key("nbxtensor.ops.x.matmul_kernel", TURN) is None


def test_a_launch_missing_its_key_is_served_the_square_config(directory):
    t = _Tuner()
    assert any(C.apply(L.FLASH, t, TURN, lookup_key=k, memory_gb=18) for k in C.lookup_keys(L.FLASH, t, TURN))
    assert t.cache[TURN].kwargs == {"BLOCK_M": 32, "BLOCK_N": 32}


def test_a_certified_turn_key_of_its_own_wins_over_the_default(directory):
    t = _Tuner()
    assert C.apply(L.FLASH, t, OWN, memory_gb=18)                               # read first, at the launch site
    assert t.cache[OWN].kwargs == {"BLOCK_M": 16, "BLOCK_N": 16}


def test_a_warm_machine_s_earlier_sweep_gives_way_to_the_square_key(directory, monkeypatch):
    import triton
    t = _Tuner()
    t.cache[TURN] = triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2)   # an earlier sweep
    assert C.override_seeded([(L.FLASH, t)]) == 1
    assert t.cache[TURN].kwargs == {"BLOCK_M": 32, "BLOCK_N": 32}


def test_a_seeded_key_the_directory_cannot_read_keeps_its_local_setting(monkeypatch, caplog):
    import triton
    monkeypatch.setattr(C, "enabled", lambda: True)
    monkeypatch.setattr(C, "active_profile", lambda: ("a-vendor", "a-profile"))
    monkeypatch.setattr(C, "executing_memory_gb", lambda *a: 18)
    t, stale = _Tuner(), (20, 128, 64, "not-a-dtype")
    t.cache[stale] = local = triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2)
    with pytest.raises(C.UnreadableKey):
        C.lookup(L.FLASH, t, stale, any_class=True)                             # the named refusal the seed reads past
    with caplog.at_level("WARNING"):
        assert C.override_seeded([(L.FLASH, t)]) == 0
    assert t.cache[stale] is local
    assert "not-a-dtype" in caplog.text and "local setting stays" in caplog.text    # said, never silent


def test_any_other_error_of_the_seeded_lookup_surfaces(directory, monkeypatch):
    import triton

    def broken(qual, tuner, key, kwargs=None):
        raise ValueError("a defect in the lookup keys")
    monkeypatch.setattr(C, "lookup_keys", broken)
    t = _Tuner()
    t.cache[TURN] = triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2)
    with pytest.raises(ValueError, match="a defect in the lookup keys"):
        C.override_seeded([(L.FLASH, t)])
