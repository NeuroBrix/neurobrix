"""A key the rule gives to the matrix unit is certified on the unit's route, and its entry names the unit BY THE RULE.

Library 863bf47 (`autotune_certified.ruled_unit_tile`): which unit runs a launch is never a timing; the certificate
only chooses WHICH of the profile's declared unit tiles. Until this test the certifier kept the unit tile only when
it timed faster than the best FMA configuration (`if ums < ms`), so a ruled key whose unit was slower got an entry
naming the FMA kernel, which the runtime then refused by name. Card-less: the tiles are every device file's own
declared `matrix_unit.mm` rows (native and split), the sweep's results and timings are given, and the entry the
certifier keeps is read back through the library's own `ruled_unit_tile`.
"""
from pathlib import Path

import pytest
import yaml

import nbxtensor
from nbxtensor.ops import autotune_cache as ATC
from nbxtensor.ops import autotune_certified as C

from neurobrix.kernels import autotune_certify as AC


def _declared_unit_tile_sets():
    """Every declared set of GEMM unit tiles of every device file the library ships (none named here)."""
    sets = []
    for path in sorted((Path(nbxtensor.__file__).parent / "backend" / "devices").glob("*/*.yml")):
        mu = (yaml.safe_load(path.read_text()) or {}).get("matrix_unit") or {}
        for rows in (mu.get("mm"), (mu.get("fp32_split") or {}).get("mm")):
            if rows:
                sets.append(pytest.param([{f: int(r[f]) for f in C._UNIT_TILE_FIELDS} for r in rows],
                                         id=f"{path.parent.name}-{path.stem}-{len(sets)}"))
    return sets


TILE_SETS = _declared_unit_tile_sets()


def _served_tile(monkeypatch, entry, rows):
    """The tile the library's runtime rule serves for a launch whose certificate is `entry`, and what it refused."""
    tuner = object()
    C.reset_refused_unit_entries()
    C._UNIT_CHOICES.clear()
    monkeypatch.setitem(C._UNIT_QUALS, id(tuner), "nbxtensor.ops.matmul.matmul_kernel")
    monkeypatch.setattr(ATC, "key_of", lambda *a, **k: ("a", "ruled", "key"))
    monkeypatch.setattr(C, "executing_memory_gb", lambda *a, **k: None)
    monkeypatch.setattr(C, "lookup", lambda *a, **k: entry)
    try:
        return C.ruled_unit_tile(tuner, (), {}, dict(rows[-1]), rows), C.refused_unit_entries()
    finally:
        C._UNIT_CHOICES.clear()


def _kept(runs, units, fma_ms):
    state = {}
    AC.keep_ruled_unit(state, runs, units, fma_ms, "a ruled key")
    return state


@pytest.mark.skipif(not TILE_SETS, reason="no device file declares GEMM unit tiles")
@pytest.mark.parametrize("rows", TILE_SETS)
def test_a_unit_slower_than_the_fma_kernel_is_still_the_one_the_entry_names(monkeypatch, rows):
    runs = [(t, None) for t in rows]
    # every unit tile proven, and every one SLOWER than the FMA configuration's best: the rule still gives it the launch
    units = [(t, 1e-6, 10.0 + i) for i, t in enumerate(rows)]
    state = _kept(runs, units, fma_ms=1.0)
    assert state["unit"]["tile"] == rows[0], "the fastest of the declared tiles, never the FMA kernel"
    assert state["unit_measured"]["chosen"] == "unit"
    entry = {"config": {"kwargs": {}}, "unit": state["unit"]}
    tile, refused = _served_tile(monkeypatch, entry, rows)
    assert tile == rows[0] and not refused, "the runtime serves the certified tile and refuses nothing"


@pytest.mark.skipif(not TILE_SETS, reason="no device file declares GEMM unit tiles")
@pytest.mark.parametrize("rows", TILE_SETS)
def test_the_timing_chooses_only_among_the_declared_tiles(monkeypatch, rows):
    runs = [(t, None) for t in rows]
    units = [(t, 1e-6, 5.0 - i) for i, t in enumerate(rows)]       # the last declared tile is the fastest
    state = _kept(runs, units, fma_ms=0.1)
    assert state["unit"]["tile"] == rows[-1]
    tile, refused = _served_tile(monkeypatch, {"config": {"kwargs": {}}, "unit": state["unit"]}, rows)
    assert tile == rows[-1] and not refused


@pytest.mark.skipif(not TILE_SETS, reason="no device file declares GEMM unit tiles")
@pytest.mark.parametrize("rows", TILE_SETS)
def test_a_ruled_key_none_of_whose_tiles_proves_is_refused_not_filed_as_fma(rows):
    with pytest.raises(RuntimeError, match="rule gives this launch to the matrix unit"):
        _kept([(t, None) for t in rows], [], fma_ms=1.0)


def test_a_launch_the_rule_leaves_to_the_fma_kernel_names_no_unit():
    assert _kept([], [], fma_ms=1.0) == {}


def _unit_profiles():
    out = []
    for path in sorted((Path(nbxtensor.__file__).parent / "backend" / "devices").glob("*/*.yml")):
        prof = yaml.safe_load(path.read_text()) or {}
        if ((prof.get("matrix_unit") or {}).get("mm")) and ((prof.get("autotune") or {}).get("buckets") or {}).get("M"):
            out.append(pytest.param(prof, id=f"{path.parent.name}-{path.stem}"))
    return out


@pytest.mark.parametrize("prof", _unit_profiles())
def test_a_ruled_key_is_drawn_at_its_bucket_top_on_the_unit_route(monkeypatch, prof):
    """(b): the key is drawn at the top of its bucket under the BOUND ladder, where the rule gives it to the unit, and
    the launch there has unit runs (the route `keep_ruled_unit` reads)."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops import _configs as K
    from nbxtensor.ops import autotune_bucket as B
    from nbxtensor.ops import autotune_harness as H
    from nbxtensor.ops import launch_keys as LK
    monkeypatch.setattr(K, "_ACTIVE_PROFILE", dict(prof))
    tile_m = max(int(r["block_m"]) for r in prof["matrix_unit"]["mm"])
    top = B.bucket_of("M", tile_m + 1)
    assert B.bucket_of("M", top) == top
    key = (top, 256, 256, False, False, "fp16", "fp16", "fp16")
    m = H.gemm_rows(key, ["fp16", "fp16", "fp16"], unit_keyed=True)
    assert m == top and LK.matrix_unit_operands(NBXDtype.float16, NBXDtype.float16, m) is not None
