"""A container's directory carries its manifest's model name, or the engine refuses it.

The name of a model is the name of its Hugging Face repository — never invented, never
hand-suffixed, never renamed — and two traces of one repository under two names are a
duplication whatever the graphs (the owner, 2026-09-26). The shared cache held four such
directories on that day (`PixArt-XL-1024` over a manifest declaring `PixArt-XL-2-1024-MS`, the
Sigma and Sana pairs, a `.pre-G-backup`). This door refuses the state at every entry: the
directory short-cut (`ensure_extracted`), the runtime loader, and extraction from a `.nbx`
whose slot (its parent directory's name) is not what its manifest declares.

Seen RED on main f29981a8: every cell below passed the misnamed container through.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from neurobrix.nbx.cache import NBXCache, ensure_extracted, refuse_misnamed

CORE = {
    "manifest.json": {"model_name": "Repo-Name", "family": "llm", "nbx_version": "0.1", "components": {}},
    "topology.json": {"stages": []},
    "runtime/variables.json": {},
    "runtime/defaults.json": {},
}


def _container(root, dirname):
    d = root / dirname
    for rel, doc in CORE.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(json.dumps(doc))
    return d


def test_the_brick_refuses_a_directory_named_otherwise(tmp_path):
    refuse_misnamed(tmp_path / "Repo-Name", CORE["manifest.json"])
    with pytest.raises(RuntimeError, match="declares model_name 'Repo-Name'"):
        refuse_misnamed(tmp_path / "Repo-Name-v2", CORE["manifest.json"])
    with pytest.raises(RuntimeError, match="declares no model_name"):
        refuse_misnamed(tmp_path / "Repo-Name", {})


def test_the_directory_short_cut_refuses_a_misnamed_container(tmp_path):
    good = _container(tmp_path, "Repo-Name")
    assert ensure_extracted(good) == good
    bad = _container(tmp_path, "Repo-Name.pre-G-backup")
    with pytest.raises(RuntimeError, match="its directory is named 'Repo-Name.pre-G-backup'"):
        ensure_extracted(bad)


def test_the_runtime_loader_refuses_a_misnamed_container(tmp_path):
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    bad = _container(tmp_path, "Hand-Name")
    with pytest.raises(RuntimeError, match="declares model_name 'Repo-Name'"):
        NBXRuntimeLoader().load(str(bad))


def test_extraction_refuses_a_slot_the_manifest_does_not_name(tmp_path):
    """`get_cache_path` keys the slot on the .nbx's parent directory; a build placed under a
    hand-chosen directory must not install under it."""
    def nbx_under(dirname):
        d = tmp_path / "builds" / dirname
        d.mkdir(parents=True)
        f = d / "model.nbx"
        with zipfile.ZipFile(f, "w") as z:
            for rel, doc in CORE.items():
                z.writestr(rel, json.dumps(doc))
        return f
    cache = NBXCache(cache_dir=tmp_path / "cache") if "cache_dir" in NBXCache.__init__.__code__.co_varnames else None
    if cache is None:
        pytest.skip("NBXCache takes no cache_dir")
    ok = cache.extract(nbx_under("Repo-Name"))
    assert ok.name == "Repo-Name" and (ok / "manifest.json").exists()
    with pytest.raises(RuntimeError, match="its directory is named 'Repo-Name-hand'"):
        cache.extract(nbx_under("Repo-Name-hand"))
    assert not (tmp_path / "cache" / "Repo-Name-hand" / "manifest.json").exists(), "refused, yet installed"
