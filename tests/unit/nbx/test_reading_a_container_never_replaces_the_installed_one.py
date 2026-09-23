"""Reading a `.nbx` must not overwrite the container already installed in the cache.

`NBXContainer.load()`, `NBXLoader` and `core/runtime/factory` all reach `ensure_extracted()`.
They are READ paths — none of them means "replace the canonical copy of this model". But
`extract()` re-unpacks whenever the `.nbx` is newer than the cache, so handing any of them a
freshly built container silently replaced the installed one.

MEASURED, 2026-09-22 17:44
--------------------------
A call made to GATE a new build's symbolic dims printed

    [Cache] Extracting model.nbx -> ~/.neurobrix/<cache>/mochi-1-preview
    [Cache] Done: 43 files, 41.05GB extracted

and replaced the canonical mochi-1-preview. Both states survived as files, so nothing was
lost — but nothing about the call said "replace a container", and nothing refused.

WHY THE EXISTING DOOR COULD NOT SEE IT
--------------------------------------
The repository's blocking hook refuses a SHELL COMMAND that names the cache path. This write
came from inside a library that resolved the destination itself, from a path in
`nbx/builds/`. No argument on the command line named the cache. A door that watches the
command line cannot see a library that computes where it writes — so the door belongs at the
library seam, which is where it now is.

THE HARMFUL STATE, not the outcome
----------------------------------
"A container this cache did not get from THIS .nbx is replaced by it." Fresh install is
allowed; re-extracting the SAME recorded source is the ordinary update path and is allowed;
anything else must be declared. Only 2 of the 59 containers in this cache carry
`.cache_meta.json` at all, so an ABSENT record is the common case and the least intended
place to overwrite — it refuses rather than assumes.
"""
from __future__ import annotations

import json

import pytest

from neurobrix.nbx.cache import NBXCache


def _cache(tmp_path):
    c = NBXCache.__new__(NBXCache)
    c.cache_dir = tmp_path / "cache"
    c.cache_dir.mkdir(parents=True, exist_ok=True)
    return c


def _installed(tmp_path, name="mochi-1-preview", source=None):
    """A container already sitting in the cache, optionally with its source recorded."""
    d = tmp_path / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"model_name": name}))
    if source is not None:
        (d / ".cache_meta.json").write_text(json.dumps({"source": str(source)}))
    return d


def _nbx(tmp_path, where):
    p = tmp_path / where
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a real container")
    return p


# ───────────────────────── the refusal ─────────────────────────

def test_a_different_nbx_over_an_installed_container_is_REFUSED(tmp_path):
    """The incident, exactly: cache holds one built from the export, a build elsewhere is read."""
    c = _cache(tmp_path)
    d = _installed(tmp_path, source=tmp_path / "models/video/mochi-1-preview/model.nbx")
    incoming = _nbx(tmp_path, "builds/2026_09_22/video/mochi-1-preview/model.nbx")
    with pytest.raises(RuntimeError) as e:
        c._refuse_incidental_replacement(incoming, d, declared=False)
    msg = str(e.value)
    assert "REPLACE" in msg
    assert str(incoming) in msg, "the refusal does not name the .nbx that would win"


def test_an_installed_container_with_NO_RECORD_is_refused(tmp_path):
    """57 of 59 containers here have no .cache_meta.json. Unknown provenance is the case
    where a replacement is least intended, so absence refuses rather than assumes."""
    c = _cache(tmp_path)
    d = _installed(tmp_path, source=None)
    with pytest.raises(RuntimeError) as e:
        c._refuse_incidental_replacement(_nbx(tmp_path, "b/model.nbx"), d, declared=False)
    assert "no .cache_meta.json" in str(e.value)


def test_the_refusal_names_both_ways_out(tmp_path):
    """A refusal that does not name what satisfies it is a wall, not a door."""
    c = _cache(tmp_path)
    d = _installed(tmp_path, source=None)
    with pytest.raises(RuntimeError) as e:
        c._refuse_incidental_replacement(_nbx(tmp_path, "b/model.nbx"), d, declared=False)
    msg = str(e.value)
    assert "NEUROBRIX_CACHE" in msg, "it does not say how to READ without replacing"
    assert NBXCache.REPLACE_ENV in msg, "it does not name the deliberate opening"


# ───────────────────────── what must still work ─────────────────────────

def test_a_fresh_install_is_allowed(tmp_path):
    """Nothing cached: this is how a container gets there at all."""
    c = _cache(tmp_path)
    empty = tmp_path / "cache" / "brand-new"
    empty.mkdir(parents=True)
    c._refuse_incidental_replacement(_nbx(tmp_path, "b/model.nbx"), empty, declared=False)


def test_re_extracting_the_SAME_recorded_source_is_allowed(tmp_path):
    """The ordinary update: the same artefact rebuilt in place."""
    c = _cache(tmp_path)
    src = _nbx(tmp_path, "builds/x/model.nbx")
    d = _installed(tmp_path, source=src)
    c._refuse_incidental_replacement(src, d, declared=False)


@pytest.mark.parametrize("how", ["declared", "env"])
def test_a_DECLARED_replacement_goes_through(tmp_path, monkeypatch, how):
    c = _cache(tmp_path)
    d = _installed(tmp_path, source=None)
    incoming = _nbx(tmp_path, "b/model.nbx")
    if how == "env":
        monkeypatch.setenv(NBXCache.REPLACE_ENV, "1")
        c._refuse_incidental_replacement(incoming, d, declared=False)
    else:
        c._refuse_incidental_replacement(incoming, d, declared=True)


def test_the_opening_is_exact_not_truthy(tmp_path, monkeypatch):
    """A door with a sloppy opening is not a door: only "1" opens it."""
    c = _cache(tmp_path)
    d = _installed(tmp_path, source=None)
    monkeypatch.setenv(NBXCache.REPLACE_ENV, "0")
    with pytest.raises(RuntimeError):
        c._refuse_incidental_replacement(_nbx(tmp_path, "b/model.nbx"), d, declared=False)


# ───────────────────────── the door must not touch the RUNTIME path ─────────────────────────

def test_an_already_extracted_directory_never_reaches_the_door(tmp_path):
    """`neurobrix run <model>` passes the extracted DIRECTORY (`cli/utils.find_model`), and
    `ensure_extracted` returns it without calling `extract()` at all. The door guards the
    transport path, not the runtime path — 57 of this cache's 59 containers carry no
    `.cache_meta.json`, so a door that fired on the runtime path would refuse nearly every
    run on this machine. Measured after the fix: all 59 load, 0 refused."""
    from neurobrix.nbx.cache import ensure_extracted
    d = tmp_path / "cache" / "some-model"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    assert ensure_extracted(d) == d


def test_a_directory_without_a_manifest_is_still_refused(tmp_path):
    """The pre-existing refusal, unchanged: a directory that is not a container says so."""
    from neurobrix.nbx.cache import ensure_extracted
    d = tmp_path / "not-a-container"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        ensure_extracted(d)


def test_the_door_fires_ABOVE_the_cached_short_circuit(tmp_path):
    """Both directions are harmful and only one is a replacement. When the cache is NEWER
    than the requested .nbx, `is_cached()` is True and the old code returned the cached
    tree — a DIFFERENT container under the requested name, with nothing printed. The slot is
    keyed on the .nbx's parent directory NAME, so two builds of one model always collide."""
    c = _cache(tmp_path)
    src_old = _nbx(tmp_path, "models/video/m/model.nbx")
    d = _installed(tmp_path, name="m", source=src_old)
    newer = _nbx(tmp_path, "builds/x/m/model.nbx")
    # the cache tree was written after `newer`, so is_cached() would short-circuit
    with pytest.raises(RuntimeError) as e:
        c.extract(newer)
    assert "silently serve it in place of what was asked for" in str(e.value)
