"""An artifact whose sha256 is not the one the container declares installs nothing: no file in the cache, nothing
half-written, no other host asked; the refusal says where the expected value came from. The same for a carried file
installed for a model, and for a file in the cache that changed since.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
from tests.unit.processors import toy
from tests.unit.processors.conftest import env_of, neurobrix

MISMATCH = (f"Error: The {toy.LABEL} sub-component that arrived is not the one this model declares (its checksum "
            "differs from the one the model's container configuration states). Nothing was installed. Run the command "
            "again; if it happens twice, write to contact@neurobrix.org.")


def _nothing_in(places):
    return sorted(p.name for p in (places["NEUROBRIX_CACHE"] / "processors").glob("*") if not p.name.endswith(".lock"))


def test_a_download_whose_checksum_differs_is_final_and_leaves_nothing(places):
    declared, served = toy.artifact(), toy.artifact(padding=7)
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(declared, size=len(served))])
    hub, hf = toy.Host({toy.FILE: served}), toy.Host({toy.FILE: declared})
    try:
        r = neurobrix("processor", "fetch", toy.hub_name("toy"), env=env_of(places, hub.url, hf.url))
    finally:
        hub.close(), hf.close()
    assert r.returncode == 1
    assert r.stderr.splitlines()[-1] == MISMATCH
    assert _nothing_in(places) == []
    assert hub.gets == {toy.FILE: 1} and hf.gets == {}, "a checksum mismatch asked another host"


def test_a_carried_file_that_is_not_the_declared_one_installs_nothing(places, tmp_path):
    declared = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(declared)])
    carried = tmp_path / toy.FILE
    carried.write_bytes(toy.artifact(padding=3))
    r = neurobrix("processor", "install", str(carried), "--for", toy.hub_name("toy"),
                  env=env_of(places, toy.dead_url(), toy.dead_url()))
    assert r.returncode == 1 and r.stderr.splitlines()[-1] == MISMATCH
    assert _nothing_in(places) == []


def test_a_cached_file_that_changed_is_refused_and_removed(places):
    declared = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(declared)])
    cached = places["NEUROBRIX_CACHE"] / "processors" / toy.FILE
    cached.parent.mkdir()
    corrupted = declared.replace(b"Each letter", b"Each lettem", 1)
    assert corrupted != declared
    cached.write_bytes(corrupted)
    r = neurobrix("processor", "fetch", toy.hub_name("toy"), env=env_of(places, toy.dead_url(), toy.dead_url()))
    assert r.returncode == 1 and r.stderr.splitlines()[-1] == MISMATCH
    assert _nothing_in(places) == []
