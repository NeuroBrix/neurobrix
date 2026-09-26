"""A model being reinstalled stays readable until the instant it is replaced.

The cache is often a mounted export shared by several machines. Three races lived
in it, each written twice — once in `cli/commands/registry.py` and once in
`nbx/cache.py`:

* two installers shared one staging name and deleted each other's trees,
* the live tree was removed BEFORE the rename, so the model was absent for the
  whole length of a recursive delete,
* `NBXCache.extract` did not stage at all and unpacked in place at the final
  name, where a `manifest.json` appears from the first member on.

These tests pin the contract of `nbx/atomic_install`, the one brick both call
sites now use. Each was seen failing first: the injection that turns it red is
named beside it, so a future reader can reproduce the red before trusting the
green.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from neurobrix.nbx.atomic_install import (
    InstallHeldByAnother,
    installing,
    is_install_artifact,
    lock_path,
    staging_path,
)


def _model(root: Path, name: str, marker: str) -> Path:
    """A directory shaped like an installed model: a manifest and a weight."""
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": marker, "model_name": name}))
    (d / "weights.safetensors").write_bytes(marker.encode() * 4096)
    return d


def test_the_live_tree_is_readable_at_every_instant_of_a_reinstall(tmp_path):
    """The heart of it: poll the model while it is being replaced.

    Injection that turns this red — restore the original body of the swap:

        if cache_path.exists():
            shutil.rmtree(cache_path)
        os.replace(staging, cache_path)

    With that, the poller below reads `manifest.json` missing (or the directory
    gone) for the whole duration of the rmtree, and the assertion on
    `seen_missing` fires. Seen red on 2026-09-18 with a 4 000-file tree, which
    makes the window wide enough to hit on the first run.
    """
    cache = tmp_path / "cache"
    target = _model(cache, "some-model", "OLD")
    # Enough members that a recursive delete is not instantaneous.
    for i in range(400):
        (target / f"shard_{i}.bin").write_bytes(b"old" * 512)

    seen_missing: list[str] = []
    seen_versions: set[str] = set()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                seen_versions.add(json.loads((target / "manifest.json").read_text())["version"])
            except FileNotFoundError:
                seen_missing.append("manifest.json absent")
            except (ValueError, OSError) as e:
                seen_missing.append(f"unreadable: {e}")
            time.sleep(0.001)

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        with installing(target, label="test") as staging:
            (staging / "manifest.json").write_text(json.dumps({"version": "NEW", "model_name": target.name}))
            (staging / "weights.safetensors").write_bytes(b"NEW" * 4096)
            time.sleep(0.05)          # the extraction the reader must survive
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not seen_missing, (
        "the model was unreadable during its own reinstall: "
        f"{seen_missing[:3]} ({len(seen_missing)} observations)")
    assert seen_versions == {"OLD", "NEW"} or seen_versions == {"NEW"}, seen_versions
    assert json.loads((target / "manifest.json").read_text())["version"] == "NEW"


def test_a_second_installer_is_refused_by_name_not_left_to_collide(tmp_path):
    """Injection: drop the `os.mkdir(lock)` / FileExistsError branch — the second
    `installing()` then enters, and both write the same target."""
    cache = tmp_path / "cache"
    target = cache / "shared-model"
    entered_second = False
    with installing(target, label="first") as staging:
        (staging / "manifest.json").write_text("{}")
        with pytest.raises(InstallHeldByAnother) as e:
            with installing(target, label="second"):
                entered_second = True        # pragma: no cover
    assert not entered_second
    # The refusal must be actionable: who holds it, and where the lock is.
    assert "pid" in str(e.value) and str(lock_path(target)) in str(e.value)


def test_a_failed_install_leaves_the_working_copy_exactly_as_it_was(tmp_path):
    """Injection: move the `os.replace(cache_path, aside)` above the `yield` —
    the old tree is then already gone when the body raises."""
    cache = tmp_path / "cache"
    target = _model(cache, "kept-model", "OLD")

    with pytest.raises(RuntimeError, match="extraction blew up"):
        with installing(target, label="doomed") as staging:
            (staging / "manifest.json").write_text(json.dumps({"version": "HALF"}))
            raise RuntimeError("extraction blew up")

    assert json.loads((target / "manifest.json").read_text())["version"] == "OLD"
    assert (target / "weights.safetensors").read_bytes() == b"OLD" * 4096


def test_nothing_of_the_install_is_left_beside_the_model(tmp_path):
    """Lock, staging tree and the aside copy are all gone once the swap is done.

    Injection: drop the `finally` clause — the lock survives and the NEXT install
    of the same model is refused for ever.
    """
    cache = tmp_path / "cache"
    target = _model(cache, "tidy-model", "OLD")
    with installing(target, label="tidy") as staging:
        (staging / "manifest.json").write_text("{}")
    assert sorted(p.name for p in cache.iterdir()) == ["tidy-model"]
    # And a second install therefore succeeds.
    with installing(target, label="again") as staging:
        (staging / "manifest.json").write_text("{}")


def test_two_installers_on_one_machine_never_share_a_staging_directory(tmp_path):
    """The old name was `<model>.installing` for everyone. Injection: return
    `cache_path.with_name(cache_path.name + ".installing")` from `staging_path`."""
    target = tmp_path / "cache" / "m"
    mine = staging_path(target)
    assert mine != target.with_name("m.installing"), (
        "the staging name carries neither host nor pid: two installers collide")
    assert str(os.getpid()) in mine.name


@pytest.mark.parametrize("name", [
    "model.installing",
    "model.installing.host-a.1234",
    "model.lock",
    "model.replaced.host-a.1234.1758200000",
])
def test_every_shape_an_install_leaves_is_skipped_by_discovery(name):
    """Each of these can carry a `manifest.json` while being incomplete or on its
    way out. Injection: `name.endswith('.installing')`, the original test — the
    tagged staging name and the aside both slip through and are offered as
    models."""
    assert is_install_artifact(name), f"{name} would be listed as an installed model"


def test_a_real_model_name_is_not_mistaken_for_an_install_artifact():
    """The guard must not eat names that merely contain the words.

    Without this, a model legitimately called `lock` or one whose name ends in
    `.replaced` would vanish from every listing, and the previous test would
    still be green — a guard that refuses everything passes every red-side check.
    """
    for name in ["Wan2.1-T2V-1.3B-Diffusers", "real-esrgan-x8", "locknet", "installing-model"]:
        assert not is_install_artifact(name), name


def test_the_cache_extractor_stages_and_does_not_unpack_at_the_final_name(tmp_path):
    """`NBXCache.extract` is the call site that never staged at all.

    Injection: restore `shutil.rmtree(cache_path); cache_path.mkdir(...)` and
    extract in place — the observer below then sees the final directory carry a
    manifest while members are still arriving.
    """
    import zipfile
    from neurobrix.nbx.cache import NBXCache

    cache_dir = tmp_path / "cache"
    nbx_dir = tmp_path / "store" / "demo-model"
    nbx_dir.mkdir(parents=True)
    nbx = nbx_dir / "model.nbx"
    with zipfile.ZipFile(nbx, "w") as z:
        z.writestr("manifest.json", json.dumps({"version": "NEW"}))
        for i in range(200):
            z.writestr(f"shard_{i}.bin", "x" * 2048)

    c = NBXCache(cache_dir=cache_dir)
    final = c.get_cache_path(nbx)
    _model(cache_dir, final.name, "OLD")

    observations: list[int] = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if (final / "manifest.json").exists():
                observations.append(len(list(final.iterdir())))
            time.sleep(0.001)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    try:
        out = c.extract(nbx)
    finally:
        stop.set()
        w.join(timeout=5)

    assert out == final
    assert json.loads((final / "manifest.json").read_text())["version"] == "NEW"
    # Every observation is either the complete OLD tree (2 files) or the complete
    # NEW one (200 shards + manifest + the `.cache_meta.json` the extractor writes
    # last = 202). A count in between is a half-written model made visible.
    assert set(observations) <= {2, 202}, (
        f"a partially extracted model was visible at {final}: sizes {sorted(set(observations))}")


# ---------------------------------------------------------------------------
# 2026-09-18, second pass. The cell above went RED once in the merged-tree gate:
#
#   AssertionError: the model was unreadable during its own reinstall:
#   ['manifest.json absent'] (1 observations)
#
# One observation out of thousands of polls, on a loaded machine — and one is
# enough, because the claim is "at every instant". The two-rename swap is correct
# but not instantaneous: between `rename(cache -> aside)` and
# `rename(staging -> cache)` the name does not exist. I had written that window
# off as "two syscalls", which is a description of its size, not an argument that
# it is absent.
#
# `renameat2(RENAME_EXCHANGE)` closes it: one syscall, no window at all.
# ---------------------------------------------------------------------------

def test_the_exchange_path_is_actually_taken_here():
    """A fallback that is always taken is a fallback that is the implementation.

    `_exchange` returns False on a kernel or filesystem without RENAME_EXCHANGE,
    which is deliberate — but then the window is back, and nothing would say so.
    This asserts which path this machine's cache filesystem takes, so a silent
    demotion to the two-rename form is visible rather than assumed.
    """
    from neurobrix.nbx.atomic_install import _exchange
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        a.mkdir(); b.mkdir()
        (a / "m").write_text("A"); (b / "m").write_text("B")
        took_it = _exchange(a, b)
        if not took_it:
            pytest.skip("this filesystem has no RENAME_EXCHANGE; the portable "
                        "two-rename swap is in use and its window is open")
        assert (a / "m").read_text() == "B" and (b / "m").read_text() == "A", (
            "_exchange reported success without swapping the two paths")


def test_twenty_five_reinstalls_and_the_model_is_never_absent(tmp_path):
    """The window was found under load, so the cell that pins it applies load.

    One reinstall gave one absence in thousands of polls. Twenty-five give the
    poller twenty-five chances at a window this small, which is what it takes for
    the red to be reproducible rather than occasional.

    Injection: make `_exchange` return False unconditionally, so the portable
    two-rename path is used, and this reports absences.
    """
    cache = tmp_path / "cache"
    target = _model(cache, "hot-model", "v0")
    for i in range(80):
        (target / f"f_{i}.bin").write_bytes(b"x" * 256)

    absences: list[str] = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                json.loads((target / "manifest.json").read_text())
            except FileNotFoundError:
                absences.append("absent")
            except (ValueError, OSError):
                absences.append("unreadable")

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        for n in range(25):
            with installing(target, label=f"round-{n}") as staging:
                (staging / "manifest.json").write_text(json.dumps({"version": f"v{n + 1}"}))
                for i in range(80):
                    (staging / f"f_{i}.bin").write_bytes(b"y" * 256)
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not absences, (
        f"{len(absences)} observations of the model missing across 25 reinstalls: "
        f"{absences[:3]}")
    assert json.loads((target / "manifest.json").read_text())["version"] == "v25"
