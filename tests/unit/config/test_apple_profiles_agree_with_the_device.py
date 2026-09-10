"""Every Apple profile must still agree with the hardware it describes.

The Apple profiles are being cut one file per reported variant rather than one
per family, because the spread inside a family is real: in the M5 generation
alone the GPU goes from eight or ten cores on the base chip to eighty on the
Ultra, and bandwidth from 153 GB/s to 1200. Eight times, inside one family —
and the desktop, which is the priority, is the end of that scale an average
serves worst. The backend already names the variant: it builds its arch string
from the device name, so `Apple M4 Pro` becomes `apple-m4-pro`, not
`apple-m4`.

Complete and standalone files, with no pointer and no runtime inheritance, buy
that precision at a price: a correctness fix applied in one file and forgotten
in the nineteen others is nineteen silent wrongs — the class this project
spends its days hunting.

**This test is the answer to that price.** It reads every Apple profile and
confronts the HARD limits — the ones that are properties of the architecture
and not of the variant — against what the device and the backend actually
report for this target. A copy left behind becomes a red test, never a false
picture.

It is written before the first per-variant file, deliberately.
"""
from pathlib import Path

import pytest
import yaml

APPLE = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors" / "apple"

#: Keys that are properties of the Apple GPU architecture rather than of one
#: chip. Every variant file must carry them, and carry the SAME value: a
#: threadgroup holds 1024 threads on an M1 and on an M5 Ultra alike.
_ARCHITECTURAL = (
    ("memory", "max_shared_memory_per_block"),
    ("block_sizes", "default"),
    ("autotune_screen_rtol",),
    ("autotune_screen_max_bytes",),
    ("shader", "language_version"),
)


def _profiles() -> dict:
    out = {}
    for f in sorted(APPLE.glob("*.yml")):
        out[f.name] = yaml.safe_load(f.read_text()) or {}
    return out


def _dig(doc, path):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _device():
    try:
        import Metal
    except Exception:
        return None
    return Metal.MTLCreateSystemDefaultDevice()


# ------------------------------------------------------------ completeness

def test_there_is_at_least_one_apple_profile():
    assert _profiles(), f"no Apple profile under {APPLE}"


@pytest.mark.parametrize("path", _ARCHITECTURAL, ids=lambda p: ".".join(p))
def test_every_profile_declares_the_architectural_limits(path):
    """A standalone file that omits a hard limit is not standalone: it is a
    file with a hole that something else will have to fill, which is the
    inheritance this cut exists to remove."""
    missing = [name for name, doc in _profiles().items() if _dig(doc, path) is None]
    assert not missing, (
        f"{'.'.join(path)} is absent from {missing}. Each variant file is "
        f"complete on its own — no pointer, no runtime inheritance — so an "
        f"absent hard limit is a hole, not a default.")


@pytest.mark.parametrize("path", _ARCHITECTURAL, ids=lambda p: ".".join(p))
def test_the_profiles_agree_on_what_belongs_to_the_architecture(path):
    """The copy that fell behind, made red.

    These values describe the GPU family, not the chip. If two files disagree,
    one of them was edited and the other was not — and the one that was not is
    serving a stale number to whichever machine it matches.
    """
    import json

    def canonical(v):
        """Order-insensitive, so two files that write the same mapping in a
        different order are not called divergent. `repr()` was: a yaml
        round-trip that sorts keys made an identical profile look stale."""
        return json.dumps(v, sort_keys=True, default=str)

    seen = {}
    for name, doc in _profiles().items():
        v = _dig(doc, path)
        if v is not None:
            seen.setdefault(canonical(v), []).append(name)
    assert len(seen) <= 1, (
        f"{'.'.join(path)} differs between Apple profiles: "
        + "; ".join(f"{val} in {files}" for val, files in seen.items())
        + ". These are architectural, so a difference means a copy was left "
          "behind rather than a chip that differs.")


# ---------------------------------------------- confronted with the device

@pytest.mark.skipif(_device() is None, reason="not an Apple device")
def test_the_shared_memory_budget_is_what_the_device_reports():
    dev = _device()
    reported = int(dev.maxThreadgroupMemoryLength())
    for name, doc in _profiles().items():
        declared = _dig(doc, ("memory", "max_shared_memory_per_block"))
        assert declared == reported, (
            f"{name} declares {declared} bytes of threadgroup memory; this "
            f"device reports {reported}. A profile that promises more than "
            f"the hardware gives produces kernels that will not launch.")


@pytest.mark.skipif(_device() is None, reason="not an Apple device")
def test_no_declared_tile_exceeds_the_threadgroup_this_device_reports():
    """The 1024 that most of the refusals in the register bump into.

    Read from the device rather than written here, so a future Apple GPU with
    a different ceiling moves this test instead of contradicting it.
    """
    dev = _device()
    ceiling = int(dev.maxThreadsPerThreadgroup().width)
    for name, doc in _profiles().items():
        blocks = doc.get("block_sizes") or {}
        assert int(blocks.get("default", 0)) <= ceiling, (
            f"{name}: block_sizes.default {blocks.get('default')} exceeds the "
            f"{ceiling}-thread threadgroup this device reports")
        for tile_name, tile in blocks.items():
            if not isinstance(tile, dict):
                continue
            warps = tile.get("num_warps")
            if warps is not None:
                assert int(warps) * 32 <= ceiling, (
                    f"{name}: block_sizes.{tile_name} asks for {warps} warps "
                    f"({int(warps) * 32} threads) against a {ceiling}-thread "
                    f"threadgroup")


#: The three tiles the engine reads from the profile, and for each one WHICH
#: extent is the one being reduced. The kernels stage that reduce one element
#: per thread, so it is that extent — not the tile's area — that cannot exceed
#: the threadgroup. Named per tile because it differs per kernel: argmax scans
#: a row, mv reduces along BLOCK_M, gemv_vec along BLOCK_K.
_REDUCED_EXTENT = {
    "argmax": "tile_n",
    "mv": "block_m",
    "gemv_vec": "block_k",
}


@pytest.mark.skipif(_device() is None, reason="not an Apple device")
@pytest.mark.parametrize("tile_name", sorted(_REDUCED_EXTENT))
def test_every_profile_declares_the_tiles_the_engine_reads(tile_name):
    """A silent profile is not a neutral profile for these three.

    Absent, each key falls back to the module default — 4096/4 warps for
    argmax, 256/4 for mv, 8x256/4 for gemv_vec — which asks 4096, 256 and 2048
    elements of 128 threads. On a backend that stages the reduce one element
    per thread that is a REFUSAL, not a slow path. So the profile that says
    nothing is the profile that makes these kernels refuse, and the omission
    has to be a red test rather than a quiet inheritance.
    """
    for name, doc in _profiles().items():
        tile = (doc.get("block_sizes") or {}).get(tile_name)
        assert isinstance(tile, dict), (
            f"{name}: block_sizes.{tile_name} is absent, so this profile "
            f"inherits the module default, which does not fit an Apple "
            f"threadgroup")
        assert tile.get("num_warps"), f"{name}: block_sizes.{tile_name} has no num_warps"
        extent = _REDUCED_EXTENT[tile_name]
        assert tile.get(extent), f"{name}: block_sizes.{tile_name} has no {extent}"


@pytest.mark.skipif(_device() is None, reason="not an Apple device")
@pytest.mark.parametrize("tile_name", sorted(_REDUCED_EXTENT))
def test_the_reduced_extent_fits_the_threads_that_reduce_it(tile_name):
    """The relation the tile keys exist to express, confronted with the device.

    `num_warps * 32 <= ceiling` is checked elsewhere and is not this: a tile
    may respect the threadgroup ceiling and still ask more elements of it than
    it has threads. That is exactly the shape of the refusals in the register
    ("a 2048-element tile needs 2048 threads and a threadgroup holds at most
    1024"), so it is asserted from the device's own number.
    """
    ceiling = int(_device().maxThreadsPerThreadgroup().width)
    extent_key = _REDUCED_EXTENT[tile_name]
    for name, doc in _profiles().items():
        tile = (doc.get("block_sizes") or {}).get(tile_name)
        if not isinstance(tile, dict):
            continue                       # the absence is the test above
        threads = int(tile["num_warps"]) * 32
        extent = int(tile[extent_key])
        assert threads <= ceiling, (
            f"{name}: block_sizes.{tile_name} asks {threads} threads against "
            f"the {ceiling} this device reports")
        assert extent <= threads, (
            f"{name}: block_sizes.{tile_name}.{extent_key} is {extent} but "
            f"only {threads} threads reduce it — one element per thread is "
            f"what the kernel does, so this tile cannot be served")


@pytest.mark.skipif(_device() is None, reason="not an Apple device")
def test_this_device_has_a_profile_or_the_fallback_is_deliberate():
    """Detection is exact-variant-first with a family fallback, announced.

    The backend names the variant from the device: `Apple M4 Pro` becomes
    `apple-m4-pro`. This test does not require that every chip ever made has a
    file — an unknown chip must RUN, correctly if not optimally, because
    refusing would contradict a cascade that never refuses. It requires that
    when a file for this exact variant exists, it is the one that matches.
    """
    dev = _device()
    variant = dev.name().lower().replace(" ", "-")          # apple-m4-pro
    names = set(_profiles())
    exact = f"{variant.replace('-', '_')}.yml"
    if exact not in names:
        pytest.skip(f"no per-variant file for {variant} yet; the family "
                    f"profile serves it, which is the declared fallback")
    doc = _profiles()[exact]
    # `compute_capability`, not `device_prefix`: the latter is the TORCH
    # device string ("mps") and names no chip. The selector in
    # ops/_configs.py calls a profile exact when its compute_capability
    # equals the reported target, and that is what makes a per-variant file
    # win over the family one.
    assert (doc.get("compute_capability") or "").strip().lower() == variant, (
        f"{exact} exists but declares compute_capability "
        f"{doc.get('compute_capability')!r}; the backend reports {variant!r}, "
        f"so this file would never be the exact match and the family profile "
        f"would serve this machine instead")
