"""Prism prices a latent at the VAE scale the container declares — the one derivation the CLI binds
the request with and the executor sizes its latents with (`container_size.vae_scale_factor`).

MEASURED 2026-10-05: SANA-Video_2B_720p declares 32 (`spatial_compression_ratio`; patch_size 4 on
top of its 4 blocks); Prism took 8 from the VAE's arch (`2 ** (len(block_out_channels) - 1)`) and
priced its transformer at 84x168 latents where the runtime runs 21x42 — 16x over: a 16 GB plan of
layer streaming, a guidance split and op-level bands, where the right scale plans lazy_sequential.

Injection (seen red, then restored green): `_declared_vae_scale` returning the arch formula ->
SANA-Video 8, RED; every other container of the cache agrees with the brick either way.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.runtime.resolution.container_size import vae_scale_factor

CACHE = cache_dir()


def _container(path):
    return SimpleNamespace(cache_path=str(path))


def _write(root: Path, rel: str, data) -> None:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data))


def test_a_patchifying_vae_is_priced_at_its_declared_compression(tmp_path):
    """Four blocks (the arch says 8), a declared 32: Prism prices at 32."""
    _write(tmp_path, "manifest.json", {"name": "m"})
    _write(tmp_path, "runtime/defaults.json", {"spatial_compression_ratio": 32})
    _write(tmp_path, "topology.json", {"components": {"vae": {}, "transformer": {}}})
    _write(tmp_path, "components/vae/config.json",
           {"decoder_block_out_channels": [128, 256, 512, 1024], "patch_size": 4})
    assert PrismSolver._declared_vae_scale(_container(tmp_path)) == 32


def test_a_container_without_a_latent_space_leaves_the_request_alone(tmp_path):
    _write(tmp_path, "manifest.json", {"name": "m"})
    assert PrismSolver._declared_vae_scale(_container(tmp_path)) is None
    assert PrismSolver._declared_vae_scale(_container("")) is None


def test_sana_video_is_priced_at_32():
    if not (CACHE / "SANA-Video_2B_720p_diffusers").exists():
        pytest.skip("SANA-Video_2B_720p_diffusers is not in this machine's cache")
    assert PrismSolver._declared_vae_scale(_container(CACHE / "SANA-Video_2B_720p_diffusers")) == 32


def test_prism_and_the_runtime_agree_on_every_container_of_the_cache():
    def read(p):
        return json.loads(p.read_text()) if p.exists() else {}
    seen = 0
    for d in sorted(CACHE.iterdir()) if CACHE.exists() else []:
        if not (d / "manifest.json").exists():
            continue
        seen += 1
        runtime = vae_scale_factor(read(d / "manifest.json"), read(d / "runtime/defaults.json"),
                                   read(d / "topology.json").get("components") or {})
        assert PrismSolver._declared_vae_scale(_container(d)) == (int(runtime) if runtime else None), d.name
    if not seen:
        pytest.skip("no container in this machine's cache")
