"""The CPU as an execution device reads its facts from config/vendors/cpu/<architecture>.yml (Prism's cpu_execution rung,
the compiled branch). The key census (2026-11-01): the paged KV cache's two memory keys are all the CPU path reads. Every
cpu/ vendor file states them, and the paged cache resolves on it. Before this file a card-less launch was refused with
"No hardware profile for cpu/<arch>" (the Dell, 2026-10-31 20:15).

Injection: a key removed from cpu/arm64.yml -> the first test is red; the file removed -> the second is red."""
from pathlib import Path

import pytest
import yaml

VENDORS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors" / "cpu"
CPU_PATH_KEYS = ("state_page_granule_bytes", "async_release")      # read by core/state/price.paged_cache_params


@pytest.mark.parametrize("path", sorted(VENDORS.glob("*.yml")), ids=lambda p: p.stem)
def test_every_cpu_vendor_file_states_the_keys_the_cpu_path_reads(path):
    memory = (yaml.safe_load(path.read_text()) or {}).get("memory") or {}
    missing = [k for k in CPU_PATH_KEYS if k not in memory]
    assert not missing, f"cpu/{path.name} states no memory.{', memory.'.join(missing)}"
    assert int(memory["state_page_granule_bytes"]) > 0 and isinstance(memory["async_release"], bool)


def test_the_paged_cache_resolves_on_an_arm64_host():
    from neurobrix.core.config.loader import get_vendor_config
    from neurobrix.core.state.price import paged_cache_params
    from nbxtensor.ops.paged_decode_attn import PAGED_DECODE_TILE_CANDIDATES
    tokens, async_release = paged_cache_params(4, 64, 64, "float32", get_vendor_config("cpu", "arm64").get("memory"),
                                               64 << 20, PAGED_DECODE_TILE_CANDIDATES)
    assert tokens > 0 and async_release is False
