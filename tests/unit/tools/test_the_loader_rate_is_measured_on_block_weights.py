"""tools/profile_facts.py measures the engine's own loader (`weight_load_gbps`) on a component's first BLOCK weights up
to a stored budget (the host-held path loads block weights only). What the test would do if the selection were
wrong: a non-block weight measured, the budget overshot by more than its last tensor, or a component without shards
not refused — red."""
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import profile_facts as PF  # noqa: E402


def _shard(path: Path, tensors):
    """A safetensors file of uint8 tensors {name: nbytes}, in the given order."""
    header, off = {}, 0
    for name, n in tensors:
        header[name] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def test_the_measured_set_is_block_weights_up_to_the_budget(tmp_path):
    w = tmp_path / "components" / "c" / "weights"
    w.mkdir(parents=True)
    _shard(w / "shard_000.safetensors", [("embed.weight", 4000), ("block.0.w", 3000), ("block.1.w", 3000),
                                         ("block.2.w", 3000), ("final_norm.weight", 10)])
    keys, stored, shards = PF.block_keys(tmp_path, "c", 5000 / 1e9)
    assert keys == ["block.0.w", "block.1.w"], keys
    assert stored == 6000 and [s.name for s in shards] == ["shard_000.safetensors"]


def test_a_component_without_shards_is_refused_by_name(tmp_path):
    (tmp_path / "components" / "x" / "weights").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no shard under"):
        PF.block_keys(tmp_path, "x", 1.0)
