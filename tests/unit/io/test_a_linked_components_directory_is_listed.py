"""A container whose components/ is a symlink into a shared store lists its weight files (compiled load: 'No weight
files found' when Path.rglob skipped the link)."""
from neurobrix.core.io.weight_loader import listed_files


def test_a_linked_components_directory_is_listed(tmp_path):
    store = tmp_path / "store" / "transformer" / "weights"
    store.mkdir(parents=True)
    (store / "shard_000.safetensors").write_bytes(b"x")
    box = tmp_path / "box"
    box.mkdir()
    (box / "manifest.json").write_text("{}")
    (box / "components").symlink_to(tmp_path / "store", target_is_directory=True)
    assert sorted(listed_files(box)) == ["components/transformer/weights/shard_000.safetensors", "manifest.json"]
