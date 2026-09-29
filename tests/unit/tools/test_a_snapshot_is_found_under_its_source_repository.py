"""The retrace finds a model's snapshot under the SOURCE repository its registry entry names, first.

The owner's naming rule (2026-09-28 03:06): a model keeps its maker's model name and its source repository
is recorded beside it (`hf_repo`). Sana_1600M_4Kpx_BF16 is built from `..._diffusers`, while the directory
under the model's own name is the original checkpoint; the retrace looked up names only and refused "no
COMPLETE snapshot" (2026-09-29). Injection: the source repository dropped from `snapshot_names` -> RED.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


def test_the_source_repository_comes_first(tmp_path, monkeypatch):
    forge = tmp_path / "forge"
    (forge / "config").mkdir(parents=True)
    (forge / "config" / "model_registry.yml").write_text(
        "diffusers:\n  Sana_1600M_4Kpx_BF16:\n    hf_repo: \"Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers\"\n")
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")
    assert R.registry_source_repository("Sana_1600M_4Kpx_BF16") == "Sana_1600M_4Kpx_BF16_diffusers"
    fake = type("M", (), {"registry_name": "Sana_1600M_4Kpx_BF16", "name": "Sana_1600M_4Kpx_BF16"})()
    assert R.Model.snapshot_names(fake) == ["Sana_1600M_4Kpx_BF16_diffusers", "Sana_1600M_4Kpx_BF16"]


def test_a_model_the_registry_does_not_name_keeps_its_own_name(tmp_path, monkeypatch):
    forge = tmp_path / "forge"
    (forge / "config").mkdir(parents=True)
    (forge / "config" / "model_registry.yml").write_text("llm:\n  Other: {hf_repo: \"a/b\"}\n")
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")
    assert R.registry_source_repository("Kokoro-82M") is None
