"""A snapshot is downloaded once and read in place. Nothing fetches a second copy.

`core/module/audio/output_processor.py` called
`snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")` — a bare hub id, which
downloads into `~/.cache/huggingface/hub` on every machine that runs it. On
2026-09-17 that cache held 4.0 G across 25 entries on a local disk that had
reached 0 MB the same day, and **every one of those models already had its
snapshot on the NAS**: the 76 MB snac blob was byte-identical
(sha 4b8164cc…) to `hf_snapshots/snac_24khz/pytorch_model.bin`.

The full R34 remedy — baking the codec into the container so no third-party
package is imported at runtime — is its own workstream. This stops the duplication.
"""

import ast
from pathlib import Path

import pytest

from neurobrix.core.workspace import (
    SnapshotNotPresent,
    snapshot_path,
    snapshots_root,
)

REPO = Path(__file__).resolve().parents[3]
OUTPUT_PROCESSOR = REPO / "src" / "neurobrix" / "core" / "module" / "audio" / "output_processor.py"


def test_the_root_comes_from_the_engine_configuration():
    import yaml
    cfg = yaml.safe_load((REPO / "src" / "neurobrix" / "config" / "system.yml").read_text())
    assert str(snapshots_root()) == cfg["paths"]["snapshots"]


def test_an_explicit_environment_variable_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUROBRIX_HF_SNAPSHOTS", str(tmp_path))
    assert snapshots_root() == tmp_path


def test_a_missing_snapshot_refuses_and_names_the_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUROBRIX_HF_SNAPSHOTS", str(tmp_path))
    with pytest.raises(SnapshotNotPresent) as e:
        snapshot_path("not_here", "nor_here")
    msg = str(e.value)
    assert "not_here" in msg and "nor_here" in msg and str(tmp_path) in msg
    assert "hub cache" in msg


def test_several_names_are_tried_because_disk_and_hub_spellings_differ(monkeypatch, tmp_path):
    (tmp_path / "second_name").mkdir()
    monkeypatch.setenv("NEUROBRIX_HF_SNAPSHOTS", str(tmp_path))
    assert snapshot_path("first_name", "second_name") == tmp_path / "second_name"


def test_snapshot_path_takes_no_default_parameter():
    """A caller able to pass one would be the fallback this removes."""
    import inspect
    assert "default" not in inspect.signature(snapshot_path).parameters


def test_no_bare_hub_id_is_handed_to_from_pretrained_in_the_audio_path():
    """Scanned over the AST, not the text: this file's own comment quotes the
    removed hub id while explaining it, and a line-based scan reads that as the
    defect still being present — which happened three times on 2026-09-17."""
    tree = ast.parse(OUTPUT_PROCESSOR.read_text())
    docstrings = set()
    # Only the node types that CARRY a docstring: `getattr(node, "body")` over
    # every node hits `IfExp.body`, which is one expression and not a list.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    constants = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and id(n) not in docstrings]
    assert "hubertsiuzdak/snac_24khz" not in constants, \
        "a bare hub id downloads a second copy into the Hugging Face cache"
    assert "snapshot_path" in OUTPUT_PROCESSOR.read_text()
