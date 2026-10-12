"""retrace_zoo keys a trace by the registry ENTRY a cached container belongs to, never by the
cache's name when that is the source repository's name. Wan2.1-T2V-1.3B-Diffusers (cached under
its repository's name, the registry entry renamed Wan2.1-T2V-1.3B on 2026-09-28) traced its graphs
under the repository's name and the build, naming the container by the entry, refused it
(retrace6, 2026-10-12 03:05). Seen RED on 8d6eceb4 (registry_name = the cache name).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402

REG = """
video:
  Wan2.1-T2V-1.3B:
    hf_repo: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    repo_format_suffix: "-Diffusers"
  Twin-A:
    hf_repo: "Org/Shared-Repo"
  Twin-B:
    hf_repo: "Org/Shared-Repo"
image:
  PixArt-XL-2-1024-MS:
    hf_repo: "PixArt-alpha/PixArt-XL-2-1024-MS"
"""


@pytest.fixture
def forge(tmp_path, monkeypatch):
    (tmp_path / "forge" / "config").mkdir(parents=True)
    (tmp_path / "forge" / "config" / "model_registry.yml").write_text(REG)
    monkeypatch.setattr(R, "FORGE", tmp_path / "forge" / "forge.py")
    return tmp_path


@pytest.mark.parametrize("cached,entry", [
    ("Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B"),   # the repository's name -> the entry
    ("Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B"),             # the entry itself
    ("PixArt-XL-2-1024-MS", "PixArt-XL-2-1024-MS"),     # entry == repository
    ("Sana-1600M-MultiLing", "Sana_1600M_1024px_MultiLing"),  # the declared alias
    ("Not-Registered", "Not-Registered"),               # left to the trace's own refusal
])
def test_the_entry_of_a_cached_name(forge, cached, entry):
    assert R.registry_entry_name(cached) == entry


def test_a_repository_shared_by_two_entries_is_refused(forge):
    with pytest.raises(ValueError, match="2 registry entries"):
        R.registry_entry_name("Shared-Repo")


def test_the_trace_runs_under_the_entry(forge, monkeypatch):
    tmp_path = forge
    monkeypatch.setattr(R, "PY", sys.executable)
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(C, "family_of", lambda n: "video")
    ran = []
    monkeypatch.setattr(R, "run", lambda cmd, *a, **k: ran.append([str(c) for c in cmd]) or 0)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"),
                                 models_root=str(tmp_path / "builds"), tmp=str(tmp_path / "tmp"), gpu="3",
                                 src=str(tmp_path / "src"), extra=[], timeout=10, trace_timeout=10)
    m = R.Model("Wan2.1-T2V-1.3B-Diffusers", args)
    assert m.name == "Wan2.1-T2V-1.3B-Diffusers"           # the old arm still reads the cached container
    monkeypatch.setattr(m, "snapshot", lambda: tmp_path)
    monkeypatch.setattr(m, "fixed_point_door", lambda: None)
    assert m.step_trace() is True
    (cmd,) = ran
    assert cmd[cmd.index("--model") + 1] == "Wan2.1-T2V-1.3B", cmd
