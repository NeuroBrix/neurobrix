"""The retrace resolves a model's hub entry from the LIVE registry, never a frozen map.

parakeet-tdt-1.1b, 2026-10-12: the tool read validation_outputs/retrace_2026_09_07/hub_map.json, asked the
registry for nvidia/Parakeet-TDT-1.1B and got 404 — the registry had renamed the entry to its maker's
repository, nvidia/parakeet-tdt-1.1b. The entry is now the registry's org/name equal, ignoring case, to
the `hf_repo` the build toolchain's registry names; a repository the registry does not list is None (a
new entry), stated by name. Injection: the match made case-sensitive -> RED (canary: hf_repo nvidia/canary-qwen-2.5b, listed
nvidia/Canary-Qwen-2.5B; the ambiguity refusal).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


def _registry(tmp_path, monkeypatch, text):
    forge = tmp_path / "forge"
    (forge / "config").mkdir(parents=True)
    (forge / "config" / "model_registry.yml").write_text(text)
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")


def _stub_hub(monkeypatch, slugs, page_size=2, fail=False):
    """A registry serving `slugs` in pages of `page_size`, as the server pages them."""
    import requests
    calls = []

    def get(url, params=None, timeout=None, **_):
        calls.append(params)
        if fail:
            raise requests.ConnectionError("no route to host")
        assert url.endswith("/api/models"), url
        page = int(params["page"])
        chunk = slugs[(page - 1) * page_size: page * page_size]
        return _Resp({"models": [{"slug": s, "org": s.split("/")[0], "name": s.split("/")[1]} for s in chunk],
                      "total": len(slugs), "page": page, "limit": page_size})
    monkeypatch.setattr(requests, "get", get)
    return calls


REG = ("stt:\n  parakeet-tdt-1.1b: {hf_repo: \"nvidia/parakeet-tdt-1.1b\"}\n"
       "  canary-qwen-2.5b: {hf_repo: \"nvidia/canary-qwen-2.5b\"}\n"
       "diffusers:\n  Sana_1600M_4Kpx_BF16: {hf_repo: \"Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers\"}\n"
       "upscaler:\n  bare: {checkpoint_file: x.pth}\n")
LIVE = ["ResembleAI/chatterbox", "nvidia/Canary-Qwen-2.5B", "openai/whisper-large", "nvidia/parakeet-tdt-1.1b",
        "Efficient-Large-Model/Sana_1600M_1024px_MultiLing"]


def test_the_entry_is_the_registrys_org_name_matched_by_hf_repo(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, REG)
    calls = _stub_hub(monkeypatch, LIVE)
    assert R.hub_entry("parakeet-tdt-1.1b", registry="http://hub") == "nvidia/parakeet-tdt-1.1b"
    assert len(calls) == 3                          # every page read (5 entries, pages of 2)
    # case-insensitive, and the registry's own spelling is returned
    assert R.hub_entry("canary-qwen-2.5b", registry="http://hub") == "nvidia/Canary-Qwen-2.5B"


def test_a_repository_the_registry_does_not_list_is_a_new_entry(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, REG)
    _stub_hub(monkeypatch, LIVE)
    assert R.hub_entry("Sana_1600M_4Kpx_BF16", registry="http://hub") is None
    assert R.hub_entry("bare", registry="http://hub") is None              # no hf_repo: no lookup at all
    assert R.hub_entry("not-in-the-registry", registry="http://hub") is None


def test_no_frozen_map_is_read(tmp_path, monkeypatch):
    assert not hasattr(R, "HUB_MAP")
    src = (Path(R.__file__)).read_text()
    assert "hub_map.json\")" not in src and "HUB_MAP" not in src


def test_an_unreachable_registry_is_named_never_a_guess(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, REG)
    _stub_hub(monkeypatch, LIVE, fail=True)
    with pytest.raises(R.HubUnreachable, match="did not list its models"):
        R.hub_entry("parakeet-tdt-1.1b", registry="http://hub")


def test_two_entries_equal_but_for_case_are_refused(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, REG)
    _stub_hub(monkeypatch, LIVE + ["NVIDIA/Parakeet-TDT-1.1B"])
    with pytest.raises(R.HubUnreachable, match="ambiguous"):
        R.hub_entry("parakeet-tdt-1.1b", registry="http://hub")


def test_an_absent_toolchain_registry_is_refused_not_read_as_new(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FORGE", tmp_path / "nowhere" / "forge.py")
    _stub_hub(monkeypatch, LIVE)
    with pytest.raises(R.HubUnreachable, match="absent"):
        R.hub_entry("parakeet-tdt-1.1b", registry="http://hub")


def test_a_card_list_reaches_forge_as_its_unified_vmm_pool(tmp_path):
    """`--gpu 0,1,3` is the POOL: every visible card joins Forge's unified VMM (forge/vmm/vmm_pool.py,
    worker.py `pool_applies` -> `vmm_pool_primary_gpu`), so the list must reach the trace whole."""
    import types
    m = types.SimpleNamespace(args=types.SimpleNamespace(gpu="0,1,3", src=None, tmp=str(tmp_path / "t"),
                                                         models_root=str(tmp_path / "b")))
    assert R.Model.env(m, tree=False)["CUDA_VISIBLE_DEVICES"] == "0,1,3"
