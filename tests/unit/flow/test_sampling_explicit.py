"""A sampling value equal to the model default OR the family default is
inherited, never "explicit": the family rung of the default cascade fills
`global.*` for a model whose defaults carry no such key (audio_llm top_k=50,
2026-09-05 — every model of the family was refused for a value nobody asked for)."""
import types

import pytest

from neurobrix.core.flow import audio_llm as AL


class _Resolver:
    def __init__(self, values):
        self._v = values

    def get(self, name, default=None):
        return self._v.get(name, default)


def _ctx(family, values):
    return types.SimpleNamespace(pkg=types.SimpleNamespace(manifest={"family": family}),
                                 variable_resolver=_Resolver(values))


def test_family_default_is_inherited_not_explicit(monkeypatch):
    from neurobrix.core.config import loader
    monkeypatch.setattr(loader, "get_family_config", lambda fam: {"defaults": {"top_k": 50, "temperature": 0.7}})
    defaults = {"temperature": 0.0, "top_k": None}
    cfg, explicit = AL._effective_sampling(_ctx("audio_llm", {"global.top_k": 50, "global.temperature": 0.0}), defaults)
    assert cfg["top_k"] == 50 and explicit == set()


def test_a_value_matching_neither_rung_is_explicit(monkeypatch):
    from neurobrix.core.config import loader
    monkeypatch.setattr(loader, "get_family_config", lambda fam: {"defaults": {"top_k": 50}})
    cfg, explicit = AL._effective_sampling(_ctx("audio_llm", {"global.top_k": 20}), {"top_k": None})
    assert explicit == {"top_k"}


def test_without_a_resolver_nothing_is_explicit():
    cfg, explicit = AL._effective_sampling(types.SimpleNamespace(), {"top_k": 7})
    assert cfg["top_k"] == 7 and explicit == set()
