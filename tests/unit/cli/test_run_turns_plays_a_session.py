"""`neurobrix run --turns FILE` plays a list of user messages as one chat session through the entry point (the
validation path of the session-owned KV cache, docs/design/prism_kv.md §5bis; the rule: a model is validated through
`python -m neurobrix run`).

Before: an llm run required --prompt (config/families/llm.yml `inputs.required`), so a multi-turn conversation could
only be played through the daemon. What is read here: the llm family accepts --turns in place of --prompt and still
refuses a run naming neither; the turns file is a JSON list of non-empty strings, anything else refused by name, and a
--prompt beside it is refused rather than ignored.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.cli.commands.run import load_turns
from neurobrix.core.runtime.output_dispatch import validate_required_inputs


def _args(**kw):
    base = dict(prompt=None, turns=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_an_llm_run_takes_turns_in_place_of_a_prompt():
    validate_required_inputs("llm", _args(turns="t.json"))
    validate_required_inputs("llm", _args(prompt="hello"))
    with pytest.raises(RuntimeError, match="--prompt or --turns"):
        validate_required_inputs("llm", _args())


def test_the_turns_file_is_a_list_of_messages(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(["hello", "and then?"]))
    assert load_turns(_args(turns=str(good))) == ["hello", "and then?"]
    assert load_turns(_args()) is None
    for bad in ({"turns": ["a"]}, [], ["a", ""], ["a", 3]):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="JSON list"):
            load_turns(_args(turns=str(p)))
    with pytest.raises(SystemExit, match="--prompt beside it"):
        load_turns(_args(turns=str(good), prompt="x"))
