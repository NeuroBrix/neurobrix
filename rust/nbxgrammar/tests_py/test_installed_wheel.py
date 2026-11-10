"""The installed wheel's smoke test (CI runs it on every platform's wheel): the library loads from the package, a JSON
schema and a description compile, and a byte vocabulary walks a valid output to its end; a construct the engine
cannot enforce is refused by name."""
import json

import numpy as np
import pytest

import nbxgrammar as G

EOS = 256


def _vocab():
    return G.Vocab([bytes([b]) for b in range(256)] + [None], [EOS])


def _walk(grammar, text):
    v, m = _vocab(), G.Matcher(grammar)
    for b in text.encode():
        w = m.mask(v).view(np.uint32)
        assert (w[b >> 5] >> (b & 31)) & 1, (text, b)
        m.advance(v, b)
    return bool((m.mask(v).view(np.uint32)[EOS >> 5] >> (EOS & 31)) & 1)


def test_a_schema_walks_to_its_end():
    g = G.Grammar.json_schema(json.dumps({"type": "object", "properties": {"city": {"type": "string"},
                                          "n": {"type": "integer"}}, "required": ["city", "n"]}))
    assert _walk(g, '{"city": "Paris", "n": 3}')


def test_a_description_with_a_literal_and_a_schema():
    g = G.Grammar.description(json.dumps({"seq": [{"lit": "<a>"}, {"schema": {"enum": ["x", "y"]}}]}))
    assert _walk(g, '<a>"y"')


def test_an_unenforceable_construct_is_refused_by_name():
    with pytest.raises(ValueError, match="pattern"):
        G.Grammar.json_schema(json.dumps({"type": "string", "pattern": "a+"}))


def test_the_library_carries_no_path_of_the_machine_that_built_it():
    """The wheel is published: its library names no build path (macOS wrote the absolute build path as the install name
    by default, which also made two builds of one tree differ) — setup.py remaps the crate's root and sets @rpath."""
    from pathlib import Path
    lib = next(p for p in Path(G.__file__).parent.iterdir() if p.suffix in (".dylib", ".so", ".dll"))
    data = lib.read_bytes()
    assert b"/Users/" not in data and b"/home/" not in data and b"C:\\Users\\" not in data
