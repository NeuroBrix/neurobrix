"""The engine's one call gives the declared entry processor, loaded from the verified bytes of its artifact under a
module name that carries its sha256 (never its own name), bound to its declaration: a variant it does not list, an
interface newer than the engine's, a part the container lacks are each refused by their sentence, the last two
before any byte is asked.

Red without the change: `neurobrix.processors` does not exist."""
import sys

import pytest

from tests.unit.processors import toy


class _Get:
    """An HTTP GET that serves the toy artifact from memory, and counts."""

    def __init__(self, data):
        self.data, self.asked = data, []

    def __call__(self, url, **kw):
        self.asked.append(url)
        data = self.data

        class Reply:
            status_code, headers = 200, {"content-length": str(len(data))}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield data
        return Reply()


def _open(places, records, get):
    from neurobrix.processors import open_declared
    entry = toy.install_model(places["NEUROBRIX_CACHE"], "toy", records)
    return open_declared({"processors": records}, entry, ["text_tagger"], get=get)


def test_the_entry_processor_answers_from_its_verified_bytes(places):
    data = toy.artifact()
    get = _Get(data)
    held = _open(places, [toy.declaration(data)], get)
    assert held.units("Ab c", "en-us") == ["A", "B", "C"]
    assert held.units("Ab", "en-gb") == ["A'", "B'"]
    assert held.count("Ab c", "en-us") == 3
    assert len(get.asked) == 1
    assert toy.NAME not in sys.modules and "reader" not in sys.modules
    assert f"_nbxp_{toy.declaration(data)['sha256'][:16]}.reader" in sys.modules


def test_a_variant_the_declaration_does_not_list_is_refused(places):
    from neurobrix.processors.messages import ProcessorRefused
    data = toy.artifact()
    held = _open(places, [toy.declaration(data)], _Get(data))
    with pytest.raises(ProcessorRefused, match=r'does not read "fr-fr"\. It reads: en-us, en-gb\.$'):
        held.units("a", "fr-fr")


@pytest.mark.parametrize("over, sentence", [
    ({"interface": 2}, "needs a newer NeuroBrix to run its toy letters sub-component"),
    ({"sources": ["elsewhere"]}, "needs a newer NeuroBrix to run its toy letters sub-component"),
    ({"components": {"tagger": "absent_part"}}, "needs the model's absent_part part"),
])
def test_what_the_engine_cannot_honour_is_refused_before_a_byte_is_asked(places, over, sentence):
    from neurobrix.processors.messages import ProcessorRefused
    data = toy.artifact()
    get = _Get(data)
    with pytest.raises(ProcessorRefused, match=sentence):
        _open(places, [toy.declaration(data, **over)], get)
    assert get.asked == []


@pytest.mark.parametrize("over, said", [
    ({"sha256": "ABC"}, "'sha256' that is not 64 lower-case hex digits"),
    ({"file": "../x.nbxp"}, "which is not a bare file name"),
    ({"size": 0}, "no positive integer 'size'"),
    ({"variants": []}, "no non-empty list of texts 'variants'"),
    ({"uses": {"unknown_words": "nobody"}}, "uses nobody, which it does not declare"),
])
def test_a_declaration_of_the_wrong_shape_is_refused_by_name(over, said):
    from neurobrix.processors.declaration import DeclarationRefused, declared
    with pytest.raises(DeclarationRefused, match=said):
        declared({"processors": [toy.declaration(toy.artifact(), **over)]})


def test_no_key_is_no_processor():
    from neurobrix.processors.declaration import declared
    assert declared({"model_name": "x"}) == []
