"""The declared hosts are tried in their order; the second only when the first does not ANSWER (no connection, a
5xx). A first host that answers "not here" (404) is final: the sub-component is said unpublished where the model
says, and the second is not asked.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
from tests.unit.processors import toy
from tests.unit.processors.conftest import env_of, neurobrix


def _fetch(places, hub_url, hf_url):
    return neurobrix("processor", "fetch", toy.hub_name("toy"), env=env_of(places, hub_url, hf_url))


def test_an_unreachable_first_host_hands_over_to_the_second(places, host):
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(toy.artifact())])
    r = _fetch(places, toy.dead_url(), host.url)
    assert r.returncode == 0, r.stderr
    assert host.gets == {toy.FILE: 1}


def test_a_first_host_in_error_hands_over_to_the_second(places, host):
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(toy.artifact())])
    failing = toy.Host({}, status=503)
    try:
        r = _fetch(places, failing.url, host.url)
    finally:
        failing.close()
    assert r.returncode == 0, r.stderr
    assert failing.gets == {toy.FILE: 1} and host.gets == {toy.FILE: 1}


def test_a_first_host_that_answers_not_here_is_final(places, host):
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(toy.artifact())])
    empty = toy.Host({})
    try:
        r = _fetch(places, empty.url, host.url)
    finally:
        empty.close()
    assert r.returncode == 1
    assert r.stderr.splitlines()[-1] == (
        f"Error: This model needs the {toy.LABEL} sub-component ({toy.NAME} {toy.VERSION}), which is not published "
        f'where this model says. Update NeuroBrix, or install it from a file with "neurobrix processor install <file>".')
    assert host.gets == {}


def test_a_model_that_declares_only_the_second_host_never_asks_the_first(places, host):
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(toy.artifact(), sources=["hf"])])
    first = toy.Host({toy.FILE: toy.artifact()})
    try:
        r = _fetch(places, first.url, host.url)
    finally:
        first.close()
    assert r.returncode == 0, r.stderr
    assert first.gets == {} and host.gets == {toy.FILE: 1}
