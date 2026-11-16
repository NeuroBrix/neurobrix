"""A model that declares a processor, on a machine that reaches neither declared host and has nothing in its cache,
is refused in ONE sentence that names the sub-component and the two offline commands; nothing is written, no URL
and no stack of calls is shown.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
from tests.unit.processors import toy
from tests.unit.processors.conftest import env_of, neurobrix


def test_no_host_answers_and_the_refusal_names_the_offline_commands(places):
    data = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(data)])
    r = neurobrix("processor", "fetch", toy.hub_name("toy"), env=env_of(places, toy.dead_url(), toy.dead_url()))
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert r.stderr.splitlines() == [
        f"Error: This model needs the {toy.LABEL} sub-component and this machine could not reach the download. "
        f'On a connected machine run "neurobrix processor fetch {toy.hub_name("toy")} --to <folder>", copy '
        f'{toy.FILE} here, then run "neurobrix processor install {toy.FILE}".']
    assert "http" not in r.stderr and "Traceback" not in r.stderr
    assert not list((places["NEUROBRIX_CACHE"] / "processors").glob("*.nbxp"))
