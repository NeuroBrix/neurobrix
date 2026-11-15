"""On a pipe (CI, a daemon's log), a model's first run says two lines about its processor, once each, and draws no
bar; its later runs say nothing.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
from tests.unit.processors import toy
from tests.unit.processors.conftest import env_of, neurobrix


def test_the_first_run_says_two_lines_and_the_next_one_nothing(places, host):
    data = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(data)])
    env = env_of(places, host.url, toy.dead_url())
    first = neurobrix("processor", "fetch", toy.hub_name("toy"), env=env)
    assert first.returncode == 0, first.stderr
    assert first.stderr.splitlines() == [
        f"This model needs the {toy.LABEL} sub-component. Downloading it once (1 KB)...",
        f"{toy.LABEL} ready, checksum verified"]
    assert "\r" not in first.stderr and first.stdout == ""
    second = neurobrix("processor", "fetch", toy.hub_name("toy"), env=env)
    assert (second.returncode, second.stdout, second.stderr) == (0, "", "")
    assert host.gets == {toy.FILE: 1}
