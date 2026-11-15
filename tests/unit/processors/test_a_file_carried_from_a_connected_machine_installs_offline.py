"""The offline path: `processor fetch <model> --to <folder>` on a connected machine writes every artifact the model
declares; `processor install <file> --for <model>` on a machine with no network puts it in the cache, checked at
once, and the model's next run is silent. Without `--for` the file is stored and checked at the first run.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
from tests.unit.processors import toy
from tests.unit.processors.conftest import env_of, neurobrix


def test_fetched_to_a_folder_then_installed_on_a_machine_with_no_network(places, host, tmp_path):
    data = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(data)])
    folder = tmp_path / "carry"
    connected = neurobrix("processor", "fetch", toy.hub_name("toy"), "--to", str(folder),
                          env=env_of(places, host.url, toy.dead_url()))
    assert connected.returncode == 0, connected.stderr
    assert connected.stdout.splitlines() == [str(folder / toy.FILE)]
    assert not (places["NEUROBRIX_CACHE"] / "processors" / toy.FILE).exists(), "--to wrote into the cache"
    offline = env_of(places, toy.dead_url(), toy.dead_url())
    r = neurobrix("processor", "install", str(folder / toy.FILE), "--for", toy.hub_name("toy"), env=offline)
    assert r.returncode == 0 and r.stderr.splitlines() == [f"{toy.LABEL} ready, checksum verified"], r.stderr
    again = neurobrix("processor", "fetch", toy.hub_name("toy"), env=offline)
    assert (again.returncode, again.stderr) == (0, "")
    listed = neurobrix("processor", "list", env=offline)
    assert toy.FILE in listed.stdout and toy.declaration(data)["sha256"] in listed.stdout


def test_installed_without_a_model_it_is_checked_at_the_first_run(places, tmp_path):
    data = toy.artifact()
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(data)])
    carried = tmp_path / toy.FILE
    carried.write_bytes(data)
    offline = env_of(places, toy.dead_url(), toy.dead_url())
    r = neurobrix("processor", "install", str(carried), env=offline)
    assert (r.returncode, r.stderr) == (0, "")
    again = neurobrix("processor", "fetch", toy.hub_name("toy"), env=offline)
    assert (again.returncode, again.stderr) == (0, "")
