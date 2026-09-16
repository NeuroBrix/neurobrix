"""A request without --seed runs under the declared default seed on the Triton
engine, as it does on the ATen branch — the CLI writes `global.seed: None`
into the request, and the default comes from the merged defaults (family 42
for video), not the container's own file. CogVideoX-2b drew a different
initial latent on every seedless run (fingerprint walk, 2026-09-16: first
differing op aten.view::2, the latent) while its 39 000 other ops agreed.

Injection: the old form `inputs.get("global.seed", pkg_defaults.get("seed"))`
restored → the first test read None — RED."""
from neurobrix.core.runtime.executor import RuntimeExecutor


def test_a_present_none_seed_falls_back_to_the_merged_default():
    assert RuntimeExecutor.run_seed({"global.seed": None}, {"seed": 42}) == 42


def test_an_explicit_seed_wins():
    assert RuntimeExecutor.run_seed({"global.seed": 7}, {"seed": 42}) == 7


def test_no_seed_anywhere_stays_unseeded_by_name():
    assert RuntimeExecutor.run_seed({"global.seed": None}, {}) is None
    assert RuntimeExecutor.run_seed({}, {}) is None
