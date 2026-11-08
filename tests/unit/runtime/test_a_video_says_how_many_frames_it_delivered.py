"""A video that carries fewer frames than the request asked for says so.

A VAE with temporal stride r represents r*k+1 frames, so `--num-frames 8` on a
stride-4 model becomes 2 latent frames and comes back as 5. The container's
shape contract does that arithmetic correctly and silently; the file then
answers a question about eight frames with five. Measured 2026-09-16 (vitrine,
Wan2.1-T2V-1.3B: five frames written, nothing said).

Injection: the notice removed → the first test read an empty log — RED."""
from types import SimpleNamespace

from neurobrix.core.runtime.output_dispatch import say_frames_delivered


def _executor(asked):
    return SimpleNamespace(variable_resolver=SimpleNamespace(resolved={"global.num_frames": asked}))


def test_a_short_delivery_is_named_with_both_counts(engine_log):
    say_frames_delivered(SimpleNamespace(defaults={}), _executor(8), 5)
    out = engine_log()
    assert "5 frames written" in out and "8 asked" in out and "temporal grid" in out


def test_an_exact_delivery_says_nothing(engine_log):
    say_frames_delivered(SimpleNamespace(defaults={}), _executor(9), 9)
    assert engine_log() == ""


def test_the_container_default_answers_when_the_request_carries_none(engine_log):
    say_frames_delivered(SimpleNamespace(defaults={"num_frames": 81}),
                         SimpleNamespace(variable_resolver=SimpleNamespace(resolved={})), 5)
    assert "81 asked" in engine_log()


def test_nothing_to_compare_says_nothing(engine_log):
    say_frames_delivered(SimpleNamespace(defaults={}), SimpleNamespace(variable_resolver=None), 5)
    assert engine_log() == ""
