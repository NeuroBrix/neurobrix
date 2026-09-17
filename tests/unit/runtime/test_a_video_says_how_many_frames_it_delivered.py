"""A video that carries fewer frames than the request asked for says so.

A VAE with temporal stride r represents r*k+1 frames, so `--num-frames 8` on a
stride-4 model becomes 2 latent frames and comes back as 5. The container's
shape contract does that arithmetic correctly and silently; the file then
answers a question about eight frames with five. Measured 2026-09-16 (vitrine,
Wan2.1-T2V-1.3B: five frames written, nothing said).

Injection: the notice removed → the first test read an empty stdout — RED."""
from types import SimpleNamespace

from neurobrix.core.runtime.output_dispatch import say_frames_delivered


def _executor(asked):
    return SimpleNamespace(variable_resolver=SimpleNamespace(resolved={"global.num_frames": asked}))


def test_a_short_delivery_is_named_with_both_counts(capsys):
    say_frames_delivered(SimpleNamespace(defaults={}), _executor(8), 5)
    out = capsys.readouterr().out
    assert "5 frames written" in out and "8 asked" in out and "temporal grid" in out


def test_an_exact_delivery_says_nothing(capsys):
    say_frames_delivered(SimpleNamespace(defaults={}), _executor(9), 9)
    assert capsys.readouterr().out == ""


def test_the_container_default_answers_when_the_request_carries_none(capsys):
    say_frames_delivered(SimpleNamespace(defaults={"num_frames": 81}),
                         SimpleNamespace(variable_resolver=SimpleNamespace(resolved={})), 5)
    assert "81 asked" in capsys.readouterr().out


def test_nothing_to_compare_says_nothing(capsys):
    say_frames_delivered(SimpleNamespace(defaults={}), SimpleNamespace(variable_resolver=None), 5)
    assert capsys.readouterr().out == ""
