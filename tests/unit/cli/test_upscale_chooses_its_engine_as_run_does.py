"""`neurobrix upscale` chooses its engine exactly as `neurobrix run` does (cli/engine_choice.py).

On a host with no GPU, `neurobrix upscale --model xinntao/Real-ESRGAN-RealESRGAN_x4plus ...` refused with
"no hardware profile with a device; the Triton branch's fp64 is read from its vendor profile", with or without
`--mode compiled`, while `neurobrix run --input-image` answered the same model (2026-11-13, a real 0.6.14 run): upscale
had its own `--mode` (default `compiled` whatever was installed) and made the Triton branch's hardware declaration for
every mode. These tests hold the one rule: the same flags on both parsers, the same choice from the same flags, and the
Triton declaration never made for the PyTorch engine.
"""
import logging

import pytest

from neurobrix.cli import create_parser
from neurobrix.cli.engine_choice import choose_engine, declare_for

_FLAGS = ("compiled", "sequential", "triton", "triton_sequential")


def _parse(argv):
    return create_parser().parse_args(argv)


def _run(*flags):
    return _parse(["run", "--model", "m", "--prompt", "p", *flags])


def _upscale(*flags):
    return _parse(["upscale", "--model", "m", "--input", "a.png", "--output", "b.png", *flags])


def test_upscale_takes_the_engine_flags_run_takes():
    for flag in ("--compiled", "--sequential", "--triton", "--triton-sequential"):
        r, u = _run(flag), _upscale(flag)
        assert [getattr(r, d) for d in _FLAGS] == [getattr(u, d) for d in _FLAGS], flag


def test_with_no_flag_upscale_and_run_choose_the_same_engine(monkeypatch):
    # the PyTorch-branch answer of default_mode() (torch present), whatever this machine has installed
    monkeypatch.setattr("neurobrix.core.runtime.engines.default_mode",
                        lambda: ("compiled", "engine: compiled (PyTorch installed; --triton runs the Triton engine)"))
    monkeypatch.setattr("neurobrix.core.runtime.engines.require_engine", lambda mode: None)
    log = logging.getLogger("t")
    assert choose_engine(_upscale(), log) == choose_engine(_run(), log) == "compiled"


def test_mode_is_the_flag_and_two_choices_are_refused(monkeypatch):
    from neurobrix.cli.commands.upscale import _mode_as_flag
    monkeypatch.setattr("neurobrix.core.runtime.engines.require_engine", lambda mode: None)
    log = logging.getLogger("t")
    a = _upscale("--mode", "sequential")
    _mode_as_flag(a)
    assert choose_engine(a, log) == "sequential"
    b = _upscale("--mode", "sequential", "--compiled")
    _mode_as_flag(b)
    assert choose_engine(b, log) is None              # two engines asked: refused, as two flags are on `run`


def test_the_pytorch_engine_never_makes_the_triton_declaration(monkeypatch):
    """A CPU host's profile has no device: the Triton declaration refuses it. The compiled engine must not ask."""
    import neurobrix.triton.dtype as tdt

    def _cpu_host(_profile):
        raise RuntimeError("ZERO FALLBACK: no hardware profile with a device")

    monkeypatch.setattr(tdt, "declare_hardware_profile", _cpu_host)
    declare_for("compiled", object())                 # no declaration, no refusal
    with pytest.raises(RuntimeError, match="no hardware profile with a device"):
        declare_for("triton", object())               # the gate is real: a Triton mode does declare
