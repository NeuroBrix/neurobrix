"""`neurobrix calibrate` records the engine that RAN its census, chosen as `run` chooses it (qa 1274-1: with Triton the
default engine, calibrate with no flag ran Triton and labelled its record "compiled", the PyTorch reference's label,
saved where the reference census goes).

What this test does if the code is wrong:
1. with no engine flag the record is labelled as the PyTorch reference (no ":triton") while the run took Triton;
2. with --compiled the record is not the reference's (a suffix), or with --triton-sequential not its own.
No device, no model: the run and the census are stand-ins; the label is read from the record calibrate builds.
Run: python -m pytest tests/unit/cli/test_calibrate_records_the_engine_it_ran.py
"""
from types import SimpleNamespace

import pytest

from neurobrix.cli import create_parser
from neurobrix.cli.commands import calibrate as C


@pytest.fixture
def census(tmp_path, monkeypatch):
    from neurobrix.core.dtype import calibration as cal
    ran, built = [], []
    monkeypatch.setattr("neurobrix.cli.utils.resolve_model_arg", lambda m: m)
    monkeypatch.setattr("neurobrix.serving.client.DaemonClient.is_running", staticmethod(lambda: False))
    monkeypatch.setattr(C, "_identity_of", lambda m: ("llm", "a-model"))
    monkeypatch.setattr(C, "_apply_family_stimulus", lambda args, family: {"prompt": "p"})
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path)
    monkeypatch.setattr("neurobrix.core.runtime.output_dispatch.resolve_output_path", lambda *a: str(tmp_path / "o"))
    monkeypatch.setattr("neurobrix.core.runtime.engines.require_engine", lambda mode: None)
    monkeypatch.setattr("neurobrix.cli.engine_choice._triton_backend_present", lambda log: True)

    def run(args):
        from neurobrix.cli.engine_choice import choose_engine
        ran.append(choose_engine(args, C_log))          # the engine the run itself takes from the same request
        return 0
    C_log = __import__("logging").getLogger("t")
    monkeypatch.setattr("neurobrix.cli.commands.run.cmd_run", run)
    one = SimpleNamespace(dag={}, passes=1, signature="s", finalize=lambda: {}, non_finite_ops=lambda: [])
    monkeypatch.setattr(cal, "begin_calibration", lambda: {"lm": one})
    monkeypatch.setattr(cal, "end_calibration", lambda: None)

    def build(*a, **k):
        built.append(k["reference"])
        return SimpleNamespace(save=lambda p: None, max_abs={}, passes=1, non_finite=[])
    monkeypatch.setattr(cal.CalibrationRecord, "build", staticmethod(build))
    return SimpleNamespace(ran=ran, built=built)


def _calibrate(*flags):
    return C.cmd_calibrate(create_parser().parse_args(["calibrate", "--model", "m", *flags]))


@pytest.mark.parametrize("flags, engine", [((), "triton"), (("--compiled",), "compiled"),
                                           (("--triton-sequential",), "triton_sequential")])
def test_the_record_names_the_engine_that_ran(census, flags, engine):
    from neurobrix.core.config.loader import get_precision_calibration_policy
    reference = str(get_precision_calibration_policy()["reference"])
    assert _calibrate(*flags) == 0
    assert census.ran == [engine]                      # what the run took
    assert census.built == [reference if engine == "compiled" else f"{reference}:{engine}"]   # what the record says
