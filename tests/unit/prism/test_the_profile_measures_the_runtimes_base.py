"""The hardware profile carries the runtime's own base memory, MEASURED on the machine per engine, and
a plan's host estimate starts from it — or says it is unmeasured, never a number nobody measured.

The base is not a constant of the code (the owner's rule, 2026-09-27 14:27: everything data-driven,
carried by the container and the hardware profiles): on this rack the compiled runtime measured 690 MB
with a CUDA context and 557 MB without, the Triton runtime 226 MB and 114 MB. Before this branch the
profile has no `runtime_base_mb` and autodetect measures nothing: these fail.
"""
from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism.cpu_config import CPUConfig


def test_both_engines_are_measured_here():
    base = A._measure_runtime_base_mb()
    assert set(base) == {"compiled", "triton"} and all(isinstance(v, int) and v > 0 for v in base.values())


def test_the_profile_carries_it_and_an_old_profile_reads_as_unmeasured():
    cpu = {"model": "x", "cores": 1, "threads": 1, "ram_mb": 1024, "architecture": "x86_64"}
    assert CPUConfig.from_yaml_dict(cpu).runtime_base_mb == {}
    assert CPUConfig.from_yaml_dict({**cpu, "runtime_base_mb": {"compiled": 690, "triton": 226}}).runtime_base_mb == {
        "compiled": 690, "triton": 226}


def test_the_solver_prices_the_profiles_base():
    src = (A.__file__.replace("autodetect.py", "solver.py"))
    text = open(src).read()
    assert 'getattr(profile.cpu, "runtime_base_mb", None)' in text and "plan.host_footprint = host_footprint(" in text
