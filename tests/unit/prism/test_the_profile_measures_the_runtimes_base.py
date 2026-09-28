"""The hardware profile carries what each engine's device work adds, MEASURED on the machine per engine,
and a plan's host estimate adds it to the planning process's own resident memory — or says it is
unmeasured, never a number nobody measured.

The base is not a constant of the code (the owner's rule, 2026-09-27 14:27: everything data-driven,
carried by the container and the hardware profiles). On this rack, from the planning state (the CLI
imported, 33 MB), the compiled engine adds about 800 MB (torch, a CUDA context, cuBLAS, cuDNN, SDPA) and
the Triton engine about 360 MB (a kernel compiled and launched). Before this branch the profile has no
`runtime_base_mb` and autodetect measures nothing: these fail.
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


def test_a_probe_started_by_a_large_process_measures_itself_not_its_parent():
    # Linux carries ru_maxrss across fork and exec: under a 2.4 GB test process every probe read its
    # parent's size at both ends and the base came out 0 (2026-09-27). The kernel's per-image counters
    # (VmRSS / VmHWM) start fresh at exec; this holds 1.5 GB and asks for a positive base below it.
    from neurobrix.core.prism import host_footprint as H
    held = bytearray(1536 << 20)
    for i in range(0, len(held), 4096):
        held[i] = 1
    base = A._measure_runtime_base_mb()
    assert all(0 < v < 1536 for v in base.values()), base
    assert H.resident_bytes_now() >= 1536 << 20
