"""A device whose runtime keeps no peak counter (Apple's MPS answers only what its driver holds NOW) has its peak read
through the run by a sampler, so the native [Peak] of a Metal run is a figure, never "na" (rule 3; every native Metal
cell of 2026-10-27 wrote none). No card: the driver is a function here."""
import time

from neurobrix.core.device_utils import SampledDevicePeak


def test_the_sampler_keeps_the_largest_reading_of_the_span():
    seq = iter([100, 300, 900, 200] + [50] * 10_000)
    s = SampledDevicePeak(lambda: next(seq), period_s=0.0005).start()
    time.sleep(0.05)
    assert s.stop() == 900


def test_the_native_peak_line_reads_the_sampler_where_cuda_is_absent():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "cli" / "commands" / "run.py").read_text()
    assert "sampled_device_peak()" in src and "_sampled_peak.stop()" in src
