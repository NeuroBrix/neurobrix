"""The host estimate prices what the output boundary holds while it saves the run's result.

Measured on card 2 (2026-09-27, phase-marked RSS): real-esrgan-x8's peak comes AFTER "Total execution",
+~330 MB in `output_dispatch.save_image` — its 3584x3584 RGB output held as float32 twice (the array and
the `* 255` temporary) plus the 8-bit image. The plan knows the output's size at the request (its graph
outputs, resolved at the request's symbols) and the family declares its output format; neither is a
number about a model. Before this branch neither the per-element cost nor the profiler's output count
exists: these fail.
"""
from types import SimpleNamespace as NS

from neurobrix.core.prism import host_footprint as H
from neurobrix.core.prism.profiler import ActivationProfiler
from neurobrix.core.runtime.output_dispatch import host_bytes_per_output_element

MB = 1 << 20


def test_each_save_path_is_priced_from_its_code():
    assert host_bytes_per_output_element("image") == 2 * 4 + 1      # float32 twice + uint8
    assert host_bytes_per_output_element("video") == 2 * 4 + 1
    assert host_bytes_per_output_element("tts") == 2 * 4 + 2        # waveform twice + PCM_16
    assert host_bytes_per_output_element("llm") == 0                # tokens


def test_a_mode_dependent_family_is_priced_at_its_dearest_format():
    assert host_bytes_per_output_element("multimodal") == max(
        host_bytes_per_output_element(f) for f in ("image", "tts", "llm"))


def test_the_profiler_counts_its_graph_outputs_at_the_binding():
    sym = {"type": "symbol", "id": "s1", "trace": 23}
    dag = {"ops": {}, "execution_order": [], "output_tensor_ids": ["t0"],
           "tensors": {"t0": {"shape": [1, 3, 23, 23], "dtype": "float16",
                              "symbolic_shape": {"dims": [1, 3, sym, sym], "concrete": [1, 3, 23, 23]}}}}
    assert ActivationProfiler(dag).estimate_peak_memory().output_elements == 3 * 23 * 23


def test_the_output_term_is_part_of_the_figure():
    plan = NS(components={"m": NS(device="cuda:0", dtype="float16", shard_map={})},
              component_memory={}, loading_mode="lazy")
    f = H.host_footprint(plan, {"m": {}}, {"m": {}}, "triton", 0, {"float16": 2}, lambda k: False,
                         output_bytes=346 * MB)
    assert f["output_bytes"] == 346 * MB and f["total_bytes"] == 346 * MB
    assert "output 346 MB" in H.summary(f)
