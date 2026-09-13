"""`--explain-plan`: the plan a user can read, rendered from the plan object.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_explain_plan.py
"""
from neurobrix.core.prism.solver import ComponentAllocation, ComponentMemory, ExecutionPlan, explain_plan


def _plan(**kw):
    comps = {"transformer": ComponentAllocation(name="transformer", devices=["cuda:2", "cuda:3"], dtype="float16",
                                                memory_mb=30000, architecture="dit", vendor="x", sharded=True),
             "vae": ComponentAllocation(name="vae", devices=["cuda:2"], dtype="float16", memory_mb=500,
                                        architecture="vae", vendor="x")}
    mem = {"transformer": ComponentMemory("transformer", 28 * 2**30, 2 * 2**30, 2**28, peak_op_uid="aten.bmm::17",
                                          activation_profiled=True),
           "vae": ComponentMemory("vae", 2**29, 2**28, 2**26)}
    base = dict(components=comps, target_dtype="float16", total_memory_mb=31000, strategy="weight_sharding",
                component_memory=mem, selection_reason="weight_sharding scored 700 ahead of zero3",
                candidates=[("weight_sharding", 700.0), ("zero3", 100.0)],
                rejected=[("single_gpu", 1000.0, "KV cache does not fit: 3 GB needed, 1 GB left")])
    base.update(kw)
    return ExecutionPlan(**base)


def test_the_rendering_names_the_choice_the_reason_the_field_and_the_refusals():
    text = explain_plan(_plan())
    assert "strategy        weight_sharding" in text
    assert "weight_sharding scored 700 ahead of zero3" in text
    assert "weight_sharding=700" in text and "zero3=100" in text
    assert "refused         single_gpu (scored 1000): KV cache does not fit" in text
    assert "transformer          -> cuda:2, cuda:3  sharded" in text
    assert "peak at aten.bmm::17" in text
    assert "[activations estimated, not profiled]" in text.split("vae")[1], "an unprofiled estimate says so"


def test_a_plan_without_a_reason_or_candidates_says_so_rather_than_reading_as_explained():
    text = explain_plan(_plan(selection_reason="", candidates=[], rejected=[]))
    assert "no reason recorded" in text and "candidates      none recorded" in text
    assert "tiling          none planned" in text


def test_a_planned_component_tiling_is_printed():
    text = explain_plan(_plan(component_tiling={"vae": {"axis": "temporal", "tiles": 24, "tile_frames": 8}}))
    assert "component tiling vae: {'axis': 'temporal', 'tiles': 24, 'tile_frames': 8}" in text


def test_the_flag_is_on_the_run_command():
    import sys
    from neurobrix.cli import main
    # argparse exits 2 on an unknown flag; a parse of `run --explain-plan --model x`
    # that reaches the command means the flag exists. We stop before the command
    # runs by asking for --help of the same subcommand.
    import subprocess
    r = subprocess.run([sys.executable, "-c", "from neurobrix.cli import main; main()", "run", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--explain-plan" in r.stdout
