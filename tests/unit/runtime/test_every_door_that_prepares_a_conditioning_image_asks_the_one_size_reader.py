"""Every door that prepares a conditioning image asks the one size reader (`core/runtime/size_policy.py`): cli run's
plan (`_request_image_hw`) and its image fit (handed the size cmd_run resolved, not the raw --height/--width), the
daemon and the plan's VLM pricing (both through `prepare_image_inputs`), and the executor. Read from the source, as the
neighbouring door tests do.

What would turn it red: a door that fits the image at the raw request size again, B4's `declared_condition_size` or
the executor's `_conditioning_image_size` back (the image's own size as the run's), the LANCZOS default back in the
DSP, or a second rounding formula somewhere else."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def _read(rel):
    return (SRC / rel).read_text()


def test_run_fits_the_image_at_the_size_it_resolved():
    tree = ast.parse(_read("cli/commands/run.py"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "prepare_image_inputs"]
    assert calls
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw["height"], ast.Name) and kw["height"].id == "height", ast.dump(kw["height"])
        assert isinstance(kw["width"], ast.Name) and kw["width"].id == "width", ast.dump(kw["width"])


def test_run_plans_a_conditioning_image_through_the_reader():
    run = _read("cli/commands/run.py")
    helper = run.split("def _request_image_hw(", 1)[1].split("\ndef ", 1)[0]
    assert "conditioning_image_size(" in helper and "is_conditioning_image(" in helper
    assert "_img_hw = _request_image_hw(args, cache_path)" in run


def test_the_image_door_reads_the_size_and_the_fit_from_the_container():
    ip = _read("core/module/vision/input_processor.py")
    door = ip.split("def prepare_image_inputs(", 1)[1].split("\ndef ", 1)[0]
    assert "conditioning_image_size(cache_path, image_path, height, width)" in door and "image_fit(fit" in door
    assert "declared_condition_size" not in ip
    assert "resolve_size(" in ip.split("def conditioning_image_size(", 1)[1].split("\ndef ", 1)[0]


def test_the_daemon_and_the_plan_go_through_the_same_door():
    assert "prepare_image_inputs(" in _read("serving/engine.py")
    for branch in ("pytorch", "triton"):                      # Prism's flow bindings, one twin per branch
        assert "prepare_image_inputs(" in _read(f"{branch}/prism/flow_bindings.py"), branch


def test_the_executor_no_longer_reads_the_run_size_off_the_image():
    ex = _read("core/runtime/executor.py")
    assert "_conditioning_image_size" not in ex
    assert "resolve_size(" in ex.split("def _conditioning_run_size(", 1)[1].split("\n    def ", 1)[0]


def test_the_dsp_has_no_default_resample_and_one_rounding_formula_exists():
    dsp = _read("core/module/vision/image_dsp.py")
    fit = dsp.split("def i2v_vae_condition_np(", 1)[1].split("\ndef ", 1)[0]
    assert "LANCZOS" not in fit and "img.height" not in fit and "img.width" not in fit
    hits = [p for p in SRC.rglob("*.py") if "max_area * ar" in p.read_text()]
    assert [p.name for p in hits] == ["size_policy.py"], hits
