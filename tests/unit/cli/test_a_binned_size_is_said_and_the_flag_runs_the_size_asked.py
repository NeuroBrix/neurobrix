"""Resolution binning is the vendor's DEFAULT, said, never a lock (orders 23/11/2026, the Sana 4K row 123: asked
1024x1024, the shipped container's recorded binning generated at 4096x4096 and resized back, with no word).

What this test does if the code is wrong:
1. a request binning resizes is planned without the one launch line that says it ("resolution binning (vendor
   default): 1024x1024 asked, generated at 4096x4096, resized back");
2. `--no-resolution-binning` does not reach the executor's `bin_request` (the run would still generate at the bin),
   or the plan does not size the request as asked;
3. `--set global.use_resolution_binning=false` is read by the run but not by the plan.

Run: python -m pytest tests/unit/cli/test_a_binned_size_is_said_and_the_flag_runs_the_size_asked.py
"""
import json
import logging

from neurobrix.cli import create_parser
from neurobrix.cli.commands import run as R
from neurobrix.core.runtime.resolution.resolution_binning import REQUEST_FLAG, bin_request, binning_line

#: the shape of what the shipped Sana_1600M_4Kpx container records (topology.json, flow.resolution_binning)
TOPOLOGY = {"flow": {"resolution_binning": {
    "default": True, "source": "SanaPipeline.__call__(use_resolution_binning)",
    "bins": {"0.5": [2896, 5792], "1.0": [4096, 4096], "2.0": [5792, 2896]},
    "classify": "nearest_ratio", "restore": "cover_resize_center_crop",
    "interpolate": {"mode": "bilinear", "align_corners": False}}}}


def _args(*extra):
    return create_parser().parse_args(["run", "--model", "M", "--prompt", "p", "--height", "1024", "--width", "1024",
                                       *extra])


def test_the_binned_size_is_said_in_one_line():
    merged = {"height": 1024, "width": 1024}
    line = binning_line(bin_request(TOPOLOGY, merged, None))
    assert line.startswith("resolution binning (vendor default): 1024x1024 asked, generated at 4096x4096, resized back")
    assert merged == {"height": 4096, "width": 4096}
    assert binning_line(bin_request(TOPOLOGY, {"height": 4096, "width": 4096}, None)) is None   # the bin itself
    assert binning_line(None) is None


def test_the_flag_reaches_bin_request_and_the_size_asked_runs():
    args = _args("--no-resolution-binning")
    assert R.binning_asked(args) is False
    inputs = R.binning_inputs(args)
    assert inputs == {REQUEST_FLAG: False} and REQUEST_FLAG == "global.use_resolution_binning"
    merged = {"height": 1024, "width": 1024}
    assert bin_request(TOPOLOGY, merged, inputs.get(REQUEST_FLAG)) is None
    assert merged == {"height": 1024, "width": 1024}
    plain = _args()
    assert R.binning_asked(plain) is None and R.binning_inputs(plain) == {}


def test_the_plan_reads_the_set_spelling_too():
    assert R.binning_asked(_args("--set", "global.use_resolution_binning=false")) is False
    assert R.binning_asked(_args("--set", "global.use_resolution_binning=true")) is True


def test_the_plan_says_the_line_once_at_launch(tmp_path, caplog):
    (tmp_path / "topology.json").write_text(json.dumps(TOPOLOGY))
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps({"height": 4096, "width": 4096}))
    with caplog.at_level(logging.INFO):
        try:
            R.request_input_config(_args(), {"family": "image"}, "image", tmp_path)
        except Exception:
            pass                                   # the rest of the plan needs a whole container: the line is said first
    said = [r.getMessage() for r in caplog.records if "resolution binning" in r.getMessage()]
    assert said == [binning_line(bin_request(TOPOLOGY, {"height": 1024, "width": 1024}, None))]
