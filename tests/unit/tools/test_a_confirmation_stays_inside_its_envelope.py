"""A confirmation request moves up to the smallest size the vendor states the model supports.

The supervisor, 2026-09-29 15:20 (reading (b)): the half size is kept when it is inside the
model's documented envelope; under the stated minimum it moves to that minimum; a vendor that
states none keeps the half size. The envelope is a registry value the container carries.

What would this file do if the code were wrong? The envelope ignored -> the 160x352 case keeps
its size, RED; the minimum applied above it too -> the inside case moves, RED; a `min: null`
read as a size -> TypeError, RED.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import trace_request as T  # noqa: E402


def _topo(env):
    return {"extracted_values": {"_global": {"envelope": env} if env is not None else {}}}


def test_a_half_size_under_the_stated_minimum_moves_up_to_it():
    stated = {"min": {"height": 480, "width": 720}, "source": ["no support for other resolutions"]}
    assert T.inside_envelope((160, 352), _topo(stated)) == (480, 720)


def test_a_half_size_inside_the_envelope_is_kept():
    stated = {"min": {"height": 192, "width": 336}, "source": ["256px"]}
    assert T.inside_envelope((288, 512), _topo(stated)) == (288, 512)
    assert T.inside_envelope((336, 192), _topo(stated)) == (336, 192)     # orientation-free


def test_no_stated_minimum_or_no_value_keeps_the_half_size():
    assert T.inside_envelope((160, 416), _topo({"min": None, "source": ["states none"]})) == (160, 416)
    assert T.inside_envelope((160, 416), _topo(None)) == (160, 416)
