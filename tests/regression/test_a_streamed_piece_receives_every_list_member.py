"""A streamed piece receives every tensor an op names inside a LIST argument.

The Mac's two Flex.1-alpha rows (df2588e7), triton-sequential, 4 096 and 16 384 MB:

    [triton-sequential] Failed at aten.mul::24 (aten::mul): ValueError: Cannot broadcast
    (1, 24, 512, 128) and (1, 1, 4608, 128)

Reproduced here on the landed engine (96288dfc) with the pinned Mac plan, and in the PIECES only —
the transformer run whole is fine. Flex's joint attention concatenates its 512 text queries with
its 4 096 image queries (`aten.cat::19`, a `tensor_tuple` argument) before the RoPE `mul`. When
the image half is a SEAM (produced by the previous piece), the piece holds it as
`input::aten.mul::97::out_0`; `build_segment_graph` aliased `input_tensor_ids` and a plain
`tensor` argument, but not an id inside a `tensor_tuple`, so the list still named the raw id.
Triton-sequential then resolved it to None and handed the kernel a shorter list — `cat` joined the
text queries with nothing. Two defects: the aliasing walk (now the one `rewire_arg` the triton
sequence also uses) and a list argument that let a missing element through (now refused).

Each cell runs the transformer WHOLE and in the pieces the Mac's plan cuts, same inputs, and
requires the output bit-identical, in both triton engines, at both of the Mac's rungs. The
inputs are the traced ones (`SEQ trace`): Flex's trace names both token axes `seq_len` and a
coordinate width of 3 `seq_len` as well, so no request can be bound by name — a Forge
symbol-naming defect (owed-proofs), which is also why this cell runs at the trace size only.
The binding gate (`tests/unit/prism/test_every_streamed_piece_binds_every_symbol_it_uses.py`)
checks the aliasing itself for every piece at three sizes.

Needs a card: `CUDA_VISIBLE_DEVICES=<n>`. Without one this FAILS.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "streamed_component_vs_whole.py"
CELLS = ([(m, r) for m in ("triton_sequential", "triton") for r in (4096, 16384)]
         # R30: the builder's fix changes the torch engines' pieces too (their tensor lists
         # resolve through the TensorResolver, which already refused a missing id).
         + [(m, 4096) for m in ("sequential", "compiled")])


@pytest.mark.slow
@pytest.mark.parametrize("mode,rung", CELLS, ids=[f"{m}-{r}" for m, r in CELLS])
def test_flex_pieces_compute_what_the_whole_transformer_computes(mode, rung, tmp_path):
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.fail("this cell executes the component on a card: set CUDA_VISIBLE_DEVICES=<n>.",
                    pytrace=False)
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [os.environ["NEUROBRIX_PYTHON"], str(TOOL), "Flex.1-alpha", "transformer", mode, "trace",
         "trace", "18186", str(rung), str(out), "1024", "1024"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert report["pieces"] >= 2, report
    for name, o in report["outputs"].items():
        assert not o.get("missing_from_pieces"), (name, o)
        assert o["whole_vs_whole_max_abs_diff"] == 0.0, (name, o)
        assert o["bit_identical"], f"{mode} at {rung} MB: pieces differ from whole: {name} {o}"
