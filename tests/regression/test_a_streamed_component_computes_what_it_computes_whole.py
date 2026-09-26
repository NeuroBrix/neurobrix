"""A component streamed in the pieces Prism planned computes, on the card, exactly what it computes
whole — the executed half of the streaming gate (the bound half is
`tests/unit/prism/test_every_streamed_piece_binds_every_symbol_it_uses.py`).

The T5 text encoder of the PixArt-XL-2-1024-MS container (the second trace of the same repository, `PixArt-XL-1024`, went on 2026-09-26; its cells with it) — formerly `PixArt-XL-1024`
(the 2026-05-20 build the Mac rendered) and `PixArt-XL-2-1024-MS` (the 2026-09-21 retrace, the name
Hugging Face uses) — planned under the Mac's own reading (11 198 MB free) with the 8 192 MB rung
imposed, at the Mac's 2048x1024 request: the plan that died in piece 3 on the Mac (8e786e70). Each
cell loads the encoder once WHOLE and once through the real `LayerStreamingStrategy` over the
planned pieces, same inputs, and requires the outputs BIT-identical.

Seen failing on main (fa6e13d2 engine, this harness, card 2):
  PixArt-XL-1024       triton_sequential b1  UnboundSymbolError: symbol 's1' (seq_len, binds from
                                             input::attention_mask::dim_1) ... Bound: ['s3']  — the Mac's
  PixArt-XL-2-1024-MS  triton_sequential b1  pieces run, rel L2 0.42 % from whole (seam dtype)

Two defects this sees, both measured 2026-09-24 on card 3:
  * a symbol a piece uses bound from nothing it receives — UnboundSymbolError, the Mac's failure;
  * a seam tensor cast to the compute dtype at the piece's entry, narrowing an fp32 island to bf16:
    every piece then diverged 0.15-0.25 % from the same ops run whole (rel L2 0.46 % at the output),
    while the whole component run twice was bit-identical.

Sizes (rule 1): batch at its trace value 1, at 2, and far at 8, in every engine where the WHOLE
component runs at that size. Two holes, named because the whole component fails there before any
piece exists (not streaming defects; a cell there would judge the whole, not the pieces):
  * compiled mode, batch 2 and 8, both containers: the whole T5 fails
    (`aten.add::7 ... tensor a (128) must match tensor b (64)`) — a compiled-engine batch defect;
  * PixArt-XL-1024, torch sequential, batch 2 and 8: the whole T5 fails on the 2026-05-20 build's
    mis-traced dims (`s3*4096` for an extent of 120) — the retrace does not carry them.
Sequence length is NOT varied: PixArt-XL-1024 froze its T5 relative-position bias at
[1, 64, 120, 120] at trace (named, a tracer defect of that build).

Needs a card: `CUDA_VISIBLE_DEVICES=<n>`. Without one this FAILS with that sentence — a streaming
gate that skipped is one that never ran.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "streamed_component_vs_whole.py"
CELLS = ([("PixArt-XL-2-1024-MS", m, b) for m in ("triton_sequential", "triton", "sequential")
          for b in (1, 2, 8)]
         + [("PixArt-XL-2-1024-MS", "compiled", 1)]
)


@pytest.mark.slow
@pytest.mark.parametrize("model,mode,batch", CELLS, ids=[f"{m}-{e}-b{b}" for m, e, b in CELLS])
def test_the_pieces_compute_what_the_whole_computes(model, mode, batch, tmp_path):
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.fail("this cell executes the component on a card: set CUDA_VISIBLE_DEVICES=<n>. A "
                    "streaming gate that did not run is not a streaming gate that passed.",
                    pytrace=False)
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [os.environ["NEUROBRIX_PYTHON"], str(TOOL), model, "text_encoder", mode, str(batch), "120",
         "11198", "8192", str(out), "2048", "1024"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=2400)
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert report["pieces"] >= 2, report
    for name, o in report["outputs"].items():
        assert not o.get("missing_from_pieces"), (name, o)
        assert o["whole_vs_whole_max_abs_diff"] == 0.0, (
            f"{name}: the WHOLE component is not reproducible run to run ({o}); a difference "
            f"between whole and pieces could not be attributed")
        assert o["bit_identical"], (f"{mode} batch {batch}: the pieces compute something else than "
                                    f"the whole component: {name} {o}")
