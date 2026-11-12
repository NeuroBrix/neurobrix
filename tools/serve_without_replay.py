#!/usr/bin/env python3
"""serve_without_replay.py — the engine's entry point with its frozen-plan replay OFF, in this process only: the gate
serve cell's ORACLE for a conversation (tools/gate_serve_cell.py). Never a served configuration: the engine has no
switch for its replay, by design (triton/replay.py `ENABLED`); this instrument sets the module's flag before the entry
point runs. The replay is built to give the direct path's bytes, so the same requests must answer the same texts —
a later turn that does not is a replay defect (2026-11-12: a served MoE answered turns 3-5 empty).

    serve_without_replay.py <the arguments of `python -m neurobrix`>
"""
import runpy
import sys

import neurobrix.triton.replay as _replay

_replay.ENABLED = False
sys.argv = ["neurobrix"] + sys.argv[1:]
runpy.run_module("neurobrix", run_name="__main__")
