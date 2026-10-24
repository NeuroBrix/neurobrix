"""R33: Prism prices a guided speech LM's decode cache (`lm_facts.decode_sequences` for a tts_llm flow reads the CFG
cascade) without loading torch. The cascade sat in `core/cfg/engine.py`, which imports torch, so a `--triton`
chatterbox run loaded torch at its plan and the shared unload then synchronised an NBX device through `torch.cuda`
("Torch not compiled with CUDA enabled", Mac 2026-10-23). What this test does if the code were wrong: torch is in
sys.modules after the call -> red."""
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"

PROBE = r"""
import sys, json
from neurobrix.core.runtime.lm_facts import decode_sequences
topo = {"flow": {"type": "tts_llm"}, "extracted_values": {"_global": {"guidance_scale": 1.5}}}
n = decode_sequences(topo, {}, None, family="audio")
print(json.dumps({"n": n, "torch": "torch" in sys.modules}))
"""


def test_the_cascade_reads_without_torch():
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                       env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["torch"] is False, "the guided decode's pricing loaded torch"
    assert out["n"] in (1, 2)


def test_the_engine_and_prism_read_the_same_cascade():
    sys.path.insert(0, str(SRC))
    from neurobrix.core import guidance_cascade as G
    from neurobrix.core.cfg import engine as E
    assert E.guidance_cascade is G.guidance_cascade
